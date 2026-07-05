################################################################################
#                       skforecast.feature_selection                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import re
from itertools import chain
import warnings
import numpy as np
import pandas as pd
from ..utils import deepcopy_forecaster


def select_features(
    forecaster: object,
    selector: object,
    y: pd.Series | pd.DataFrame,
    exog: pd.Series | pd.DataFrame | None = None,
    select_only: str | list[str] | None = None,
    force_inclusion: list[str] | str | None = None,
    subsample: int | float = 0.5,
    random_state: int = 123,
    verbose: bool = True
) -> tuple[list[int], list[str], list[str], list[str]]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Three groups of features are
    evaluated: autoregressive features (lags and window features), exogenous
    features and calendar features. By default, the selection process is performed
    on the three sets of features at the same time, so that the most relevant
    autoregressive, exogenous and calendar features are selected. However, using
    the `select_only` argument, the selection process can focus only on one or more
    of these groups without taking into account the others. Therefore, all features
    in the remaining groups will remain in the model. It is also possible to force
    the inclusion of certain features in the final list of selected features using
    the `force_inclusion` parameter.

    If encoded, calendar features are evaluated at the encoded-column level (e.g. 
    `month_sin`, `month_cos`), but they are returned at the source-feature level 
    (e.g. `month`). A source calendar feature is kept whenever at least one of 
    its encoded columns is selected.

    Parameters
    ----------
    forecaster : ForecasterRecursive, ForecasterDirect
        Forecaster model. If forecaster is a ForecasterDirect, the
        selector will only be applied to the features of the first step.
    selector : object
        A feature selector from sklearn.feature_selection.
    y : pandas Series, pandas DataFrame
        Target time series to which the feature selection will be applied.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    select_only : str, list, default None
        Decide what type of features to include in the selection process.

        - If `'autoreg'` (or `['autoreg']`), only autoregressive features (lags
        and window features) are evaluated by the selector. All exogenous and
        calendar features are kept and returned in `selected_exog` and
        `selected_calendar_features`.
        - If `'exog'` (or `['exog']`), only exogenous features are evaluated. All
        autoregressive and calendar features are kept and returned in
        `selected_lags`, `selected_window_features` and
        `selected_calendar_features`.
        - If `'calendar'` (or `['calendar']`), only calendar features are
        evaluated. All autoregressive and exogenous features are kept and
        returned in `selected_lags`, `selected_window_features` and
        `selected_exog`.
        - If `list`, any combination of `'autoreg'`, `'exog'` and `'calendar'`.
        Only the groups listed are evaluated by the selector; the remaining
        groups are kept unchanged.
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default None
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.

        For calendar features, `force_inclusion` is matched against the encoded
        column names (e.g. `month_sin`); forcing any encoded column keeps its
        source calendar feature in `selected_calendar_features`.
    subsample : int, float, default 0.5
        Proportion of records to use for feature selection.
    random_state : int, default 123
        Sets a seed for the random subsample so that the subsampling process 
        is always deterministic.
    verbose : bool, default True
        Print information about feature selection process.

    Returns
    -------
    selected_lags : list
        List of selected lags.
    selected_window_features : list
        List of selected window features.
    selected_exog : list
        List of selected exogenous features.
    selected_calendar_features : list
        List of selected calendar features (source-level names, without the
        encoding suffix). Empty list if the forecaster has no calendar features.

    """

    forecaster_name = type(forecaster).__name__
    valid_forecasters = ['ForecasterRecursive', 'ForecasterDirect']

    if forecaster_name not in valid_forecasters:
        raise TypeError(
            f"`forecaster` must be one of the following classes: {valid_forecasters}."
        )

    valid_select_only = ['autoreg', 'exog', 'calendar']
    if select_only is None:
        select_only_list = valid_select_only
    else:
        select_only_list = (
            [select_only] if isinstance(select_only, str) else select_only
        )
        if not isinstance(select_only_list, list):
            raise TypeError(
                "`select_only` must be a str, a list of str, or None."
            )
        if not set(select_only_list).issubset(valid_select_only):
            raise ValueError(
                "`select_only` must be one or more of the following values: "
                "'autoreg', 'exog', 'calendar', or None."
            )

    eval_autoreg = 'autoreg' in select_only_list
    eval_exog = 'exog' in select_only_list
    eval_calendar = 'calendar' in select_only_list

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy_forecaster(forecaster)
    forecaster.is_fitted = False
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
    if forecaster_name == 'ForecasterDirect':
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step          = 1,
                               X_train       = X_train,
                               y_train       = y_train,
                               remove_suffix = True
                           )
    
    lags_cols = []
    window_features_cols = []
    autoreg_cols = []
    if forecaster.lags is not None:
        lags_cols = forecaster.lags_names
        autoreg_cols.extend(lags_cols)
    if forecaster.window_features is not None:
        window_features_cols = forecaster.window_features_names
        autoreg_cols.extend(window_features_cols)

    # `create_train_X_y` fit-transforms `calendar_features`, so its encoded
    # output column names (e.g. `month_sin`) are available in `feature_names_out_`.
    # `calendar_features_names` holds the source-level feature names (e.g. `month`).
    calendar_cols = []
    calendar_features_names = []
    if forecaster.calendar_features is not None:
        calendar_cols = list(forecaster.calendar_features.feature_names_out_)
        calendar_features_names = list(forecaster.calendar_features_names)

    # Use sets for O(1) lookup instead of O(n) list search
    autoreg_cols_set = set(autoreg_cols)
    calendar_cols_set = set(calendar_cols)
    exog_cols = [
        col for col in X_train.columns
        if col not in autoreg_cols_set and col not in calendar_cols_set
    ]
    exog_cols_set = set(exog_cols)

    forced_autoreg = []
    forced_exog = []
    forced_calendar = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols_set]
            forced_exog = [col for col in force_inclusion if col in exog_cols_set]
            forced_calendar = [col for col in force_inclusion if col in calendar_cols_set]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]
            forced_calendar = [col for col in calendar_cols if re.match(force_inclusion, col)]

    # Groups not evaluated by the selector are kept fixed and removed from the
    # selection matrix so the selector only sees the requested groups.
    cols_to_drop = []
    if not eval_autoreg:
        cols_to_drop.extend(autoreg_cols)
    if not eval_exog:
        cols_to_drop.extend(exog_cols)
    if not eval_calendar:
        cols_to_drop.extend(calendar_cols)
    
    if cols_to_drop:
        X_train = X_train.drop(columns=cols_to_drop)

    if X_train.shape[1] == 0:
        raise ValueError(
            "No features remain to be evaluated by the selector. The group(s) "
            "requested in `select_only` contain no features. Make sure the "
            "forecaster includes features for the selected group(s)."
        )

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.integers(low=0, high=len(X_train), size=subsample)
    X_train_sample = X_train.iloc[sample, :]
    y_train_sample = y_train.iloc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if eval_autoreg:
        selected_autoreg = [
            feature for feature in selected_features if feature in autoreg_cols_set
        ]
    else:
        selected_autoreg = autoreg_cols

    if eval_exog:
        selected_exog = [
            feature for feature in selected_features if feature in exog_cols_set
        ]
    else:
        selected_exog = exog_cols

    if eval_calendar:
        selected_calendar_cols = [
            feature for feature in selected_features if feature in calendar_cols_set
        ]
    else:
        selected_calendar_cols = calendar_cols

    if force_inclusion is not None: 
        if eval_exog:
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            exog_cols_order = {col: i for i, col in enumerate(exog_cols)}
            selected_exog.sort(key=lambda x: exog_cols_order[x])
        if eval_autoreg:
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            autoreg_cols_order = {col: i for i, col in enumerate(autoreg_cols)}
            selected_autoreg.sort(key=lambda x: autoreg_cols_order[x])
        if eval_calendar:
            forced_calendar_not_selected = set(forced_calendar) - set(selected_features)
            selected_calendar_cols.extend(forced_calendar_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            calendar_cols_order = {col: i for i, col in enumerate(calendar_cols)}
            selected_calendar_cols.sort(key=lambda x: calendar_cols_order[x])

    if len(selected_autoreg) == 0:
        warnings.warn(
            "No autoregressive features have been selected. Since a Forecaster "
            "cannot be created without them, be sure to include at least one "
            "using the `force_inclusion` parameter."
        )
        selected_lags = []
        selected_window_features = []
    else:
        lags_cols_set = set(lags_cols)
        window_features_cols_set = set(window_features_cols)
        selected_lags = [
            int(feature.replace('lag_', '')) 
            for feature in selected_autoreg if feature in lags_cols_set
        ]
        selected_window_features = [
            feature for feature in selected_autoreg if feature in window_features_cols_set
        ]

    # Map the selected encoded calendar columns back to their source calendar
    # features (e.g. `month_sin`, `month_cos` -> `month`). A source feature is
    # kept if at least one of its encoded columns is selected. Source features
    # are matched longest-first to disambiguate names that share a prefix
    # (e.g. `week` and `day_of_week`).
    source_features_sorted = sorted(calendar_features_names, key=len, reverse=True)
    selected_calendar_features = []
    seen_calendar_features = set()
    for col in selected_calendar_cols:
        for feature in source_features_sorted:
            if col == feature or col.startswith(f"{feature}_"):
                if feature not in seen_calendar_features:
                    seen_calendar_features.add(feature)
                    selected_calendar_features.append(feature)
                break
    calendar_features_order = {
        feature: i for i, feature in enumerate(calendar_features_names)
    }
    selected_calendar_features.sort(key=lambda x: calendar_features_order[x])

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols) + len(calendar_cols)}") 
        print(f"    Lags            (n={len(lags_cols)})")
        print(f"    Window features (n={len(window_features_cols)})")
        print(f"    Exog            (n={len(exog_cols)})")
        print(f"    Calendar        (n={len(calendar_cols)})")
        n_selected = (
            len(selected_lags) + len(selected_window_features)
            + len(selected_exog) + len(selected_calendar_features)
        )
        print(f"Number of features selected: {n_selected}")
        print(f"    Lags            (n={len(selected_lags)}) : {selected_lags}")
        print(f"    Window features (n={len(selected_window_features)}) : {selected_window_features}")
        print(f"    Exog            (n={len(selected_exog)}) : {selected_exog}")
        print(f"    Calendar        (n={len(selected_calendar_features)}) : {selected_calendar_features}")

    return selected_lags, selected_window_features, selected_exog, selected_calendar_features


def select_features_multiseries(
    forecaster: object,
    selector: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    select_only: str | list[str] | None = None,
    force_inclusion: list[str] | str | None = None,
    subsample: int | float = 0.5,
    random_state: int = 123,
    verbose: bool = True,
) -> tuple[list[int] | dict[str, list[int]], list[str], list[str], list[str]]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Three groups of features are
    evaluated: autoregressive features (lags and window features), exogenous
    features and calendar features. By default, the selection process is performed
    on the three sets of features at the same time, so that the most relevant
    autoregressive, exogenous and calendar features are selected. However, using
    the `select_only` argument, the selection process can focus only on one or more
    of these groups without taking into account the others. Therefore, all features
    in the remaining groups will remain in the model. It is also possible to force
    the inclusion of certain features in the final list of selected features using
    the `force_inclusion` parameter.

    If encoded, calendar features are evaluated at the encoded-column level (e.g. 
    `month_sin`, `month_cos`), but they are returned at the source-feature level 
    (e.g. `month`). A source calendar feature is kept whenever at least one of 
    its encoded columns is selected.

    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate
        Forecaster model. If forecaster is a ForecasterDirectMultiVariate, the
        selector will only be applied to the features of the first step.
    selector : object
        A feature selector from sklearn.feature_selection.
    series : pandas DataFrame, dict
        Target time series to which the feature selection will be applied.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    select_only : str, list, default None
        Decide what type of features to include in the selection process. 
        
        - If `'autoreg'` (or `['autoreg']`), only autoregressive features (lags
        and window features) are evaluated by the selector. All exogenous and
        calendar features are kept and returned in `selected_exog` and
        `selected_calendar_features`.
        - If `'exog'` (or `['exog']`), only exogenous features are evaluated. All
        autoregressive and calendar features are kept and returned in
        `selected_lags`, `selected_window_features` and
        `selected_calendar_features`.
        - If `'calendar'` (or `['calendar']`), only calendar features are
        evaluated. All autoregressive and exogenous features are kept and
        returned in `selected_lags`, `selected_window_features` and
        `selected_exog`.
        - If `list`, any combination of `'autoreg'`, `'exog'` and `'calendar'`.
        Only the groups listed are evaluated by the selector; the remaining
        groups are kept unchanged.
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default None
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.

        For calendar features, `force_inclusion` is matched against the encoded
        column names (e.g. `month_sin`); forcing any encoded column keeps its
        source calendar feature in `selected_calendar_features`.
    subsample : int, float, default 0.5
        Proportion of records to use for feature selection.
    random_state : int, default 123
        Sets a seed for the random subsample so that the subsampling process 
        is always deterministic.
    verbose : bool, default True
        Print information about feature selection process.

    Returns
    -------
    selected_lags : list, dict
        List of selected lags. If the forecaster is a ForecasterDirectMultiVariate,
        the output is a dict with the selected lags for each series, {series_name: lags},
        as the lags can be different for each series.
    selected_window_features : list
        List of selected window features.
    selected_exog : list
        List of selected exogenous features.
    selected_calendar_features : list
        List of selected calendar features (source-level names, without the
        encoding suffix). Empty list if the forecaster has no calendar features.

    """

    forecaster_name = type(forecaster).__name__
    valid_forecasters = [
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirectMultiVariate'
    ]

    if forecaster_name not in valid_forecasters:
        raise TypeError(
            f"`forecaster` must be one of the following classes: {valid_forecasters}."
        )

    valid_select_only = ['autoreg', 'exog', 'calendar']
    if select_only is None:
        select_only_list = valid_select_only
    else:
        select_only_list = (
            [select_only] if isinstance(select_only, str) else select_only
        )
        if not isinstance(select_only_list, list):
            raise TypeError(
                "`select_only` must be a str, a list of str, or None."
            )
        if not set(select_only_list).issubset(valid_select_only):
            raise ValueError(
                "`select_only` must be one or more of the following values: "
                "'autoreg', 'exog', 'calendar', or None."
            )

    eval_autoreg = 'autoreg' in select_only_list
    eval_exog = 'exog' in select_only_list
    eval_calendar = 'calendar' in select_only_list

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy_forecaster(forecaster)
    forecaster.is_fitted = False
    if forecaster_name == 'ForecasterDirectMultiVariate':
        X_train, y_train = forecaster.create_train_X_y(series=series, exog=exog)
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step          = 1,
                               X_train       = X_train,
                               y_train       = y_train,
                               remove_suffix = True
                           )
        lags_cols = list(
            chain(*[v for v in forecaster.lags_names.values() if v is not None])
        )
        window_features_cols = forecaster.X_train_window_features_names_out_
        encoding_cols = []
    else:
        output = forecaster._create_train_X_y(series=series, exog=exog)
        X_train = output[0]
        y_train = output[1]
        lags_cols = forecaster.lags_names
        window_features_cols = output[7]  # X_train_window_features_names_out_ output
        if forecaster.encoding == 'onehot':
            encoding_cols = output[4]  # X_train_series_names_in_ output
        else:
            encoding_cols = ['_level_skforecast']
    
    lags_cols = [] if lags_cols is None else lags_cols
    window_features_cols = [] if window_features_cols is None else window_features_cols
    autoreg_cols = []
    if forecaster.lags is not None:
        autoreg_cols.extend(lags_cols)
    if forecaster.window_features is not None:
        autoreg_cols.extend(window_features_cols)

    # `create_train_X_y` fit-transforms `calendar_features`, so its encoded
    # output column names (e.g. `month_sin`) are available in X_train.
    # `calendar_features_names` holds the source-level feature names (e.g. `month`).
    calendar_cols = []
    calendar_features_names = []
    if forecaster.calendar_features is not None:
        calendar_cols = list(forecaster.calendar_features.feature_names_out_)
        calendar_features_names = list(forecaster.calendar_features_names)

    # Use sets for O(1) lookup instead of O(n) list search
    autoreg_cols_set = set(autoreg_cols)
    calendar_cols_set = set(calendar_cols)
    encoding_cols_set = set(encoding_cols)
    exog_cols = [
        col
        for col in X_train.columns
        if col not in autoreg_cols_set
        and col not in calendar_cols_set
        and col not in encoding_cols_set
    ]
    exog_cols_set = set(exog_cols)

    forced_autoreg = []
    forced_exog = []
    forced_calendar = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols_set]
            forced_exog = [col for col in force_inclusion if col in exog_cols_set]
            forced_calendar = [col for col in force_inclusion if col in calendar_cols_set]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]
            forced_calendar = [col for col in calendar_cols if re.match(force_inclusion, col)]

    # Groups not evaluated by the selector are kept fixed and removed from the
    # selection matrix so the selector only sees the requested groups. Encoding
    # columns are always removed.
    cols_to_drop = list(encoding_cols)
    if not eval_autoreg:
        cols_to_drop.extend(autoreg_cols)
    if not eval_exog:
        cols_to_drop.extend(exog_cols)
    if not eval_calendar:
        cols_to_drop.extend(calendar_cols)

    if cols_to_drop:
        X_train = X_train.drop(columns=cols_to_drop)

    if X_train.shape[1] == 0:
        raise ValueError(
            "No features remain to be evaluated by the selector. The group(s) "
            "requested in `select_only` contain no features. Make sure the "
            "forecaster includes features for the selected group(s)."
        )

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.integers(low=0, high=len(X_train), size=subsample)
    X_train_sample = X_train.iloc[sample, :]
    y_train_sample = y_train.iloc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if eval_autoreg:
        selected_autoreg = [
            feature
            for feature in selected_features
            if feature in autoreg_cols_set
        ]
    else:
        selected_autoreg = autoreg_cols

    if eval_exog:
        selected_exog = [
            feature
            for feature in selected_features
            if feature in exog_cols_set
        ]
    else:
        selected_exog = exog_cols

    if eval_calendar:
        selected_calendar_cols = [
            feature
            for feature in selected_features
            if feature in calendar_cols_set
        ]
    else:
        selected_calendar_cols = calendar_cols

    if force_inclusion is not None: 
        if eval_exog:
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            exog_cols_order = {col: i for i, col in enumerate(exog_cols)}
            selected_exog.sort(key=lambda x: exog_cols_order[x])
        if eval_autoreg:
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            autoreg_cols_order = {col: i for i, col in enumerate(autoreg_cols)}
            selected_autoreg.sort(key=lambda x: autoreg_cols_order[x])
        if eval_calendar:
            forced_calendar_not_selected = set(forced_calendar) - set(selected_features)
            selected_calendar_cols.extend(forced_calendar_not_selected)
            # Use dict for O(1) index lookup instead of O(n) list.index()
            calendar_cols_order = {col: i for i, col in enumerate(calendar_cols)}
            selected_calendar_cols.sort(key=lambda x: calendar_cols_order[x])

    if len(selected_autoreg) == 0:
        warnings.warn(
            "No autoregressive features have been selected. Since a Forecaster "
            "cannot be created without them, be sure to include at least one "
            "using the `force_inclusion` parameter."
        )
        selected_lags = []
        selected_window_features = []
        verbose_selected_lags = []
    else:
        lags_cols_set = set(lags_cols)
        window_features_cols_set = set(window_features_cols)
        if forecaster_name == 'ForecasterDirectMultiVariate':
            selected_lags = {
                series_name: (
                    [
                        int(feature.replace(f"{series_name}_lag_", ""))
                        for feature in selected_autoreg
                        if feature in lags_names
                    ]
                    if lags_names is not None
                    else []
                )
                for series_name, lags_names in forecaster.lags_names.items()
            }
            verbose_selected_lags = [
                feature for feature in selected_autoreg if feature in lags_cols_set
            ]
        else:
            selected_lags = [
                int(feature.replace('lag_', '')) 
                for feature in selected_autoreg 
                if feature in lags_cols_set
            ]
            verbose_selected_lags = selected_lags

        selected_window_features = [
            feature for feature in selected_autoreg 
            if feature in window_features_cols_set
        ]

    # Map the selected encoded calendar columns back to their source calendar
    # features (e.g. `month_sin`, `month_cos` -> `month`). A source feature is
    # kept if at least one of its encoded columns is selected. Source features
    # are matched longest-first to disambiguate names that share a prefix
    # (e.g. `week` and `day_of_week`).
    source_features_sorted = sorted(calendar_features_names, key=len, reverse=True)
    selected_calendar_features = []
    seen_calendar_features = set()
    for col in selected_calendar_cols:
        for feature in source_features_sorted:
            if col == feature or col.startswith(f"{feature}_"):
                if feature not in seen_calendar_features:
                    seen_calendar_features.add(feature)
                    selected_calendar_features.append(feature)
                break
    calendar_features_order = {
        feature: i for i, feature in enumerate(calendar_features_names)
    }
    selected_calendar_features.sort(key=lambda x: calendar_features_order[x])

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols) + len(calendar_cols)}") 
        print(f"    Lags            (n={len(lags_cols)})")
        print(f"    Window features (n={len(window_features_cols)})")
        print(f"    Exog            (n={len(exog_cols)})")
        print(f"    Calendar        (n={len(calendar_cols)})")
        n_selected = (
            len(verbose_selected_lags) + len(selected_window_features)
            + len(selected_exog) + len(selected_calendar_features)
        )
        print(f"Number of features selected: {n_selected}")
        print(f"    Lags            (n={len(verbose_selected_lags)}) : {verbose_selected_lags}")
        print(f"    Window features (n={len(selected_window_features)}) : {selected_window_features}")
        print(f"    Exog            (n={len(selected_exog)}) : {selected_exog}")
        print(f"    Calendar        (n={len(selected_calendar_features)}) : {selected_calendar_features}")

    return selected_lags, selected_window_features, selected_exog, selected_calendar_features
