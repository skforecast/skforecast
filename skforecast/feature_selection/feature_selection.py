################################################################################
#                       skforecast.feature_selection                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import re
from copy import deepcopy
from itertools import chain
import warnings
import numpy as np
import pandas as pd


def select_features(
    forecaster: object,
    selector: object,
    y: pd.Series | pd.DataFrame,
    exog: pd.Series | pd.DataFrame | None = None,
    select_only: str | None = None,
    force_inclusion: list[str] | str | None = None,
    subsample: int | float = 0.5,
    random_state: int = 123,
    verbose: bool = True
) -> tuple[list[int], list[str], list[str]]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Two groups of features are
    evaluated: autoregressive features (lags and window features) and exogenous
    features. By default, the selection process is performed on both sets of features
    at the same time, so that the most relevant autoregressive and exogenous features
    are selected. However, using the `select_only` argument, the selection process
    can focus only on the autoregressive or exogenous features without taking into
    account the other features. Therefore, all other features will remain in the model. 
    It is also possible to force the inclusion of certain features in the final list
    of selected features using the `force_inclusion` parameter.

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
    select_only : str, default None
        Decide what type of features to include in the selection process. 
        
        - If `'autoreg'`, only autoregressive features (lags and window features)
        are evaluated by the selector. All exogenous features are included in the
        output `selected_exog`.
        - If `'exog'`, only exogenous features are evaluated without the presence
        of autoregressive features. All autoregressive features are included 
        in the outputs `selected_lags` and `selected_window_features`.
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default None
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.
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

    """

    forecaster_name = type(forecaster).__name__
    valid_forecasters = ['ForecasterRecursive', 'ForecasterDirect']

    if forecaster_name not in valid_forecasters:
        raise TypeError(
            f"`forecaster` must be one of the following classes: {valid_forecasters}."
        )
    
    if select_only not in ['autoreg', 'exog', None]:
        raise ValueError(
            "`select_only` must be one of the following values: 'autoreg', 'exog', None."
        )

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy(forecaster)
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

    exog_cols = [col for col in X_train.columns if col not in autoreg_cols]

    forced_autoreg = []
    forced_exog = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols]
            forced_exog = [col for col in force_inclusion if col in exog_cols]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]

    if select_only == 'autoreg':
        X_train = X_train.drop(columns=exog_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols)

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.integers(low=0, high=len(X_train), size=subsample)
    X_train_sample = X_train.iloc[sample, :]
    y_train_sample = y_train.iloc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if select_only == 'exog':
        selected_autoreg = autoreg_cols
    else:
        selected_autoreg = [
            feature
            for feature in selected_features
            if feature in autoreg_cols
        ]

    if select_only == 'autoreg':
        selected_exog = exog_cols
    else:
        selected_exog = [
            feature
            for feature in selected_features
            if feature in exog_cols
        ]

    if force_inclusion is not None: 
        if select_only != 'autoreg':
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            selected_exog.sort(key=exog_cols.index)
        if select_only != 'exog':
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            selected_autoreg.sort(key=autoreg_cols.index)

    if len(selected_autoreg) == 0:
        warnings.warn(
            "No autoregressive features have been selected. Since a Forecaster "
            "cannot be created without them, be sure to include at least one "
            "using the `force_inclusion` parameter."
        )
        selected_lags = []
        selected_window_features = []
    else:
        selected_lags = [
            int(feature.replace('lag_', '')) 
            for feature in selected_autoreg if feature in lags_cols
        ]
        selected_window_features = [
            feature for feature in selected_autoreg if feature in window_features_cols
        ]

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols)}") 
        print(f"    Lags            (n={len(lags_cols)})")
        print(f"    Window features (n={len(window_features_cols)})")
        print(f"    Exog            (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Lags            (n={len(selected_lags)}) : {selected_lags}")
        print(f"    Window features (n={len(selected_window_features)}) : {selected_window_features}")
        print(f"    Exog            (n={len(selected_exog)}) : {selected_exog}")

    return selected_lags, selected_window_features, selected_exog


def select_features_multiseries(
    forecaster: object,
    selector: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    select_only: str | None = None,
    force_inclusion: list[str] | str | None = None,
    subsample: int | float = 0.5,
    random_state: int = 123,
    verbose: bool = True,
) -> tuple[list[int] | dict[str, int], list[str], list[str]]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Two groups of features are
    evaluated: autoregressive features and exogenous features. By default, the 
    selection process is performed on both sets of features at the same time, 
    so that the most relevant autoregressive and exogenous features are selected. 
    However, using the `select_only` argument, the selection process can focus 
    only on the autoregressive or exogenous features without taking into account 
    the other features. Therefore, all other features will remain in the model. 
    It is also possible to force the inclusion of certain features in the final 
    list of selected features using the `force_inclusion` parameter.

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
    select_only : str, default None
        Decide what type of features to include in the selection process. 
        
        - If `'autoreg'`, only autoregressive features (lags and window features) 
        are evaluated by the selector. All exogenous features are 
        included in the output `selected_exog`.
        - If `'exog'`, only exogenous features are evaluated without the presence
        of autoregressive features. All autoregressive features are included 
        in the outputs `selected_lags` and `selected_window_features`.
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default None
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.
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
    
    if select_only not in ['autoreg', 'exog', None]:
        raise ValueError(
            "`select_only` must be one of the following values: 'autoreg', 'exog', None."
        )

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy(forecaster)
    forecaster.is_fitted = False
    output = forecaster._create_train_X_y(series=series, exog=exog)
    X_train = output[0]
    y_train = output[1]
    if forecaster_name == 'ForecasterDirectMultiVariate':
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
        lags_cols = forecaster.lags_names
        window_features_cols = output[6]  # X_train_window_features_names_out_ output
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
    
    exog_cols = [
        col
        for col in X_train.columns
        if col not in autoreg_cols and col not in encoding_cols
    ]

    forced_autoreg = []
    forced_exog = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols]
            forced_exog = [col for col in force_inclusion if col in exog_cols]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]

    if select_only == 'autoreg':
        X_train = X_train.drop(columns=exog_cols + encoding_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols + encoding_cols)
    else:
        X_train = X_train.drop(columns=encoding_cols)

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.integers(low=0, high=len(X_train), size=subsample)
    X_train_sample = X_train.iloc[sample, :]
    y_train_sample = y_train.iloc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if select_only == 'exog':
        selected_autoreg = autoreg_cols
    else:
        selected_autoreg = [
            feature
            for feature in selected_features
            if feature in autoreg_cols
        ]

    if select_only == 'autoreg':
        selected_exog = exog_cols
    else:
        selected_exog = [
            feature
            for feature in selected_features
            if feature in exog_cols
        ]

    if force_inclusion is not None: 
        if select_only != 'autoreg':
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            selected_exog.sort(key=exog_cols.index)
        if select_only != 'exog':
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            selected_autoreg.sort(key=autoreg_cols.index)

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
                feature for feature in selected_autoreg if feature in lags_cols
            ]
        else:
            selected_lags = [
                int(feature.replace('lag_', '')) 
                for feature in selected_autoreg 
                if feature in lags_cols
            ]
            verbose_selected_lags = selected_lags

        selected_window_features = [
            feature for feature in selected_autoreg 
            if feature in window_features_cols
        ]

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols)}") 
        print(f"    Lags            (n={len(lags_cols)})")
        print(f"    Window features (n={len(window_features_cols)})")
        print(f"    Exog            (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Lags            (n={len(verbose_selected_lags)}) : {verbose_selected_lags}")
        print(f"    Window features (n={len(selected_window_features)}) : {selected_window_features}")
        print(f"    Exog            (n={len(selected_exog)}) : {selected_exog}")

    return selected_lags, selected_window_features, selected_exog
