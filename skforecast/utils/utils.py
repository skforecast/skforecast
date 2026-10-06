################################################################################
#                               skforecast.utils                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from copy import copy, deepcopy
import dis
from functools import partial, wraps
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import inspect
from packaging.requirements import Requirement
from pathlib import Path
import platform
import sys
import textwrap
from typing import Any, Callable, ParamSpec, TypeVar
import uuid
import warnings
import zoneinfo
import joblib
import pickle
import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline
from .. import __version__
from ..exceptions import warn_skforecast_categories
from ..exceptions import (
    DataTypeWarning,
    IgnoredArgumentWarning,
    MissingExogWarning,
    MissingValuesWarning,
    SaveLoadSkforecastWarning,
    SkforecastVersionWarning,
    UnknownLevelWarning,
    InputTypeWarning
)

# Type variables for the manage_warnings decorator. ParamSpec preserves the
# decorated function's parameter signature, and TypeVar preserves its return
# type, so that type checkers see the original signatures through the wrapper.
P = ParamSpec('P')
R = TypeVar('R')

# sklearn estimators that natively support NaN values in the input features.
# Tree-based models gained this support in scikit-learn 1.3 (decision trees),
# 1.4 (random forests) and 1.6 (extra trees), all at or below the minimum
# version required by skforecast.
_SKLEARN_NAN_TOLERANT_ESTIMATORS = frozenset({
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'ExtraTreeClassifier',
    'ExtraTreeRegressor',
    'ExtraTreesClassifier',
    'ExtraTreesRegressor',
    'HistGradientBoostingClassifier',
    'HistGradientBoostingRegressor',
    'RandomForestClassifier',
    'RandomForestRegressor',
})

optional_dependencies = {
    'stats': [
        'statsmodels>=0.13, <0.15'
    ],
    'plotting': [
        'matplotlib>=3.7, <3.12', 
        'statsmodels>=0.13, <0.15'
    ],
        'deeplearning': [
        'keras>=3.0, <4.0',
        'matplotlib>=3.7, <3.12',
    ]
}


def initialize_lags(
    forecaster_name: str,
    lags: Any
) -> tuple[np.ndarray[int] | None, list[str] | None, int | None]:
    """
    Check lags argument input and generate the corresponding numpy ndarray.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    lags : Any
        Lags used as predictors.

    Returns
    -------
    lags : numpy ndarray, None
        Lags used as predictors.
    lags_names : list, None
        Names of the lags used as predictors.
    max_lag : int, None
        Maximum value of the lags.
    
    """

    lags_names = None
    max_lag = None
    if lags is not None:
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError("Minimum value of lags allowed is 1.")
            lags = np.arange(1, lags + 1)

        if isinstance(lags, (list, tuple, range)):
            lags = np.array(lags)
        
        if isinstance(lags, np.ndarray):
            if lags.size == 0:
                return None, None, None
            if lags.ndim != 1:
                raise ValueError("`lags` must be a 1-dimensional array.")
            if not np.issubdtype(lags.dtype, np.integer):
                raise TypeError("All values in `lags` must be integers.")
            if np.any(lags < 1):
                raise ValueError("Minimum value of lags allowed is 1.")
        else:
            if forecaster_name == 'ForecasterDirectMultiVariate':
                raise TypeError(
                    f"`lags` argument must be a dict, int, 1d numpy ndarray, range, "
                    f"tuple or list. Got {type(lags)}."
                )
            else:
                raise TypeError(
                    f"`lags` argument must be an int, 1d numpy ndarray, range, "
                    f"tuple or list. Got {type(lags)}."
                )
        
        lags = np.sort(lags)
        lags_names = [f'lag_{i}' for i in lags]
        max_lag = max(lags)

    return lags, lags_names, max_lag


def initialize_window_features(
    window_features: Any
) -> tuple[list[object] | None, list[str] | None, int | None]:
    """
    Check window_features argument input and generate the corresponding list.

    Parameters
    ----------
    window_features : Any
        Classes used to create window features.

    Returns
    -------
    window_features : list, None
        List of classes used to create window features.
    window_features_names : list, None
        List with all the features names of the window features.
    max_size_window_features : int, None
        Maximum value of the `window_sizes` attribute of all classes.
    
    """

    needed_atts = ['window_sizes', 'features_names']
    needed_methods = ['transform_batch', 'transform']

    max_window_sizes = None
    window_features_names = None
    max_size_window_features = None
    if window_features is not None:
        if isinstance(window_features, list) and len(window_features) < 1:
            raise ValueError(
                "Argument `window_features` must contain at least one element."
            )
        if not isinstance(window_features, list):
            window_features = [window_features]

        link_to_docs = (
            "\nVisit the documentation for more information about how to create "
            "custom window features:\n"
            "https://skforecast.org/latest/user_guides/window-features-and-custom-features.html#create-your-custom-window-features"
        )
        
        max_window_sizes = []
        window_features_names = []
        needed_atts_set = set(needed_atts)
        needed_methods_set = set(needed_methods)
        for wf in window_features:
            wf_name = type(wf).__name__
            atts_methods = set(dir(wf))
            if not needed_atts_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the attributes: {needed_atts}." + link_to_docs
                )
            if not needed_methods_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the methods: {needed_methods}." + link_to_docs
                )
            
            window_sizes = wf.window_sizes
            if not isinstance(window_sizes, (int, list)):
                raise TypeError(
                    f"Attribute `window_sizes` of {wf_name} must be an int or a list "
                    f"of ints. Got {type(window_sizes)}." + link_to_docs
                )
            
            if isinstance(window_sizes, int):
                if window_sizes < 1:
                    raise ValueError(
                        f"If argument `window_sizes` is an integer, it must be equal to or "
                        f"greater than 1. Got {window_sizes} from {wf_name}." + link_to_docs
                    )
                max_window_sizes.append(window_sizes)
            else:
                if not all(isinstance(ws, int) for ws in window_sizes) or not all(
                    ws >= 1 for ws in window_sizes
                ):                    
                    raise ValueError(
                        f"If argument `window_sizes` is a list, all elements must be integers "
                        f"equal to or greater than 1. Got {window_sizes} from {wf_name}." + link_to_docs
                    )
                max_window_sizes.append(max(window_sizes))

            features_names = wf.features_names
            if not isinstance(features_names, list):
                raise TypeError(
                    f"Attribute `features_names` of {wf_name} must be a list "
                    f"of strings. Got {type(features_names)}." + link_to_docs
                )
            if not all(isinstance(fn, str) for fn in features_names):
                raise TypeError(
                    f"If argument `features_names` is a list, all elements "
                    f"must be strings. Got {features_names} from {wf_name}." + link_to_docs
                )
            window_features_names.extend(features_names)

        max_size_window_features = max(max_window_sizes)
        if len(set(window_features_names)) != len(window_features_names):
            raise ValueError(
                f"All window features names must be unique. Got {window_features_names}."
            )

    return window_features, window_features_names, max_size_window_features


def _get_source_code(fun: Callable) -> str | None:
    """
    Return the source code of a function, or `None` if it is not available,
    for example for a `functools.partial`, a callable object or a function
    defined in an interactive console.

    Parameters
    ----------
    fun : Callable
        Function whose source code is returned.

    Returns
    -------
    source_code : str, None
        Source code of the function, or `None` if it is not available.

    """

    try:
        source_code = inspect.getsource(fun)
    except (OSError, TypeError):
        source_code = None

    return source_code


def initialize_weights(
    forecaster_name: str,
    estimator: object,
    weight_func: Callable | dict[str, Callable],
    series_weights: dict[str, float]
) -> tuple[Callable | dict[str, Callable] | None, str | dict[str, str] | None, dict[str, float] | None]:
    """
    Check weights arguments, `weight_func` and `series_weights` for the different 
    forecasters. Create `source_code_weight_func`, source code of the custom 
    function(s) used to create weights.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    estimator : estimator or pipeline compatible with the scikit-learn API
        Estimator of the forecaster.
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.
    series_weights : dict
        Argument `series_weights` of the forecaster.

    Returns
    -------
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.
    source_code_weight_func : str, dict
        Argument `source_code_weight_func` of the forecaster. It is `None` for a
        function whose source code is not available (e.g. a `functools.partial`
        or a callable object).
    series_weights : dict
        Argument `series_weights` of the forecaster. Only ForecasterRecursiveMultiSeries.
    
    """

    source_code_weight_func = None

    if weight_func is not None:

        if forecaster_name in ['ForecasterRecursiveMultiSeries']:
            if not isinstance(weight_func, (Callable, dict)):
                raise TypeError(
                    f"Argument `weight_func` must be a Callable or a dict of "
                    f"Callables. Got {type(weight_func)}."
                )
        elif not isinstance(weight_func, Callable):
            raise TypeError(
                f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
            )
        
        if isinstance(weight_func, dict):
            source_code_weight_func = {}
            for key in weight_func:
                source_code_weight_func[key] = _get_source_code(weight_func[key])
        else:
            source_code_weight_func = _get_source_code(weight_func)

        if 'sample_weight' not in inspect.signature(estimator.fit).parameters:
            warnings.warn(
                f"Argument `weight_func` is ignored since estimator {estimator} "
                f"does not accept `sample_weight` in its `fit` method.",
                IgnoredArgumentWarning
            )
            weight_func = None
            source_code_weight_func = None

    if series_weights is not None:
        if not isinstance(series_weights, dict):
            raise TypeError(
                f"Argument `series_weights` must be a dict of floats or ints."
                f"Got {type(series_weights)}."
            )
        if 'sample_weight' not in inspect.signature(estimator.fit).parameters:
            warnings.warn(
                f"Argument `series_weights` is ignored since estimator {estimator} "
                f"does not accept `sample_weight` in its `fit` method.",
                IgnoredArgumentWarning
            )
            series_weights = None

    return weight_func, source_code_weight_func, series_weights


def initialize_transformer_series(
    forecaster_name: str,
    series_names_in_: list[str],
    encoding: str | None = None,
    transformer_series: object | dict[str, object | None] | None = None
) -> dict[str, object | None]:
    """
    Initialize `transformer_series_` attribute for the Forecasters Multiseries.

    - If `transformer_series` is `None`, no transformation is applied.
    - If `transformer_series` is a scikit-learn transformer (object), the same 
    transformer is applied to all series (`series_names_in_`).
    - If `transformer_series` is a `dict`, a different transformer can be
    applied to each series. The keys of the dictionary must be the same as the
    names of the series in `series_names_in_`.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    series_names_in_ : list
        Names of the series (levels) used during training.
    encoding : str, default None
        Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).
    transformer_series : object, dict, default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. 

    Returns
    -------
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the 
        objects in `transformer_series` and is used internally to avoid overwriting.
    
    """

    if forecaster_name == 'ForecasterRecursiveMultiSeries':
        if encoding is None:
            series_names_in_ = ['_unknown_level']
        else:
            series_names_in_ = series_names_in_ + ['_unknown_level']

    if transformer_series is None:
        transformer_series_ = {serie: None for serie in series_names_in_}
    elif not isinstance(transformer_series, dict):
        transformer_series_ = {
            serie: clone(transformer_series) 
            for serie in series_names_in_
        }
    else:
        transformer_series_ = {serie: None for serie in series_names_in_}
        # Only elements already present in transformer_series_ are updated
        transformer_series_.update(
            {
                k: deepcopy(v)
                for k, v in transformer_series.items()
                if k in transformer_series_
            }
        )

        series_not_in_transformer_series = (
            set(series_names_in_) - set(transformer_series.keys())
        ) - {'_unknown_level'}
        if series_not_in_transformer_series:
            warnings.warn(
                f"{series_not_in_transformer_series} not present in `transformer_series`."
                f" No transformation is applied to these series.",
                IgnoredArgumentWarning
            )

    return transformer_series_


def initialize_differentiator_multiseries(
    series_names_in_: list[str],
    differentiator: object | dict[str, object | None] | None = None
) -> dict[str, object | None]:
    """
    Initialize `differentiator_` attribute for the ForecasterRecursiveMultiSeries.

    - If `int`, the same order of differentiation is applied to all series.
    - If `dict`, a different order of differentiation (including None) can 
    be used for each series. The keys must be the names of the series used
    to fit the forecaster. If a series is not present in the dictionary, no
    differencing is applied.
    - If `None`, no differencing is applied.

    Parameters
    ----------
    series_names_in_ : list
        Names of the series (levels) used during training.
    differentiator : TimeSeriesDifferentiator, dict, default None
        Skforecast object (or dict of objects) used to differentiate the time series.

    Returns
    -------
    differentiator_ : dict
        Dictionary with the `differentiator` for each series. It is created cloning the
        objects in `differentiator` and is used internally to avoid overwriting.
    
    """
    
    series_names_in_ = series_names_in_ + ['_unknown_level']
    if differentiator is None:
        differentiator_ = {serie: None for serie in series_names_in_}
    elif not isinstance(differentiator, dict):
        differentiator_ = {
            serie: copy(differentiator) for serie in series_names_in_
        }
    else:
        differentiator_ = {serie: None for serie in series_names_in_}
        # Only elements already present in differentiator_ are updated
        differentiator_.update(
            {
                k: deepcopy(v)
                for k, v in differentiator.items()
                if k in differentiator_
            }
        )

        series_not_in_differentiator = (
            set(series_names_in_) - set(differentiator.keys())
        )
        if series_not_in_differentiator:
            warnings.warn(
                f"{series_not_in_differentiator} not present in `differentiation`."
                f" No differentiation is applied to these series.",
                IgnoredArgumentWarning
            )

    return differentiator_


def check_select_fit_kwargs(
    estimator: object,
    fit_kwargs: dict[str, object] | None = None
) -> dict[str, object]:
    """
    Check if `fit_kwargs` is a dict and select only the keys that are used by
    the `fit` method of the estimator.

    Parameters
    ----------
    estimator : object
        Estimator object.
    fit_kwargs : dict, default None
        Dictionary with the arguments to pass to the `fit' method of the forecaster.

    Returns
    -------
    fit_kwargs : dict
        Dictionary with the arguments to be passed to the `fit` method of the 
        estimator after removing the unused keys.
    
    """

    if fit_kwargs is None:
        fit_kwargs = {}
    else:
        if not isinstance(fit_kwargs, dict):
            raise TypeError(
                f"Argument `fit_kwargs` must be a dict. Got {type(fit_kwargs)}."
            )
        
        fit_params = inspect.signature(estimator.fit).parameters

        # Non used keys
        non_used_keys = [
            k for k in fit_kwargs.keys() if k not in fit_params
        ]
        if non_used_keys:
            warnings.warn(
                f"Argument/s {non_used_keys} ignored since they are not used by the "
                f"estimator's `fit` method.",
                IgnoredArgumentWarning
            )

        if 'sample_weight' in fit_kwargs.keys():
            warnings.warn(
                "The `sample_weight` argument is ignored. Use `weight_func` to pass "
                "a function that defines the individual weights for each sample "
                "based on its index.",
                IgnoredArgumentWarning
            )
            del fit_kwargs['sample_weight']

        # Select only the keyword arguments allowed by the estimator's `fit` method.
        fit_kwargs = {
            k: v for k, v in fit_kwargs.items() if k in fit_params
        }

    return fit_kwargs


def configure_estimator_categorical_features(
    estimator: object,
    categorical_features_names_in_: list[str] | None,
    X_train_features_names_out_: list[str],
    fit_kwargs: dict[str, object]
) -> dict[str, object]:
    """
    Configure native categorical feature support for the estimator. Returns
    updated `fit_kwargs` with the appropriate arguments for the estimator.
    For estimators that require configuration via `set_params` (XGBoost,
    HistGradientBoosting), the estimator is modified in-place.

    Supported estimators: LGBMRegressor/LGBMClassifier,
    CatBoostRegressor/CatBoostClassifier, XGBRegressor/XGBClassifier,
    HistGradientBoostingRegressor/HistGradientBoostingClassifier (sklearn).

    Parameters
    ----------
    estimator : object
        Estimator object. If the estimator is a Pipeline, the last step is
        used.
    categorical_features_names_in_ : list, None
        Names of the categorical features. If `None` or empty, any previously
        set categorical configuration on the estimator is reset.
    X_train_features_names_out_ : list
        Names of all features in `X_train`, in column order.
    fit_kwargs : dict
        Dictionary with the arguments to pass to the `fit` method of the
        estimator. This dictionary is updated in-place and returned.

    Returns
    -------
    fit_kwargs : dict
        Updated dictionary with the categorical feature arguments added for
        the estimator's `fit` method.

    """

    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    estimator_name = type(estimator).__name__
    module = type(estimator).__module__.split('.')[0]

    if not categorical_features_names_in_:
        # Reset any previously set categorical params (from a prior fit call)
        if module == 'xgboost':
            estimator.set_params(feature_types=None, enable_categorical=False)
        elif module == 'sklearn' and estimator_name in (
            'HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'
        ):
            estimator.set_params(categorical_features='from_dtype')
        return fit_kwargs

    cat_indices = [
        X_train_features_names_out_.index(name)
        for name in categorical_features_names_in_
    ]

    if module == 'lightgbm':
        # LGBMRegressor.fit() accepts `categorical_feature` as a list of
        # int indices when X is a numpy array.
        if 'categorical_feature' in fit_kwargs:
            warnings.warn(
                "The `categorical_feature` argument in `fit_kwargs` is being "
                "overridden by the values detected from `categorical_features`. "
                f"Overridden value: {fit_kwargs['categorical_feature']}.",
                IgnoredArgumentWarning
            )
        fit_kwargs['categorical_feature'] = cat_indices

    # NOTE: https://github.com/catboost/catboost/issues/3064
    elif module == 'catboost':
        # CatBoostRegressor.fit() accepts `cat_features` as a list of int
        # indices.
        if 'cat_features' in fit_kwargs:
            warnings.warn(
                "The `cat_features` argument in `fit_kwargs` is being "
                "overridden by the values detected from `categorical_features`. "
                f"Overridden value: {fit_kwargs['cat_features']}.",
                IgnoredArgumentWarning
            )
        fit_kwargs['cat_features'] = cat_indices

    elif module == 'xgboost':
        # XGBRegressor requires `feature_types` and `enable_categorical=True`
        # set via set_params (they are constructor params, not fit params).
        prev_feature_types = estimator.get_params().get('feature_types')
        prev_enable_categorical = estimator.get_params().get('enable_categorical')
        set_cat_indices = set(cat_indices)
        feature_types = [
            'c' if i in set_cat_indices else 'q'
            for i in range(len(X_train_features_names_out_))
        ]
        estimator.set_params(
            feature_types=feature_types, enable_categorical=True
        )
        if prev_feature_types is not None:
            warnings.warn(
                "The estimator's `feature_types` and `enable_categorical` "
                "parameters have been set to handle categorical features. "
                f"Previous values: feature_types={prev_feature_types}, "
                f"enable_categorical={prev_enable_categorical}.",
                IgnoredArgumentWarning
            )

    elif module == 'sklearn' and estimator_name in (
        'HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'
    ):
        # HistGradientBoosting accepts `categorical_features` as a
        # list of int indices via set_params (constructor param).
        prev_categorical = estimator.get_params().get('categorical_features')
        estimator.set_params(categorical_features=cat_indices)
        if prev_categorical not in (None, 'from_dtype'):
            warnings.warn(
                "The estimator's `categorical_features` parameter has been "
                "set to handle categorical features. Previous value: "
                f"`categorical_features={prev_categorical}`.",
                IgnoredArgumentWarning
            )

    return fit_kwargs


def cast_catboost_categorical_columns(
    X: np.ndarray,
    fit_kwargs: dict[str, object],
    estimator: object,
) -> np.ndarray:
    """
    Cast categorical columns of `X` to integer dtype as required by CatBoost
    when `X` is a numpy array.

    NaN values produced by the internal `OrdinalEncoder`
    (`unknown_value=np.nan`, `encoded_missing_value=np.nan`) are filled with
    `-1` before casting. `-1` cannot collide with the encoder's output range
    (`0..n-1`) and is treated by CatBoost as a novel category.

    No-op if `cat_features` is not in `fit_kwargs` or the estimator (or the
    last step of a Pipeline) is not a CatBoost model.

    Parameters
    ----------
    X : numpy ndarray
        Training or test matrix with categorical columns at the indices
        listed in `fit_kwargs['cat_features']`.
    fit_kwargs : dict
        Keyword arguments to pass to `estimator.fit`. Must contain the key
        `'cat_features'` (a list of int indices) for the cast to run.
    estimator : object
        Estimator the matrix will be passed to. The cast only runs if this
        is a CatBoost estimator (or a `Pipeline` whose last step is).

    Returns
    -------
    X : numpy ndarray
        Matrix with categorical columns cast to int and NaNs replaced by
        `-1`. Returned unchanged if the cast does not apply.

    """

    if 'cat_features' not in fit_kwargs:
        return X

    target_estimator = estimator
    if isinstance(target_estimator, Pipeline):
        target_estimator = target_estimator[-1]
    if type(target_estimator).__module__.split('.')[0] != 'catboost':
        return X

    cat_idx = np.asarray(fit_kwargs['cat_features'])
    X = X.astype(object)
    cat_block = np.asarray(X[:, cat_idx], dtype=float)
    X[:, cat_idx] = np.nan_to_num(cat_block, nan=-1).astype(int)

    return X


def cast_catboost_categorical_columns_dataframe(
    X: pd.DataFrame,
    fit_kwargs: dict[str, object],
    estimator: object,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Cast categorical columns of `X` to integer dtype as required by CatBoost
    when `X` is a pandas DataFrame.

    Two dtypes are supported for categorical columns:
    * `pandas.Categorical` — converted via `.cat.codes` (NaN -> -1 by default).
    * float (with NaN from the `OrdinalEncoder`) — `fillna(-1).astype(int)`.

    No-op if `cat_features` is not in `fit_kwargs` or the estimator (or the
    last step of a Pipeline) is not a CatBoost model.

    Parameters
    ----------
    X : pandas DataFrame
        Training matrix.
    fit_kwargs : dict
        Keyword arguments to pass to `estimator.fit`. Must contain the key
        `'cat_features'` (a list of int indices) for the cast to run.
    estimator : object
        Estimator the matrix will be passed to. The cast only runs if this
        is a CatBoost estimator (or a `Pipeline` whose last step is).
    feature_names : list of str
        Column names of `X` in positional order; used to translate the
        integer indices in `fit_kwargs['cat_features']` to column labels.

    Returns
    -------
    X : pandas DataFrame
        DataFrame with categorical columns cast to int. Returned unchanged
        if the cast does not apply.

    """

    if 'cat_features' not in fit_kwargs:
        return X

    target_estimator = estimator
    if isinstance(target_estimator, Pipeline):
        target_estimator = target_estimator[-1]
    if type(target_estimator).__module__.split('.')[0] != 'catboost':
        return X

    X = X.copy()
    cat_cols = [feature_names[i] for i in fit_kwargs['cat_features']]
    for col in cat_cols:
        if hasattr(X[col].dtype, 'categories'):
            X[col] = X[col].cat.codes.astype(int)
        else:
            X[col] = X[col].fillna(-1).astype(int)

    return X


def _get_catboost_cat_feature_indices(estimator: object) -> np.ndarray:
    """
    Return the indices of the categorical features of a fitted CatBoost
    estimator (regressor or classifier). At predict time, these columns must be
    cast to integer, as `cast_catboost_categorical_columns` does at fit time.

    Parameters
    ----------
    estimator : object
        Fitted estimator.

    Returns
    -------
    cat_indices : numpy ndarray
        Indices of the categorical features. Empty if the estimator is not a
        CatBoost model or was fitted without categorical features.

    """

    if type(estimator).__module__.split('.')[0] != 'catboost':
        return np.array([], dtype=int)

    return np.array(estimator.get_cat_feature_indices(), dtype=int)


def _get_estimator_categorical_set_params(
    forecaster: object
) -> dict[str, object]:
    """
    Return the current values of the estimator-level params that
    `configure_estimator_categorical_features` sets via `set_params` for
    XGBoost and HistGradientBoosting estimators.  For all other estimators an
    empty dict is returned so callers can treat this as a no-op.

    The function selects the estimator object that is actually mutated by
    `configure_estimator_categorical_features`:
    * `ForecasterDirect` and `ForecasterDirectMultiVariate` store
      per-step clones in `estimators_[1]`.
    * All other forecasters expose the shared template via `estimator`.

    Parameters
    ----------
    forecaster : object
        Forecaster whose estimator params should be captured.

    Returns
    -------
    params : dict
        XGBoost: `{'feature_types': ..., 'enable_categorical': ...}`
        HistGradientBoosting: `{'categorical_features': ...}`
        Others: `{}`

    """

    if type(forecaster).__name__ in ('ForecasterDirect', 'ForecasterDirectMultiVariate'):
        estimator = forecaster.estimators_[1]
    else:
        estimator = forecaster.estimator

    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    module = type(estimator).__module__.split('.')[0]
    estimator_name = type(estimator).__name__

    if module == 'xgboost':
        p = estimator.get_params()
        return {
            'feature_types': p.get('feature_types'),
            'enable_categorical': p.get('enable_categorical', False),
        }
    elif module == 'sklearn' and estimator_name in (
        'HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'
    ):
        p = estimator.get_params()
        return {'categorical_features': p.get('categorical_features')}

    return {}


def _restore_estimator_categorical_set_params(
    forecaster: object,
    params: dict[str, object]
) -> None:
    """
    Restore the estimator-level params previously captured by
    `_get_estimator_categorical_set_params`.  No-op when `params` is empty.

    Parameters
    ----------
    forecaster : object
        Forecaster whose estimator params should be restored.
    params : dict
        Dict previously returned by `_get_estimator_categorical_set_params`.

    """

    if not params:
        return

    if type(forecaster).__name__ in ('ForecasterDirect', 'ForecasterDirectMultiVariate'):
        estimator = forecaster.estimators_[1]
    else:
        estimator = forecaster.estimator

    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    estimator.set_params(**params)


def check_y(
    y: Any,
    series_id: str = "`y`",
    allow_nan: bool = False
) -> None:
    """
    Raise Exception if `y` is not pandas Series or if it has missing values.
    
    Parameters
    ----------
    y : Any
        Time series values.
    series_id : str, default '`y`'
        Identifier of the series used in the warning message.
    allow_nan : bool, default False
        If `True`, skip the check for missing values.
    
    Returns
    -------
    None
    
    """
    
    if not isinstance(y, pd.Series):
        raise TypeError(
            f"{series_id} must be a pandas Series with a DatetimeIndex or a RangeIndex. "
            f"Found {type(y)}."
        )
        
    if not allow_nan:
        if y.isna().to_numpy().any():
            raise ValueError(f"{series_id} has missing values.")
    
    return


def check_exog(
    exog: pd.Series | pd.DataFrame,
    allow_nan: bool = True,
    series_id: str = "`exog`"
) -> None:
    """
    Raise Exception if `exog` is not pandas Series or pandas DataFrame.
    If `allow_nan = True`, issue a warning if `exog` contains NaN values.
    
    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variable/s included as predictor/s.
    allow_nan : bool, default True
        If True, allows the presence of NaN values in `exog`. If False (default),
        issue a warning if `exog` contains NaN values.
    series_id : str, default '`exog`'
        Identifier of the series for which the exogenous variable/s are used
        in the warning message.

    Returns
    -------
    None

    """
    
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{series_id} must be a pandas Series or DataFrame. Got {type(exog)}."
        )
    
    if isinstance(exog, pd.Series) and exog.name is None:
        raise ValueError(f"When {series_id} is a pandas Series, it must have a name.")

    if not allow_nan:
        if exog.isna().to_numpy().any():
            warnings.warn(
                f"{series_id} has missing values. Most machine learning models "
                f"do not allow missing values. Fitting the forecaster may fail.", 
                MissingValuesWarning
            )
    
    return


def get_exog_dtypes(
    exog: pd.Series | pd.DataFrame, 
) -> dict[str, type]:
    """
    Store dtypes of `exog`.

    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variable/s included as predictor/s.

    Returns
    -------
    exog_dtypes : dict
        Dictionary with the dtypes in `exog`.
    
    """

    if isinstance(exog, pd.Series):
        exog_dtypes = {exog.name: exog.dtypes}
    else:
        exog_dtypes = exog.dtypes.to_dict()
    
    return exog_dtypes


def check_exog_dtypes(
    exog: pd.Series | pd.DataFrame,
    call_check_exog: bool = True,
    series_id: str = "`exog`"
) -> None:
    """
    Raise Exception if `exog` has categorical columns with non integer values.
    This is needed when using machine learning estimators that allow categorical
    features.
    Issue a Warning if `exog` has columns that are not `int`, `float`, or `category`.
    
    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variable/s included as predictor/s.
    call_check_exog : bool, default True
        If `True`, call `check_exog` function.
    series_id : str, default '`exog`'
        Identifier of the series for which the exogenous variable/s are used
        in the warning message.

    Returns
    -------
    None

    """

    if call_check_exog:
        check_exog(exog=exog, allow_nan=False, series_id=series_id)

    valid_dtypes = ("int", "Int", "float", "Float", "uint")

    if isinstance(exog, pd.DataFrame):
        unique_dtypes = set(exog.dtypes)
        has_invalid_dtype = False
        for dtype in unique_dtypes:
            if isinstance(dtype, pd.CategoricalDtype):
                try:
                    is_integer = np.issubdtype(dtype.categories.dtype, np.integer)
                except TypeError:
                    is_integer = False
                if not is_integer:
                    raise TypeError(
                        "Categorical dtypes in exog must contain only integer values. "
                        "See skforecast docs for more info about how to include "
                        "categorical features https://skforecast.org/"
                        "latest/user_guides/categorical-features.html"
                    )
            elif not dtype.name.startswith(valid_dtypes):
                has_invalid_dtype = True
        
        if has_invalid_dtype:
            warnings.warn(
                f"{series_id} may contain only `int`, `float` or `category` dtypes. "
                f"Most machine learning models do not allow other types of values. "
                f"Fitting the forecaster may fail.", 
                DataTypeWarning
            )
    
    else:
        
        dtype_name = str(exog.dtypes)
        if not (dtype_name.startswith(valid_dtypes) or dtype_name == "category"):
            warnings.warn(
                f"{series_id} may contain only `int`, `float` or `category` dtypes. Most "
                f"machine learning models do not allow other types of values. "
                f"Fitting the forecaster may fail.", 
                DataTypeWarning
            )

        if isinstance(exog.dtype, pd.CategoricalDtype):
            try:
                is_integer = np.issubdtype(exog.cat.categories.dtype, np.integer)
            except TypeError:
                is_integer = False
            if not is_integer:
                raise TypeError(
                    "Categorical dtypes in exog must contain only integer values. "
                    "See skforecast docs for more info about how to include "
                    "categorical features https://skforecast.org/"
                    "latest/user_guides/categorical-features.html"
                )


def check_interval(
    interval: list[float] | tuple[float] | None = None,
    ensure_symmetric_intervals: bool = False,
    quantiles: list[float] | tuple[float] | None = None,
    alpha: float = None,
    alpha_literal: str | None = 'alpha'
) -> None:
    """
    Check provided confidence interval sequence is valid.

    Parameters
    ----------
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. Sequence of bounds to
        compute. Values must be between 0 and 1 inclusive. For example, interval
        of 95% should be as `interval = [0.025, 0.975]`.
    ensure_symmetric_intervals : bool, default False
        If True, ensure that the intervals are symmetric.
    quantiles : list, tuple, default None
        Sequence of quantiles to compute, which must be between 0 and 1 
        inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
        `quantiles = [0.05, 0.5, 0.95]`.
    alpha : float, default None
        The confidence intervals used in ForecasterStats are (1 - alpha) %.
    alpha_literal : str, default 'alpha'
        Literal used in the exception message when `alpha` is provided.

    Returns
    -------
    None
    
    """

    if interval is not None:
        if not isinstance(interval, (list, tuple)):
            raise TypeError(
                "`interval` must be a `list` or `tuple`. For example, interval of 95% "
                "should be as `interval = [0.025, 0.975]`."
            )

        if len(interval) != 2:
            raise ValueError(
                "`interval` must contain exactly 2 values, respectively the "
                "lower and upper interval bounds. For example, interval of 95% "
                "should be as `interval = [0.025, 0.975]`."
            )

        if (interval[0] < 0.) or (interval[0] >= 1.):
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be >= 0 and < 1."
            )

        if (interval[1] <= 0.) or (interval[1] > 1.):
            raise ValueError(
                f"Upper interval bound ({interval[1]}) must be > 0 and <= 1."
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be less than the "
                f"upper interval bound ({interval[1]})."
            )
        
        if ensure_symmetric_intervals and interval[0] + interval[1] != 1.:
            raise ValueError(
                f"Interval must be symmetric, the sum of the lower, ({interval[0]}), "
                f"and upper, ({interval[1]}), interval bounds must be equal to "
                f"1. Got {interval[0] + interval[1]}."
            )
        
    if quantiles is not None:
        if not isinstance(quantiles, (list, tuple)):
            raise TypeError(
                "`quantiles` must be a `list` or `tuple`. For example, quantiles "
                "0.05, 0.5, and 0.95 should be as `quantiles = [0.05, 0.5, 0.95]`."
            )
        
        for q in quantiles:
            if (q < 0.) or (q > 1.):
                raise ValueError(
                    "All elements in `quantiles` must be >= 0 and <= 1."
                )
    
    if alpha is not None:
        if not isinstance(alpha, float):
            raise TypeError(
                f"`{alpha_literal}` must be a `float`. For example, interval of 95% "
                f"should be as `alpha = 0.05`."
            )

        if (alpha <= 0.) or (alpha >= 1):
            raise ValueError(
                f"`{alpha_literal}` must have a value between 0 and 1. Got {alpha}."
            )


def _check_exog_alignment(
    exog_name: str,
    exog_index: pd.Index,
    expected_index: pd.Index,
    align_by_index: bool
) -> None:
    """
    Check that `exog` has a value for each of the steps predicted.

    - If `align_by_index` is `False`, `exog` is used by position, so its first
    values must follow the dates of the steps predicted without gaps. A
    `ValueError` is raised otherwise. The first date must have already been
    checked.
    - If `align_by_index` is `True` (`ForecasterRecursiveMultiSeries`), `exog`
    is aligned with the predictions by its index, so it only has to contain
    the dates of the steps predicted. A `MissingValuesWarning` is issued if
    some of them are missing, since their values are filled with NaN, and a
    `ValueError` is raised if its index has duplicated dates, since it cannot
    be aligned.

    Parameters
    ----------
    exog_name : str
        Name of `exog` used in the error and warning messages.
    exog_index : pandas Index
        Index of `exog`.
    expected_index : pandas Index
        Index of the steps predicted, from 1 to the last step. It is created
        with `expand_index` from the index of `last_window`.
    align_by_index : bool
        If `True`, `exog` is aligned with the predictions by its index, so
        missing dates issue a warning instead of an error. If `False`, `exog`
        is used by position.

    Returns
    -------
    None

    """

    # NOTE: An index with the same frequency (or step) as the steps predicted
    # that starts at the first step has no gaps.
    if len(exog_index) > 0 and exog_index[0] == expected_index[0]:
        if isinstance(expected_index, pd.RangeIndex):
            if exog_index.step == expected_index.step:
                return
        elif exog_index.freq == expected_index.freq:
            return

    last_step = len(expected_index)
    # NOTE: If `exog` has fewer values than steps, a warning or an error has
    # already been issued, so only the first `len(exog)` steps are checked.
    n_steps = min(len(exog_index), last_step)
    expected_index = expected_index[:n_steps]
    if align_by_index:
        if exog_index.has_duplicates:
            raise ValueError(
                f"The index of {exog_name} has duplicated values, for example "
                f"{exog_index[exog_index.duplicated()][0]}. Each date must "
                f"appear only once."
            )
        is_misaligned = ~expected_index.isin(exog_index)
    else:
        exog_index = exog_index[:n_steps]
        is_misaligned = exog_index != expected_index

    if is_misaligned.any():
        position = np.flatnonzero(is_misaligned)[0]
        if align_by_index:
            warnings.warn(
                f"{exog_name} has no value for some of the {last_step} steps "
                f"predicted. The first one is {expected_index[position]} "
                f"(position {position}). Missing values are filled with NaN. "
                f"Most of machine learning models do not allow missing values. "
                f"Prediction method may fail.",
                MissingValuesWarning
            )
        else:
            raise ValueError(
                f"{exog_name} must have consecutive values following the "
                f"frequency of `last_window` for the {last_step} steps predicted.\n"
                f"    Expected index at position {position} : "
                f"{expected_index[position]}.\n"
                f"    {exog_name} index at position {position} : "
                f"{exog_index[position]}.\n"
                f"If there is no data for some steps, add them to {exog_name} "
                f"explicitly as NaN, for example:\n"
                f"    exog = exog.reindex(expand_index(last_window.index, "
                f"steps={last_step}))\n"
                f"where `expand_index` is in `skforecast.utils`, and `last_window` "
                f"is the window used to predict (by default, the last window "
                f"stored in the forecaster)."
            )


def check_predict_input(
    forecaster_name: str,
    steps: int | list[int],
    is_fitted: bool,
    exog_in_: bool,
    index_type_: type,
    index_freq_: str,
    window_size: int,
    last_window: pd.Series | pd.DataFrame | None,
    last_window_exog: pd.Series | pd.DataFrame | None = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    exog_names_in_: list[str] | None = None,
    max_step: int | None = None,
    levels: str | list[str] | None = None,
    levels_forecaster: str | list[str] | None = None,
    series_names_in_: list[str] | None = None,
    encoding: str | None = None
) -> None:
    """
    Check all inputs of predict method. This is a helper function to validate
    that inputs used in predict method match attributes of a forecaster already
    trained.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    steps : int, list
        Number of future steps predicted.
    is_fitted: bool
        Tag to identify if the estimator has been fitted (trained).
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    window_size: int
        Size of the window needed to create the predictors. It is equal to 
        `max_lag`.
    last_window : pandas Series, pandas DataFrame, None
        Values of the series used to create the predictors (lags) need in the 
        first iteration of prediction (t + 1).
    last_window_exog : pandas Series, pandas DataFrame, default None
        Values of the exogenous variables aligned with `last_window` in 
        ForecasterStats predictions.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variable/s included as predictor/s.
    exog_names_in_ : list, default None
        Names of the exogenous variables used during training.
    max_step: int, default None
        Maximum number of steps allowed (`ForecasterDirect` and 
        `ForecasterDirectMultiVariate`).
    levels : str, list, default None
        Time series to be predicted (`ForecasterRecursiveMultiSeries`
        and `ForecasterRnn).
    levels_forecaster : str, list, default None
        Time series used as output data of a multiseries problem in a RNN problem
        (`ForecasterRnn`).
    series_names_in_ : list, default None
        Names of the columns used during fit (`ForecasterRecursiveMultiSeries`, 
        `ForecasterDirectMultiVariate` and `ForecasterRnn`).
    encoding : str, default None
        Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).

    Returns
    -------
    None

    """

    if not is_fitted:
        raise NotFittedError(
            "This Forecaster instance is not fitted yet. Call `fit` with "
            "appropriate arguments before using predict."
        )

    if isinstance(steps, (int, np.integer)) and steps < 1:
        raise ValueError(
            f"`steps` must be an integer greater than or equal to 1. Got {steps}."
        )

    if isinstance(steps, list) and min(steps) < 1:
        raise ValueError(
           f"The minimum value of `steps` must be equal to or greater than 1. "
           f"Got {min(steps)}."
        )

    if max_step is not None:
        if max(steps) > max_step:
            raise ValueError(
                f"The maximum value of `steps` must be less than or equal to "
                f"the value of steps defined when initializing the forecaster. "
                f"Got {max(steps)}, but the maximum is {max_step}."
            )

    if forecaster_name in ['ForecasterRecursiveMultiSeries', 'ForecasterRnn']:
        if not isinstance(levels, (type(None), str, list)):
            raise TypeError(
                "`levels` must be a `list` of column names, a `str` of a "
                "column name or `None`."
            )

        levels_to_check = (
            levels_forecaster if forecaster_name == 'ForecasterRnn'
            else series_names_in_
        )
        unknown_levels = set(levels) - set(levels_to_check)
        if forecaster_name == 'ForecasterRnn':
            if len(unknown_levels) != 0:
                raise ValueError(
                    f"`levels` names must be included in the series used during fit "
                    f"({levels_to_check}). Got {levels}."
                )
        else:
            if len(unknown_levels) != 0 and last_window is not None and encoding is not None:
                if encoding == 'onehot':
                    warnings.warn(
                        f"`levels` {unknown_levels} were not included in training. The resulting "
                        f"one-hot encoded columns for this feature will be all zeros.",
                        UnknownLevelWarning
                    )
                else:
                    warnings.warn(
                        f"`levels` {unknown_levels} were not included in training. "
                        f"Unknown levels are encoded as NaN, which may cause the "
                        f"prediction to fail if the estimator does not accept NaN values.",
                        UnknownLevelWarning
                    )

    if exog is None and exog_in_:
        raise ValueError(
            "Forecaster trained with exogenous variable/s. "
            "Same variable/s must be provided when predicting."
        )

    if exog is not None and not exog_in_:
        raise ValueError(
            "Forecaster trained without exogenous variable/s. "
            "`exog` must be `None` when predicting."
        )

    # Checks last_window
    # Check last_window type (pd.Series or pd.DataFrame according to forecaster)
    if isinstance(last_window, type(None)) and forecaster_name not in [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterRnn'
    ]:
        raise ValueError(
            "`last_window` was not stored during training. If you don't want "
            "to retrain the Forecaster, provide `last_window` as argument."
        )

    if forecaster_name in [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterDirectMultiVariate',
        'ForecasterRnn'
    ]:
        if not isinstance(last_window, pd.DataFrame):
            raise TypeError(
                f"`last_window` must be a pandas DataFrame. Got {type(last_window)}."
            )

        last_window_cols = last_window.columns.to_list()

        if (
            forecaster_name in ["ForecasterRecursiveMultiSeries", "ForecasterRnn"]
            and len(set(levels) - set(last_window_cols)) != 0
        ):
            missing_levels = set(levels) - set(last_window_cols)
            raise ValueError(
                f"`last_window` must contain a column(s) named as the level(s) to be predicted. "
                f"The following `levels` are missing in `last_window`: {missing_levels}\n"
                f"Ensure that `last_window` contains all the necessary columns "
                f"corresponding to the `levels` being predicted.\n"
                f"    Argument `levels`     : {levels}\n"
                f"    `last_window` columns : {last_window_cols}\n"
                f"Example: If `levels = ['series_1', 'series_2']`, make sure "
                f"`last_window` includes columns named 'series_1' and 'series_2'."
            )

        if forecaster_name == 'ForecasterDirectMultiVariate':
            if len(set(series_names_in_) - set(last_window_cols)) > 0:
                raise ValueError(
                    f"`last_window` columns must be the same as the `series` "
                    f"column names used to create the X_train matrix.\n"
                    f"    `last_window` columns    : {last_window_cols}\n"
                    f"    `series` columns X train : {series_names_in_}"
                )
    else:
        if not isinstance(last_window, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"`last_window` must be a pandas Series or DataFrame. "
                f"Got {type(last_window)}."
            )
        if isinstance(last_window, pd.DataFrame) and last_window.shape[1] != 1:
            raise ValueError(
                f"`last_window` must be a pandas Series or a DataFrame with a "
                f"single column. Got {last_window.shape[1]} columns."
            )

    # Check last_window len, nulls and index (type and freq)
    if len(last_window) < window_size:
        raise ValueError(
            f"`last_window` must have as many values as needed to "
            f"generate the predictors. For this forecaster it is {window_size}."
        )
    if last_window.isna().to_numpy().any():
        warnings.warn(
            "`last_window` has missing values. Most of machine learning models do "
            "not allow missing values. Prediction method may either raise an "
            "error or return NaN predictions.",
            MissingValuesWarning
        )
    
    _, last_window_index = check_extract_values_and_index(
        data=last_window, data_label='`last_window`', ignore_freq=False, return_values=False
    )
    if not isinstance(last_window_index, index_type_):
        raise TypeError(
            f"Expected index of type {index_type_} for `last_window`. "
            f"Got {type(last_window_index)}."
        )
    if isinstance(last_window_index, pd.DatetimeIndex):
        if not last_window_index.freq == index_freq_:
            raise TypeError(
                f"Expected frequency of type {index_freq_} for `last_window`. "
                f"Got {last_window_index.freq}."
            )
    else:
        if not last_window_index.step == index_freq_:
            raise TypeError(
                f"Expected step of type {index_freq_} for `last_window`. "
                f"Got {last_window_index.step}."
            )

    # Checks exog
    if exog is not None:

        # Check type, nulls and expected type
        if forecaster_name in ['ForecasterRecursiveMultiSeries']:
            if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or dict. Got {type(exog)}."
                )
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}."
                )

        if isinstance(exog, dict):
            no_exog_levels = set(levels) - set(exog.keys())
            if no_exog_levels:
                warnings.warn(
                    f"`exog` does not contain keys for levels {no_exog_levels}. "
                    f"Missing levels are filled with NaN. Most of machine learning "
                    f"models do not allow missing values. Prediction method may fail.",
                    MissingExogWarning
                )
            exogs_to_check = [
                (f"`exog` for series '{k}'", v) 
                for k, v in exog.items() 
                if v is not None and k in levels
            ]
        else:
            exogs_to_check = [('`exog`', exog)]

        last_step = max(steps) if isinstance(steps, list) else steps
        expected_index = expand_index(last_window_index, last_step)
        # NOTE: ForecasterRecursiveMultiSeries aligns `exog` with the predictions
        # by index and column, so missing values are filled with NaN and only a
        # warning is issued. The rest of forecasters use `exog` by position.
        align_by_index = forecaster_name in ['ForecasterRecursiveMultiSeries']
        for exog_name, exog_to_check in exogs_to_check:

            if not isinstance(exog_to_check, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"{exog_name} must be a pandas Series or DataFrame. Got {type(exog_to_check)}"
                )

            if exog_to_check.isna().to_numpy().any():
                warnings.warn(
                    f"{exog_name} has missing values. Most of machine learning models "
                    f"do not allow missing values. Prediction method may fail.", 
                    MissingValuesWarning
                )

            # Check exog has many values as distance to max step predicted
            if len(exog_to_check) < last_step:
                if align_by_index:
                    warnings.warn(
                        f"{exog_name} doesn't have as many values as steps "
                        f"predicted, {last_step}. Missing values are filled "
                        f"with NaN. Most of machine learning models do not "
                        f"allow missing values. Prediction method may fail.",
                        MissingValuesWarning
                    )
                else: 
                    raise ValueError(
                        f"{exog_name} must have at least as many values as "
                        f"steps predicted, {last_step}."
                    )

            # Check name/columns are in exog_names_in_
            if isinstance(exog_to_check, pd.DataFrame):
                col_missing = set(exog_names_in_).difference(set(exog_to_check.columns))
                if col_missing:
                    if align_by_index:
                        warnings.warn(
                            f"{col_missing} not present in {exog_name}. All "
                            f"values will be NaN.",
                            MissingExogWarning
                        ) 
                    else:
                        raise ValueError(
                            f"Missing columns in {exog_name}. Expected {exog_names_in_}. "
                            f"Got {exog_to_check.columns.to_list()}."
                        )
            else:
                if exog_to_check.name is None:
                    raise ValueError(
                        f"When {exog_name} is a pandas Series, it must have a name. Got None."
                    )

                if exog_to_check.name not in exog_names_in_:
                    if align_by_index:
                        warnings.warn(
                            f"'{exog_to_check.name}' was not observed during training. "
                            f"{exog_name} is ignored. Exogenous variables must be one "
                            f"of: {exog_names_in_}.",
                            IgnoredArgumentWarning
                        )
                    else:
                        raise ValueError(
                            f"'{exog_to_check.name}' was not observed during training. "
                            f"Exogenous variables must be: {exog_names_in_}."
                        )

            # Check index dtype and freq
            _, exog_index = check_extract_values_and_index(
                data=exog_to_check, data_label=exog_name, ignore_freq=True, return_values=False
            )
            if not isinstance(exog_index, index_type_):
                raise TypeError(
                    f"Expected index of type {index_type_} for {exog_name}. "
                    f"Got {type(exog_index)}."
                )

            # Check exog starts one step ahead of last_window end.
            if not align_by_index and expected_index[0] != exog_index[0]:
                raise ValueError(
                    f"To make predictions {exog_name} must start one step "
                    f"ahead of `last_window`.\n"
                    f"    `last_window` ends at : {last_window.index[-1]}.\n"
                    f"    {exog_name} starts at : {exog_index[0]}.\n"
                    f"    Expected index : {expected_index[0]}."
                )

            _check_exog_alignment(
                exog_name      = exog_name,
                exog_index     = exog_index,
                expected_index = expected_index,
                align_by_index = align_by_index
            )

    # Checks ForecasterStats
    if forecaster_name == 'ForecasterStats':
        # Check last_window_exog type, len, nulls and index (type and freq)
        if last_window_exog is not None:
            if not exog_in_:
                raise ValueError(
                    "Forecaster trained without exogenous variable/s. "
                    "`last_window_exog` must be `None` when predicting."
                )

            if not isinstance(last_window_exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`last_window_exog` must be a pandas Series or a "
                    f"pandas DataFrame. Got {type(last_window_exog)}."
                )
            if len(last_window_exog) < window_size:
                raise ValueError(
                    f"`last_window_exog` must have as many values as needed to "
                    f"generate the predictors. For this forecaster it is {window_size}."
                )
            if last_window_exog.isna().to_numpy().any():
                warnings.warn(
                    "`last_window_exog` has missing values. Most of machine learning "
                    "models do not allow missing values. Prediction method may fail.",
                    MissingValuesWarning
            )
            _, last_window_exog_index = check_extract_values_and_index(
                data=last_window_exog, data_label='`last_window_exog`', return_values=False
            )
            if not isinstance(last_window_exog_index, index_type_):
                raise TypeError(
                    f"Expected index of type {index_type_} for `last_window_exog`. "
                    f"Got {type(last_window_exog_index)}."
                )
            if isinstance(last_window_exog_index, pd.DatetimeIndex):
                if not last_window_exog_index.freq == index_freq_:
                    raise TypeError(
                        f"Expected frequency of type {index_freq_} for "
                        f"`last_window_exog`. Got {last_window_exog_index.freq}."
                    )

            # Check all columns are in the pd.DataFrame, last_window_exog
            if isinstance(last_window_exog, pd.DataFrame):
                col_missing = set(exog_names_in_).difference(set(last_window_exog.columns))
                if col_missing:
                    raise ValueError(
                        f"Missing columns in `last_window_exog`. Expected {exog_names_in_}. "
                        f"Got {last_window_exog.columns.to_list()}."
                    )
            else:
                if last_window_exog.name is None:
                    raise ValueError(
                        "When `last_window_exog` is a pandas Series, it must have a "
                        "name. Got None."
                    )

                if last_window_exog.name not in exog_names_in_:
                    raise ValueError(
                        f"'{last_window_exog.name}' was not observed during training. "
                        f"Exogenous variables must be: {exog_names_in_}."
                    )


def check_residuals_input(
    forecaster_name: str,
    use_in_sample_residuals: bool,
    in_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    out_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    use_binned_residuals: bool,
    in_sample_residuals_by_bin_: dict[str | int, np.ndarray | dict[int, np.ndarray]] | None,
    out_sample_residuals_by_bin_: dict[str | int, np.ndarray | dict[int, np.ndarray]] | None,
    levels: list[str] | None = None,
    encoding: str | None = None
) -> None:
    """
    Check residuals input arguments in Forecasters.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    use_in_sample_residuals : bool
        Indicates if in-sample or out-of-sample residuals are used.
    in_sample_residuals_ : numpy ndarray, dict
        Residuals of the model when predicting training data.
    out_sample_residuals_ : numpy ndarray, dict
        Residuals of the model when predicting non training data.
    use_binned_residuals : bool
        Indicates if residuals are binned.
    in_sample_residuals_by_bin_ : dict
        In-sample residuals binned according to the predicted value each residual
        is associated with.
    out_sample_residuals_by_bin_ : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with.
    levels : list, default None
        Names of the series (levels) to be predicted (Forecasters multiseries).
    encoding : str, default None
        Encoding used to identify the different series (ForecasterRecursiveMultiSeries).

    Returns
    -------
    None
    
    """

    forecasters_multiseries = (
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirectMultiVariate',
        'ForecasterRnn'
    )

    if use_in_sample_residuals:
        if use_binned_residuals:
            residuals = in_sample_residuals_by_bin_
            literal = "in_sample_residuals_by_bin_"
        else:
            residuals = in_sample_residuals_
            literal = "in_sample_residuals_"
        
        # Check if residuals are empty or None
        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`store_in_sample_residuals = True` when fitting the forecaster "
                f"or use the `set_in_sample_residuals()` method before predicting."
            )
            
        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`, "
                        f"most likely because they were not present in the training data. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels.",
                        UnknownLevelWarning
                    )
    else:
        if use_binned_residuals:
            residuals = out_sample_residuals_by_bin_
            literal = "out_sample_residuals_by_bin_"
        else:
            residuals = out_sample_residuals_
            literal = "out_sample_residuals_"
        
        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`use_in_sample_residuals = True` or the "
                f"`set_out_sample_residuals()` method before predicting."
            )
            
        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels. "
                        f"Otherwise, Use the `set_out_sample_residuals()` method before "
                        f"predicting to set the residuals for these levels.",
                        UnknownLevelWarning
                    )

    if forecaster_name in forecasters_multiseries:
        for level in residuals.keys():
            level_residuals = residuals[level]
            if level_residuals is None or len(level_residuals) == 0:
                raise ValueError(
                    f"Residuals for level '{level}' are None. Check `forecaster.{literal}`."
                )


def check_extract_values_and_index(
    data: pd.Series | pd.DataFrame,
    data_label: str = '`y`',
    ignore_freq: bool = False,
    return_values: bool = True
) -> tuple[np.ndarray | None, pd.Index]:
    """
    Return values and index of series separately. Check that index is a pandas
    `DatetimeIndex` or `RangeIndex`. Optionally, check that the index has a
    frequency.
    
    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Time series.
    data_label : str, default '`y`'
        Label of the data to be used in warnings and errors.
    ignore_freq : bool, default False
        If `True`, ignore the frequency of the index. If `False`, check that the
        index is a pandas `DatetimeIndex` with a frequency.
    return_values : bool, default True
        If `True` return the values of `data` as numpy ndarray. This option is
        intended to avoid copying data when it is not necessary.

    Returns
    -------
    data_values : numpy ndarray, None
        Numpy array with values of `data`.
    data_index : pandas Index
        Index of `data`.

    """
    
    if isinstance(data.index, pd.DatetimeIndex):            
        if not ignore_freq and data.index.freq is None:
            raise ValueError(
                f"{data_label} has a pandas DatetimeIndex without a frequency. "
                f"To avoid this error, set the frequency of the DatetimeIndex."
            )
        data_index = data.index
    elif isinstance(data.index, pd.RangeIndex):
        data_index = data.index
    else:
        raise TypeError(
            f"{data_label} has an unsupported index type. The index must be a "
            f"pandas DatetimeIndex or a RangeIndex. Got {type(data.index)}."
        )

    data_values = data.to_numpy(copy=True).ravel() if return_values else None

    return data_values, data_index


def input_to_frame(
    data: pd.Series | pd.DataFrame,
    input_name: str
) -> pd.DataFrame:
    """
    Convert data to a pandas DataFrame. If data is a pandas Series, it is 
    converted to a DataFrame with a single column. If data is a DataFrame, 
    it is returned as is.

    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Input data.
    input_name : str
        Name of the input data. Accepted values are 'y', 'last_window' and 'exog'.

    Returns
    -------
    data : pandas DataFrame
        Input data as a DataFrame.

    """

    output_col_name = {
        'y': 'y',
        'last_window': 'y',
        'exog': 'exog'
    }

    if isinstance(data, pd.Series):
        data = data.to_frame(
            name=data.name if data.name is not None else output_col_name[input_name]
        )

    return data


def exog_to_direct(
    exog: pd.Series | pd.DataFrame,
    steps: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms `exog` to a pandas DataFrame with the shape needed for Direct
    forecasting.
    
    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.
    steps : int
        Number of steps that will be predicted using exog.

    Returns
    -------
    exog_direct : pandas DataFrame
        Exogenous variables transformed.
    exog_direct_names : list
        Names of the columns of the exogenous variables transformed. Only 
        created if `exog` is a pandas Series or DataFrame.
    
    """

    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}.")

    if isinstance(exog, pd.Series):
        exog = exog.to_frame()

    n_rows = len(exog)
    exog_idx = exog.index
    exog_cols = exog.columns
    exog_direct = []
    for i in range(steps):
        exog_step = exog.iloc[i : n_rows - (steps - 1 - i), ]
        exog_step.index = pd.RangeIndex(len(exog_step))
        exog_step.columns = [f"{col}_step_{i + 1}" for col in exog_cols]
        exog_direct.append(exog_step)

    exog_direct = pd.concat(exog_direct, axis=1) if steps > 1 else exog_direct[0]

    exog_direct_names = exog_direct.columns.to_list()
    exog_direct.index = exog_idx[-len(exog_direct):]
    
    return exog_direct, exog_direct_names


def exog_to_direct_numpy(
    exog: np.ndarray | pd.Series | pd.DataFrame,
    steps: int
) -> tuple[np.ndarray, list[str] | None]:
    """
    Transforms `exog` to numpy ndarray with the shape needed for Direct
    forecasting.
    
    Parameters
    ----------
    exog : numpy ndarray, pandas Series, pandas DataFrame
        Exogenous variables, shape(samples,). If exog is a pandas format, the 
        direct exog names are created.
    steps : int
        Number of steps that will be predicted using exog.

    Returns
    -------
    exog_direct : numpy ndarray
        Exogenous variables transformed.
    exog_direct_names : list, None
        Names of the columns of the exogenous variables transformed. Only 
        created if `exog` is a pandas Series or DataFrame.

    """

    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog_cols = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
        exog_direct_names = [
            f"{col}_step_{i + 1}" for i in range(steps) for col in exog_cols
        ]
        exog = exog.to_numpy()
    else:
        exog_direct_names = None
        if not isinstance(exog, np.ndarray):
            raise TypeError(
                f"`exog` must be a numpy ndarray, pandas Series or DataFrame. "
                f"Got {type(exog)}."
            )

    if exog.ndim == 1:
        exog = np.expand_dims(exog, axis=1)

    n_rows = len(exog)
    exog_direct = [exog[i : n_rows - (steps - 1 - i)] for i in range(steps)]
    exog_direct = np.concatenate(exog_direct, axis=1) if steps > 1 else exog_direct[0]
    
    return exog_direct, exog_direct_names


def _is_utc_anchored_index(
    index: pd.DatetimeIndex,
    freq: pd.DateOffset
) -> bool:
    """
    Check whether a timezone-aware DatetimeIndex advances in fixed UTC steps
    instead of local calendar steps.

    With a timezone that observes daylight saving time, a frequency of days
    or longer (`'D'`, `'W'`, `'MS'`...) can follow two conventions that share
    the same `freq`: local calendar steps, where the local time of day is
    constant (e.g. data stamped at local midnight), or fixed UTC steps, where
    the UTC time of day is constant and the local time shifts one hour at each
    daylight saving change (e.g. data stamped at UTC midnight and then
    converted to a local timezone). `pandas.date_range` always assumes the
    first one.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Index used to identify the convention.
    freq : pandas DateOffset
        Frequency of the index.

    Returns
    -------
    is_utc_anchored : bool
        `True` if the index advances in fixed UTC steps.

    Notes
    -----
    If `index` contains a daylight saving change, the convention is identified
    from the time of day that remains constant. If it does not, both
    conventions are indistinguishable, and the index is considered UTC
    anchored only when its timestamps are at UTC midnight but not at local
    midnight.

    """

    if index.tz is None or len(index) == 0:
        return False

    # Intraday frequencies are always generated by pandas in fixed steps.
    if isinstance(freq, pd.offsets.Tick) and not isinstance(freq, pd.offsets.Day):
        return False

    index_utc = index.tz_convert("UTC")
    index_local = index.tz_localize(None)
    time_utc = (index_utc - index_utc.normalize()).unique()
    time_local = (index_local - index_local.normalize()).unique()

    if len(time_utc) != 1:
        return False
    if len(time_local) != 1:
        return True

    midnight = pd.Timedelta(0)
    is_utc_anchored = time_utc[0] == midnight and time_local[0] != midnight

    return is_utc_anchored


def _date_range_from_index(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: pd.DateOffset
) -> pd.DatetimeIndex:
    """
    Create a date range between `start` and `end` following the same time
    convention as `index`. It behaves as `pandas.date_range` unless `index` is
    a timezone-aware index that advances in fixed UTC steps, in which case the
    range is generated in UTC and converted back to the timezone of `index`.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Index used to identify the time convention.
    start : pandas Timestamp
        Left bound of the range.
    end : pandas Timestamp
        Right bound of the range.
    freq : pandas DateOffset
        Frequency of the range.

    Returns
    -------
    span_index : pandas DatetimeIndex
        Date range between `start` and `end`.

    """

    if _is_utc_anchored_index(index=index, freq=freq):
        span_index = pd.date_range(
                         start = start.tz_convert("UTC"),
                         end   = end.tz_convert("UTC"),
                         freq  = freq
                     ).tz_convert(index.tz)
    else:
        span_index = pd.date_range(start=start, end=end, freq=freq)

    return span_index


def date_to_index_position(
    index: pd.Index,
    date_input: int | str | pd.Timestamp,
    method: str = 'prediction',
    date_literal: str = 'steps',
    kwargs_pd_to_datetime: dict = {}
) -> int:
    """
    Transform a datetime string or pandas Timestamp to an integer. The integer
    represents the position of the datetime in the index.
    
    Parameters
    ----------
    index : pandas Index
        Original datetime index (must be a pandas DatetimeIndex if `date_input` 
        is not an int).
    date_input : int, str, pandas Timestamp
        Datetime to transform to integer.
        
        + If int, returns the same integer.
        + If str or pandas Timestamp, it is converted and expanded into the index.
    method : str, default 'prediction'
        Can be 'prediction' or 'validation'. 
        
        + If 'prediction', the date must be later than the last date in the index.
        + If 'validation', the date must be within the index range.
    date_literal : str, default 'steps'
        Variable name used in error messages.
    kwargs_pd_to_datetime : dict, default {}
        Additional keyword arguments to pass to `pd.to_datetime()`.
    
    Returns
    -------
    output : int
        `date_input` transformed to integer position in the `index`.
        
        + If `date_input` is an integer, it returns the same integer.
        + If method is 'prediction', number of steps to predict from the last
        date in the index.
        + If method is 'validation', position plus one of the date in the index,
        this is done to include the target date in the training set when using 
        pandas iloc with slices.
    
    """

    if method not in ['prediction', 'validation']:
        raise ValueError("`method` must be 'prediction' or 'validation'.")
    
    if isinstance(date_input, (str, pd.Timestamp)):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                f"Index must be a pandas DatetimeIndex when `{date_literal}` is "
                f"not an integer. Check input series or last window."
            )
        
        target_date = pd.to_datetime(date_input, **kwargs_pd_to_datetime)
        last_date = pd.to_datetime(index[-1])

        if method == 'prediction':
            if target_date <= last_date:
                raise ValueError(
                    "If `steps` is a date, it must be greater than the last date "
                    "in the index."
                )
            span_index = _date_range_from_index(
                             index = index,
                             start = last_date,
                             end   = target_date,
                             freq  = index.freq
                         )
            output = len(span_index) - 1
        elif method == 'validation':
            first_date = pd.to_datetime(index[0])
            if target_date < first_date or target_date > last_date:
                raise ValueError(
                    "If `initial_train_size` is a date, it must be greater than "
                    "the first date in the index and less than the last date."
                )
            span_index = _date_range_from_index(
                             index = index,
                             start = first_date,
                             end   = target_date,
                             freq  = index.freq
                         )
            output = len(span_index)

    elif isinstance(date_input, (int, np.integer)):
        output = date_input

    else:
        raise TypeError(
            f"`{date_literal}` must be an integer, string, or pandas Timestamp."
        )
    
    return output


def expand_index(
    index: pd.Index | None, 
    steps: int
) -> pd.Index:
    """
    Create a new index of length `steps` starting at the end of the index.
    
    Parameters
    ----------
    index : pandas Index, None
        Original index.
    steps : int
        Number of steps to expand.

    Returns
    -------
    new_index : pandas Index
        New index.

    """

    if not isinstance(steps, (int, np.integer)):
        raise TypeError(f"`steps` must be an integer. Got {type(steps)}.")

    if isinstance(index, pd.Index):
        
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freq
            if freq is None:
                inferred = pd.infer_freq(index) if len(index) >= 3 else None
                freq = (
                    pd.tseries.frequencies.to_offset(inferred)
                    if inferred is not None
                    else None
                )
            if freq is None:
                raise ValueError(
                    "Could not infer a frequency from `index`. This can happen "
                    "when the index has fewer than 3 observations or is "
                    "irregularly spaced. Set an explicit frequency (e.g. "
                    "`index.freq = 'D'` or `series = series.asfreq('D')`) "
                    "before calling this function."
                )
            if _is_utc_anchored_index(index=index, freq=freq):
                # Timezone-aware index that advances in fixed UTC steps: the
                # new index is generated in UTC to keep the same convention.
                new_index = pd.date_range(
                                start   = index[-1].tz_convert("UTC") + freq,
                                periods = steps,
                                freq    = freq
                            ).tz_convert(index.tz)
            else:
                # NOTE: The range starts at the last date and drops it. Adding
                # `freq` to it would add a fixed 24 hours with daily frequencies,
                # which shifts the local time of day at a daylight saving change.
                new_index = pd.date_range(
                                start   = index[-1],
                                periods = steps + 1,
                                freq    = freq
                            )[1:]
        elif isinstance(index, pd.RangeIndex):
            new_index = pd.RangeIndex(
                            start = index[-1] + index.step,
                            stop  = index[-1] + index.step + steps * index.step,
                            step  = index.step
                        )
        else:
            raise TypeError(
                "Argument `index` must be a pandas DatetimeIndex or RangeIndex."
            )
    else:
        new_index = pd.RangeIndex(
                        start = 0,
                        stop  = steps
                    )
    
    return new_index


def transform_numpy(
    array: np.ndarray,
    transformer: object | None,
    fit: bool = False,
    inverse_transform: bool = False,
    force_single_column: bool = False
) -> np.ndarray:
    """
    Transform raw values of a numpy ndarray with a scikit-learn alike 
    transformer, preprocessor or ColumnTransformer. The transformer used must 
    have the following methods: fit, transform, fit_transform and 
    inverse_transform. ColumnTransformers are not allowed since they do not 
    have inverse_transform method.

    Parameters
    ----------
    array : numpy ndarray
        Array to be transformed.
    transformer : scikit-learn alike transformer, preprocessor, or ColumnTransformer.
        Scikit-learn alike transformer (preprocessor) with methods: fit, transform,
        fit_transform and inverse_transform.
    fit : bool, default False
        Train the transformer before applying it.
    inverse_transform : bool, default False
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.
    force_single_column : bool, default False
        If `True`, raise an error if the transformer generates more than one
        column. This ensures that the output array is always 1D or single-column.

    Returns
    -------
    array_transformed : numpy ndarray
        Transformed array.

    """

    if transformer is None:
        return array
    
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"`array` argument must be a numpy ndarray. Got {type(array)}"
        )
    
    original_ndim = array.ndim
    original_shape = array.shape
    reshaped_for_inverse = False
    
    if original_ndim == 1:
        array = array.reshape(-1, 1)

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", 
            message="X does not have valid feature names", 
            category=UserWarning
        )
        if inverse_transform:
            # Vectorized inverse transformation for 2D arrays with multiple columns.
            # Reshape to single column, transform, and reshape back.
            # This is faster than applying the transformer column by column.
            if array.shape[1] > 1:
                array = array.reshape(-1, 1)
                reshaped_for_inverse = True
            array_transformed = transformer.inverse_transform(array)
        elif fit:
            array_transformed = transformer.fit_transform(array)
        else:
            array_transformed = transformer.transform(array)

    if hasattr(array_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense
        array_transformed = array_transformed.toarray()

    if isinstance(array_transformed, (pd.Series, pd.DataFrame)):
        array_transformed = array_transformed.to_numpy()

    if force_single_column and array_transformed.ndim > 1 and array_transformed.shape[1] > 1:
        raise ValueError(
            f"`transformer_y` and `transformer_series` must return a single column. "
            f"The transformer generated {array_transformed.shape[1]} columns. "
            f"Transformers that expand target series into multiple feature "
            f"columns are not supported; use `window_features` or pass "
            f"those features through `exog` instead."
        )

    # Reshape back to original shape only if we reshaped for inverse_transform
    if reshaped_for_inverse:
        array_transformed = array_transformed.reshape(original_shape)

    if original_ndim == 1:
        array_transformed = array_transformed.ravel()

    return array_transformed


def transform_series(
    series: pd.Series,
    transformer: object | None,
    fit: bool = False,
    inverse_transform: bool = False,
    force_single_column: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Transform raw values of pandas Series with a scikit-learn alike 
    transformer, preprocessor or ColumnTransformer. The transformer used must 
    have the following methods: fit, transform, fit_transform and 
    inverse_transform. ColumnTransformers are not allowed since they do not 
    have inverse_transform method.

    Parameters
    ----------
    series : pandas Series
        Series to be transformed.
    transformer : scikit-learn alike transformer, preprocessor, or ColumnTransformer.
        Scikit-learn alike transformer (preprocessor) with methods: fit, transform,
        fit_transform and inverse_transform.
    fit : bool, default False
        Train the transformer before applying it.
    inverse_transform : bool, default False
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.
    force_single_column : bool, default False
        If `True`, raise an error if the transformer generates more than one
        column. This ensures that the output is always a pandas Series.

    Returns
    -------
    series_transformed : pandas Series, pandas DataFrame
        Transformed Series. Depending on the transformer used, the output may 
        be a Series or a DataFrame.

    """
    
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"`series` argument must be a pandas Series. Got {type(series)}."
        )
        
    if transformer is None:
        return series

    series_name = series.name if series.name is not None else 'no_name'
    data = series.to_frame(name=series_name)

    # If argument feature_names_in_ exits, is overwritten to allow using the 
    # transformer on other series than those that were passed during fit.
    if not fit and hasattr(transformer, 'feature_names_in_') and transformer.feature_names_in_[0] != data.columns[0]:
        transformer = deepcopy(transformer)
        transformer.feature_names_in_ = np.array([data.columns[0]], dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if inverse_transform:
            values_transformed = transformer.inverse_transform(data)
        elif fit:
            values_transformed = transformer.fit_transform(data)
        else:
            values_transformed = transformer.transform(data)   

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense array.
        values_transformed = values_transformed.toarray()
    
    if isinstance(values_transformed, np.ndarray) and values_transformed.shape[1] == 1:
        series_transformed = pd.Series(
                                 data  = values_transformed.ravel(),
                                 index = data.index,
                                 name  = data.columns[0]
                             )
    elif isinstance(values_transformed, pd.DataFrame) and values_transformed.shape[1] == 1:
        series_transformed = values_transformed.squeeze()
    else:
        if force_single_column:
            raise ValueError(
                f"`transformer_y` and `transformer_series` must return a single column. "
                f"The transformer generated {values_transformed.shape[1]} columns. "
                f"Transformers that expand target series into multiple feature "
                f"columns are not supported; use `window_features` or pass "
                f"those features through `exog` instead."
            )
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names_out = transformer.get_feature_names_out()
            if len(feature_names_out) != values_transformed.shape[1]:
                feature_names_out = [f'transformed_{i}' for i in range(values_transformed.shape[1])]
        else:
            feature_names_out = [f'transformed_{i}' for i in range(values_transformed.shape[1])]

        series_transformed = pd.DataFrame(
                                 data    = values_transformed,
                                 index   = data.index,
                                 columns = feature_names_out
                             )

    return series_transformed


def transform_dataframe(
    df: pd.DataFrame,
    transformer: object | None,
    fit: bool = False,
    inverse_transform: bool = False,
    force_single_column: bool = False
) -> pd.DataFrame:
    """
    Transform raw values of pandas DataFrame with a scikit-learn alike 
    transformer, preprocessor or ColumnTransformer. The transformer used must 
    have the following methods: fit, transform, fit_transform and 
    inverse_transform. ColumnTransformers are not allowed since they do not 
    have inverse_transform method.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to be transformed.
    transformer : scikit-learn alike transformer, preprocessor, or ColumnTransformer.
        Scikit-learn alike transformer (preprocessor) with methods: fit, transform,
        fit_transform and inverse_transform.
    fit : bool, default False
        Train the transformer before applying it.
    inverse_transform : bool, default False
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.
    force_single_column : bool, default False
        If `True`, raise an error if the transformer generates more than one
        column. This ensures that the output DataFrame has a single column.

    Returns
    -------
    df_transformed : pandas DataFrame
        Transformed DataFrame.

    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"`df` argument must be a pandas DataFrame. Got {type(df)}"
        )

    if transformer is None:
        return df

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )
 
    if inverse_transform:
        values_transformed = transformer.inverse_transform(df)
    elif fit:
        values_transformed = transformer.fit_transform(df)
    else:
        values_transformed = transformer.transform(df)

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense
        values_transformed = values_transformed.toarray()

    if isinstance(values_transformed, pd.DataFrame):
        df_transformed = values_transformed
    else:
        values_transformed = np.asarray(values_transformed)
        if values_transformed.ndim == 1:
            values_transformed = values_transformed.reshape(-1, 1)

        feature_names_out = (
            transformer.get_feature_names_out()
            if hasattr(transformer, 'get_feature_names_out')
            else df.columns
        )
        if len(feature_names_out) != values_transformed.shape[1]:
            feature_names_out = [f'transformed_{i}' for i in range(values_transformed.shape[1])]

        df_transformed = pd.DataFrame(
                             data    = values_transformed,
                             index   = df.index,
                             columns = feature_names_out
                         )

    if force_single_column and df_transformed.shape[1] > 1:
        raise ValueError(
            f"`transformer_y` and `transformer_series` must return a single column. "
            f"The transformer generated {df_transformed.shape[1]} columns. "
            f"Transformers that expand target series into multiple feature "
            f"columns are not supported; use `window_features` or pass "
            f"those features through `exog` instead."
        )

    return df_transformed


def manage_warnings(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that safely manages skforecast warning suppression using
    `warnings.catch_warnings()` context manager. If the decorated function
    receives a `suppress_warnings=True` keyword argument, all skforecast
    warnings are suppressed within its execution scope. Warning filter state
    is automatically saved and restored, making this safe for nested calls
    and exception scenarios.

    By using `warnings.catch_warnings()`, the filter state is saved on entry and 
    restored on exit — even if an exception is raised — so nested decorated 
    functions never interfere with each other's suppression settings.

    The decorator's type signature uses module-level type variables:

    - `P` (`ParamSpec`): Captures the full parameter specification
      (positional and keyword arguments) of the decorated function, so that
      type checkers preserve the original call signature through the wrapper.
    - `R` (`TypeVar`): Captures the return type of the decorated function,
      ensuring the wrapper advertises the same return type.

    Parameters
    ----------
    func : Callable[P, R]
        The function to decorate. Expected to accept a `suppress_warnings`
        keyword argument.

    Returns
    -------
    Callable[P, R]
        The wrapped function with safe warning management.

    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        suppress = kwargs.get('suppress_warnings', False)
        with warnings.catch_warnings():
            if suppress:
                for category in warn_skforecast_categories:
                    warnings.filterwarnings('ignore', category=category)
            return func(*args, **kwargs)
    return wrapper


def _decompose_offset(offset: Any) -> Any:
    """
    Replace a generic `pandas.DateOffset` (e.g. `pd.DateOffset(days=7)`), which
    skops cannot serialize, with a plain dict.

    The other pandas offsets (e.g. `Day`, `MonthBegin` or `CustomBusinessDay`)
    and any other value are returned unchanged.

    Parameters
    ----------
    offset : object
        Value to decompose.

    Returns
    -------
    offset : object
        Plain dict with the `n`, `normalize` and keyword arguments of the
        `pandas.DateOffset`, or the value itself. The `offset_type_` key marks a
        decomposed offset.

    """

    if type(offset) is pd.DateOffset:
        offset = {
            'offset_type_': 'DateOffset',
            'n': offset.n,
            'normalize': offset.normalize,
            'kwds': offset.kwds,
        }

    return offset


def _compose_offset(offset: Any) -> Any:
    """
    Rebuild a `pandas.DateOffset` from the dict produced by `_decompose_offset`.

    Parameters
    ----------
    offset : object
        Plain dict representation of the `pandas.DateOffset`, as returned by
        `_decompose_offset`, or a value that was not decomposed.

    Returns
    -------
    offset : object
        Reconstructed `pandas.DateOffset`. A value that was not decomposed is
        returned unchanged.

    """

    if isinstance(offset, dict) and offset.get('offset_type_') == 'DateOffset':
        offset = pd.DateOffset(
            n=offset['n'], normalize=offset['normalize'], **offset['kwds']
        )

    return offset


def _decompose_index(index: pd.Index) -> dict[str, Any]:
    """
    Decompose a pandas Index into a plain dict that skops can serialize.

    A `DatetimeIndex` stores its values as integers since the epoch (UTC)
    together with their unit, time zone name and frequency, a `RangeIndex`
    stores its `start`, `stop`, and `step`, and any other index type stores its
    values as a list.

    Parameters
    ----------
    index : pandas Index
        Index to decompose.

    Returns
    -------
    payload : dict
        Plain dict representation of the index. The `index_type_` key
        (`'datetime'`, `'range'`, or `'other'`) selects how it is rebuilt.

    """

    if isinstance(index, pd.DatetimeIndex):
        # NOTE: The time zone is stored by name, since skops cannot serialize
        # the time zone objects. A time zone that cannot be rebuilt from its
        # name (e.g. from dateutil, or a datetime.timezone with a custom name)
        # raises an error here, instead of failing or changing when loading.
        tz = None if index.tz is None else str(index.tz)
        tz_zoneinfo = isinstance(index.tz, zoneinfo.ZoneInfo)
        if tz is not None:
            try:
                tz_rebuilt = zoneinfo.ZoneInfo(tz) if tz_zoneinfo else tz
                is_rebuilt = (
                    pd.DatetimeTZDtype(unit=index.unit, tz=tz_rebuilt) == index.dtype
                )
            except (KeyError, ValueError):
                is_rebuilt = False
            if not is_rebuilt:
                raise ValueError(
                    f"The time zone {index.tz!r} of the index cannot be saved with "
                    f"backend='skops' because it cannot be rebuilt from its name "
                    f"{tz!r}. Convert the index to a named time zone (e.g. "
                    f"'Europe/Madrid') or use another backend."
                )
        payload = {
            'index_type_': 'datetime',
            'index': index.asi8,
            'unit': index.unit,
            'tz': tz,
            'tz_zoneinfo': tz_zoneinfo,
            'freq': _decompose_offset(index.freq),
            'index_name': index.name,
        }
    elif isinstance(index, pd.RangeIndex):
        payload = {
            'index_type_': 'range',
            'range': [index.start, index.stop, index.step],
            'index_name': index.name,
        }
    else:
        payload = {
            'index_type_': 'other',
            'index': index.to_list(),
            'index_name': index.name,
        }

    return payload


def _compose_index(payload: dict[str, Any]) -> pd.Index:
    """
    Rebuild a pandas Index from the dict produced by `_decompose_index`.

    Parameters
    ----------
    payload : dict
        Plain dict representation of the index, as returned by
        `_decompose_index`.

    Returns
    -------
    index : pandas Index
        Reconstructed index, matching the original type: `DatetimeIndex` (with
        its time zone and frequency restored), `RangeIndex`, or a generic
        `Index`.

    """

    if payload['index_type_'] == 'datetime':
        if 'unit' in payload:
            values = np.asarray(payload['index'], dtype=np.int64)
            index = pd.DatetimeIndex(values.view(f"M8[{payload['unit']}]"))
            if payload['tz'] is not None:
                tz = payload['tz']
                if payload['tz_zoneinfo']:
                    tz = zoneinfo.ZoneInfo(tz)
                index = index.tz_localize('UTC').tz_convert(tz)
        else:
            # NOTE: Files saved with skforecast < 0.26 store the timestamps as
            # strings, which only keep the UTC offset of each one. They are
            # rebuilt with the offset of the first one (or without time zone).
            index = pd.to_datetime(payload['index'], format='ISO8601', utc=True)
            index = index.tz_convert(pd.Timestamp(payload['index'][0]).tz)
        index = pd.DatetimeIndex(
            index, freq=_compose_offset(payload['freq']), name=payload['index_name']
        )
    elif payload['index_type_'] == 'range':
        start, stop, step = payload['range']
        index = pd.RangeIndex(
            start=start, stop=stop, step=step, name=payload['index_name']
        )
    else:
        index = pd.Index(payload['index'], name=payload['index_name'])

    return index


def _decompose_pandas_object(
    obj: pd.Series | pd.DataFrame | pd.Index
) -> dict[str, Any]:
    """
    Decompose a pandas object into a plain dict that skops can serialize.

    The values are kept as a numpy array (serialized natively by skops, preserving
    dtype) and the index is decomposed with `_decompose_index`. The `object_type_`
    marker records the original type so `_compose_pandas_object` can invert it.

    Note that `to_numpy()` collapses a DataFrame to a single dtype, so per-column
    dtypes are not preserved. This is fine for the attributes this is used on
    (`last_window_` is homogeneous numeric, `training_range_` is an Index).

    Parameters
    ----------
    obj : pandas Series, pandas DataFrame, pandas Index
        Object to decompose.

    Returns
    -------
    payload : dict
        Plain dict representation of `obj`.

    """

    if isinstance(obj, pd.DataFrame):
        payload = {
            'object_type_': 'DataFrame',
            'data': obj.to_numpy(),
            'columns': obj.columns.to_list(),
            **_decompose_index(obj.index),
        }
    elif isinstance(obj, pd.Series):
        payload = {
            'object_type_': 'Series',
            'data': obj.to_numpy(),
            'name': obj.name,
            **_decompose_index(obj.index),
        }
    else:
        payload = {
            'object_type_': 'Index',
            **_decompose_index(obj),
        }

    return payload


def _compose_pandas_object(
    payload: dict[str, Any]
) -> pd.Series | pd.DataFrame | pd.Index:
    """
    Rebuild a pandas object from the dict produced by `_decompose_pandas_object`.

    Parameters
    ----------
    payload : dict
        Plain dict representation of the pandas object, including the
        `object_type_` type marker.

    Returns
    -------
    obj : pandas Series, pandas DataFrame, pandas Index
        Reconstructed pandas object, matching the original type.

    """

    index = _compose_index(payload)

    if payload['object_type_'] == 'DataFrame':
        obj = pd.DataFrame(
            data=payload['data'], index=index, columns=payload['columns']
        )
    elif payload['object_type_'] == 'Series':
        obj = pd.Series(data=payload['data'], index=index, name=payload['name'])
    else:
        obj = index

    return obj


def _decompose_dtype(dtype: Any) -> Any:
    """
    Replace a pandas dtype that skops cannot serialize with a plain dict.

    A `CategoricalDtype` stores its categories as a numpy ndarray and whether
    they are ordered, while an `ArrowDtype` or a `DatetimeTZDtype` stores its
    name. Any other dtype is returned unchanged.

    Parameters
    ----------
    dtype : object
        Dtype to decompose.

    Returns
    -------
    dtype : object
        Plain dict representation of the dtype, or the dtype itself if skops
        can serialize it. The `dtype_type_` key (`'category'` or `'name'`)
        selects how it is rebuilt.

    """

    if isinstance(dtype, pd.CategoricalDtype):
        dtype = {
            'dtype_type_': 'category',
            'categories': dtype.categories.to_numpy(),
            'ordered': dtype.ordered,
        }
    elif isinstance(dtype, (pd.ArrowDtype, pd.DatetimeTZDtype)):
        dtype = {'dtype_type_': 'name', 'name': str(dtype)}

    return dtype


def _compose_dtype(dtype: Any) -> Any:
    """
    Rebuild a pandas dtype from the dict produced by `_decompose_dtype`.

    Parameters
    ----------
    dtype : object
        Plain dict representation of the dtype, as returned by
        `_decompose_dtype`, or a dtype that was not decomposed.

    Returns
    -------
    dtype : object
        Reconstructed dtype. A dtype that was not decomposed is returned
        unchanged.

    """

    if isinstance(dtype, dict) and dtype.get('dtype_type_') == 'category':
        dtype = pd.CategoricalDtype(
            categories=dtype['categories'], ordered=dtype['ordered']
        )
    elif isinstance(dtype, dict) and dtype.get('dtype_type_') == 'name':
        dtype = pd.api.types.pandas_dtype(dtype['name'])

    return dtype


def _skops_decompose_forecaster(forecaster: object) -> object:
    """
    Return a shallow copy of a forecaster whose attributes that skops cannot
    serialize are replaced with plain dicts. The forecaster itself is not
    modified.

    The decomposed attributes are:

    - `last_window_` and `training_range_`, which may be a pandas object
    (single-series forecasters) or a dict of pandas objects (multi-series
    forecasters).
    - `exog_dtypes_in_` and `exog_dtypes_out_`, whose categorical, pyarrow and
    time zone aware dtypes are decomposed with `_decompose_dtype`.
    - `index_freq_`, `offset` and `window_size`, when they are a generic
    `pandas.DateOffset` (`offset` and `window_size` in `ForecasterEquivalentDate`).

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster created with skforecast library.

    Returns
    -------
    forecaster_decomposed : Forecaster
        Shallow copy of the forecaster with the decomposed attributes.

    """

    forecaster_decomposed = copy(forecaster)
    for attr in ('last_window_', 'training_range_'):
        value = getattr(forecaster, attr, None)
        if isinstance(value, dict):
            value = {k: _decompose_pandas_object(v) for k, v in value.items()}
        elif isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = _decompose_pandas_object(value)
        else:
            continue
        setattr(forecaster_decomposed, attr, value)
    for attr in ('exog_dtypes_in_', 'exog_dtypes_out_'):
        value = getattr(forecaster, attr, None)
        if isinstance(value, dict):
            value = {k: _decompose_dtype(v) for k, v in value.items()}
            setattr(forecaster_decomposed, attr, value)
    for attr in ('index_freq_', 'offset', 'window_size'):
        if hasattr(forecaster, attr):
            value = _decompose_offset(getattr(forecaster, attr))
            setattr(forecaster_decomposed, attr, value)

    return forecaster_decomposed


def _skops_reconstruct_forecaster(forecaster: object) -> None:
    """
    Rebuild the attributes of a forecaster decomposed by
    `_skops_decompose_forecaster`.

    Operates in place on `last_window_`, `training_range_`, `exog_dtypes_in_`,
    `exog_dtypes_out_`, `index_freq_`, `offset` and `window_size`. The
    `object_type_` marker key distinguishes a single decomposed object from a
    multi-series dict of decomposed objects.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster loaded from a skops file.

    Returns
    -------
    None

    """

    for attr in ('last_window_', 'training_range_'):
        value = getattr(forecaster, attr, None)
        if not isinstance(value, dict):
            continue
        if 'object_type_' in value and 'index_type_' in value:
            value = _compose_pandas_object(value)
        else:
            value = {k: _compose_pandas_object(v) for k, v in value.items()}
        setattr(forecaster, attr, value)
    for attr in ('exog_dtypes_in_', 'exog_dtypes_out_'):
        value = getattr(forecaster, attr, None)
        if isinstance(value, dict):
            value = {k: _compose_dtype(v) for k, v in value.items()}
            setattr(forecaster, attr, value)
    for attr in ('index_freq_', 'offset', 'window_size'):
        if hasattr(forecaster, attr):
            setattr(forecaster, attr, _compose_offset(getattr(forecaster, attr)))


def _get_source_with_imports(fun: Callable) -> tuple[str, list[str]]:
    """
    Return the source code of a function preceded by the import statements of
    the modules, functions and classes it uses from its global namespace, so
    that the code can be saved as a module that works on its own.

    Parameters
    ----------
    fun : Callable
        Function whose source code is returned.

    Returns
    -------
    source_code : str
        Source code of the function, preceded by the import statements.
    names_not_imported : list
        Names of the other objects the function uses from outside its body,
        which cannot be written as an import: global variables that are not a
        module, function or class of an importable module, and variables of
        an enclosing function.

    """

    def get_global_names(code):
        # NOTE: The names are read from the bytecode, including nested code
        # (comprehensions, generator expressions and lambdas), because
        # `inspect.getclosurevars` misses nested code and takes attribute names
        # (e.g. `month` in `index.month`) as global variables.
        global_names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME')
        }
        for constant in code.co_consts:
            if inspect.iscode(constant):
                global_names |= get_global_names(constant)

        return global_names

    # NOTE: The source code is compiled as a module, so that the names used in
    # the signature (annotations and default values) and in the decorators,
    # which are evaluated when the module is imported, are also read.
    # `dont_inherit` avoids `from __future__ import annotations` of this module,
    # which would compile the annotations as strings.
    source_code = inspect.getsource(fun)
    module_code = compile(
        textwrap.dedent(source_code), '<string>', 'exec', dont_inherit=True
    )
    imports = []
    names_not_imported = list(fun.__code__.co_freevars)
    for name in sorted(get_global_names(module_code) - set(names_not_imported)):
        if name not in fun.__globals__:
            # Builtins and undefined names
            continue
        value = fun.__globals__[name]
        if value is fun:
            continue
        if inspect.ismodule(value):
            if value.__name__ == name:
                imports.append(f"import {name}")
            else:
                imports.append(f"import {value.__name__} as {name}")
            continue
        module_name = getattr(value, '__module__', None)
        object_name = getattr(value, '__qualname__', None)
        module = sys.modules.get(module_name) if isinstance(module_name, str) else None
        is_importable = (
            module_name != '__main__'
            and isinstance(object_name, str)
            and getattr(module, object_name, None) is value
        )
        if is_importable and object_name == name:
            imports.append(f"from {module_name} import {name}")
        elif is_importable:
            imports.append(f"from {module_name} import {object_name} as {name}")
        else:
            names_not_imported.append(name)

    if imports:
        source_code = "\n".join(sorted(imports)) + "\n\n\n" + source_code

    return source_code, names_not_imported


@manage_warnings
def save_forecaster(
    forecaster: object,
    file_name: str,
    backend: str = 'joblib',
    save_custom_functions: bool = True,
    verbose: bool = False,
    suppress_warnings: bool = False
) -> None:
    """
    Save forecaster model to disk. Custom functions used to create weights that
    are defined in the `'__main__'` namespace (e.g. a notebook or a script run
    directly) are saved as .py files next to the forecaster file, since they
    cannot be re-imported when the forecaster is loaded in a different session.
    Functions imported from a module are restored automatically and are not
    exported. When `backend='cloudpickle'`, custom functions are embedded in the
    saved file and no .py files are created.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster created with skforecast library.
    file_name : str
        File name given to the object. The extension of the `backend` is added
        to the name (e.g. `'model_v1.2'` is saved as `'model_v1.2.joblib'`). If
        the name already ends with a backend extension (`.joblib`, `.pkl`,
        `.pickle`, `.cloudpickle` or `.skops`), it is replaced by the extension
        of the `backend`.
    backend : str, default 'joblib'
        Serialization backend used to save the forecaster.

        - If `'joblib'`, the forecaster is saved using joblib (extension
        `.joblib`).
        - If `'pickle'`, the forecaster is saved using pickle (extension
        `.pkl`).
        - If `'cloudpickle'`, the forecaster is saved using cloudpickle
        (extension `.cloudpickle`). Custom functions and user-defined classes
        are embedded in the file, so no separate `.py` files are needed.
        Requires `cloudpickle` to be installed.
        - If `'skops'`, the forecaster is saved using skops (extension
        `.skops`), a secure format that does not execute arbitrary code on
        load. The attributes that skops cannot serialize (`last_window_` and
        `training_range_`, the categorical, pyarrow and time zone aware dtypes
        of the exogenous variables, and generic `pandas.DateOffset` objects)
        are decomposed into plain types before saving and rebuilt on load. The
        time zone of the index is stored by its name, so it must be a named time
        zone (e.g. `'Europe/Madrid'` or `'UTC'`). Not supported for
        `ForecasterStats`, `ForecasterRnn`, or `ForecasterFoundation`, whose
        underlying estimators (statsmodels, Keras, or a foundation model) embed
        objects that skops cannot serialize. Requires `skops` to be installed.
    save_custom_functions : bool, default True
        If True, save custom functions used in the forecaster (weight_func) as
        .py files in the folder of `file_name`, but only those defined in the
        `'__main__'` namespace. These functions need to be imported in the
        environment where the forecaster is going to be loaded (e.g. with
        `from models.custom_weights import custom_weights` if the forecaster is
        saved in the folder `models`). Has no effect when `backend='cloudpickle'`.
    verbose : bool, default False
        Print summary about the forecaster saved.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed. See
        skforecast.exceptions.warn_skforecast_categories for more information.

    Returns
    -------
    None

    """

    valid_backends = {'joblib', 'pickle', 'cloudpickle', 'skops'}
    if backend not in valid_backends:
        raise ValueError(
            f"Invalid `backend` argument: '{backend}'. Valid options are: "
            f"{', '.join(repr(b) for b in sorted(valid_backends))}."
        )

    backend_extensions = {
        'joblib': '.joblib',
        'pickle': '.pkl',
        'cloudpickle': '.cloudpickle',
        'skops': '.skops'
    }
    # NOTE: Only a known backend extension is replaced, so that the dots in the
    # name are kept (e.g. 'model_v1.2' is saved as 'model_v1.2.joblib').
    known_extensions = {'.joblib', '.pkl', '.pickle', '.cloudpickle', '.skops'}
    file_name = Path(file_name)
    if file_name.suffix.lower() in known_extensions:
        file_name = file_name.with_suffix(backend_extensions[backend])
    else:
        file_name = file_name.with_name(file_name.name + backend_extensions[backend])

    # Save forecaster
    if backend == 'joblib':
        joblib.dump(forecaster, filename=file_name)
    elif backend == 'pickle':
        with open(file_name, 'wb') as file:
            pickle.dump(forecaster, file)
    elif backend == 'cloudpickle':
        try:
            import cloudpickle
        except ImportError as exc:
            raise ImportError(
                "'cloudpickle' is required for backend='cloudpickle' but is not "
                "installed. Install it with: pip install cloudpickle"
            ) from exc
        with open(file_name, 'wb') as file:
            cloudpickle.dump(forecaster, file)
    elif backend == 'skops':
        unsupported_forecasters = {
            'ForecasterStats', 'ForecasterRnn', 'ForecasterFoundation'
        }
        if type(forecaster).__name__ in unsupported_forecasters:
            raise NotImplementedError(
                f"backend='skops' is not supported for {type(forecaster).__name__}. "
                f"Its underlying estimator (statsmodels, Keras, or a foundation "
                f"model) embeds objects that skops cannot serialize. Use "
                f"backend='joblib', 'pickle', or 'cloudpickle' instead."
            )
        try:
            import skops.io
        except ImportError as exc:
            raise ImportError(
                "'skops' is required for backend='skops' but is not installed. "
                "Install it with: pip install skops"
            ) from exc
        skops.io.dump(_skops_decompose_forecaster(forecaster), file_name)

    if backend != 'cloudpickle':
        if hasattr(forecaster, 'weight_func') and forecaster.weight_func is not None:
            funs = (
                set(forecaster.weight_func.values())
                if isinstance(forecaster.weight_func, dict)
                else {forecaster.weight_func}
            )
            # NOTE: Only functions defined in the '__main__' namespace (notebook/script)
            # cannot be re-imported when the forecaster is loaded in a different
            # session, so they are the only ones that need the .py export / warning.
            # Functions from importable modules are restored automatically by
            # joblib/pickle (by reference). A `functools.partial` is restored from
            # the function it wraps, so that function is the one exported. Lambda
            # functions and callable objects cannot be exported as a module.
            main_funs = set()
            main_callables_not_exportable = []
            for fun in funs:
                if isinstance(fun, partial):
                    fun = fun.func
                if getattr(fun, '__module__', None) != '__main__':
                    continue
                if inspect.isfunction(fun) and fun.__name__.isidentifier():
                    main_funs.add(fun)
                else:
                    main_callables_not_exportable.append(fun)
            main_funs = sorted(main_funs, key=lambda f: f.__name__)
            if save_custom_functions:
                if main_funs:
                    saved_files = []
                    for fun in main_funs:
                        fun_file_name = file_name.parent / f"{fun.__name__}.py"
                        source_code, names_not_imported = _get_source_with_imports(fun)
                        with open(fun_file_name, 'w', encoding='utf-8') as file:
                            file.write(source_code)
                        saved_files.append(fun_file_name)
                        if names_not_imported:
                            warnings.warn(
                                f"The custom function '{fun.__name__}' uses objects "
                                f"defined outside its body that cannot be saved in "
                                f"'{fun_file_name}': "
                                f"{', '.join(repr(n) for n in names_not_imported)}. "
                                f"Define them inside the function, or save the "
                                f"forecaster with backend='cloudpickle', which "
                                f"stores the function together with the objects "
                                f"it uses.",
                                SaveLoadSkforecastWarning
                            )
                    saved_files_names = ', '.join(f"'{f}'" for f in saved_files)
                    warnings.warn(
                        "Custom function(s) used to create weights are defined in "
                        "the '__main__' namespace and have been saved as: "
                        f"{saved_files_names}. These files "
                        "must be imported before loading the forecaster.\n"
                        "Visit the documentation for more information: "
                        "https://skforecast.org/latest/user_guides/save-load-forecaster.html"
                        "#saving-and-loading-a-forecaster-model-with-custom-features",
                        SaveLoadSkforecastWarning
                    )
                if main_callables_not_exportable:
                    callables_names = ', '.join(
                        repr(getattr(f, '__name__', type(f).__name__))
                        for f in main_callables_not_exportable
                    )
                    warnings.warn(
                        "Custom callable(s) used to create weights are defined in "
                        "the '__main__' namespace but cannot be saved as .py files "
                        f"(lambda functions or callable objects): {callables_names}. "
                        "Define them as named functions, or save the forecaster "
                        "with backend='cloudpickle', which stores them in the file.",
                        SaveLoadSkforecastWarning
                    )
            elif main_funs or main_callables_not_exportable:
                warnings.warn(
                    "Custom function(s) used to create weights are defined in "
                    "the '__main__' namespace and have not been saved. To save "
                    "them automatically, set `save_custom_functions=True`. "
                    "Otherwise, ensure they are importable before loading the "
                    "forecaster.",
                    SaveLoadSkforecastWarning
                )

        if hasattr(forecaster, 'window_features') and forecaster.window_features is not None:
            skforecast_classes = {'RollingFeatures', 'RollingFeaturesClassification'}
            custom_classes = set(forecaster.window_features_class_names) - skforecast_classes
            if custom_classes:
                warnings.warn(
                    "The Forecaster includes custom user-defined classes in the "
                    "`window_features` argument. These classes are not saved automatically "
                    "when saving the Forecaster. Please ensure you save these classes "
                    "manually and import them before loading the Forecaster.\n"
                    "    Custom classes: " + ', '.join(custom_classes) + "\n"
                    "Visit the documentation for more information: "
                    "https://skforecast.org/latest/user_guides/save-load-forecaster.html#saving-and-loading-a-forecaster-model-with-custom-features",
                    SaveLoadSkforecastWarning
                )

    if verbose:
        forecaster.summary()


@manage_warnings
def load_forecaster(
    file_name: str,
    backend: str | None = None,
    trusted: bool | list[str] = False,
    verbose: bool = True,
    suppress_warnings: bool = False
) -> object:
    """
    Load forecaster model from disk. If the forecaster was saved with
    custom user-defined classes as window features or custom functions
    to create weights, these objects must be available in the environment
    where the forecaster is going to be loaded.

    Parameters
    ----------
    file_name : str
        Object file name.
    backend : str, None, default None
        Serialization backend used to load the forecaster. When `None`, the
        backend is inferred from the file extension:

        - `.joblib` : `'joblib'`
        - `.pkl` or `.pickle` : `'pickle'`
        - `.cloudpickle` : `'cloudpickle'`
        - `.skops` : `'skops'`
    trusted : bool, list, default False
        Types that skops is allowed to reconstruct when loading the file. Only
        used when `backend='skops'`, ignored otherwise. Controls the `trusted`
        argument of `skops.io.load`:

        - If `False`, only skops' built-in safe types are trusted. Because all
        skforecast forecasters contain types that are not trusted by default,
        loading raises an `UntrustedTypesFoundException` listing them. This is
        the secure default: review the listed types before trusting them.
        - If a list of str, additionally trust those type names. Obtain the
        candidate list with `skops.io.get_untrusted_types(file=file_name)` and
        pass it after reviewing it.
        - If `True`, trust every type found in the file. Use this only for files
        from a source you trust, as it removes skops' security guarantee.

        **New in version 0.23.0**
    verbose : bool, default True
        Print summary about the forecaster loaded.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed. See
        skforecast.exceptions.warn_skforecast_categories for more information.

    Returns
    -------
    forecaster : Forecaster
        Forecaster created with skforecast library.

    """

    extension_backend_map = {
        '.cloudpickle': 'cloudpickle',
        '.joblib': 'joblib',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.skops': 'skops',
    }
    valid_backends = {'joblib', 'pickle', 'cloudpickle', 'skops'}

    if backend is None:
        suffix = Path(file_name).suffix.lower()
        if suffix not in extension_backend_map:
            raise ValueError(
                f"Cannot infer backend from file extension '{suffix}'. "
                f"Recognized extensions: "
                f"{', '.join(repr(e) for e in sorted(extension_backend_map))}. "
                f"Provide the `backend` argument explicitly."
            )
        backend = extension_backend_map[suffix]
    elif backend not in valid_backends:
        raise ValueError(
            f"Invalid `backend` argument: '{backend}'. Valid options are: "
            f"{', '.join(repr(b) for b in sorted(valid_backends))}."
        )

    if backend == 'joblib':
        forecaster = joblib.load(filename=Path(file_name))
    elif backend == 'pickle':
        with open(file_name, 'rb') as file:
            forecaster = pickle.load(file)
    elif backend == 'cloudpickle':
        try:
            import cloudpickle  # noqa: F401 — needed to unpickle cloudpickle files
        except ImportError as exc:
            raise ImportError(
                "'cloudpickle' is required for backend='cloudpickle' but is not "
                "installed. Install it with: pip install cloudpickle"
            ) from exc
        with open(file_name, 'rb') as file:
            forecaster = pickle.load(file)
    elif backend == 'skops':
        try:
            import skops.io
            from skops.io.exceptions import UntrustedTypesFoundException
        except ImportError as exc:
            raise ImportError(
                "'skops' is required for backend='skops' but is not installed. "
                "Install it with: pip install skops"
            ) from exc
        # skops is the secure backend: by default (`trusted=False`) only its
        # built-in safe types are loaded and any other type must be reviewed and
        # trusted explicitly. `skops.io.load` only accepts a list of type names
        # (or None), so the friendly `trusted` argument is mapped here: `False`
        # -> None (strict), `True` -> all types found in the file, list -> as is.
        # The attributes decomposed into plain types by `save_forecaster` are
        # rebuilt.
        if trusted is False:
            trusted_types = None
        elif trusted is True:
            trusted_types = skops.io.get_untrusted_types(file=file_name)
        else:
            trusted_types = trusted
        try:
            forecaster = skops.io.load(file=file_name, trusted=trusted_types)
        except UntrustedTypesFoundException as exc:
            exc.args = (
                f"{exc.args[0]}\n"
                f"skops does not load these types unless you explicitly trust them. "
                f"To review the full list of untrusted types in the file, run "
                f"`skops.io.get_untrusted_types(file='{file_name}')`. If you trust the "
                f"source of '{file_name}', reload with `load_forecaster(..., trusted=True)` "
                f"to trust them all, or pass the reviewed list via `trusted=[...]`.",
            )
            raise
        _skops_reconstruct_forecaster(forecaster)

    forecaster_v = forecaster.skforecast_version

    if forecaster_v != __version__:
        warnings.warn(
            f"The skforecast version installed in the environment differs "
            f"from the version used to create the forecaster.\n"
            f"    Installed Version  : {__version__}\n"
            f"    Forecaster Version : {forecaster_v}\n"
            f"This may create incompatibilities when using the library.",
            SkforecastVersionWarning
        )

    if verbose:
        forecaster.summary()

    return forecaster


def _find_optional_dependency(
    package_name: str, 
    optional_dependencies: dict[str, list[str]] = optional_dependencies
) -> tuple[str, str]:
    """
    Find if a package is an optional dependency. If True, find the version and 
    the extension it belongs to.

    Parameters
    ----------
    package_name : str
        Name of the package to check.
    optional_dependencies : dict, default `optional_dependencies`
        Skforecast optional dependencies.

    Returns
    -------
    extra: str
        Name of the extra extension where the optional dependency is needed.
    package_version: str
        Name and versions of the dependency.

    """

    for extra, packages in optional_dependencies.items():
        package_version = [
            package for package in packages
            if Requirement(package).name == package_name
        ]
        if package_version:
            return extra, package_version[0]

    raise ValueError(
        f"'{package_name}' is not listed in optional_dependencies."
    )


def check_optional_dependency(
    package_name: str
) -> None:
    """
    Check if an optional dependency is installed, if not raise an ImportError  
    with installation instructions.

    Parameters
    ----------
    package_name : str
        Name of the package to check.

    Returns
    -------
    None
    
    """

    if find_spec(package_name) is None:
        try:
            extra, package_version = _find_optional_dependency(package_name=package_name)
            msg = (
                f"\n'{package_name}' is an optional dependency not included in the default "
                f"skforecast installation. Please run: `pip install \"{package_version}\"` to install it."
                f"\n\nAlternately, you can install it by running `pip install skforecast[{extra}]`"
            )
        except ValueError:
            msg = f"\n'{package_name}' is needed but not installed. Please install it."
        
        raise ImportError(msg)


def multivariate_time_series_corr(
    time_series: pd.Series,
    other: pd.DataFrame,
    lags: int | list[int] | np.ndarray[int],
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation between a time_series and the lagged values of other 
    time series. 

    Parameters
    ----------
    time_series : pandas Series
        Target time series.
    other : pandas DataFrame
        Time series whose lagged values are correlated to `time_series`.
    lags : int, list, numpy ndarray
        Lags to be included in the correlation analysis.
    method : str, default 'pearson'
        - 'pearson': standard correlation coefficient.
        - 'kendall': Kendall Tau correlation coefficient.
        - 'spearman': Spearman rank correlation.

    Returns
    -------
    corr : pandas DataFrame
        Correlation values.

    """

    if not len(time_series) == len(other):
        raise ValueError("`time_series` and `other` must have the same length.")

    if not (time_series.index == other.index).all():
        raise ValueError("`time_series` and `other` must have the same index.")

    if isinstance(lags, int):
        lags = range(lags)

    corr = {}
    for col in other.columns:
        lag_values = {}
        for lag in lags:
            lag_values[lag] = other[col].shift(lag)

        lag_values = pd.DataFrame(lag_values)
        lag_values.insert(0, None, time_series)
        corr[col] = lag_values.corr(method=method).iloc[1:, 0]

    corr = pd.DataFrame(corr)
    corr.index = corr.index.astype('int64')
    corr.index.name = "lag"
    
    return corr


def select_n_jobs_fit_forecaster(
    forecaster_name: str,
    estimator: object
) -> int:
    """
    Select the optimal number of jobs to use in the fitting process. This
    selection is based on heuristics and is not guaranteed to be optimal. 
    
    The number of jobs is chosen as follows:
    
    - If forecaster_name is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and estimator_name is a linear estimator then `n_jobs = 1`, 
    otherwise `n_jobs = max(1, cpu_count() - 1)`.
    - If estimator is a `LGBMRegressor(n_jobs=1)`, then `n_jobs = max(1, cpu_count() - 1)`.
    - If estimator is a `LGBMRegressor` with internal n_jobs != 1, then `n_jobs = 1`.
    This is because `lightgbm` is highly optimized for gradient boosting and
    parallelizes operations at a very fine-grained level, making additional
    parallelization unnecessary and potentially harmful due to resource contention.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    estimator : estimator or pipeline compatible with the scikit-learn API
        An instance of an estimator or pipeline compatible with the scikit-learn API.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.
    
    """

    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    if forecaster_name in {'ForecasterDirect', 'ForecasterDirectMultiVariate'}:
        if isinstance(estimator, LinearModel):
            n_jobs = 1
        elif type(estimator).__name__ == 'LGBMRegressor':
            n_jobs = max(1, joblib.cpu_count() - 1) if estimator.n_jobs == 1 else 1
        else:
            n_jobs = max(1, joblib.cpu_count() - 1)
    else:
        n_jobs = 1

    return n_jobs


def set_cpu_gpu_device(
    estimator: object, 
    device: str | None = 'cpu'
) -> str | None:
    """
    Set the `device` parameter of an XGBoost or LightGBM regressor and return
    its previous value, so that it can be restored afterwards. Recursive
    forecasters use it to predict on CPU, since they predict one row at a time.

    Parameters
    ----------
    estimator : object
        Estimator whose device is set. Only `XGBRegressor` and `LGBMRegressor`
        are modified. For any other estimator, nothing is done and `None` is
        returned.
    device : str, None, default 'cpu'
        Device to set, passed to the estimator as is (for example `'cpu'`,
        `'gpu'`, `'cuda'` or `'cuda:0'`). To restore the original device, pass
        the value returned by a previous call. If `None`, the device is not
        changed.

    Returns
    -------
    original_device : str, None
        Device of the estimator before the call. `None` if the estimator is not
        supported or its device is not set (both libraries then use the CPU).

    """

    if type(estimator).__name__ not in ('XGBRegressor', 'LGBMRegressor'):
        return None

    original_device = getattr(estimator, 'device', None)

    # NOTE: A device that is not set already means CPU in XGBoost and LightGBM,
    # so it is left unset instead of setting 'cpu'.
    current_device = 'cpu' if original_device is None else original_device
    if device is not None and device != current_device:
        estimator.set_params(device=device)

    return original_device


def _build_predict_function(
    estimator: object,
) -> Callable:
    """
    Build an optimized predict callable for a fitted estimator. The returned
    function takes a 2D numpy array `X` of shape `(n_samples, n_features)` and
    returns predictions as a 1D numpy array of shape `(n_samples,)`.

    Fast prediction paths (bypassing sklearn's `predict` overhead) are used
    for the following estimator types:

    - Linear models of scikit-learn inheriting from `LinearModel` (`np.dot`)
    - `LGBMRegressor` (`booster_.predict`)
    - `XGBRegressor` (`get_booster().inplace_predict`, with the same
    `iteration_range` and `missing` as `XGBRegressor.predict`). The 'gblinear'
    booster does not support `inplace_predict` and uses `estimator.predict`.
    - `RandomForestRegressor` (per-tree `tree_.predict`)
    - `DecisionTreeRegressor` (`tree_.predict`)

    For `CatBoostRegressor` with categorical features, the categorical column
    indices are resolved once at build time and the array is cast to `object`
    dtype with those columns converted to `int` before each prediction call.
    CatBoost requires integer values (not float) for categorical features when
    the input is a numpy array. For `CatBoostRegressor` without categorical
    features, the array is passed directly and its `writeable` flag is restored
    after prediction, since CatBoost borrows an F-contiguous array without a
    copy and leaves it read-only.

    For any other estimator the standard `estimator.predict` method is used.
    This includes user subclasses of scikit-learn estimators, since they may
    override `predict`.

    Parameters
    ----------
    estimator : object
        A fitted scikit-learn compatible estimator.

    Returns
    -------
    predict_fn : callable
        A function `predict_fn(X) -> np.ndarray` where `X` has shape
        `(n_samples, n_features)` and the output has shape `(n_samples,)`.
    
    """

    estimator_name = type(estimator).__name__
    # NOTE: The fast paths of scikit-learn estimators skip their `predict`
    # method. User subclasses may override it, so they use the generic fallback.
    is_sklearn_class = type(estimator).__module__.startswith('sklearn.')

    if is_sklearn_class and isinstance(estimator, LinearModel):
        coef = estimator.coef_
        intercept = estimator.intercept_

        # NOTE: np.dot does not validate its input, so NaN in `X` propagates to the
        # prediction instead of raising as sklearn's `predict` would. Inputs must be
        # validated upstream (see `check_predict_input`).
        def predict_fn(X):
            return np.dot(X, coef) + intercept

        return predict_fn

    if estimator_name == 'LGBMRegressor':
        booster = estimator.booster_

        def predict_fn(X):
            return booster.predict(X)

        return predict_fn

    # NOTE: `inplace_predict` is not supported by the 'gblinear' booster, which
    # uses the generic fallback (as `XGBRegressor.predict` does).
    if estimator_name == 'XGBRegressor' and estimator.booster != 'gblinear':
        booster = estimator.get_booster()
        # Same arguments as `XGBRegressor.predict`: only the trees up to the
        # best iteration when early stopping is used, and the user `missing` value.
        try:
            iteration_range = (0, estimator.best_iteration + 1)
        except AttributeError:
            iteration_range = (0, 0)
        missing = estimator.missing

        def predict_fn(X):
            return booster.inplace_predict(
                X, iteration_range=iteration_range, missing=missing
            )

        return predict_fn

    if is_sklearn_class and estimator_name == 'RandomForestRegressor':
        trees = estimator.estimators_

        def predict_fn(X):
            # ascontiguousarray gives a C-contiguous float32 copy (the cast
            # copies anyway), which is the layout tree traversal expects.
            X_f32 = np.ascontiguousarray(X, dtype=np.float32)
            preds = [tree.tree_.predict(X_f32)[:, 0] for tree in trees]
            return np.mean(preds, axis=0)

        return predict_fn

    if is_sklearn_class and estimator_name == 'DecisionTreeRegressor':
        tree_ = estimator.tree_

        def predict_fn(X):
            # ascontiguousarray gives a C-contiguous float32 copy (the cast
            # copies anyway), which is the layout tree traversal expects.
            return tree_.predict(np.ascontiguousarray(X, dtype=np.float32))[:, 0]

        return predict_fn

    if estimator_name == 'CatBoostRegressor':
        # CatBoost requires integer values (not float) for categorical features
        # when X is a numpy array. This requires casting the array to object
        # dtype and converting the categorical columns to int before each prediction call.
        cat_indices = _get_catboost_cat_feature_indices(estimator)
        if len(cat_indices) > 0:
            def predict_fn(X):
                X_obj = X.astype(object)
                X_obj[:, cat_indices] = np.nan_to_num(X[:, cat_indices], nan=-1).astype(int)
                return estimator.predict(X_obj).ravel()

            return predict_fn

        # Without categorical features CatBoost ingests the numpy array
        # directly. When the array is F-contiguous it is borrowed without a
        # copy and left read-only after predict, so the writeable flag is
        # restored to let callers keep filling the array in place across steps.
        def predict_fn(X):
            preds = estimator.predict(X).ravel()
            if not X.flags.writeable:
                X.flags.writeable = True
            return preds

        return predict_fn

    # Generic fallback
    def predict_fn(X):
        return estimator.predict(X).ravel()

    return predict_fn


def check_preprocess_series(
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Index]]:
    """
    Check and preprocess `series` argument in `ForecasterRecursiveMultiSeries` class.

    - If `series` is a wide-format pandas DataFrame, each column represents a
    different time series, and the index must be either a `DatetimeIndex` or 
    a `RangeIndex` with frequency or step size, as appropriate
    - If `series` is a long-format pandas DataFrame with a MultiIndex, the 
    first level of the index must contain the series IDs, and the second 
    level must be a `DatetimeIndex` with the same frequency across all series.
    - If series is a dictionary, each key must be a series ID, and each value 
    must be a named pandas Series. All series must have the same index, which 
    must be either a `DatetimeIndex` or a `RangeIndex`, and they must share the 
    same frequency or step size, as appropriate.

    When `series` is a pandas DataFrame, it is converted to a dictionary of pandas 
    Series, where the keys are the series IDs and the values are the Series with 
    the same index as the original DataFrame.
    
    Parameters
    ----------
    series : pandas DataFrame, dict
        Training time series.

    Returns
    -------
    series_dict : dict
        Dictionary with the series used during training.
    series_indexes : dict
        Dictionary with the index of each series.
    
    """

    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
            f"Got {type(series)}."
        )

    if isinstance(series, pd.DataFrame):

        if not isinstance(series.index, pd.MultiIndex):
            _, _ = check_extract_values_and_index(
                data=series, data_label='`series`', return_values=False
            )
            series = series.copy()
            series.index.name = None
            series_dict = series.to_dict(orient='series')
        else:
            if not isinstance(series.index.levels[1], pd.DatetimeIndex):
                raise TypeError(
                    f"The second level of the MultiIndex in `series` must be a "
                    f"pandas DatetimeIndex with the same frequency for each series. "
                    f"Found {type(series.index.levels[1])}."
                )
            
            first_col = series.columns[0]
            if len(series.columns) != 1:
                warnings.warn(
                    f"`series` DataFrame has multiple columns. Only the values of "
                    f"first column, '{first_col}', will be used as series values. "
                    f"All other columns will be ignored.",
                    IgnoredArgumentWarning
                )

            series = series.copy()
            series.index = series.index.set_names([series.index.names[0], None])
            series_dict = {
                series_id: group[first_col].droplevel(0).rename(series_id)
                for series_id, group in series.groupby(level=0, sort=True, observed=True)
            }
        
        warnings.warn(
            "Passing a DataFrame (either wide or long format) as `series` requires "
            "additional internal transformations, which can increase computational "
            "time. It is recommended to use a dictionary of pandas Series instead. "
            "For more details, see: "
            "https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html#input-data",
            InputTypeWarning
        )

    else:

        not_valid_series = [
            k 
            for k, v in series.items()
            if not isinstance(v, (pd.Series, pd.DataFrame))
        ]
        if not_valid_series:
            raise TypeError(
                f"If `series` is a dictionary, all series must be a named "
                f"pandas Series or a pandas DataFrame with a single column. "
                f"Review series: {not_valid_series}"
            )

        series_dict = {
            k: v.copy()
            for k, v in series.items()
        }

    not_valid_index = []
    indexes_freq = set()
    series_indexes = {}
    for k, v in series_dict.items():
        if isinstance(v, pd.DataFrame):
            if v.shape[1] != 1:
                raise ValueError(
                    f"If `series` is a dictionary, all series must be a named "
                    f"pandas Series or a pandas DataFrame with a single column. "
                    f"Review series: '{k}'"
                )
            series_dict[k] = v.iloc[:, 0]

        series_dict[k].name = k
        idx = v.index
        if isinstance(idx, pd.DatetimeIndex):
            indexes_freq.add(idx.freq)
        elif isinstance(idx, pd.RangeIndex):
            indexes_freq.add(idx.step)
        else:
            not_valid_index.append(k)

        if v.isna().to_numpy().all():
            raise ValueError(f"All values of series '{k}' are NaN.")

        series_indexes[k] = idx

    if not_valid_index:
        raise TypeError(
            f"If `series` is a dictionary, all series must have a Pandas "
            f"RangeIndex or DatetimeIndex with the same step/frequency. "
            f"Review series: {not_valid_index}"
        )
    if None in indexes_freq:
        raise ValueError(
            "If `series` is a dictionary, all series must have a Pandas "
            "RangeIndex or DatetimeIndex with the same step/frequency. "
            "If it a MultiIndex DataFrame, the second level must be a DatetimeIndex "
            "with the same frequency for each series. Found series with no "
            "frequency or step."
        )
    if not len(indexes_freq) == 1:
        raise ValueError(
            f"If `series` is a dictionary, all series must have a Pandas "
            f"RangeIndex or DatetimeIndex with the same step/frequency. "
            f"If it a MultiIndex DataFrame, the second level must be a DatetimeIndex "
            f"with the same frequency for each series. "
            f"Found frequencies: {sorted(indexes_freq)}"
        )

    return series_dict, series_indexes


def check_preprocess_exog_multiseries(
    series_names_in_: list[str],
    series_index_type: type,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame | None],
    exog_dict: dict[str, pd.Series | pd.DataFrame | None],
) -> tuple[dict[str, pd.DataFrame | None], list[str]]:
    """
    Check and preprocess `exog` argument in `ForecasterRecursiveMultiSeries` class.

    - If `exog` is a wide-format pandas DataFrame, it must share the same 
    index type as series. Each column represents a different exogenous variable, 
    and the same values are applied to all time series.
    - If `exog` is a long-format pandas Series or DataFrame with a MultiIndex, 
    the first level contains the series IDs to which it belongs, and the 
    second level contains a pandas DatetimeIndex. One column must be created
    for each exogenous variable.
    - If `exog` is a dictionary, each key must be the series ID to which it 
    belongs, and each value must be a named pandas Series/DataFrame with
    the same index type as `series` or None. While it is not necessary for 
    all values to include all the exogenous variables, the dtypes must be 
    consistent for the same exogenous variable across all series.

    When `exog` is a pandas DataFrame, it is converted to a dictionary of pandas 
    DataFrames, where the keys are the series IDs and the values are the Series 
    with the same index as the original DataFrame.

    Parameters
    ----------
    series_names_in_ : list
        Names of the series (levels) used during training.
    series_index_type : type
        Index type of the series used during training.
    exog : pandas Series, pandas DataFrame, dict
        Exogenous variable/s used during training.
    exog_dict : dict
        Dictionary with the exogenous variable/s used during training.

    Returns
    -------
    exog_dict : dict
        Dictionary with the exogenous variable/s used during training.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    
    """

    if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
        raise TypeError(
            f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
            f"Series/DataFrames or None. Got {type(exog)}."
        )

    if isinstance(exog, (pd.Series, pd.DataFrame)): 
        
        exog = exog.copy().to_frame() if isinstance(exog, pd.Series) else exog.copy()
        if isinstance(exog.index, pd.MultiIndex):
            if not isinstance(exog.index.levels[1], pd.DatetimeIndex):
                raise TypeError(
                    f"When input data are pandas MultiIndex DataFrame, "
                    f"`series` and `exog` second level index must be a "
                    f"pandas DatetimeIndex. Found `exog` index type: "
                    f"{type(exog.index.levels[1])}."
                )
            exog.index = exog.index.set_names([exog.index.names[0], None])
            exog_dict.update(
                {
                    series_id: group.droplevel(0)
                    for series_id, group in exog.groupby(level=0, sort=True, observed=True)
                    if series_id in series_names_in_
                }
            )
            series_ids_in_exog = exog.index.remove_unused_levels().levels[0]
            warnings.warn(
                "Using a long-format DataFrame as `exog` requires additional transformations, "
                "which can increase computational time. It is recommended to use a dictionary of "
                "Series or DataFrames instead. For more information, see: "
                "https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#input-data",
                InputTypeWarning
            )
        else:
            if not isinstance(exog.index, series_index_type):
                raise TypeError(
                    f"`exog` must have the same index type as `series`, pandas "
                    f"RangeIndex or pandas DatetimeIndex.\n"
                    f"    `series` index type : {series_index_type}.\n"
                    f"    `exog`   index type : {type(exog.index)}."
                )
            exog_dict = {series_id: exog for series_id in series_names_in_}
            series_ids_in_exog = series_names_in_

    else:

        not_valid_exog = [
            k 
            for k, v in exog.items()
            if not isinstance(v, (pd.Series, pd.DataFrame, type(None)))
        ]
        if not_valid_exog:
            raise TypeError(
                f"If `exog` is a dictionary, all exog must be a named pandas "
                f"Series, a pandas DataFrame or None. Review exog: {not_valid_exog}"
            )

        # NOTE: Only elements already present in exog_dict are updated. Copy is
        # needed to avoid modifying the original exog.
        exog_dict.update(
            {
                k: v.copy()
                for k, v in exog.items()
                if k in series_names_in_ and v is not None
            }
        )
        series_ids_in_exog = exog.keys()

    series_not_in_exog = set(series_names_in_) - set(series_ids_in_exog)
    if series_not_in_exog:
        warnings.warn(
            f"No `exog` for series {series_not_in_exog}. All values "
            f"of the exogenous variables for these series will be NaN.",
            MissingExogWarning
        )

    for k, v in exog_dict.items():
        if v is not None:
            check_exog(exog=v, allow_nan=True)
            if isinstance(v, pd.Series):
                v = v.to_frame()
            exog_dict[k] = v

    not_valid_index = [
        k
        for k, v in exog_dict.items()
        if v is not None and not isinstance(v.index, series_index_type)
    ]
    if not_valid_index:
        raise TypeError(
            f"All exog must have the same index type as `series`, which can be "
            f"either a pandas RangeIndex or a pandas DatetimeIndex. If either "
            f"`series` or `exog` is a pandas DataFrame with a MultiIndex, then "
            f"both must be pandas DatetimeIndex. Review exog for series: {not_valid_index}."
        )
    
    if isinstance(exog, dict):
        # NOTE: Check that all exog have the same dtypes for common columns
        exog_dtypes_buffer = pd.DataFrame(
            {k: df.dtypes for k, df in exog_dict.items() if df is not None}
        )
        exog_dtypes_nunique = exog_dtypes_buffer.nunique(axis=1)
        if not (exog_dtypes_nunique == 1).all():
            non_unique_dtypes_exogs = exog_dtypes_nunique[exog_dtypes_nunique != 1].index.to_list()
            raise TypeError(
                f"Exog/s: {non_unique_dtypes_exogs} have different dtypes in different "
                f"series. If any of these variables are categorical, note that this "
                f"error can also occur when their internal categories "
                f"(`series.cat.categories`) differ between series. Please ensure "
                f"that all series have the same categories (and category order) "
                f"for each categorical variable."
            )

        exog_names_in_ = list(
            set(
                column
                for df in exog_dict.values()
                if df is not None
                for column in df.columns.to_list()
            )
        )
    else:
        exog_names_in_ = list(exog.columns) if isinstance(exog, pd.DataFrame) else [exog.name]

    if len(set(exog_names_in_) - set(series_names_in_)) != len(exog_names_in_):
        raise ValueError(
            f"`exog` cannot contain a column named the same as one of the series.\n"
            f"    `series` columns : {series_names_in_}.\n"
            f"    `exog`   columns : {exog_names_in_}."
        )

    return exog_dict, exog_names_in_


def align_series_and_exog_multiseries(
    series_dict: dict[str, pd.Series],
    exog_dict: dict[str, pd.DataFrame | None],
    trim_series_nan: bool = True,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame | None]]:
    """
    Align series and exog according to their index. If needed, reindexing is
    applied. Heading and trailing NaNs are removed from all series in 
    `series_dict` when `trim_series_nan` is `True`.

    Parameters
    ----------
    series_dict : dict
        Dictionary with the series used during training.
    exog_dict : dict, default None
        Dictionary with the exogenous variable/s used during training.
    trim_series_nan : bool, default True
        If `True`, leading and trailing NaNs are removed from each series
        and exog is reindexed accordingly. If `False`, NaN trimming is
        skipped and only exog reindexing is performed.

    Returns
    -------
    series_dict : dict
        Dictionary with the series used during training.
    exog_dict : dict
        Dictionary with the exogenous variable/s used during training.
    
    """

    for k in series_dict.keys():
        if trim_series_nan and (
            np.isnan(series_dict[k].iat[0]) or np.isnan(series_dict[k].iat[-1])
        ):
            first_valid_index = series_dict[k].first_valid_index()
            last_valid_index = series_dict[k].last_valid_index()
            series_dict[k] = series_dict[k].loc[first_valid_index : last_valid_index]
        else:
            first_valid_index = series_dict[k].index[0]
            last_valid_index = series_dict[k].index[-1]

        if exog_dict[k] is not None:
            if not series_dict[k].index.equals(exog_dict[k].index):
                exog_dict[k] = exog_dict[k].loc[first_valid_index:last_valid_index]
                if exog_dict[k].empty:
                    warnings.warn(
                        f"`exog` for series '{k}' is empty after aligning "
                        f"with the series index. Exog values will be NaN.",
                        MissingValuesWarning
                    )
                    exog_dict[k] = None
                elif len(exog_dict[k]) != len(series_dict[k]):
                    warnings.warn(
                        f"`exog` for series '{k}' doesn't have values for "
                        f"all the dates in the series. Missing values will be "
                        f"filled with NaN.",
                        MissingValuesWarning
                    )
                    exog_dict[k] = exog_dict[k].reindex(
                        series_dict[k].index, fill_value = np.nan
                    )

    return series_dict, exog_dict


def prepare_levels_multiseries(
    X_train_series_names_in_: list[str],
    levels: str | list[str] | None = None
) -> tuple[list[str], bool]:
    """
    Prepare list of levels to be predicted in multiseries Forecasters.

    Parameters
    ----------
    X_train_series_names_in_ : list
        Names of the series (levels) included in the matrix `X_train`.
    levels : str, list, default None
        Names of the series (levels) to be predicted.

    Returns
    -------
    levels : list
        Names of the series (levels) to be predicted.
    input_levels_is_list : bool
        Indicates if input levels argument is a list.

    """

    input_levels_is_list = False
    if levels is None:
        levels = X_train_series_names_in_
    elif isinstance(levels, str):
        levels = [levels]
    else:
        input_levels_is_list = True

    return levels, input_levels_is_list


def preprocess_levels_self_last_window_multiseries(
    levels: list[str],
    input_levels_is_list: bool,
    last_window_: dict[str, pd.Series],
) -> tuple[list[str], pd.DataFrame]:
    """
    Preprocess `levels` and `last_window` (when using self.last_window_) arguments 
    in multiseries Forecasters when predicting. Only levels whose last window 
    ends at the same datetime index will be predicted together.

    Parameters
    ----------
    levels : list
        Names of the series (levels) to be predicted.
    input_levels_is_list : bool
        Indicates if input levels argument is a list.
    last_window_ : dict
        Dictionary with the last window of each series (self.last_window_).

    Returns
    -------
    levels : list
        Names of the series (levels) to be predicted.
    last_window : pandas DataFrame
        Series values used to create the predictors (lags) needed in the 
        first iteration of the prediction (t + 1).

    """

    if not levels:
        raise ValueError(
            "No series to predict. `levels` is an empty list. Provide at least "
            "one series name in `levels`, or set it to `None` to predict all "
            "the series stored in the `last_window_` attribute."
        )

    available_last_windows = set() if last_window_ is None else set(last_window_.keys())
    not_available_last_window = set(levels) - available_last_windows
    if not_available_last_window:
        levels = [
            level for level in levels 
            if level not in not_available_last_window
        ]
        if not levels:
            raise ValueError(
                f"No series to predict. None of the series {not_available_last_window} "
                f"are present in `last_window_` attribute. Provide `last_window` "
                f"as argument in predict method."
            )
        else:
            warnings.warn(
                f"Levels {not_available_last_window} are excluded from "
                f"prediction since they were not stored in `last_window_` "
                f"attribute during training. If you don't want to retrain "
                f"the Forecaster, provide `last_window` as argument.",
                IgnoredArgumentWarning
            )

    last_index_levels = [
        v.index[-1] 
        for k, v in last_window_.items()
        if k in levels
    ]
    if len(set(last_index_levels)) > 1:
        max_index_levels = max(last_index_levels)
        selected_levels = [
            k
            for k, v in last_window_.items()
            if k in levels and v.index[-1] == max_index_levels
        ]

        series_excluded_from_last_window = set(levels) - set(selected_levels)
        levels = selected_levels

        if input_levels_is_list and series_excluded_from_last_window:
            warnings.warn(
                f"Only series whose last window ends at the same index "
                f"can be predicted together. Series that do not reach "
                f"the maximum index, '{max_index_levels}', are excluded "
                f"from prediction: {series_excluded_from_last_window}.",
                IgnoredArgumentWarning
            )

    last_window = pd.DataFrame(
        {k: v 
         for k, v in last_window_.items() 
         if k in levels}
    )

    return levels, last_window


def prepare_steps_direct(
    max_step: int | list[int] | np.ndarray[int],
    steps: int | list[int] | None = None
) -> list[int]:
    """
    Prepare list of steps to be predicted in Direct Forecasters.

    Parameters
    ----------
    max_step : int, list, numpy ndarray
        Maximum number of future steps the forecaster will predict 
        when using predict methods.
    steps : int, list, None, default None
        Predict n steps. The value of `steps` must be less than or equal to the 
        value of steps defined when initializing the forecaster. Starts at 1.
    
        - If `int`: Only steps within the range of 1 to int are predicted.
        - If `list`: List of ints. Only the steps contained in the list 
        are predicted.
        - If `None`: As many steps are predicted as were defined at 
        initialization.

    Returns
    -------
    steps_direct : list
        Steps to be predicted.

    """

    if isinstance(steps, int):
        steps_direct = list(range(1, steps + 1))
    elif steps is None:
        if isinstance(max_step, int):
            steps_direct = list(range(1, max_step + 1))
        else:
            steps_direct = [int(s) for s in max_step]
    elif isinstance(steps, list):
        steps_direct = []
        for step in steps:
            if not isinstance(step, (int, np.integer)):
                raise TypeError(
                    f"`steps` argument must be an int, a list of ints or `None`. "
                    f"Got {type(steps)}."
                )
            steps_direct.append(int(step))

    return steps_direct


def get_style_repr_html(
    is_fitted: bool = False
) -> tuple[str, str]:
    """
    Return style and unique_id for HTML representation.

    Parameters
    ----------
    is_fitted : bool, default False
        Indicates if the object has been fitted.
    
    Returns
    -------
    style : str
        CSS style.
    unique_id : str
        Unique id for the HTML container.
    
    """

    unique_id = str(uuid.uuid4()).replace('-', '')
    background_color = "#f0f8ff" if is_fitted else "#f9f1e2"
    section_color = "#b3dbfd" if is_fitted else "#fae3b3"

    style = f"""
    <style>
        .container-{unique_id} {{
            font-family: 'Arial', sans-serif;
            font-size: 0.9em;
            color: #333333;
            border: 1px solid #ddd;
            background-color: {background_color};
            padding: 5px 15px;
            border-radius: 8px;
            max-width: 600px;
            #margin: auto;
        }}
        .container-{unique_id} h2 {{
            font-size: 1.5em;
            color: #222222;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-bottom: 15px;
            margin-top: 5px;
        }}
        .container-{unique_id} details {{
            margin: 10px 0;
            border-color: {section_color};
        }}
        .container-{unique_id} summary {{
            font-weight: bold;
            font-size: 1.1em;
            color: #000000;
            cursor: pointer;
            margin-bottom: 5px;
            background-color: {section_color};
            padding: 5px;
            border-radius: 5px;
        }}
        .container-{unique_id} summary:hover {{
            color: #000000;
            background-color: #e0e0e0;
        }}
        .container-{unique_id} ul {{
            font-family: 'Courier New', monospace;
            list-style-type: none;
            padding-left: 20px;
            margin: 10px 0;
            line-height: normal;
        }}
        .container-{unique_id} li {{
            margin: 5px 0;
            font-family: 'Courier New', monospace;
        }}
        .container-{unique_id} li strong {{
            font-weight: bold;
            color: #444444;
        }}
        .container-{unique_id} li::before {{
            content: "- ";
            color: #666666;
        }}
        .container-{unique_id} a {{
            color: #001633;
            text-decoration: none;
        }}
        .container-{unique_id} a:hover {{
            color: #359ccb; 
        }}
    </style>
    """

    return style, unique_id


def show_versions(
    as_str: bool = False
) -> str | None:
    """
    Print useful debugging information.

    Parameters
    ----------
    as_str : bool, default False
        If True, return the output as a string instead of printing.

    Returns
    -------
    vers_info : str
        The output string if `as_str` is True, otherwise None.

    Notes
    -----
    Adapted from the scikit-learn 1.7.2 show_versions function.
    https://github.com/scikit-learn/scikit-learn/
    Copyright (c) 2007-2025 The scikit-learn developers, BSD-3

    Examples
    --------
    >>> from skforecast.utils import show_versions
    >>> vers_info = show_versions(as_str=True)

    """

    deps = [
        "pip",
        "setuptools",
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "scipy",
        "optuna",
        "joblib",
        "numba",
        "rich",
        "statsmodels",
        "matplotlib",
        "keras",
        "torch",
        "lightgbm",
        "xgboost",
        "catboost",
        "skops",
        "cloudpickle",
    ]
    
    sys_info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }
    
    lines = ["\nSystem:"]
    for k, stat in sys_info.items():
        lines.append(f"{k:<11}: {stat}")

    deps_info = {"skforecast": __version__}
    for mod_name in deps:
        try:
            deps_info[mod_name] = version(mod_name)
        except PackageNotFoundError:
            deps_info[mod_name] = None

    lines.append("\nPython dependencies:")
    for k, stat in deps_info.items():
        lines.append(f"{k:<13}: {stat}")

    vers_info = "\n".join(lines)

    if as_str:
        return vers_info
    else:
        print(vers_info)
        return None


def deepcopy_forecaster(
    forecaster: object,
    include_in_sample_residuals: bool = False,
    include_out_sample_residuals: bool = False,
    include_last_window: bool = False,
) -> object:
    """
    Create a lightweight deep copy of a forecaster by temporarily
    replacing heavy fitted attributes with lightweight placeholders
    before copying.

    Estimators are always replaced with unfitted clones (same
    hyperparameters) to avoid copying expensive fitted state (e.g.,
    tree structures, model weights). For sklearn-compatible estimators
    `sklearn.base.clone` is used; for statistical models
    (`ForecasterStats`) `copy.copy` is used instead. Additional
    heavy attributes (residuals and last window) can be optionally
    included via parameters.

    Parameters
    ----------
    forecaster : object
        Forecaster object to copy. Can be any skforecast forecaster:
        `ForecasterRecursive`, `ForecasterDirect`, `ForecasterRecursiveMultiSeries`,
        `ForecasterDirectMultiVariate`, `ForecasterStats` or
        `ForecasterFoundation`.
    include_in_sample_residuals : bool, default `False`
        If `True`, `in_sample_residuals_` and `in_sample_residuals_by_bin_` are 
        preserved in the copy. These are recomputed during `fit()`, so they can 
        safely be excluded when the copy will be re-fitted.
    include_out_sample_residuals : bool, default `False`
        If `True`, `out_sample_residuals_` and `out_sample_residuals_by_bin_` are 
        preserved in the copy. These are user-provided via `set_out_sample_residuals()`
        and are NOT recomputed during `fit()`, so they must be included when the 
        copy needs them for prediction intervals with `use_in_sample_residuals=False`.
    include_last_window : bool, default `False`
        If `True`, `last_window_` is preserved in the copy. For most forecasters 
        this stores only the last `window_size` observations (small), but for
        `ForecasterStats` it contains ALL training data.

    Returns
    -------
    forecaster_copy : object
        Lightweight deep copy of the forecaster with unfitted estimator(s) and 
        optionally without residuals and last window.

    """

    # Save references to heavy attributes before replacing them
    saved = {}

    # 1. Replace fitted estimator with unfitted clone (same hyperparameters)
    if hasattr(forecaster, 'estimator') and forecaster.estimator is not None:
        saved['estimator'] = forecaster.estimator
        if type(forecaster).__name__ == 'ForecasterRnn':
            forecaster.estimator = deepcopy(forecaster.estimator)
        else:
            forecaster.estimator = clone(forecaster.estimator)

    # 2. Replace fitted estimators collection
    if hasattr(forecaster, 'estimators_') and forecaster.estimators_ is not None:
        saved['estimators_'] = forecaster.estimators_
        if isinstance(forecaster.estimators_, dict):
            # ForecasterDirect, ForecasterDirectMultiVariate: dict of fitted estimators
            forecaster.estimators_ = {
                step: clone(forecaster.estimator)
                for step in forecaster.estimators_
            }
        elif isinstance(forecaster.estimators_, list):
            # ForecasterStats: list of fitted stats models
            forecaster.estimators_ = [
                clone(est) for est in forecaster.estimators
            ]

    # 3. Optionally replace residuals with None
    _residual_attrs = []
    if not include_in_sample_residuals:
        _residual_attrs += ['in_sample_residuals_', 'in_sample_residuals_by_bin_']
    if not include_out_sample_residuals:
        _residual_attrs += ['out_sample_residuals_', 'out_sample_residuals_by_bin_']

    for attr in _residual_attrs:
        if hasattr(forecaster, attr) and getattr(forecaster, attr) is not None:
            saved[attr] = getattr(forecaster, attr)
            setattr(forecaster, attr, None)

    # 4. Optionally replace last_window_ with None
    if (
        not include_last_window
        and hasattr(forecaster, 'last_window_')
        and forecaster.last_window_ is not None
    ):
        saved['last_window_'] = forecaster.last_window_
        forecaster.last_window_ = None

    # Perform the (now lightweight) deep copy
    forecaster_copy = deepcopy(forecaster)

    # Restore original heavy attributes on the original forecaster
    for attr, value in saved.items():
        setattr(forecaster, attr, value)

    return forecaster_copy


def scale_correction_factor_differentiation(
    correction_factor: float | np.ndarray,
    steps: int,
    differentiation_order: int
) -> np.ndarray:
    """
    Scale the conformal prediction correction factor to account for the
    variance growth introduced by inverting the differentiation. When
    differentiation of order `d` is reverted via cumulative sums, a constant
    correction factor would accumulate linearly, producing intervals that
    grow as `h` instead of the theoretically correct growth rate.

    The scaling is derived from the MA(infinity) representation of the
    inverse difference operator `(1-B)^{-d}`, whose coefficients are
    `psi_j = comb(j + d - 1, d - 1)`. The correction factor at step `h`
    is scaled by `sqrt(sum_{j=0}^{h-1} psi_j^2)`, which for `d=1`
    simplifies to `sqrt(h)`.

    Parameters
    ----------
    correction_factor : float, numpy ndarray
        Correction factor from the conformal prediction method. Can be a
        scalar (non-binned residuals) or a 1D array with one value per step
        (binned residuals).
    steps : int
        Number of forecast steps.
    differentiation_order : int
        Order of differentiation applied to the series.

    Returns
    -------
    correction_factor_scaled : numpy ndarray
        Scaled correction factor with one value per step.

    """

    steps_array = np.arange(1, steps + 1)
    scaling_factor = np.sqrt(
        np.cumsum(
            comb(steps_array + differentiation_order - 2, differentiation_order - 1)
            ** 2
        )
    )

    return correction_factor * scaling_factor


def estimator_has_native_nan_support(estimator: object) -> bool:
    """
    Check whether an estimator natively supports NaN values in its input
    features.

    Uses the same module-based family detection as
    `configure_estimator_categorical_features`. Recognized families are
    lightgbm, catboost, xgboost, and sklearn's tree-based estimators
    (`DecisionTree`, `ExtraTree`, `ExtraTrees`, `RandomForest` and
    `HistGradientBoosting`, both regressors and classifiers). If `estimator`
    is a Pipeline, the last step is inspected.

    Parameters
    ----------
    estimator : object
        Estimator object. If the estimator is a Pipeline, the last step is used.

    Returns
    -------
    has_native_nan_support : bool
        `True` if the estimator natively supports NaN inputs, otherwise `False`.

    Notes
    -----
    Detection is based on the estimator family and class name, not on its
    hyperparameters. Uncommon configurations that disable NaN support are not
    detected, in which case the estimator raises its own error.

    """

    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    estimator_name = type(estimator).__name__
    module = type(estimator).__module__.split('.')[0]

    if module in ('lightgbm', 'catboost', 'xgboost'):
        return True

    if module == 'sklearn' and estimator_name in _SKLEARN_NAN_TOLERANT_ESTIMATORS:
        return True

    return False
