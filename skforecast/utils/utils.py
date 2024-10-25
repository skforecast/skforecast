################################################################################
#                               skforecast.utils                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import importlib
import inspect
import warnings
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
import sklearn.linear_model
import sklearn.pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import skforecast
from ..exceptions import warn_skforecast_categories
from ..exceptions import (
    MissingValuesWarning,
    MissingExogWarning,
    DataTypeWarning,
    UnknownLevelWarning,
    IgnoredArgumentWarning,
    SaveLoadSkforecastWarning,
    SkforecastVersionWarning
)

optional_dependencies = {
    'sarimax': [
        'statsmodels>=0.12, <0.15'
    ],
    'deeplearning': [
        'matplotlib>=3.3, <3.10',
        'keras>=2.6, <4.0',
    ],
    'plotting': [
        'matplotlib>=3.3, <3.10', 
        'seaborn>=0.11, <0.14', 
        'statsmodels>=0.12, <0.15'
    ]
}


def initialize_lags(
    forecaster_name: str,
    lags: Any
) -> Union[Optional[np.ndarray], Optional[list], Optional[int]]:
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
            if lags.ndim != 1:
                raise ValueError("`lags` must be a 1-dimensional array.")
            if lags.size == 0:
                raise ValueError("Argument `lags` must contain at least one value.")
            if not np.issubdtype(lags.dtype, np.integer):
                raise TypeError("All values in `lags` must be integers.")
            if np.any(lags < 1):
                raise ValueError("Minimum value of lags allowed is 1.")
        else:
            if forecaster_name != 'ForecasterDirectMultiVariate':
                raise TypeError(
                    (f"`lags` argument must be an int, 1d numpy ndarray, range, "
                     f"tuple or list. Got {type(lags)}.")
                )
            else:
                raise TypeError(
                    (f"`lags` argument must be a dict, int, 1d numpy ndarray, range, "
                     f"tuple or list. Got {type(lags)}.")
                )
        
        lags_names = [f'lag_{i}' for i in lags]
        max_lag = max(lags)

    return lags, lags_names, max_lag


def initialize_window_features(
    window_features: Any
) -> Union[Optional[list], Optional[list], Optional[int]]:
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

        # TODO: Add link to documentation in the error message: how to create a 
        # custom window feature class
        max_window_sizes = []
        window_features_names = []
        for wf in window_features:
            wf_name = type(wf).__name__
            atts_methods = set([a for a in dir(wf)])
            if not set(needed_atts).issubset(atts_methods):
                raise ValueError(
                    (f"{wf_name} must have the attributes: {needed_atts}.")
                )
            if not set(needed_methods).issubset(atts_methods):
                raise ValueError(
                    (f"{wf_name} must have the methods: {needed_methods}.")
                )
            
            window_sizes = wf.window_sizes
            if not isinstance(window_sizes, (int, list)):
                raise TypeError(
                    (f"Attribute `window_sizes` of {wf_name} must be an int or a list "
                     f"of ints. Got {type(window_sizes)}.")
                )
            
            if isinstance(window_sizes, int):
                if window_sizes < 1:
                    raise ValueError(
                        (f"If argument `window_sizes` is an integer, it must be equal "
                         f"to or greater than 1. Got {window_sizes} from {wf_name}.")
                    )
                max_window_sizes.append(window_sizes)
            else:
                if not all(isinstance(ws, int) for ws in window_sizes) or not all(
                    ws >= 1 for ws in window_sizes
                ):                    
                    raise ValueError(
                        (f"If argument `window_sizes` is a list, all elements must be integers "
                         f"equal to or greater than 1. Got {window_sizes} from {wf_name}.")
                    )
                max_window_sizes.append(max(window_sizes))

            features_names = wf.features_names
            if not isinstance(features_names, (str, list)):
                raise TypeError(
                    (f"Attribute `features_names` of {wf_name} must be a str or "
                     f"a list of strings. Got {type(features_names)}.")
                )
            if isinstance(features_names, str):
                window_features_names.append(features_names)
            else:
                if not all(isinstance(fn, str) for fn in features_names):
                    raise TypeError(
                        (f"If argument `features_names` is a list, all elements "
                         f"must be strings. Got {features_names} from {wf_name}.")
                    )
                window_features_names.extend(features_names)

        max_size_window_features = max(max_window_sizes)
        if len(set(window_features_names)) != len(window_features_names):
            raise ValueError(
                (f"All window features names must be unique. Got {window_features_names}.")
            )

    return window_features, window_features_names, max_size_window_features


def initialize_weights(
    forecaster_name: str,
    regressor: object,
    weight_func: Union[Callable, dict],
    series_weights: dict
) -> Tuple[Union[Callable, dict], Union[str, dict], dict]:
    """
    Check weights arguments, `weight_func` and `series_weights` for the different 
    forecasters. Create `source_code_weight_func`, source code of the custom 
    function(s) used to create weights.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    regressor : regressor or pipeline compatible with the scikit-learn API
        Regressor of the forecaster.
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.
    series_weights : dict
        Argument `series_weights` of the forecaster.

    Returns
    -------
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.
    source_code_weight_func : str, dict
        Argument `source_code_weight_func` of the forecaster.
    series_weights : dict
        Argument `series_weights` of the forecaster.
    
    """

    source_code_weight_func = None

    if weight_func is not None:

        if forecaster_name in ['ForecasterRecursiveMultiSeries']:
            if not isinstance(weight_func, (Callable, dict)):
                raise TypeError(
                    (f"Argument `weight_func` must be a Callable or a dict of "
                     f"Callables. Got {type(weight_func)}.")
                )
        elif not isinstance(weight_func, Callable):
            raise TypeError(
                f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
            )
        
        if isinstance(weight_func, dict):
            source_code_weight_func = {}
            for key in weight_func:
                source_code_weight_func[key] = inspect.getsource(weight_func[key])
        else:
            source_code_weight_func = inspect.getsource(weight_func)

        if 'sample_weight' not in inspect.signature(regressor.fit).parameters:
            warnings.warn(
                (f"Argument `weight_func` is ignored since regressor {regressor} "
                 f"does not accept `sample_weight` in its `fit` method."),
                 IgnoredArgumentWarning
            )
            weight_func = None
            source_code_weight_func = None

    if series_weights is not None:
        if not isinstance(series_weights, dict):
            raise TypeError(
                (f"Argument `series_weights` must be a dict of floats or ints."
                 f"Got {type(series_weights)}.")
            )
        if 'sample_weight' not in inspect.signature(regressor.fit).parameters:
            warnings.warn(
                (f"Argument `series_weights` is ignored since regressor {regressor} "
                 f"does not accept `sample_weight` in its `fit` method."),
                 IgnoredArgumentWarning
            )
            series_weights = None

    return weight_func, source_code_weight_func, series_weights


def initialize_transformer_series(
    forecaster_name: str,
    series_names_in_: list,
    encoding: Optional[str] = None,
    transformer_series: Optional[Union[object, dict]] = None
) -> dict:
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
    encoding : str, default `None`
        Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).
    transformer_series : object, dict, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. 

    Returns
    -------
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the 
        objects in `transformer_series` and is used internally to avoid overwriting.
    
    """

    multiseries_forecasters = [
        'ForecasterRecursiveMultiSeries'
    ]

    if forecaster_name in multiseries_forecasters:
        if encoding is None:
            series_names_in_ = ['_unknown_level']
        else:
            series_names_in_ = series_names_in_ + ['_unknown_level']

    if transformer_series is None:
        transformer_series_ = {serie: None for serie in series_names_in_}
    elif not isinstance(transformer_series, dict):
        transformer_series_ = {serie: clone(transformer_series) 
                               for serie in series_names_in_}
    else:
        transformer_series_ = {serie: None for serie in series_names_in_}
        # Only elements already present in transformer_series_ are updated
        transformer_series_.update(
            (k, v) for k, v in deepcopy(transformer_series).items() 
            if k in transformer_series_
        )

        series_not_in_transformer_series = (
            set(series_names_in_) - set(transformer_series.keys())
        ) - {'_unknown_level'}
        if series_not_in_transformer_series:
            warnings.warn(
                (f"{series_not_in_transformer_series} not present in `transformer_series`."
                f" No transformation is applied to these series."),
                IgnoredArgumentWarning
            )

    return transformer_series_


def check_select_fit_kwargs(
    regressor: object,
    fit_kwargs: Optional[dict] = None
) -> dict:
    """
    Check if `fit_kwargs` is a dict and select only the keys that are used by
    the `fit` method of the regressor.

    Parameters
    ----------
    regressor : object
        Regressor object.
    fit_kwargs : dict, default `None`
        Dictionary with the arguments to pass to the `fit' method of the forecaster.

    Returns
    -------
    fit_kwargs : dict
        Dictionary with the arguments to be passed to the `fit` method of the 
        regressor after removing the unused keys.
    
    """

    if fit_kwargs is None:
        fit_kwargs = {}
    else:
        if not isinstance(fit_kwargs, dict):
            raise TypeError(
                f"Argument `fit_kwargs` must be a dict. Got {type(fit_kwargs)}."
            )

        # Non used keys
        non_used_keys = [k for k in fit_kwargs.keys()
                         if k not in inspect.signature(regressor.fit).parameters]
        if non_used_keys:
            warnings.warn(
                (f"Argument/s {non_used_keys} ignored since they are not used by the "
                 f"regressor's `fit` method."),
                 IgnoredArgumentWarning
            )

        if 'sample_weight' in fit_kwargs.keys():
            warnings.warn(
                ("The `sample_weight` argument is ignored. Use `weight_func` to pass "
                 "a function that defines the individual weights for each sample "
                 "based on its index."),
                 IgnoredArgumentWarning
            )
            del fit_kwargs['sample_weight']

        # Select only the keyword arguments allowed by the regressor's `fit` method.
        fit_kwargs = {k: v for k, v in fit_kwargs.items()
                      if k in inspect.signature(regressor.fit).parameters}

    return fit_kwargs


def check_y(
    y: Any,
    series_id: str = "`y`"
) -> None:
    """
    Raise Exception if `y` is not pandas Series or if it has missing values.
    
    Parameters
    ----------
    y : Any
        Time series values.
    series_id : str, default '`y`'
        Identifier of the series used in the warning message.
    
    Returns
    -------
    None
    
    """
    
    if not isinstance(y, pd.Series):
        raise TypeError(f"{series_id} must be a pandas Series.")
        
    if y.isnull().any():
        raise ValueError(f"{series_id} has missing values.")
    
    return


def check_exog(
    exog: Union[pd.Series, pd.DataFrame],
    allow_nan: bool = True,
    series_id: str = "`exog`"
) -> None:
    """
    Raise Exception if `exog` is not pandas Series or pandas DataFrame.
    If `allow_nan = True`, issue a warning if `exog` contains NaN values.
    
    Parameters
    ----------
    exog : pandas DataFrame, pandas Series
        Exogenous variable/s included as predictor/s.
    allow_nan : bool, default `True`
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
        if exog.isnull().any().any():
            warnings.warn(
                (f"{series_id} has missing values. Most machine learning models "
                 f"do not allow missing values. Fitting the forecaster may fail."), 
                 MissingValuesWarning
            )
    
    return


def get_exog_dtypes(
    exog: Union[pd.DataFrame, pd.Series]
) -> dict:
    """
    Store dtypes of `exog`.

    Parameters
    ----------
    exog : pandas DataFrame, pandas Series
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
    exog: Union[pd.DataFrame, pd.Series],
    call_check_exog: bool = True,
    series_id: str = "`exog`"
) -> None:
    """
    Raise Exception if `exog` has categorical columns with non integer values.
    This is needed when using machine learning regressors that allow categorical
    features.
    Issue a Warning if `exog` has columns that are not `init`, `float`, or `category`.
    
    Parameters
    ----------
    exog : pandas DataFrame, pandas Series
        Exogenous variable/s included as predictor/s.
    call_check_exog : bool, default `True`
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

    if isinstance(exog, pd.DataFrame):
        if not exog.select_dtypes(exclude=[np.number, 'category']).columns.empty:
            warnings.warn(
                (f"{series_id} may contain only `int`, `float` or `category` dtypes. "
                 f"Most machine learning models do not allow other types of values. "
                 f"Fitting the forecaster may fail."), 
                 DataTypeWarning
            )
        for col in exog.select_dtypes(include='category'):
            if exog[col].cat.categories.dtype not in [int, np.int32, np.int64]:
                raise TypeError(
                    ("Categorical dtypes in exog must contain only integer values. "
                     "See skforecast docs for more info about how to include "
                     "categorical features https://skforecast.org/"
                     "latest/user_guides/categorical-features.html")
                )
    else:
        if exog.dtype.name not in ['int', 'int8', 'int16', 'int32', 'int64', 'float', 
        'float16', 'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'category']:
            warnings.warn(
                (f"{series_id} may contain only `int`, `float` or `category` dtypes. Most "
                 f"machine learning models do not allow other types of values. "
                 f"Fitting the forecaster may fail."), 
                 DataTypeWarning
            )
        if exog.dtype.name == 'category' and exog.cat.categories.dtype not in [int,
        np.int32, np.int64]:
            raise TypeError(
                ("Categorical dtypes in exog must contain only integer values. "
                 "See skforecast docs for more info about how to include "
                 "categorical features https://skforecast.org/"
                 "latest/user_guides/categorical-features.html")
            )
         
    return


def check_interval(
    interval: list = None,
    quantiles: float = None,
    alpha: float = None
) -> None:
    """
    Check provided confidence interval sequence is valid.

    Parameters
    ----------
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`.
    quantiles : list, default `None`
        Sequence of quantiles to compute, which must be between 0 and 1 
        inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
        `quantiles = [0.05, 0.5, 0.95]`.
    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %.

    Returns
    -------
    None
    
    """

    if interval is not None:
        if not isinstance(interval, list):
            raise TypeError(
                ("`interval` must be a `list`. For example, interval of 95% "
                 "should be as `interval = [2.5, 97.5]`.")
            )

        if len(interval) != 2:
            raise ValueError(
                ("`interval` must contain exactly 2 values, respectively the "
                 "lower and upper interval bounds. For example, interval of 95% "
                 "should be as `interval = [2.5, 97.5]`.")
            )

        if (interval[0] < 0.) or (interval[0] >= 100.):
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be >= 0 and < 100."
            )

        if (interval[1] <= 0.) or (interval[1] > 100.):
            raise ValueError(
                f"Upper interval bound ({interval[1]}) must be > 0 and <= 100."
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                (f"Lower interval bound ({interval[0]}) must be less than the "
                 f"upper interval bound ({interval[1]}).")
            )
        
    if quantiles is not None:
        if not isinstance(quantiles, list):
            raise TypeError(
                ("`quantiles` must be a `list`. For example, quantiles 0.05, "
                 "0.5, and 0.95 should be as `quantiles = [0.05, 0.5, 0.95]`.")
            )
        
        for q in quantiles:
            if (q < 0.) or (q > 1.):
                raise ValueError(
                    ("All elements in `quantiles` must be >= 0 and <= 1.")
                )
    
    if alpha is not None:
        if not isinstance(alpha, float):
            raise TypeError(
                ("`alpha` must be a `float`. For example, interval of 95% "
                 "should be as `alpha = 0.05`.")
            )

        if (alpha <= 0.) or (alpha >= 1):
            raise ValueError(
                f"`alpha` must have a value between 0 and 1. Got {alpha}."
            )

    return


def check_predict_input(
    forecaster_name: str,
    steps: Union[int, list],
    is_fitted: bool,
    exog_in_: bool,
    index_type_: type,
    index_freq_: str,
    window_size: int,
    last_window: Union[pd.Series, pd.DataFrame, None],
    last_window_exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    exog_type_in_: Optional[type] = None,
    exog_names_in_: Optional[list] = None,
    interval: Optional[list] = None,
    alpha: Optional[float] = None,
    max_steps: Optional[int] = None,
    levels: Optional[Union[str, list]] = None,
    levels_forecaster: Optional[Union[str, list]] = None,
    series_names_in_: Optional[list] = None,
    encoding: Optional[str] = None
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
        Tag to identify if the regressor has been fitted (trained).
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
    last_window_exog : pandas Series, pandas DataFrame, default `None`
        Values of the exogenous variables aligned with `last_window` in 
        ForecasterSarimax predictions.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s.
    exog_type_in_ : type, default `None`
        Type of exogenous variable/s used in training.
    exog_names_in_ : list, default `None`
        Names of the exogenous variables used during training.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`.
    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %.
    max_steps: int, default `None`
        Maximum number of steps allowed (`ForecasterDirect` and 
        `ForecasterDirectMultiVariate`).
    levels : str, list, default `None`
        Time series to be predicted (`ForecasterRecursiveMultiSeries`
        and `ForecasterRnn).
    levels_forecaster : str, list, default `None`
        Time series used as output data of a multiseries problem in a RNN problem
        (`ForecasterRnn`).
    series_names_in_ : list, default `None`
        Names of the columns used during fit (`ForecasterRecursiveMultiSeries`, 
        `ForecasterDirectMultiVariate` and `ForecasterRnn`).
    encoding : str, default `None`
        Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).

    Returns
    -------
    None

    """

    if not is_fitted:
        raise NotFittedError(
            ("This Forecaster instance is not fitted yet. Call `fit` with "
             "appropriate arguments before using predict.")
        )

    if isinstance(steps, (int, np.integer)) and steps < 1:
        raise ValueError(
            f"`steps` must be an integer greater than or equal to 1. Got {steps}."
        )

    if isinstance(steps, list) and min(steps) < 1:
        raise ValueError(
           (f"The minimum value of `steps` must be equal to or greater than 1. "
            f"Got {min(steps)}.")
        )

    if max_steps is not None:
        if max(steps) > max_steps:
            raise ValueError(
                (f"The maximum value of `steps` must be less than or equal to "
                 f"the value of steps defined when initializing the forecaster. "
                 f"Got {max(steps)}, but the maximum is {max_steps}.")
            )

    if interval is not None or alpha is not None:
        check_interval(interval=interval, alpha=alpha)

    if forecaster_name in ['ForecasterRecursiveMultiSeries', 
                           'ForecasterRnn']:
        if not isinstance(levels, (type(None), str, list)):
            raise TypeError(
                ("`levels` must be a `list` of column names, a `str` of a "
                 "column name or `None`.")
            )

        levels_to_check = (
            levels_forecaster if forecaster_name == 'ForecasterRnn'
            else series_names_in_
        )
        unknown_levels = set(levels) - set(levels_to_check)
        if forecaster_name == 'ForecasterRnn':
            if len(unknown_levels) != 0:
                raise ValueError(
                    (f"`levels` names must be included in the series used during fit "
                     f"({levels_to_check}). Got {levels}.")
                )
        else:
            if len(unknown_levels) != 0 and last_window is not None and encoding is not None:
                if encoding == 'onehot':
                    warnings.warn(
                        (f"`levels` {unknown_levels} were not included in training. The resulting "
                         f"one-hot encoded columns for this feature will be all zeros."),
                         UnknownLevelWarning
                    )
                else:
                    warnings.warn(
                        (f"`levels` {unknown_levels} were not included in training. "
                         f"Unknown levels are encoded as NaN, which may cause the "
                         f"prediction to fail if the regressor does not accept NaN values."),
                         UnknownLevelWarning
                    )

    if exog is None and exog_in_:
        raise ValueError(
            ("Forecaster trained with exogenous variable/s. "
             "Same variable/s must be provided when predicting.")
        )

    if exog is not None and not exog_in_:
        raise ValueError(
            ("Forecaster trained without exogenous variable/s. "
             "`exog` must be `None` when predicting.")
        )

    # Checks last_window
    # Check last_window type (pd.Series or pd.DataFrame according to forecaster)
    if isinstance(last_window, type(None)) and forecaster_name not in [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterRnn'
    ]:
        raise ValueError(
            ("`last_window` was not stored during training. If you don't want "
             "to retrain the Forecaster, provide `last_window` as argument.")
        )

    if forecaster_name in ['ForecasterRecursiveMultiSeries', 
                           'ForecasterDirectMultiVariate',
                           'ForecasterRnn']:
        if not isinstance(last_window, pd.DataFrame):
            raise TypeError(
                f"`last_window` must be a pandas DataFrame. Got {type(last_window)}."
            )

        last_window_cols = last_window.columns.to_list()

        if forecaster_name in ['ForecasterRecursiveMultiSeries', 
                               'ForecasterRnn'] and \
            len(set(levels) - set(last_window_cols)) != 0:
            raise ValueError(
                (f"`last_window` must contain a column(s) named as the level(s) "
                 f"to be predicted.\n"
                 f"    `levels` : {levels}\n"
                 f"    `last_window` columns : {last_window_cols}")
            )

        if forecaster_name == 'ForecasterDirectMultiVariate':
            if len(set(series_names_in_) - set(last_window_cols)) > 0:
                raise ValueError(
                    (f"`last_window` columns must be the same as the `series` "
                     f"column names used to create the X_train matrix.\n"
                     f"    `last_window` columns    : {last_window_cols}\n"
                     f"    `series` columns X train : {series_names_in_}")
                )
    else:
        if not isinstance(last_window, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"`last_window` must be a pandas Series or DataFrame. "
                f"Got {type(last_window)}."
            )

    # Check last_window len, nulls and index (type and freq)
    if len(last_window) < window_size:
        raise ValueError(
            (f"`last_window` must have as many values as needed to "
             f"generate the predictors. For this forecaster it is {window_size}.")
        )
    if last_window.isnull().any().all():
        warnings.warn(
            ("`last_window` has missing values. Most of machine learning models do "
             "not allow missing values. Prediction method may fail."), 
             MissingValuesWarning
        )
    _, last_window_index = preprocess_last_window(
                               last_window   = last_window.iloc[:0],
                               return_values = False
                           ) 
    if not isinstance(last_window_index, index_type_):
        raise TypeError(
            (f"Expected index of type {index_type_} for `last_window`. "
             f"Got {type(last_window_index)}.")
        )
    if isinstance(last_window_index, pd.DatetimeIndex):
        if not last_window_index.freqstr == index_freq_:
            raise TypeError(
                (f"Expected frequency of type {index_freq_} for `last_window`. "
                 f"Got {last_window_index.freqstr}.")
            )

    # Checks exog
    if exog is not None:

        # Check type, nulls and expected type
        if forecaster_name in ['ForecasterRecursiveMultiSeries']:
            if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or dict. Got {type(exog)}."
                )
            if exog_type_in_ == dict and not isinstance(exog, dict):
                raise TypeError(
                    f"Expected type for `exog`: {exog_type_in_}. Got {type(exog)}."
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
                    (f"`exog` does not contain keys for levels {no_exog_levels}. "
                     f"Missing levels are filled with NaN. Most of machine learning "
                     f"models do not allow missing values. Prediction method may fail."),
                     MissingExogWarning
                )
            exogs_to_check = [
                (f"`exog` for series '{k}'", v) 
                for k, v in exog.items() 
                if v is not None and k in levels
            ]
        else:
            exogs_to_check = [('`exog`', exog)]

        for exog_name, exog_to_check in exogs_to_check:

            if not isinstance(exog_to_check, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"{exog_name} must be a pandas Series or DataFrame. Got {type(exog_to_check)}"
                )

            if exog_to_check.isnull().any().any():
                warnings.warn(
                    (f"{exog_name} has missing values. Most of machine learning models "
                     f"do not allow missing values. Prediction method may fail."), 
                     MissingValuesWarning
                )

            # Check exog has many values as distance to max step predicted
            last_step = max(steps) if isinstance(steps, list) else steps
            if len(exog_to_check) < last_step:
                if forecaster_name in ['ForecasterRecursiveMultiSeries']:
                    warnings.warn(
                        (f"{exog_name} doesn't have as many values as steps "
                         f"predicted, {last_step}. Missing values are filled "
                         f"with NaN. Most of machine learning models do not "
                         f"allow missing values. Prediction method may fail."),
                         MissingValuesWarning
                    )
                else: 
                    raise ValueError(
                        (f"{exog_name} must have at least as many values as "
                         f"steps predicted, {last_step}.")
                    )

            # Check name/columns are in exog_names_in_
            if isinstance(exog_to_check, pd.DataFrame):
                col_missing = set(exog_names_in_).difference(set(exog_to_check.columns))
                if col_missing:
                    if forecaster_name in ['ForecasterRecursiveMultiSeries']:
                        warnings.warn(
                            (f"{col_missing} not present in {exog_name}. All "
                             f"values will be NaN."),
                             MissingExogWarning
                        ) 
                    else:
                        raise ValueError(
                            (f"Missing columns in {exog_name}. Expected {exog_names_in_}. "
                             f"Got {exog_to_check.columns.to_list()}.")
                        )
            else:
                if exog_to_check.name is None:
                    raise ValueError(
                        (f"When {exog_name} is a pandas Series, it must have a name. Got None.")
                    )

                if exog_to_check.name not in exog_names_in_:
                    if forecaster_name in ['ForecasterRecursiveMultiSeries']:
                        warnings.warn(
                            (f"'{exog_to_check.name}' was not observed during training. "
                             f"{exog_name} is ignored. Exogenous variables must be one "
                             f"of: {exog_names_in_}."),
                             IgnoredArgumentWarning
                        )
                    else:
                        raise ValueError(
                            (f"'{exog_to_check.name}' was not observed during training. "
                             f"Exogenous variables must be: {exog_names_in_}.")
                        )

            # Check index dtype and freq
            _, exog_index = preprocess_exog(
                                exog          = exog_to_check.iloc[:0, ],
                                return_values = False
                            )
            if not isinstance(exog_index, index_type_):
                raise TypeError(
                    (f"Expected index of type {index_type_} for {exog_name}. "
                     f"Got {type(exog_index)}.")
                )
            if forecaster_name not in ['ForecasterRecursiveMultiSeries']:
                if isinstance(exog_index, pd.DatetimeIndex):
                    if not exog_index.freqstr == index_freq_:
                        raise TypeError(
                            (f"Expected frequency of type {index_freq_} for {exog_name}. "
                             f"Got {exog_index.freqstr}.")
                        )

            # Check exog starts one step ahead of last_window end.
            expected_index = expand_index(last_window.index, 1)[0]
            if expected_index != exog_to_check.index[0]:
                if forecaster_name in ['ForecasterRecursiveMultiSeries']:
                    warnings.warn(
                        (f"To make predictions {exog_name} must start one step "
                         f"ahead of `last_window`. Missing values are filled "
                         f"with NaN.\n"
                         f"    `last_window` ends at : {last_window.index[-1]}.\n"
                         f"    {exog_name} starts at : {exog_to_check.index[0]}.\n"
                         f"     Expected index       : {expected_index}."),
                         MissingValuesWarning
                    )  
                else:
                    raise ValueError(
                        (f"To make predictions {exog_name} must start one step "
                         f"ahead of `last_window`.\n"
                         f"    `last_window` ends at : {last_window.index[-1]}.\n"
                         f"    {exog_name} starts at : {exog_to_check.index[0]}.\n"
                         f"     Expected index : {expected_index}.")
                    )

    # Checks ForecasterSarimax
    if forecaster_name == 'ForecasterSarimax':
        # Check last_window_exog type, len, nulls and index (type and freq)
        if last_window_exog is not None:
            if not exog_in_:
                raise ValueError(
                    ("Forecaster trained without exogenous variable/s. "
                     "`last_window_exog` must be `None` when predicting.")
                )

            if not isinstance(last_window_exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    (f"`last_window_exog` must be a pandas Series or a "
                     f"pandas DataFrame. Got {type(last_window_exog)}.")
                )
            if len(last_window_exog) < window_size:
                raise ValueError(
                    (f"`last_window_exog` must have as many values as needed to "
                     f"generate the predictors. For this forecaster it is {window_size}.")
                )
            if last_window_exog.isnull().any().all():
                warnings.warn(
                    ("`last_window_exog` has missing values. Most of machine learning "
                     "models do not allow missing values. Prediction method may fail."),
                     MissingValuesWarning
            )
            _, last_window_exog_index = preprocess_last_window(
                                            last_window   = last_window_exog.iloc[:0],
                                            return_values = False
                                        ) 
            if not isinstance(last_window_exog_index, index_type_):
                raise TypeError(
                    (f"Expected index of type {index_type_} for `last_window_exog`. "
                     f"Got {type(last_window_exog_index)}.")
                )
            if isinstance(last_window_exog_index, pd.DatetimeIndex):
                if not last_window_exog_index.freqstr == index_freq_:
                    raise TypeError(
                        (f"Expected frequency of type {index_freq_} for "
                         f"`last_window_exog`. Got {last_window_exog_index.freqstr}.")
                    )

            # Check all columns are in the pd.DataFrame, last_window_exog
            if isinstance(last_window_exog, pd.DataFrame):
                col_missing = set(exog_names_in_).difference(set(last_window_exog.columns))
                if col_missing:
                    raise ValueError(
                        (f"Missing columns in `last_window_exog`. Expected {exog_names_in_}. "
                         f"Got {last_window_exog.columns.to_list()}.") 
                    )
            else:
                if last_window_exog.name is None:
                    raise ValueError(
                        ("When `last_window_exog` is a pandas Series, it must have a "
                         "name. Got None.")
                    )

                if last_window_exog.name not in exog_names_in_:
                    raise ValueError(
                        (f"'{last_window_exog.name}' was not observed during training. "
                         f"Exogenous variables must be: {exog_names_in_}.")
                    )

    return


def preprocess_y(
    y: Union[pd.Series, pd.DataFrame],
    return_values: bool = True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Return values and index of series separately. Index is overwritten 
    according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.
    
    Parameters
    ----------
    y : pandas Series, pandas DataFrame
        Time series.
    return_values : bool, default `True`
        If `True` return the values of `y` as numpy ndarray. This option is 
        intended to avoid copying data when it is not necessary.

    Returns
    -------
    y_values : None, numpy ndarray
        Numpy array with values of `y`.
    y_index : pandas Index
        Index of `y` modified according to the rules.
    
    """
    
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
        y_index = y.index
    elif isinstance(y.index, pd.RangeIndex):
        y_index = y.index
    elif isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
        warnings.warn(
            ("Series has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )
    else:
        warnings.warn(
            ("Series has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )

    y_values = y.to_numpy(copy=True).ravel() if return_values else None

    return y_values, y_index


def preprocess_last_window(
    last_window: Union[pd.Series, pd.DataFrame],
    return_values: bool = True
 ) -> Tuple[np.ndarray, pd.Index]:
    """
    Return values and index of series separately. Index is overwritten 
    according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.
    
    Parameters
    ----------
    last_window : pandas Series, pandas DataFrame
        Time series values.
    return_values : bool, default `True`
        If `True` return the values of `last_window` as numpy ndarray. This option 
        is intended to avoid copying data when it is not necessary.

    Returns
    -------
    last_window_values : numpy ndarray
        Numpy array with values of `last_window`.
    last_window_index : pandas Index
        Index of `last_window` modified according to the rules.
    
    """
    
    if isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is not None:
        last_window_index = last_window.index
    elif isinstance(last_window.index, pd.RangeIndex):
        last_window_index = last_window.index
    elif isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is None:
        warnings.warn(
            ("`last_window` has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                            )
    else:
        warnings.warn(
            ("`last_window` has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                            )

    last_window_values = last_window.to_numpy(copy=True).ravel() if return_values else None

    return last_window_values, last_window_index


def preprocess_exog(
    exog: Union[pd.Series, pd.DataFrame],
    return_values: bool = True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Return values and index of series or data frame separately. Index is
    overwritten  according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.

    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.
    return_values : bool, default `True`
        If `True` return the values of `exog` as numpy ndarray. This option is 
        intended to avoid copying data when it is not necessary.

    Returns
    -------
    exog_values : None, numpy ndarray
        Numpy array with values of `exog`.
    exog_index : pandas Index
        Index of `exog` modified according to the rules.
    
    """
    
    if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is not None:
        exog_index = exog.index
    elif isinstance(exog.index, pd.RangeIndex):
        exog_index = exog.index
    elif isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is None:
        warnings.warn(
            ("`exog` has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    else:
        warnings.warn(
            ("`exog` has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    exog_values = exog.to_numpy(copy=True) if return_values else None

    return exog_values, exog_index


def input_to_frame(
    data: Union[pd.Series, pd.DataFrame],
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


def cast_exog_dtypes(
    exog: Union[pd.Series, pd.DataFrame],
    exog_dtypes: dict,
) -> Union[pd.Series, pd.DataFrame]:  # pragma: no cover
    """
    Cast `exog` to a specified types. This is done because, for a forecaster to 
    accept a categorical exog, it must contain only integer values. Due to the 
    internal modifications of numpy, the values may be casted to `float`, so 
    they have to be re-converted to `int`.

    - If `exog` is a pandas Series, `exog_dtypes` must be a dict with a 
    single value.
    - If `exog_dtypes` is `category` but the current type of `exog` is `float`, 
    then the type is cast to `int` and then to `category`. 

    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.
    exog_dtypes: dict
        Dictionary with name and type of the series or data frame columns.

    Returns
    -------
    exog : pandas Series, pandas DataFrame
        Exogenous variables casted to the indicated dtypes.

    """

    # Remove keys from exog_dtypes not in exog.columns
    exog_dtypes = {k: v for k, v in exog_dtypes.items() if k in exog.columns}
    
    if isinstance(exog, pd.Series) and exog.dtypes != list(exog_dtypes.values())[0]:
        exog = exog.astype(list(exog_dtypes.values())[0])
    elif isinstance(exog, pd.DataFrame):
        for col, initial_dtype in exog_dtypes.items():
            if exog[col].dtypes != initial_dtype:
                if initial_dtype == "category" and exog[col].dtypes == float:
                    exog[col] = exog[col].astype(int).astype("category")
                else:
                    exog[col] = exog[col].astype(initial_dtype)

    return exog


def exog_to_direct(
    exog: Union[pd.Series, pd.DataFrame],
    steps: int
) -> Union[pd.DataFrame, list]:
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

    if len(exog_direct) > 1:
        exog_direct = pd.concat(exog_direct, axis=1, copy=False)
    else:
        exog_direct = exog_direct[0]

    exog_direct_names = exog_direct.columns.to_list()
    exog_direct.index = exog_idx[-len(exog_direct):]
    
    return exog_direct, exog_direct_names


def exog_to_direct_numpy(
    exog: Union[np.ndarray, pd.Series, pd.DataFrame],
    steps: int
) -> Tuple[np.ndarray, Optional[list]]:
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
    exog_direct = []
    for i in range(steps):
        exog_step = exog[i : n_rows - (steps - 1 - i)]
        exog_direct.append(exog_step)

    if len(exog_direct) > 1:
        exog_direct = np.concatenate(exog_direct, axis=1)
    else:
        exog_direct = exog_direct[0]
    
    return exog_direct, exog_direct_names


def date_to_index_position(
    index: pd.Index,
    date_input: Union[int, str, pd.Timestamp],
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
    date_literal : str, default 'steps'
        Variable name used in error messages.
    kwargs_pd_to_datetime : dict, default {}
        Additional keyword arguments to pass to `pd.to_datetime()`.
    
    Returns
    -------
    date_position : int
        Integer representing the position of the datetime in the index.
    
    """
    
    if isinstance(date_input, (str, pd.Timestamp)):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                f"Index must be a pandas DatetimeIndex when `{date_literal}` is "
                f"not an integer. Check input series or last window."
            )
        
        target_date = pd.to_datetime(date_input, **kwargs_pd_to_datetime)
        last_date = pd.to_datetime(index[-1])
        if target_date <= last_date:
            raise ValueError(
                "The provided date must be later than the last date in the index."
            )
        
        steps_diff = pd.date_range(start=last_date, end=target_date, freq=index.freq)
        date_position = len(steps_diff) - 1
    
    elif isinstance(date_input, (int, np.integer)):
        date_position = date_input
    else:
        raise TypeError(
            f"`{date_literal}` must be an integer, string, or pandas Timestamp."
        )
    
    return date_position


def expand_index(
    index: Union[pd.Index, None], 
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
            new_index = pd.date_range(
                            start   = index[-1] + index.freq,
                            periods = steps,
                            freq    = index.freq
                        )
        elif isinstance(index, pd.RangeIndex):
            new_index = pd.RangeIndex(
                            start = index[-1] + 1,
                            stop  = index[-1] + 1 + steps
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
    transformer,
    fit: bool = False,
    inverse_transform: bool = False
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
    fit : bool, default `False`
        Train the transformer before applying it.
    inverse_transform : bool, default `False`
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

    Returns
    -------
    array_transformed : numpy ndarray
        Transformed array.

    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"`array` argument must be a numpy ndarray. Got {type(array)}"
        )

    if transformer is None:
        return array
    
    array_ndim = array.ndim
    if array_ndim == 1:
        array = array.reshape(-1, 1)

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    if not inverse_transform:
        if fit:
            array_transformed = transformer.fit_transform(array)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    message="X does not have valid feature names", 
                    category=UserWarning
                )
                array_transformed = transformer.transform(array)
    else:
        array_transformed = transformer.inverse_transform(array)

    if hasattr(array_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense
        array_transformed = array_transformed.toarray()

    if array_ndim == 1:
        array_transformed = array_transformed.ravel()

    return array_transformed


def transform_series(
    series: pd.Series,
    transformer,
    fit: bool = False,
    inverse_transform: bool = False
) -> Union[pd.Series, pd.DataFrame]:
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
    fit : bool, default `False`
        Train the transformer before applying it.
    inverse_transform : bool, default `False`
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

    Returns
    -------
    series_transformed : pandas Series, pandas DataFrame
        Transformed Series. Depending on the transformer used, the output may 
        be a Series or a DataFrame.

    """
    
    if not isinstance(series, pd.Series):
        raise TypeError(
            (f"`series` argument must be a pandas Series. Got {type(series)}.")
        )
        
    if transformer is None:
        return series

    if series.name is None:
        series.name = 'no_name'
        
    data = series.to_frame()

    if fit and hasattr(transformer, 'fit'):
        transformer.fit(data)

    # If argument feature_names_in_ exits, is overwritten to allow using the 
    # transformer on other series than those that were passed during fit.
    if hasattr(transformer, 'feature_names_in_') and transformer.feature_names_in_[0] != data.columns[0]:
        transformer = deepcopy(transformer)
        transformer.feature_names_in_ = np.array([data.columns[0]], dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if inverse_transform:
            values_transformed = transformer.inverse_transform(data)
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
        series_transformed = pd.DataFrame(
                                 data    = values_transformed,
                                 index   = data.index,
                                 columns = transformer.get_feature_names_out()
                             )

    return series_transformed


def transform_dataframe(
    df: pd.DataFrame,
    transformer,
    fit: bool = False,
    inverse_transform: bool = False
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
    fit : bool, default `False`
        Train the transformer before applying it.
    inverse_transform : bool, default `False`
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

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
 
    if not inverse_transform:
        if fit:
            values_transformed = transformer.fit_transform(df)
        else:
            values_transformed = transformer.transform(df)
    else:
        values_transformed = transformer.inverse_transform(df)

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense
        values_transformed = values_transformed.toarray()

    if hasattr(transformer, 'get_feature_names_out'):
        feature_names_out = transformer.get_feature_names_out()
    elif hasattr(transformer, 'categories_'):   
        feature_names_out = transformer.categories_
    else:
        feature_names_out = df.columns

    df_transformed = pd.DataFrame(
                         data    = values_transformed,
                         index   = df.index,
                         columns = feature_names_out
                     )

    return df_transformed


def save_forecaster(
    forecaster: object, 
    file_name: str,
    save_custom_functions: bool = True, 
    verbose: bool = True
) -> None:
    """
    Save forecaster model using joblib. If custom functions are used to create
    weights, they are saved as .py files.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster created with skforecast library.
    file_name : str
        File name given to the object.
    save_custom_functions : bool, default True
        If True, save custom functions used in the forecaster (weight_func) as 
        .py files. Custom functions need to be available in the environment 
        where the forecaster is going to be loaded.
    verbose : bool, default True
        Print summary about the forecaster saved.

    Returns
    -------
    None

    """
    
    # TODO: Ver con Ximo, esto si no tiene sufijo o no es .joblib lo cambia
    file_name = Path(file_name).with_suffix('.joblib')

    # Save forecaster
    joblib.dump(forecaster, filename=file_name)

    if save_custom_functions:
        # Save custom functions to create weights
        if hasattr(forecaster, 'weight_func') and forecaster.weight_func is not None:
            if isinstance(forecaster.weight_func, dict):
                for fun in set(forecaster.weight_func.values()):
                    file_name = fun.__name__ + '.py'
                    with open(file_name, 'w') as file:
                        file.write(inspect.getsource(fun))
            else:
                file_name = forecaster.weight_func.__name__ + '.py'
                with open(file_name, 'w') as file:
                    file.write(inspect.getsource(forecaster.weight_func))
    else:
        if hasattr(forecaster, 'weight_func') and forecaster.weight_func is not None:
            warnings.warn(
                "Custom function(s) used to create weights are not saved. To save them, "
                "set `save_custom_functions` to `True`.",
                SaveLoadSkforecastWarning
            )

    # TODO: Include docs link in the warning message
    if hasattr(forecaster, 'window_features') and forecaster.window_features is not None:
        skforecast_classes = {'RollingFeatures'}
        custom_classes = set(forecaster.window_features_class_names) - skforecast_classes
        if custom_classes:
            warnings.warn(
                "The Forecaster includes custom user-defined classes in the "
                "`window_features` argument. These classes are not saved automatically "
                "when saving the Forecaster. Please ensure you save these classes "
                "manually and import them before loading the Forecaster.\n"
                "    Custom classes: " + ', '.join(custom_classes),
                SaveLoadSkforecastWarning
            )

    if verbose:
        forecaster.summary()


def load_forecaster(
    file_name: str,
    verbose: bool = True
) -> object:
    """
    Load forecaster model using joblib. If the forecaster was saved with 
    custom user-defined classes as as window features or custom
    functions to create weights, these objects must be available
    in the environment where the forecaster is going to be loaded.

    Parameters
    ----------
    file_name: str
        Object file name.
    verbose: bool, default `True`
        Print summary about the forecaster loaded.

    Returns
    -------
    forecaster: Forecaster
        Forecaster created with skforecast library.
    
    """

    forecaster = joblib.load(filename=Path(file_name))

    skforecast_v = skforecast.__version__
    forecaster_v = forecaster.skforecast_version

    if forecaster_v != skforecast_v:
        warnings.warn(
            f"The skforecast version installed in the environment differs "
            f"from the version used to create the forecaster.\n"
            f"    Installed Version  : {skforecast_v}\n"
            f"    Forecaster Version : {forecaster_v}\n"
            f"This may create incompatibilities when using the library.",
             SkforecastVersionWarning
        )

    if verbose:
        forecaster.summary()

    return forecaster


def _find_optional_dependency(
    package_name: str, 
    optional_dependencies: dict = optional_dependencies
) -> Tuple[str, str]:
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
    package_version: srt
        Name and versions of the dependency.

    """

    for extra, packages in optional_dependencies.items():
        package_version = [package for package in packages if package_name in package]
        if package_version:
            return extra, package_version[0]


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

    if importlib.util.find_spec(package_name) is None:
        try:
            extra, package_version = _find_optional_dependency(package_name=package_name)
            msg = (
                f"\n'{package_name}' is an optional dependency not included in the default "
                f"skforecast installation. Please run: `pip install \"{package_version}\"` to install it."
                f"\n\nAlternately, you can install it by running `pip install skforecast[{extra}]`"
            )
        except:
            msg = f"\n'{package_name}' is needed but not installed. Please install it."
        
        raise ImportError(msg)


def multivariate_time_series_corr(
    time_series: pd.Series,
    other: pd.DataFrame,
    lags: Union[int, list, np.array],
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
    regressor_name: str,
) -> int:
    """
    Select the optimal number of jobs to use in the fitting process. This
    selection is based on heuristics and is not guaranteed to be optimal. 
    
    The number of jobs is chosen as follows:
    
    - If forecaster_name is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and regressor_name is a linear regressor then `n_jobs = 1`, otherwise `n_jobs = cpu_count() - 1`.

    - If `LGBMRegressor` then `n_jobs = 1`. This is because `lightgbm` 
    is highly optimized for gradient boosting and parallelizes operations at a very 
    fine-grained level, making additional parallelization unnecessary and 
    potentially harmful due to resource contention.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    regressor_name : str
        The type of regressor.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.
    
    """

    linear_regressors = [
        regressor_name
        for regressor_name in dir(sklearn.linear_model)
        if not regressor_name.startswith('_')
    ]

    if forecaster_name in ['ForecasterDirect', 
                           'ForecasterDirectMultiVariate']:
        if regressor_name in linear_regressors or regressor_name == 'LGBMRegressor':
            n_jobs = 1
        else:
            n_jobs = joblib.cpu_count() - 1
    else:
        n_jobs = 1

    return n_jobs


def check_preprocess_series(
    series: Union[pd.DataFrame, dict],
) -> Tuple[dict, pd.Index]:
    """
    Check and preprocess `series` argument in `ForecasterRecursiveMultiSeries` class.

    - If `series` is a pandas DataFrame, it is converted to a dict of pandas 
    Series and index is overwritten according to the rules of preprocess_y.
    - If `series` is a dict, all values are converted to pandas Series. Checks
    if all index are pandas DatetimeIndex and, at least, one Series has a non-null
    frequency. No multiple frequency is allowed.

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

    if isinstance(series, pd.DataFrame):

        _, series_index = preprocess_y(y=series, return_values=False)
        series = series.copy()
        series.index = series_index
        series_dict = series.to_dict("series")

    elif isinstance(series, dict):

        not_valid_series = [
            k 
            for k, v in series.items()
            if not isinstance(v, (pd.Series, pd.DataFrame))
        ]
        if not_valid_series:
            raise TypeError(
                (f"If `series` is a dictionary, all series must be a named "
                 f"pandas Series or a pandas DataFrame with a single column. "
                 f"Review series: {not_valid_series}")
            )

        series_dict = {
            k: v.copy()
            for k, v in series.items()
        }

        for k, v in series_dict.items():
            if isinstance(v, pd.DataFrame):
                if v.shape[1] != 1:
                    raise ValueError(
                        (f"If `series` is a dictionary, all series must be a named "
                         f"pandas Series or a pandas DataFrame with a single column. "
                         f"Review series: '{k}'")
                    )
                series_dict[k] = v.iloc[:, 0]

            series_dict[k].name = k

        not_valid_index = [
            k 
            for k, v in series_dict.items()
            if not isinstance(v.index, pd.DatetimeIndex)
        ]
        if not_valid_index:
            raise TypeError(
                (f"If `series` is a dictionary, all series must have a Pandas "
                 f"DatetimeIndex as index with the same frequency. "
                 f"Review series: {not_valid_index}")
            )

        indexes_freq = [f"{v.index.freq}" for v in series_dict.values()]
        indexes_freq = sorted(set(indexes_freq))
        if not len(indexes_freq) == 1:
            raise ValueError(
                (f"If `series` is a dictionary, all series must have a Pandas "
                 f"DatetimeIndex as index with the same frequency. "
                 f"Found frequencies: {indexes_freq}")
            )
    else:
        raise TypeError(
            (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
             f"Got {type(series)}.")
        )

    for k, v in series_dict.items():
        if np.isnan(v).all():
            raise ValueError(f"All values of series '{k}' are NaN.")

    series_indexes = {
        k: v.index
        for k, v in series_dict.items()
    }

    return series_dict, series_indexes


def check_preprocess_exog_multiseries(
    input_series_is_dict: bool,
    series_indexes: dict,
    series_names_in_: list,
    exog: Union[pd.Series, pd.DataFrame, dict],
    exog_dict: dict,
) -> Tuple[dict, list]:
    """
    Check and preprocess `exog` argument in `ForecasterRecursiveMultiSeries` class.

    - If input series is a pandas DataFrame (input_series_is_dict = False),  
    checks that input exog (pandas Series, DataFrame or dict) has the same index 
    (type, length and frequency). Index is overwritten according to the rules 
    of preprocess_exog. Create a dict of exog with the same keys as series.
    - If input series is a dict (input_series_is_dict = True), then input 
    exog must be a dict. Check exog has a pandas DatetimeIndex and convert all
    values to pandas DataFrames.

    Parameters
    ----------
    input_series_is_dict : bool
        Indicates if input series argument is a dict.
    series_indexes : dict
        Dictionary with the index of each series.
    series_names_in_ : list
        Names of the series (levels) used during training.
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
            (f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
             f"Series/DataFrames or None. Got {type(exog)}.")
        )

    if not input_series_is_dict:
        # If input series is a pandas DataFrame, all index are the same.
        # Select the first index to check exog
        series_index = series_indexes[series_names_in_[0]]

    if isinstance(exog, (pd.Series, pd.DataFrame)): 

        if input_series_is_dict:
            raise TypeError(
                (f"`exog` must be a dict of DataFrames or Series if "
                 f"`series` is a dict. Got {type(exog)}.")
            )

        _, exog_index = preprocess_exog(exog=exog, return_values=False)
        exog = exog.copy().to_frame() if isinstance(exog, pd.Series) else exog.copy()
        exog.index = exog_index

        if len(exog) != len(series_index):
            raise ValueError(
                (f"`exog` must have same number of samples as `series`. "
                 f"length `exog`: ({len(exog)}), length `series`: ({len(series_index)})")
            )

        if not (exog_index == series_index).all():
            raise ValueError(
                ("Different index for `series` and `exog`. They must be equal "
                 "to ensure the correct alignment of values.")
            )

        exog_dict = {serie: exog for serie in series_names_in_}

    else:

        not_valid_exog = [
            k 
            for k, v in exog.items()
            if not isinstance(v, (pd.Series, pd.DataFrame, type(None)))
        ]
        if not_valid_exog:
            raise TypeError(
                (f"If `exog` is a dictionary, all exog must be a named pandas "
                 f"Series, a pandas DataFrame or None. Review exog: {not_valid_exog}")
            )

        # Only elements already present in exog_dict are updated
        exog_dict.update(
            (k, v.copy())
            for k, v in exog.items() 
            if k in exog_dict and v is not None
        )

        series_not_in_exog = set(series_names_in_) - set(exog.keys())
        if series_not_in_exog:
            warnings.warn(
                (f"{series_not_in_exog} not present in `exog`. All values "
                 f"of the exogenous variables for these series will be NaN."),
                 MissingExogWarning
            )

        for k, v in exog_dict.items():
            if v is not None:
                check_exog(exog=v, allow_nan=True)
                if isinstance(v, pd.Series):
                    v = v.to_frame()
                exog_dict[k] = v

        if not input_series_is_dict:
            for k, v in exog_dict.items():
                if v is not None:
                    if len(v) != len(series_index):
                        raise ValueError(
                            (f"`exog` for series '{k}' must have same number of "
                             f"samples as `series`. length `exog`: ({len(v)}), "
                             f"length `series`: ({len(series_index)})")
                        )

                    _, v_index = preprocess_exog(exog=v, return_values=False)
                    exog_dict[k].index = v_index
                    if not (exog_dict[k].index == series_index).all():
                        raise ValueError(
                            (f"Different index for series '{k}' and its exog. "
                             f"When `series` is a pandas DataFrame, they must be "
                             f"equal to ensure the correct alignment of values.")
                        )
        else:
            not_valid_index = [
                k
                for k, v in exog_dict.items()
                if v is not None and not isinstance(v.index, pd.DatetimeIndex)
            ]
            if not_valid_index:
                raise TypeError(
                    (f"All exog must have a Pandas DatetimeIndex as index with the "
                     f"same frequency. Check exog for series: {not_valid_index}")
                )
            
        # Check that all exog have the same dtypes for common columns
        exog_dtypes_buffer = [df.dtypes for df in exog_dict.values() if df is not None]
        exog_dtypes_buffer = pd.concat(exog_dtypes_buffer, axis=1)
        exog_dtypes_nunique = exog_dtypes_buffer.nunique(axis=1).eq(1)
        if not exog_dtypes_nunique.all():
            non_unique_dtyeps_exogs = exog_dtypes_nunique[exog_dtypes_nunique != 1].index.to_list()
            raise TypeError(f"Exog/s: {non_unique_dtyeps_exogs} have different dtypes in different series.")

    exog_names_in_ = list(
        set(
            column
            for df in exog_dict.values()
            if df is not None
            for column in df.columns.to_list()
        )
    )

    if len(set(exog_names_in_) - set(series_names_in_)) != len(exog_names_in_):
        raise ValueError(
            (f"`exog` cannot contain a column named the same as one of the series.\n"
             f"    `series` columns : {series_names_in_}.\n"
             f"    `exog`   columns : {exog_names_in_}.")
        )

    return exog_dict, exog_names_in_


def align_series_and_exog_multiseries(
    series_dict: dict,
    input_series_is_dict: bool,
    exog_dict: dict = None
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """
    Align series and exog according to their index. If needed, reindexing is
    applied. Heading and trailing NaNs are removed from all series in 
    `series_dict`.

    - If input series is a pandas DataFrame (input_series_is_dict = False),  
    input exog (pandas Series, DataFrame or dict) must have the same index 
    (type, length and frequency). Reindexing is not applied.
    - If input series is a dict (input_series_is_dict = True), then input 
    exog must be a dict. Both must have a pandas DatetimeIndex, but can have 
    different lengths. Reindexing is applied.

    Parameters
    ----------
    series_dict : dict
        Dictionary with the series used during training.
    input_series_is_dict : bool
        Indicates if input series argument is a dict.
    exog_dict : dict, default `None`
        Dictionary with the exogenous variable/s used during training.

    Returns
    -------
    series_dict : dict
        Dictionary with the series used during training.
    exog_dict : dict
        Dictionary with the exogenous variable/s used during training.
    
    """

    for k in series_dict.keys():

        first_valid_index = series_dict[k].first_valid_index()
        last_valid_index = series_dict[k].last_valid_index()

        series_dict[k] = series_dict[k].loc[first_valid_index : last_valid_index]

        if exog_dict[k] is not None:
            if input_series_is_dict:
                index_intersection = (
                    series_dict[k].index.intersection(exog_dict[k].index)
                )
                if len(index_intersection) == 0:
                    warnings.warn(
                        (f"Series '{k}' and its `exog` do not have the same index. "
                         f"All exog values will be NaN for the period of the series."),
                         MissingValuesWarning
                    )
                elif len(index_intersection) != len(series_dict[k]):
                    warnings.warn(
                        (f"Series '{k}' and its `exog` do not have the same length. "
                         f"Exog values will be NaN for the not matched period of the series."),
                         MissingValuesWarning
                    )  
                exog_dict[k] = exog_dict[k].loc[index_intersection]
                if len(index_intersection) != len(series_dict[k]):
                    exog_dict[k] = exog_dict[k].reindex(
                                       series_dict[k].index, 
                                       fill_value = np.nan
                                   )
            else:
                exog_dict[k] = exog_dict[k].loc[first_valid_index : last_valid_index]

    return series_dict, exog_dict


def prepare_levels_multiseries(
    X_train_series_names_in_: list,
    levels: Optional[Union[str, list]] = None
) -> Tuple[list, bool]:
    """
    Prepare list of levels to be predicted in multiseries Forecasters.

    Parameters
    ----------
    X_train_series_names_in_ : list
        Names of the series (levels) included in the matrix `X_train`.
    levels : str, list, default `None`
        Names of the series (levels) to be predicted.

    Returns
    -------
    levels : list
        Names of the series (levels) to be predicted.

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
    levels: list,
    input_levels_is_list: bool,
    last_window_: dict
) -> Tuple[list, pd.DataFrame]:
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

    available_last_windows = set() if last_window_ is None else set(last_window_.keys())
    not_available_last_window = set(levels) - available_last_windows
    if not_available_last_window:
        levels = [level for level in levels 
                  if level not in not_available_last_window]
        if not levels:
            raise ValueError(
                (f"No series to predict. None of the series {not_available_last_window} "
                 f"are present in `last_window_` attribute. Provide `last_window` "
                 f"as argument in predict method.")
            )
        else:
            warnings.warn(
                (f"Levels {not_available_last_window} are excluded from "
                 f"prediction since they were not stored in `last_window_` "
                 f"attribute during training. If you don't want to retrain "
                 f"the Forecaster, provide `last_window` as argument."),
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
                (f"Only series whose last window ends at the same index "
                 f"can be predicted together. Series that do not reach "
                 f"the maximum index, '{max_index_levels}', are excluded "
                 f"from prediction: {series_excluded_from_last_window}."),
                IgnoredArgumentWarning
            )

    last_window = pd.DataFrame(
        {k: v 
         for k, v in last_window_.items() 
         if k in levels}
    )

    return levels, last_window


def prepare_residuals_multiseries(
    levels: list,
    use_in_sample_residuals: bool,
    encoding: Optional[str] = None,
    in_sample_residuals_: Optional[dict] = None,
    out_sample_residuals_: Optional[dict] = None
) -> Tuple[list, bool]:
    """
    Prepare residuals for bootstrapping prediction in multiseries Forecasters.

    Parameters
    ----------
    levels : list
        Names of the series (levels) to be predicted.
    use_in_sample_residuals : bool
        Indicates if `forecaster.in_sample_residuals_` are used.
    encoding : str, default `None`
        Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).
    in_sample_residuals_ : dict, default `None`
        Residuals of the model when predicting training data. Only stored up to
        1000 values in the form `{level: residuals}`. If `transformer_series` 
        is not `None`, residuals are stored in the transformed scale.
    out_sample_residuals_ : dict, default `None`
        Residuals of the model when predicting non-training data. Only stored
        up to 1000 values in the form `{level: residuals}`. If `transformer_series` 
        is not `None`, residuals are assumed to be in the transformed scale. Use 
        `set_out_sample_residuals()` method to set values.

    Returns
    -------
    levels : list
        Names of the series (levels) to be predicted.
    residuals : dict
        Residuals of the model for each level to use in bootstrapping prediction.

    """

    if use_in_sample_residuals:
        unknown_levels = set(levels) - set(in_sample_residuals_.keys())
        if unknown_levels and encoding is not None:
            warnings.warn(
                (f"`levels` {unknown_levels} are not present in `forecaster.in_sample_residuals_`, "
                 f"most likely because they were not present in the training data. "
                 f"A random sample of the residuals from other levels will be used. "
                 f"This can lead to inaccurate intervals for the unknown levels."),
                 UnknownLevelWarning
            )
        residuals = in_sample_residuals_.copy()
    else:
        if out_sample_residuals_ is None:
            raise ValueError(
                ("`forecaster.out_sample_residuals_` is `None`. Use "
                 "`use_in_sample_residuals=True` or the "
                 "`set_out_sample_residuals()` method before predicting.")
            )
        else:
            unknown_levels = set(levels) - set(out_sample_residuals_.keys())
            if unknown_levels and encoding is not None:
                warnings.warn(
                    (f"`levels` {unknown_levels} are not present in `forecaster.out_sample_residuals_`. "
                     f"A random sample of the residuals from other levels will be used. "
                     f"This can lead to inaccurate intervals for the unknown levels. "
                     f"Otherwise, Use the `set_out_sample_residuals()` method before "
                     f"predicting to set the residuals for these levels."),
                     UnknownLevelWarning
                )
            residuals = out_sample_residuals_.copy()

    check_residuals = (
        "forecaster.in_sample_residuals_" if use_in_sample_residuals
        else "forecaster.out_sample_residuals_"
    )
    for level in levels:
        if level in unknown_levels:
            residuals[level] = residuals['_unknown_level']
        if residuals[level] is None or len(residuals[level]) == 0:
            raise ValueError(
                (f"Not available residuals for level '{level}'. "
                 f"Check `{check_residuals}`.")
            )
        elif (any(element is None for element in residuals[level]) or
              np.any(np.isnan(residuals[level]))):
            raise ValueError(
                (f"forecaster residuals for level '{level}' contains `None` "
                 f"or `NaNs` values. Check `{check_residuals}`.")
            )
        
    return residuals


def prepare_steps_direct(
    max_step: int,
    steps: Optional[Union[int, list]] = None
) -> list:
    """
    Prepare list of steps to be predicted in Direct Forecasters.

    Parameters
    ----------
    max_step : int
        Maximum number of future steps the forecaster will predict 
        when using method `predict()`.
    steps : int, list, None, default `None`
        Predict n steps. The value of `steps` must be less than or equal to the 
        value of steps defined when initializing the forecaster. Starts at 1.
    
        - If `int`: Only steps within the range of 1 to int are predicted.
        - If `list`: List of ints. Only the steps contained in the list 
        are predicted.
        - If `None`: As many steps are predicted as were defined at 
        initialization.

    Returns
    -------
    steps : list
        Steps to be predicted.

    """

    if isinstance(steps, int):
        steps = list(np.arange(steps) + 1)
    elif steps is None:
        steps = list(np.arange(max_step) + 1)
    elif isinstance(steps, list):
        steps = list(np.array(steps))
    
    for step in steps:
        if not isinstance(step, (int, np.int64, np.int32)):
            raise TypeError(
                (f"`steps` argument must be an int, a list of ints or `None`. "
                 f"Got {type(steps)}.")
            )
    # Required since numpy 2.0
    steps = [int(step) for step in steps if step is not None]

    return steps


def set_skforecast_warnings(
    suppress_warnings: bool,
    action: str = 'default'
) -> None:
    """
    Set skforecast warnings action.

    Parameters
    ----------
    suppress_warnings : bool
        If `True`, skforecast warnings will be suppressed. If `False`, skforecast
        warnings will be shown as default. See 
        skforecast.exceptions.warn_skforecast_categories for more information.
    action : str, default `'default'`
        Action to be taken when a warning is raised. See the warnings module
        for more information.

    Returns
    -------
    None
    
    """

    if suppress_warnings:
        for category in warn_skforecast_categories:
            warnings.filterwarnings(action, category=category)
