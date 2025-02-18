################################################################################
#                           skforecast.preprocessing                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
from typing_extensions import Self
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from ..exceptions import MissingValuesWarning
from numba import njit
from copy import deepcopy


def _check_X_numpy_ndarray_1d(ensure_1d=True):
    """
    This decorator checks if the argument X is a numpy ndarray with 1 dimension.

    Parameters
    ----------
    ensure_1d : bool, default True
        Whether to ensure if X is a 1D numpy array.
    
    Returns
    -------
    decorator : Callable
        A decorator function.

    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):

            if args:
                X = args[0] 
            elif 'X' in kwargs:
                X = kwargs['X']
            else:
                raise ValueError("Methods must be called with 'X' as argument.")

            if not isinstance(X, np.ndarray):
                raise TypeError(f"'X' must be a numpy ndarray. Found {type(X)}.")
            if ensure_1d and not X.ndim == 1:
                raise ValueError(f"'X' must be a 1D array. Found {X.ndim} dimensions.")
            
            result = func(self, *args, **kwargs)
            
            return result
        
        return wrapper
    
    return decorator


class TimeSeriesDifferentiator(BaseEstimator, TransformerMixin):
    """
    Transforms a time series into a differentiated time series of a specified order
    and provides functionality to revert the differentiation. 
    
    When using a `direct` module Forecaster, the model in step 1 must be 
    used if you want to reverse the differentiation of the training time 
    series with the `inverse_transform_training` method.

    Parameters
    ----------
    order : int
        The order of differentiation to be applied.
    window_size : int, default None
        The window size used by the forecaster. This is required to revert the 
        differentiation for the target variable `y` or its predicted values.

    Attributes
    ----------
    order : int
        The order of differentiation.
    initial_values : list
        List with the first value of the time series before each differentiation.
        If `order = 2`, first value correspond with the first value of the original
        time series and the second value correspond with the first value of the
        differentiated time series of order 1. These values are necessary to 
        revert the differentiation and reconstruct the original time series.
    pre_train_values : list
        List with the first training value of the time series before each differentiation.
        For `order = 1`, the value correspond with the last value of the window used to
        create the predictors. For order > 1, the value correspond with the first
        value of the differentiated time series prior to the next differentiation.
        These values are necessary to revert the differentiation and reconstruct the
        training time series.
    last_values : list
        List with the last value of the time series before each differentiation, 
        used to revert differentiation on subsequent data windows. If `order = 2`, 
        first value correspond with the last value of the original time series 
        and the second value correspond with the last value of the differentiated 
        time series of order 1. This is essential for correctly transforming a 
        time series that follows immediately after the series used to fit the 
        transformer.

    """

    def __init__(
        self, 
        order: int = 1,
        window_size: int | None = None
    ) -> None:

        if not isinstance(order, (int, np.integer)):
            raise TypeError(
                f"Parameter `order` must be an integer greater than 0. Found {type(order)}."
            )
        if order < 1:
            raise ValueError(
                f"Parameter `order` must be an integer greater than 0. Found {order}."
            )

        if window_size is not None:
            if not isinstance(window_size, (int, np.integer)):
                raise TypeError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {type(window_size)}."
                )
            if window_size < 1:
                raise ValueError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {window_size}."
                )

        self.order = order
        self.window_size = window_size
        self.initial_values = []
        self.pre_train_values = []
        self.last_values = []

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"TimeSeriesDifferentiator(order={self.order}, window_size={self.window_size})"
        )

    @_check_X_numpy_ndarray_1d()
    def fit(
        self, 
        X: np.ndarray, 
        y: Any = None
    ) -> Self:
        """
        Fits the transformer. Stores the values needed to revert the 
        differentiation of different window of the time series, original 
        time series, training time series, and a time series that follows
        immediately after the series used to fit the transformer.

        Parameters
        ----------
        X : numpy ndarray
            Time series to be differentiated.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : TimeSeriesDifferentiator

        """

        self.initial_values = []
        self.pre_train_values = []
        self.last_values = []

        for i in range(self.order):
            if i == 0:
                self.initial_values.append(X[0])
                if self.window_size is not None:
                    self.pre_train_values.append(X[self.window_size - self.order])
                self.last_values.append(X[-1])
                X_diff = np.diff(X, n=1)
            else:
                self.initial_values.append(X_diff[0])
                if self.window_size is not None:
                    self.pre_train_values.append(X_diff[self.window_size - self.order])
                self.last_values.append(X_diff[-1])
                X_diff = np.diff(X_diff, n=1)

        return self

    @_check_X_numpy_ndarray_1d()
    def transform(
        self, 
        X: np.ndarray, 
        y: Any = None
    ) -> np.ndarray:
        """
        Transforms a time series into a differentiated time series of order n.

        Parameters
        ----------
        X : numpy ndarray
            Time series to be differentiated.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_diff : numpy ndarray
            Differentiated time series. The length of the array is the same as
            the original time series but the first n `order` values are nan.

        """

        X_diff = np.diff(X, n=self.order)
        X_diff = np.append((np.full(shape=self.order, fill_value=np.nan)), X_diff)

        return X_diff

    @_check_X_numpy_ndarray_1d()
    def inverse_transform(
        self, 
        X: np.ndarray, 
        y: Any = None
    ) -> np.ndarray:
        """
        Reverts the differentiation. To do so, the input array is assumed to be
        the same time series used to fit the transformer but differentiated.

        Parameters
        ----------
        X : numpy ndarray
            Differentiated time series.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_diff : numpy ndarray
            Reverted differentiated time series.
        
        """

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]
        for i in range(self.order):
            if i == 0:
                X_undiff = np.insert(X, 0, self.initial_values[-1])
                X_undiff = np.cumsum(X_undiff, dtype=float)
            else:
                X_undiff = np.insert(X_undiff, 0, self.initial_values[-(i + 1)])
                X_undiff = np.cumsum(X_undiff, dtype=float)

        return X_undiff

    @_check_X_numpy_ndarray_1d()
    def inverse_transform_training(
        self, 
        X: np.ndarray, 
        y: Any = None
    ) -> np.ndarray:
        """
        Reverts the differentiation. To do so, the input array is assumed to be
        the differentiated training time series generated with the original 
        time series used to fit the transformer.

        When using a `direct` module Forecaster, the model in step 1 must be 
        used if you want to reverse the differentiation of the training time 
        series with the `inverse_transform_training` method.

        Parameters
        ----------
        X : numpy ndarray
            Differentiated time series.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_diff : numpy ndarray
            Reverted differentiated time series.
        
        """

        if not self.pre_train_values:
            raise ValueError(
                "The `window_size` parameter must be set before fitting the "
                "transformer to revert the differentiation of the training "
                "time series."
            )

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]
        for i in range(self.order):
            if i == 0:
                X_undiff = np.insert(X, 0, self.pre_train_values[-1])
                X_undiff = np.cumsum(X_undiff, dtype=float)
            else:
                X_undiff = np.insert(X_undiff, 0, self.pre_train_values[-(i + 1)])
                X_undiff = np.cumsum(X_undiff, dtype=float)

        # Remove initial values as they are not part of the training time series
        X_undiff = X_undiff[self.order:]

        return X_undiff

    @_check_X_numpy_ndarray_1d(ensure_1d=False)
    def inverse_transform_next_window(
        self,
        X: np.ndarray,
        y: Any = None
    ) -> np.ndarray:
        """
        Reverts the differentiation. The input array `X` is assumed to be a 
        differentiated time series of order n that starts right after the
        the time series used to fit the transformer.

        Parameters
        ----------
        X : numpy ndarray
            Differentiated time series. It is assumed o start right after
            the time series used to fit the transformer.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_undiff : numpy ndarray
            Reverted differentiated time series.
        
        """
        
        array_ndim = X.ndim
        if array_ndim == 1:
            X = X[:, np.newaxis]

        # Remove initial rows with nan values if present
        X = X[~np.isnan(X).any(axis=1)]

        for i in range(self.order):
            if i == 0:
                X_undiff = np.cumsum(X, axis=0, dtype=float) + self.last_values[-1]
            else:
                X_undiff = np.cumsum(X_undiff, axis=0, dtype=float) + self.last_values[-(i + 1)]

        if array_ndim == 1:
            X_undiff = X_undiff.ravel()

        return X_undiff

    def set_params(self, **params):
        """
        Set the parameters of the TimeSeriesDifferentiator.
        
        Parameters
        ----------
        params : dict
            A dictionary of the parameters to set.

        Returns
        -------
        None
        
        """

        for param, value in params.items():
            setattr(self, param, value)


def series_long_to_dict(
    data: pd.DataFrame,
    series_id: str,
    index: str,
    values: str,
    freq: str,
    suppress_warnings: bool = False
) -> dict[str, pd.Series]:
    """
    Convert long format series to dictionary of pandas Series with frequency.
    Input data must be a pandas DataFrame with columns for the series identifier,
    time index, and values. The function will group the data by the series
    identifier and convert the time index to a datetime index with the given
    frequency.

    Parameters
    ----------
    data: pandas DataFrame
        Long format series.
    series_id: str
        Column name with the series identifier.
    index: str
        Column name with the time index.
    values: str
        Column name with the values.
    freq: str
        Frequency of the series.
    suppress_warnings: bool, default False
        If True, suppress warnings when a series is incomplete after setting the
        frequency.

    Returns
    -------
    series_dict: dict
        Dictionary with the series.

    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    for col in [series_id, index, values]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in `data`.")
        
    original_sizes = data.groupby(series_id, observed=True).size()
    series_dict = {}
    for k, v in data.groupby(series_id, observed=True):
        series_dict[k] = v.set_index(index)[values].asfreq(freq, fill_value=np.nan).rename(k)
        series_dict[k].index.name = None
        if not suppress_warnings and len(series_dict[k]) != original_sizes[k]:
            warnings.warn(
                f"Series '{k}' is incomplete. NaNs have been introduced after "
                f"setting the frequency.",
                MissingValuesWarning
            )

    return series_dict


def exog_long_to_dict(
    data: pd.DataFrame,
    series_id: str,
    index: str,
    freq: str,
    drop_all_nan_cols: bool = False,
    consolidate_dtypes: bool = True,
    suppress_warnings: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Convert long format exogenous variables to dictionary. Input data must be a
    pandas DataFrame with columns for the series identifier, time index, and
    exogenous variables. The function will group the data by the series identifier
    and convert the time index to a datetime index with the given frequency.

    Parameters
    ----------
    data: pandas DataFrame
        Long format exogenous variables.
    series_id: str
        Column name with the series identifier.
    index: str
        Column name with the time index.
    freq: str
        Frequency of the series.
    drop_all_nan_cols: bool, default False
        If True, drop columns with all values as NaN. This is useful when
        there are series without some exogenous variables.
    consolidate_dtypes: bool, default True
        Consolidate the data types of the exogenous variables if, after setting
        the frequency, NaNs have been introduced and the data types have changed
        to float.
    suppress_warnings: bool, default False
        If True, suppress warnings when exog is incomplete after setting the
        frequency.
        
    Returns
    -------
    exog_dict: dict
        Dictionary with the exogenous variables.

    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    for col in [series_id, index]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in `data`.")

    cols_float_dtype = set(data.select_dtypes(include=float).columns)
    original_sizes = data.groupby(series_id, observed=True).size()
    exog_dict = dict(tuple(data.groupby(series_id, observed=True)))
    exog_dict = {
        k: v.set_index(index).asfreq(freq, fill_value=np.nan).drop(columns=series_id)
        for k, v in exog_dict.items()
    }

    for k in exog_dict.keys():
        exog_dict[k].index.name = None

    nans_introduced = False
    if not suppress_warnings or consolidate_dtypes:
        for k, v in exog_dict.items():
            if len(v) != original_sizes[k]:
                nans_introduced = True
                if not suppress_warnings:
                    warnings.warn(
                        f"Exogenous variables for series '{k}' are incomplete. "
                        f"NaNs have been introduced after setting the frequency.",
                        MissingValuesWarning
                    )
                if consolidate_dtypes:
                    cols_float_dtype.update(v.select_dtypes(include=float).columns)

    if consolidate_dtypes and nans_introduced:
        new_dtypes = {k: float for k in cols_float_dtype}
        exog_dict = {k: v.astype(new_dtypes) for k, v in exog_dict.items()}

    if drop_all_nan_cols:
        exog_dict = {k: v.dropna(how="all", axis=1) for k, v in exog_dict.items()}

    return exog_dict


def create_datetime_features(
    X: pd.Series | pd.DataFrame,
    features: list[str] | None = None,
    encoding: str = "cyclical",
    max_values: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Extract datetime features from the DateTime index of a pandas DataFrame or Series.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input DataFrame or Series with a datetime index.
    features : list, default None
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second'.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical encoding of calendar features.
        When `None`, the following values are used: {'month': 12, 'week': 52, 
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24, 
        'minute': 60, 'second': 60}.

    Returns
    -------
    X_new : pandas DataFrame
        DataFrame with the extracted (and optionally encoded) datetime features.
    
    """

    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError("Input `X` must be a pandas Series or DataFrame")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Input `X` must have a pandas DatetimeIndex")
    if encoding not in ["cyclical", "onehot", None]:
        raise ValueError("Encoding must be one of 'cyclical', 'onehot' or None")

    default_features = [
        "year",
        "month",
        "week",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "weekend",
        "hour",
        "minute",
        "second",
    ]
    features = features or default_features

    default_max_values = {
        "month": 12,
        "week": 52,
        "day_of_week": 7,
        "day_of_month": 31,
        "day_of_year": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }
    max_values = max_values or default_max_values

    X_new = pd.DataFrame(index=X.index)

    datetime_attrs = {
        "year": "year",
        "month": "month",
        "week": lambda idx: idx.isocalendar().week,
        "day_of_week": "dayofweek",
        "day_of_year": "dayofyear",
        "day_of_month": "day",
        "weekend": lambda idx: (idx.weekday >= 5).astype(int),
        "hour": "hour",
        "minute": "minute",
        "second": "second",
    }

    not_supported_features = set(features) - set(datetime_attrs.keys())
    if not_supported_features:
        raise ValueError(
            f"Features {not_supported_features} are not supported. "
            f"Supported features are {list(datetime_attrs.keys())}."
        )

    for feature in features:
        attr = datetime_attrs[feature]
        X_new[feature] = (
            attr(X.index) if callable(attr) else getattr(X.index, attr).astype(int)
        )

    if encoding == "cyclical":
        cols_to_drop = []
        for feature, max_val in max_values.items():
            if feature in X_new.columns:
                X_new[f"{feature}_sin"] = np.sin(2 * np.pi * X_new[feature] / max_val)
                X_new[f"{feature}_cos"] = np.cos(2 * np.pi * X_new[feature] / max_val)
                cols_to_drop.append(feature)
        X_new = X_new.drop(columns=cols_to_drop)
    elif encoding == "onehot":
        X_new = pd.get_dummies(
            X_new, columns=features, drop_first=False, sparse=False, dtype=int
        )

    return X_new


class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting datetime features from the DateTime index of a
    pandas DataFrame or Series. It can also apply encoding to the extracted features.

    Parameters
    ----------
    features : list, default None
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second'.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical encoding of calendar features.
        When `None`, the following values are used: {'month': 12, 'week': 52, 
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24, 
        'minute': 60, 'second': 60}.
    
    Attributes
    ----------
    features : list
        List of calendar features to extract from the index.
    encoding : str
        Encoding method for the extracted features.
    max_values : dict
        Dictionary of maximum values for the cyclical encoding of calendar features.
    
    """

    def __init__(
        self,
        features: list[str] | None = None,
        encoding: str = "cyclical",
        max_values: dict[str, int] | None = None
    ) -> None:

        if encoding not in ["cyclical", "onehot", None]:
            raise ValueError("Encoding must be one of 'cyclical', 'onehot' or None")

        self.features = (
            features
            if features is not None
            else [
                "year",
                "month",
                "week",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "weekend",
                "hour",
                "minute",
                "second",
            ]
        )
        self.encoding = encoding
        self.max_values = (
            max_values
            if max_values is not None
            else {
                "month": 12,
                "week": 52,
                "day_of_week": 7,
                "day_of_month": 31,
                "day_of_year": 365,
                "hour": 24,
                "minute": 60,
                "second": 60,
            }
        )

    def fit(self, X, y=None):
        """
        A no-op method to satisfy the scikit-learn API.
        """
        return self

    def transform(
        self,
        X: pd.Series | pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create datetime features from the DateTime index of a pandas DataFrame or Series.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame
            Input DataFrame or Series with a datetime index.
        
        Returns
        -------
        X_new : pandas DataFrame
            DataFrame with the extracted (and optionally encoded) datetime features.

        """

        X_new = create_datetime_features(
                    X          = X,
                    encoding   = self.encoding,
                    features   = self.features,
                    max_values = self.max_values,
                )

        return X_new


@njit
def _np_mean_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy mean function implemented with Numba JIT.
    """
    return np.mean(x)


@njit
def _np_std_jit(x: np.ndarray, ddof: int = 1) -> float:  # pragma: no cover
    """
    Standard deviation function implemented with Numba JIT.
    If the array has only one element, the function returns 0.
    """
    if len(x) == 1:
        return 0.
    
    a_a, b_b = 0, 0
    for i in x:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(x)) - ((a_a / (len(x))) ** 2)
    var = var * (len(x) / (len(x) - ddof))
    std = np.sqrt(var)

    return std


@njit
def _np_min_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy min function implemented with Numba JIT.
    """
    return np.min(x)


@njit
def _np_max_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy max function implemented with Numba JIT.
    """
    return np.max(x)


@njit
def _np_sum_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy sum function implemented with Numba JIT.
    """
    return np.sum(x)


@njit
def _np_median_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy median function implemented with Numba JIT.
    """
    return np.median(x)


@njit
def _np_min_max_ratio_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    NumPy min-max ratio function implemented with Numba JIT.
    """
    return np.min(x) / np.max(x)


@njit
def _np_cv_jit(x: np.ndarray) -> float:  # pragma: no cover
    """
    Coefficient of variation function implemented with Numba JIT.
    If the array has only one element, the function returns 0.
    """
    if len(x) == 1:
        return 0.
    
    a_a, b_b = 0, 0
    for i in x:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(x)) - ((a_a / (len(x))) ** 2)
    var = var * (len(x) / (len(x) - 1))
    std = np.sqrt(var)

    return std / np.mean(x)


@njit
def _ewm_jit(x: np.ndarray, alpha: float = 0.3) -> float:  # pragma: no cover
    """
    Calculate the exponentially weighted mean of an array.

    Parameters
    ----------
    x : numpy ndarray
        Input array.
    alpha : float, default 0.3
        Decay factor.

    Returns
    -------
    ewm : float
        The exponentially weighted mean.
    
    """
    if not (0 < alpha <= 1):
        raise ValueError("Alpha should be in the range (0, 1].")
    
    n = len(x)
    weights = 0
    sum_weights = 0
    for i in range(n):
        weight = (1 - alpha) ** (n - 1 - i)
        weights += x[i] * weight
        sum_weights += weight

    ewm = weights / sum_weights

    return ewm


class RollingFeatures():
    """
    This class computes rolling features. To avoid data leakage, the last point 
    in the window is excluded from calculations, ('closed': 'left' and 
    'center': False).
    
    Currently, the following statistics are supported: 'mean', 'std', 'min', 'max',
    'sum', 'median', 'ratio_min_max', 'coef_variation', 'ewm'. For 'ewm', the
    alpha parameter can be set in the kwargs_stats dictionary, default is
    {'ewm': {'alpha': 0.3}}.

    Parameters
    ----------
    stats : str, list
        Statistics to compute over the rolling window. Can be a `string` or a `list`,
        and can have repeats. Available statistics are: 'mean', 'std', 'min', 'max',
        'sum', 'median', 'ratio_min_max', 'coef_variation', 'ewm'. For 'ewm', the
        alpha parameter can be set in the kwargs_stats dictionary, default is
        {'ewm': {'alpha': 0.3}}.
    window_sizes : int, list
        Size of the rolling window for each statistic. If an `int`, all stats share 
        the same window size. If a `list`, it should have the same length as stats.
    min_periods : int, list, default None
        Minimum number of observations in window required to have a value. 
        Same as the `min_periods` argument of pandas rolling. If `None`, 
        defaults to `window_sizes`.
    features_names : list, default None
        Names of the output features. If `None`, default names will be used in the 
        format 'roll_stat_window_size', for example 'roll_mean_7'.
    fillna : str, float, default None
        Fill missing values in `transform_batch` method. Available 
        methods are: 'mean', 'median', 'ffill', 'bfill', or a float value.
    kwargs_stats : dict, default {'ewm': {'alpha': 0.3}}
        Dictionary with additional arguments for the statistics. The keys are the
        statistic names and the values are dictionaries with the arguments for the
        corresponding statistic. For example, {'ewm': {'alpha': 0.3}}.
    
    Attributes
    ----------
    stats : list
        Statistics to compute over the rolling window.
    n_stats : int
        Number of statistics to compute.
    window_sizes : list
        Size of the rolling window for each statistic.
    max_window_size : int
        Maximum window size.
    min_periods : list
        Minimum number of observations in window required to have a value.
    features_names : list
        Names of the output features.
    fillna : str, float
        Method to fill missing values in `transform_batch` method.
    unique_rolling_windows : dict
        Dictionary containing unique rolling window parameters and the corresponding
        statistics.
    kwargs_stats : dict
        Dictionary with additional arguments for the statistics. 
        
    """

    def __init__(
        self, 
        stats: str | list[str],
        window_sizes: int | list[int],
        min_periods: int | list[int] | None = None,
        features_names: list[str] | None = None, 
        fillna: str | float | None = None,
        kwargs_stats: dict[str, dict[str, object]] | None = {'ewm': {'alpha': 0.3}}
    ) -> None:
        
        self._validate_params(
            stats,
            window_sizes,
            min_periods,
            features_names,
            fillna,
            kwargs_stats
        )

        if isinstance(stats, str):
            stats = [stats]
        self.stats = stats
        self.n_stats = len(stats)

        if isinstance(window_sizes, int):
            window_sizes = [window_sizes] * self.n_stats
        self.window_sizes = window_sizes
        self.max_window_size = max(window_sizes)
        
        if min_periods is None:
            min_periods = self.window_sizes
        elif isinstance(min_periods, int):
            min_periods = [min_periods] * self.n_stats
        self.min_periods = min_periods

        if features_names is None:
            features_names = []
            for stat, window_size in zip(self.stats, self.window_sizes):
                if stat not in kwargs_stats:
                    features_names.append(f"roll_{stat}_{window_size}")
                else:
                    kwargs_sufix = "_".join([f"{k}_{v}" for k, v in kwargs_stats[stat].items()])
                    features_names.append(f"roll_{stat}_{window_size}_{kwargs_sufix}")
        self.features_names = features_names

        self.fillna = fillna
        self.kwargs_stats = kwargs_stats if kwargs_stats is not None else {}

        window_params_list = []
        for i in range(len(self.stats)):
            window_params = (self.window_sizes[i], self.min_periods[i])
            window_params_list.append(window_params)

        # Find unique window parameter combinations
        unique_rolling_windows = {}
        for i, params in enumerate(window_params_list):
            key = f"{params[0]}_{params[1]}"
            if key not in unique_rolling_windows:
                unique_rolling_windows[key] = {
                    'params': {
                        'window': params[0], 
                        'min_periods': params[1], 
                        'center': False,
                        'closed': 'left'
                    },
                    'stats_idx': [], 
                    'stats_names': [], 
                    'rolling_obj': None
                }
            unique_rolling_windows[key]['stats_idx'].append(i)
            unique_rolling_windows[key]['stats_names'].append(self.features_names[i])

        self.unique_rolling_windows = unique_rolling_windows

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"RollingFeatures(\n"
            f"    stats           = {self.stats},\n"
            f"    window_sizes    = {self.window_sizes},\n"
            f"    Max window size = {self.max_window_size},\n"
            f"    min_periods     = {self.min_periods},\n"
            f"    features_names  = {self.features_names},\n"
            f"    fillna          = {self.fillna}\n"
            f"    kwargs_stats    = {self.kwargs_stats},\n"
            f")"
        )

    def _validate_params(
        self, 
        stats: str | list[str], 
        window_sizes: int | list[int],
        min_periods: int | list[int] | None = None,
        features_names: list[str] | None = None, 
        fillna: str | float | None = None,
        kwargs_stats: dict[str, dict[str, object]] | None = None
    ) -> None:
        """
        Validate the parameters of the RollingFeatures class.

        Parameters
        ----------
        stats : str, list
            Statistics to compute over the rolling window. Can be a `string` or a `list`,
            and can have repeats. Available statistics are: 'mean', 'std', 'min', 'max',
            'sum', 'median', 'ratio_min_max', 'coef_variation', 'ewm'.
        window_sizes : int, list
            Size of the rolling window for each statistic. If an `int`, all stats share 
            the same window size. If a `list`, it should have the same length as stats.
        min_periods : int, list, default None
            Minimum number of observations in window required to have a value. 
            Same as the `min_periods` argument of pandas rolling. If `None`, 
            defaults to `window_sizes`.
        features_names : list, default None
            Names of the output features. If `None`, default names will be used in the 
            format 'roll_stat_window_size', for example 'roll_mean_7'.
        fillna : str, float, default None
            Fill missing values in `transform_batch` method. Available 
            methods are: 'mean', 'median', 'ffill', 'bfill', or a float value.
        kwargs_stats : dict, default None
            Dictionary with additional arguments for the statistics. The keys are the
            statistic names and the values are dictionaries with the arguments for the
            corresponding statistic. For example, {'ewm': {'alpha': 0.3}}.

        Returns
        -------
        None

        """

        # stats
        allowed_stats = [
            'mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 
            'coef_variation', 'ewm'
        ]

        if not isinstance(stats, (str, list)):
            raise TypeError(
                f"`stats` must be a string or a list of strings. Got {type(stats)}."
            )        
        if isinstance(stats, str):
            stats = [stats]

        for stat in set(stats):
            if stat not in allowed_stats:
                raise ValueError(
                    f"Statistic '{stat}' is not allowed. Allowed stats are: {allowed_stats}."
                )
        n_stats = len(stats)
        
        # window_sizes
        if not isinstance(window_sizes, (int, list)):
            raise TypeError(
                f"`window_sizes` must be an int or a list of ints. Got {type(window_sizes)}."
            )
        
        if isinstance(window_sizes, list):
            n_window_sizes = len(window_sizes)
            if n_window_sizes != n_stats:
                raise ValueError(
                    f"Length of `window_sizes` list ({n_window_sizes}) "
                    f"must match length of `stats` list ({n_stats})."
                )
            
        # Check duplicates (stats, window_sizes)
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes] * n_stats
        if len(set(zip(stats, window_sizes))) != n_stats:
            raise ValueError(
                f"Duplicate (stat, window_size) pairs are not allowed.\n"
                f"    `stats`       : {stats}\n"
                f"    `window_sizes : {window_sizes}"
            )
        
        # min_periods
        if not isinstance(min_periods, (int, list, type(None))):
            raise TypeError(
                f"`min_periods` must be an int, list of ints, or None. Got {type(min_periods)}."
            )
        
        if min_periods is not None:
            if isinstance(min_periods, int):
                min_periods = [min_periods] * n_stats
            elif isinstance(min_periods, list):
                n_min_periods = len(min_periods)
                if n_min_periods != n_stats:
                    raise ValueError(
                        f"Length of `min_periods` list ({n_min_periods}) "
                        f"must match length of `stats` list ({n_stats})."
                    )
            
            for i, min_period in enumerate(min_periods):
                if min_period > window_sizes[i]:
                    raise ValueError(
                        "Each `min_period` must be less than or equal to its "
                        "corresponding `window_size`."
                    )
        
        # features_names
        if not isinstance(features_names, (list, type(None))):
            raise TypeError(
                f"`features_names` must be a list of strings or None. Got {type(features_names)}."
            )
        
        if isinstance(features_names, list):
            n_features_names = len(features_names)
            if n_features_names != n_stats:
                raise ValueError(
                    f"Length of `features_names` list ({n_features_names}) "
                    f"must match length of `stats` list ({n_stats})."
                )
        
        # fillna
        if fillna is not None:
            if not isinstance(fillna, (int, float, str)):
                raise TypeError(
                    f"`fillna` must be a float, string, or None. Got {type(fillna)}."
                )
            
            if isinstance(fillna, str):
                allowed_fill_strategy = ['mean', 'median', 'ffill', 'bfill']
                if fillna not in allowed_fill_strategy:
                    raise ValueError(
                        f"'{fillna}' is not allowed. Allowed `fillna` "
                        f"values are: {allowed_fill_strategy} or a float value."
                    )
        
        # kwargs_stats
        allowed_kwargs_stats = ['ewm']
        if kwargs_stats is not None:
            if not isinstance(kwargs_stats, dict):
                raise TypeError(
                    f"`kwargs_stats` must be a dictionary or None. Got {type(kwargs_stats)}."
                )
            
            for stat in kwargs_stats.keys():
                if stat not in allowed_kwargs_stats:
                    raise ValueError(
                        f"Invalid statistic '{stat}' found in `kwargs_stats`. "
                        f"Allowed statistics with additional arguments are: "
                        f"{allowed_kwargs_stats}. Please ensure all keys in "
                        f"`kwargs_stats` are among the allowed statistics."
                    )

    def _apply_stat_pandas(
        self, 
        rolling_obj: pd.core.window.rolling.Rolling, 
        stat: str
    ) -> pd.Series:
        """
        Apply the specified statistic to a pandas rolling object.

        Parameters
        ----------
        rolling_obj : pandas Rolling
            Rolling object to apply the statistic.
        stat : str
            Statistic to compute.
        
        Returns
        -------
        stat_series : pandas Series
            Series with the computed statistic.
        
        """

        if stat == 'mean':
            return rolling_obj.mean()
        elif stat == 'std':
            return rolling_obj.std()
        elif stat == 'min':
            return rolling_obj.min()
        elif stat == 'max':
            return rolling_obj.max()
        elif stat == 'sum':
            return rolling_obj.sum()
        elif stat == 'median':
            return rolling_obj.median()
        elif stat == 'ratio_min_max':
            return rolling_obj.min() / rolling_obj.max()
        elif stat == 'coef_variation':
            return rolling_obj.std() / rolling_obj.mean()
        elif stat == 'ewm':
            kwargs = self.kwargs_stats.get(stat, {})
            return rolling_obj.apply(lambda x: _ewm_jit(x.to_numpy(), **kwargs))
        else:
            raise ValueError(f"Statistic '{stat}' is not implemented.")

    def transform_batch(
        self, 
        X: pd.Series
    ) -> pd.DataFrame:
        """
        Transform an entire pandas Series using rolling windows and compute the 
        specified statistics.

        Parameters
        ----------
        X : pandas Series
            The input data series to transform.

        Returns
        -------
        rolling_features : pandas DataFrame
            A DataFrame containing the rolling features.
        
        """

        for k in self.unique_rolling_windows.keys():
            rolling_obj = X.rolling(**self.unique_rolling_windows[k]['params'])
            self.unique_rolling_windows[k]['rolling_obj'] = rolling_obj
        
        rolling_features = []
        for i, stat in enumerate(self.stats):
            window_size = self.window_sizes[i]
            min_periods = self.min_periods[i]

            key = f"{window_size}_{min_periods}"
            rolling_obj = self.unique_rolling_windows[key]['rolling_obj']

            stat_series = self._apply_stat_pandas(rolling_obj=rolling_obj, stat=stat)            
            rolling_features.append(stat_series)

        rolling_features = pd.concat(rolling_features, axis=1)
        rolling_features.columns = self.features_names
        rolling_features = rolling_features.iloc[self.max_window_size:]

        if self.fillna is not None:
            if self.fillna == 'mean':
                rolling_features = rolling_features.fillna(rolling_features.mean())
            elif self.fillna == 'median':
                rolling_features = rolling_features.fillna(rolling_features.median())
            elif self.fillna == 'ffill':
                rolling_features = rolling_features.ffill()
            elif self.fillna == 'bfill':
                rolling_features = rolling_features.bfill()
            else:
                rolling_features = rolling_features.fillna(self.fillna)
        
        return rolling_features

    def _apply_stat_numpy_jit(
        self, 
        X_window: np.ndarray, 
        stat: str
    ) -> float:
        """
        Apply the specified statistic to a numpy array using Numba JIT.

        Parameters
        ----------
        X_window : numpy array
            Array with the rolling window.
        stat : str
            Statistic to compute.

        Returns
        -------
        stat_value : float
            Value of the computed statistic.
        
        """
        
        if stat == 'mean':
            return _np_mean_jit(X_window)
        elif stat == 'std':
            return _np_std_jit(X_window)
        elif stat == 'min':
            return _np_min_jit(X_window)
        elif stat == 'max':
            return _np_max_jit(X_window)
        elif stat == 'sum':
            return _np_sum_jit(X_window)
        elif stat == 'median':
            return _np_median_jit(X_window)
        elif stat == 'ratio_min_max':
            return _np_min_max_ratio_jit(X_window)
        elif stat == 'coef_variation':
            return _np_cv_jit(X_window)
        elif stat == 'ewm':
            kwargs = self.kwargs_stats.get(stat, {})
            return _ewm_jit(X_window, **kwargs)
        else:
            raise ValueError(f"Statistic '{stat}' is not implemented.")

    def transform(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Transform a numpy array using rolling windows and compute the 
        specified statistics. The returned array will have the shape 
        (X.shape[1] if exists, n_stats). For example, if X is a flat
        array, the output will have shape (n_stats,). If X is a 2D array,
        the output will have shape (X.shape[1], n_stats).

        Parameters
        ----------
        X : numpy ndarray
            The input data array to transform.

        Returns
        -------
        rolling_features : numpy ndarray
            An array containing the computed statistics.
        
        """

        array_ndim = X.ndim
        if array_ndim == 1:
            X = X[:, np.newaxis]
            
        rolling_features = np.full(
            shape=(X.shape[1], self.n_stats), fill_value=np.nan, dtype=float
        )

        for i in range(X.shape[1]):
            for j, stat in enumerate(self.stats):
                X_window = X[-self.window_sizes[j]:, i]
                X_window = X_window[~np.isnan(X_window)]
                if len(X_window) > 0: 
                    rolling_features[i, j] = self._apply_stat_numpy_jit(X_window, stat)
                else:
                    rolling_features[i, j] = np.nan

        if array_ndim == 1:
            rolling_features = rolling_features.ravel()
        
        return rolling_features
    

class QuantileBinner:
    """
    QuantileBinner class to bin data into quantile-based bins using `numpy.percentile`.
    This class is similar to `KBinsDiscretizer` but faster for binning data into
    quantile-based bins. Bin  intervals are defined following the convention:
    bins[i-1] <= x < bins[i]. See more information in `numpy.percentile` and
    `numpy.digitize`.
    
    Parameters
    ----------
    n_bins : int
        The number of quantile-based bins to create.
    method : str, default 'linear'
        The method used to compute the quantiles. This parameter is passed to 
        `numpy.percentile`. Default is 'linear'. Valid values are "inverse_cdf",
        "averaged_inverse_cdf", "closest_observation", "interpolated_inverse_cdf",
        "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased".
    subsample : int, default 200000
        The number of samples to use for computing quantiles. If the dataset 
        has more samples than `subsample`, a random subset will be used.
    dtype : data type, default numpy.float64
        The data type to use for the bin indices. Default is `numpy.float64`.
    random_state : int, default 789654
        The random seed to use for generating a random subset of the data.
    
    Attributes
    ----------
    n_bins : int
        The number of quantile-based bins to create.
    method : str
        The method used to compute the quantiles. This parameter is passed to 
        `numpy.percentile`. Default is 'linear'. Valid values are 'linear',
        'lower', 'higher', 'midpoint', 'nearest'.
    subsample : int
        The number of samples to use for computing quantiles. If the dataset 
        has more samples than `subsample`, a random subset will be used.
    dtype : data type
        The data type to use for the bin indices. Default is `numpy.float64`.
    random_state : int
        The random seed to use for generating a random subset of the data.
    n_bins_ : int
        The number of bins learned during fitting.
    bin_edges_ : numpy ndarray
        The edges of the bins learned during fitting.
    intervals_ : dict
        A dictionary with the bin indices as keys and the corresponding bin
        intervals as values.
    
    """

    def __init__(
        self,
        n_bins: int,
        method: str = "linear",
        subsample: int = 200000,
        dtype: type = np.float64,
        random_state: int = 789654
    ):
        
        self._validate_params(
            n_bins,
            method,
            subsample,
            dtype,
            random_state
        )

        self.n_bins       = n_bins
        self.method       = method
        self.subsample    = subsample
        self.dtype        = dtype
        self.random_state = random_state
        self.n_bins_      = None
        self.bin_edges_   = None
        self.intervals_   = None

    def _validate_params(
        self,
        n_bins: int,
        method: str,
        subsample: int,
        dtype: type,
        random_state: int
    ):
        """
        Validate the parameters passed to the class initializer.
        """
    
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError(
                f"`n_bins` must be an int greater than 1. Got {n_bins}."
            )

        valid_methods = [
            "inverse_cdf",
            "averaged_inverse_cdf",
            "closest_observation",
            "interpolated_inverse_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"`method` must be one of {valid_methods}. Got {method}."
            )
        if not isinstance(subsample, int) or subsample < 1:
            raise ValueError(
                f"`subsample` must be an integer greater than or equal to 1. "
                f"Got {subsample}."
            )
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError(
                f"`random_state` must be an integer greater than or equal to 0. "
                f"Got {random_state}."
            )
        if not isinstance(dtype, type):
            raise ValueError(
                f"`dtype` must be a valid numpy dtype. Got {dtype}."
            )

    def fit(self, X: np.ndarray):
        """
        Learn the bin edges based on quantiles from the training data.
        
        Parameters
        ----------
        X : numpy ndarray
            The training data used to compute the quantiles.

        Returns
        -------
        None
        
        """

        if X.size == 0:
            raise ValueError("Input data `X` cannot be empty.")
        if len(X) > self.subsample:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.integers(0, len(X), self.subsample)]

        self.bin_edges_ = np.percentile(
            a      = X,
            q      = np.linspace(0, 100, self.n_bins + 1),
            method = self.method
        )

        self.n_bins_ = len(self.bin_edges_) - 1
        self.intervals_ = {
            int(i): (float(self.bin_edges_[i]), float(self.bin_edges_[i + 1]))
            for i in range(self.n_bins_)
        }

    def transform(self, X: np.ndarray):
        """
        Assign new data to the learned bins.
        
        Parameters
        ----------
        X : numpy ndarray
            The data to assign to the bins.
        
        Returns
        -------
        bin_indices : numpy ndarray 
            The indices of the bins each value belongs to.
            Values less than the smallest bin edge are assigned to the first bin,
            and values greater than the largest bin edge are assigned to the last bin.
       
        """

        if self.bin_edges_ is None:
            raise NotFittedError(
                "The model has not been fitted yet. Call 'fit' with training data first."
            )

        bin_indices = np.digitize(X, bins=self.bin_edges_, right=False)
        bin_indices = np.clip(bin_indices, 1, self.n_bins_).astype(self.dtype) - 1

        return bin_indices

    def fit_transform(self, X):
        """
        Fit the model to the data and return the bin indices for the same data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit and transform.
        
        Returns
        -------
        bin_indices : numpy.ndarray
            The indices of the bins each value belongs to.
            Values less than the smallest bin edge are assigned to the first bin,
            and values greater than the largest bin edge are assigned to the last bin.
        
        """

        self.fit(X)

        return self.transform(X)

    def get_params(self):
        """
        Get the parameters of the quantile binner.
        
        Parameters
        ----------
        self
        
        Returns
        -------
        params : dict
            A dictionary of the parameters of the quantile binner.
        
        """

        return {
            "n_bins": self.n_bins,
            "method": self.method,
            "subsample": self.subsample,
            "dtype": self.dtype,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """
        Set the parameters of the QuantileBinner.
        
        Parameters
        ----------
        params : dict
            A dictionary of the parameters to set.

        Returns
        -------
        None
        
        """

        for param, value in params.items():
            setattr(self, param, value)


class ConformalIntervalCalibrator:
    """
    Transformer that calibrates the prediction interval to achieve the desired 
    coverage based on conformity scores. It uses the conformal split method.

    Parameters
    ----------
    nominal_coverage : float, default 0.8
        Desired coverage. This is the desired probability that the true value 
        falls within the calibrated interval.

    Attributes
    ----------
    nominal_coverage : float
        Desired coverage. This is the desired probability that the true value 
        falls within the calibrated interval.
    correction_factor_ : dict
        Correction factor to achieve the desired coverage.
    
    """

    def __init__(self, nominal_coverage: float = 0.8):
        self.nominal_coverage = nominal_coverage
        self.correction_factor_ = {}
        self.fit_input_type_ = None
        self.fit_series_names_ = None

    def fit(
        self,
        y_true: np.ndarray | pd.Series | dict[np.ndarray | pd.Series],
        y_pred_interval: np.ndarray | pd.DataFrame | dict[np.ndarray | pd.DataFrame]
    ):
        """
        Learn the correction factor needed to achieve the desired coverage.

        Parameters
        ----------
        y_true : numpy ndarray, pandas Series, dict
            True target values. When only one series is available, it can be a
            numpy array or pandas Series. When multiple series are available, it
            must be a dictionary with series names as keys.
        y_pred_interval : numpy ndarray, pandas DataFrame, dict
            Prediction interval calculated on the true target values. It must have
            two columns: lower bound and upper bound. If multiple series are available,
            it must be a dictionary with series names as keys.

        Returns
        -------
        self

        """
        if not isinstance(y_true, (np.ndarray, pd.Series, dict)):
            raise ValueError(
                "`y_true` must be a numpy array, pandas Series, or a dictionary."
            )
        
        if not isinstance(y_pred_interval, (np.ndarray, pd.DataFrame, dict)):
            raise ValueError(
                "`y_pred_interval` must be a numpy array, pandas DataFrame, or a dictionary."
            )

        if (isinstance(y_true, dict) and not isinstance(y_pred_interval, dict)) or  (not isinstance(y_true, dict) and isinstance(y_pred_interval, dict)):
            raise ValueError(
                "y_true and y_pred_interval must be both dictionaries or both single series."
            )

        if isinstance(y_true, dict) != isinstance(y_pred_interval, dict):
            raise ValueError(
                    "`y_true` and `y_pred_interval` must both be dictionaries if one of them is."
                )
        
        if not isinstance(y_true, dict):
            self.fit_input_type_ = "single"    
            y_true = {'y': y_true}
            y_pred_interval = {'y': y_pred_interval}
        else:
            self.fit_input_type_ = "multi"

        for k in y_true.keys():
            y_true_ = deepcopy(y_true[k])
            y_pred_interval_ = deepcopy(y_pred_interval[k])

            if not isinstance(y_true_, (np.ndarray, pd.Series)):
                raise ValueError(
                    f"All values in `y_true` must be numpy arrays or pandas Series. "
                    f"Got {type(y_true_)}."
                )
            if not isinstance(y_pred_interval_, (np.ndarray, pd.DataFrame)):
                raise ValueError(
                    f"All values in `y_pred_interval` must be numpy arrays or pandas DataFrames. "
                    f"Got {type(y_pred_interval_)}."
                )
            
            y_true_ = np.asarray(y_true_)
            y_pred_interval_ = np.asarray(y_pred_interval_)
            lower_bound = y_pred_interval_[:, 0]
            upper_bound = y_pred_interval_[:, 1]
            conformity_scores = np.max(
                [
                    lower_bound - y_true_,
                    y_true_ - upper_bound,
                ],
                axis=0,
            )

            self.correction_factor_[k] = np.quantile(conformity_scores, self.nominal_coverage)

        self.fit_series_names_ = list(y_true.keys())

        return self

    def transform(
        self, 
        y_pred_interval: np.ndarray | pd.DataFrame | dict[np.ndarray | pd.DataFrame]
    ) -> np.ndarray | pd.DataFrame | dict[np.ndarray | pd.DataFrame]:
        """
        Apply the correction factor to the prediction interval to achieve the desired
        coverage.

        Parameters
        ----------
        y_pred_interval : numpy ndarray, pandas DataFrame, dict
            Prediction interval to be calibrated using conformal method. It must have
            two columns: lower bound and upper bound. If multiple series are available,
            it must be a dictionary with series names as keys.

        Returns
        -------
        y_pred_interval_conformal : pandas DataFrame, numpy ndarray, dict
            Prediction interval with the correction factor applied.
        
        """

        if not isinstance(y_pred_interval, (np.ndarray, pd.DataFrame, dict)):
            raise ValueError(
                "`y_pred_interval` must be a numpy array, pandas DataFrame, or a dictionary."
            )
        
        if isinstance(y_pred_interval, dict) and self.fit_input_type_ != "multi":
            raise ValueError(
                f"The transformer was fitted with a single series. `y_pred_interval` "
                f"must be a pandas DataFrame or numpy array with the lower and upper bounds "
                f"of the prediction interval. Got a {type(y_pred_interval)}."
            )
        
        if isinstance(y_pred_interval, (pd.DataFrame, np.ndarray)) and self.fit_input_type_ != "single":
            raise ValueError(
                f"The transformer was fitted with multiple series. `y_pred_interval` "
                f"must be a dictionary with series names as keys. Got a {type(y_pred_interval)}."
            )
        
        if not isinstance(y_pred_interval, dict):
            y_pred_interval = {'y': y_pred_interval}

        conformalized_intervals = {}

        for k in y_pred_interval.keys():

            if k not in self.fit_series_names_:
                raise ValueError(
                    f"Series '{k}' was not seen during fit. Available series are: "
                    f"{self.fit_series_names_}."
                )
            correction_factor = self.correction_factor_[k]
            
            y_pred_interval_ = deepcopy(y_pred_interval[k])

            if not isinstance(y_pred_interval_, (np.ndarray, pd.DataFrame)):
                raise ValueError(
                    f"All values in `y_pred_interval` must be numpy arrays or pandas DataFrames. "
                    f"Got {type(y_pred_interval_)}."
                )
            
            if y_pred_interval_.shape[1] != 2:
                raise ValueError(
                    "All values in `y_pred_interval` must have 2 columns that represent the "
                    "lower and upper bounds of the prediction interval."
                )
            
            columns = ["lower_bound", "upper_bound"]
            index = np.arange(y_pred_interval_.shape[0])
            if isinstance(y_pred_interval_, pd.DataFrame):
                columns = y_pred_interval_.columns
                index = y_pred_interval_.index
                y_pred_interval_ = y_pred_interval_.to_numpy()

            y_pred_interval_conformal = y_pred_interval_.copy()
            y_pred_interval_conformal[:, 0] = (
                y_pred_interval_conformal[:, 0] - correction_factor
            )
            y_pred_interval_conformal[:, 1] = (
                y_pred_interval_conformal[:, 1] + correction_factor
            )

            # If upper bound is less than lower bound, swap them
            mask = (
                y_pred_interval_conformal[:, 1]
                < y_pred_interval_conformal[:, 0]
            )
            (
                y_pred_interval_conformal[mask, 0],
                y_pred_interval_conformal[mask, 1],
            ) = (
                y_pred_interval_conformal[mask, 1],
                y_pred_interval_conformal[mask, 0],
            )

            y_pred_interval_conformal = pd.DataFrame(
                data    = y_pred_interval_conformal,
                columns = columns,
                index   = index,
            )

            conformalized_intervals[k] = y_pred_interval_conformal
        
        if self.fit_input_type_ == "single":
            conformalized_intervals = conformalized_intervals['y']

        return conformalized_intervals
