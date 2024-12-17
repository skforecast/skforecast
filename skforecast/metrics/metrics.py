################################################################################
#                                metrics                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Callable
import numpy as np
import pandas as pd
import inspect
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    median_absolute_error,
)
from statsmodels.tsa.stattools import acf, pacf


def _get_metric(metric: str) -> Callable:
    """
    Get the corresponding scikit-learn function to calculate the metric.

    Parameters
    ----------
    metric : str
        Metric used to quantify the goodness of fit of the model.

    Returns
    -------
    metric : Callable
        scikit-learn function to calculate the desired metric.

    """

    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
        "median_absolute_error",
    ]

    if metric not in allowed_metrics:
        raise ValueError((f"Allowed metrics are: {allowed_metrics}. Got {metric}."))

    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_absolute_scaled_error": mean_absolute_scaled_error,
        "root_mean_squared_scaled_error": root_mean_squared_scaled_error,
        "median_absolute_error": median_absolute_error,
    }

    metric = add_y_train_argument(metrics[metric])

    return metric


def add_y_train_argument(func: Callable) -> Callable:
    """
    Add `y_train` argument to a function if it is not already present.

    Parameters
    ----------
    func : callable
        Function to which the argument is added.

    Returns
    -------
    wrapper : callable
        Function with `y_train` argument added.
    
    """

    sig = inspect.signature(func)
    
    if "y_train" in sig.parameters:
        return func

    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("y_train", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @wraps(func)
    def wrapper(*args, y_train=None, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__signature__ = new_sig
    
    return wrapper


def mean_absolute_scaled_error(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_train: Union[list, pd.Series, np.ndarray],
) -> float:
    """
    Mean Absolute Scaled Error (MASE)

    MASE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the mean absolute error of the forecast divided by the
    mean absolute error of a naive forecast in the training set. The naive
    forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Parameters
    ----------
    y_true : pandas Series, numpy ndarray
        True values of the target variable.
    y_pred : pandas Series, numpy ndarray
        Predicted values of the target variable.
    y_train : list, pandas Series, numpy ndarray
        True values of the target variable in the training set. If `list`, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    mase : float
        MASE value.
    
    """

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("`y_true` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("`y_pred` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("`y_train` must be a list, pandas Series or numpy ndarray.")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    ("When `y_train` is a list, each element must be a pandas Series "
                     "or numpy ndarray.")
                )
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("`y_true` and `y_pred` must have at least one element.")

    if isinstance(y_train, list):
        naive_forecast = np.concatenate([np.diff(x) for x in y_train])
    else:
        naive_forecast = np.diff(y_train)

    mase = np.mean(np.abs(y_true - y_pred)) / np.nanmean(np.abs(naive_forecast))

    return mase


def root_mean_squared_scaled_error(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_train: Union[list, pd.Series, np.ndarray],
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE)

    RMSSE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the root mean squared error of the forecast divided by
    the root mean squared error of a naive forecast in the training set. The
    naive forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Parameters
    ----------
    y_true : pandas Series, numpy ndarray
        True values of the target variable.
    y_pred : pandas Series, numpy ndarray
        Predicted values of the target variable.
    y_train : list, pandas Series, numpy ndarray
        True values of the target variable in the training set. If list, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    rmsse : float
        RMSSE value.
    
    """

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("`y_true` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("`y_pred` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("`y_train` must be a list, pandas Series or numpy ndarray.")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    ("When `y_train` is a list, each element must be a pandas Series "
                     "or numpy ndarray.")
                )
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("`y_true` and `y_pred` must have at least one element.")

    if isinstance(y_train, list):
        naive_forecast = np.concatenate([np.diff(x) for x in y_train])
    else:
        naive_forecast = np.diff(y_train)
    
    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.nanmean(naive_forecast ** 2))
    
    return rmsse


def calculate_autocorrelation(
    data: pd.Series,
    n_lags: int = 50,
    sort_by: str = "partial_autocorrelation_abs",
    acf_kwargs: dict = {},
    pacf_kwargs: dict = {},
) -> pd.DataFrame:
    """
    Calculate autocorrelation and partial autocorrelation for a time series.
    This is a wrapper around statsmodels.tsa.stattools.acf and statsmodels.tsa.stattools.pacf.

    Parameters
    ----------
    data : pandas Series
        Time series to calculate autocorrelation.
    n_lags : int
        Number of lags to calculate autocorrelation.
    sort_by : str, default 'partial_autocorrelation_abs'
        Sort results by `lag', 'partial_autocorrelation_abs', 'partial_autocorrelation',
        'autocorrelation_abs' or 'autocorrelation'.
    acf_kwargs : dict, default {}
        Optional arguments to pass to statsmodels.tsa.stattools.acf.
    pacf_kwargs : dict, default {}
        Optional arguments to pass to statsmodels.tsa.stattools.pacf.

    Returns
    -------
    pd.DataFrame
        Autocorrelation and partial autocorrelation values.

    Examples
    --------
    >>> from skforecast.metrics import calculate_autocorrelation
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> calculate_autocorrelation(data = data, n_lags = 4)
    
    """

    if sort_by not in [
        "lag",
        "partial_autocorrelation_abs",
        "partial_autocorrelation",
        "autocorrelation_abs",
        "autocorrelation",
    ]:
        raise ValueError(
            "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
            "'autocorrelation_abs' or 'autocorrelation'."
        )

    if not isinstance(data, pd.Series):
        raise ValueError("`data` must be a pandas Series.")

    pacf_values = pacf(data, nlags=n_lags, **pacf_kwargs)
    acf_values = acf(data, nlags=n_lags, **acf_kwargs)

    results = pd.DataFrame(
        {
            "lag": range(n_lags + 1),
            "partial_autocorrelation_abs": np.abs(pacf_values),
            "partial_autocorrelation": pacf_values,
            "autocorrelation_abs": np.abs(acf_values),
            "autocorrelation": acf_values,
        }
    ).iloc[1:]

    if sort_by == "lag":
        results = results.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return results