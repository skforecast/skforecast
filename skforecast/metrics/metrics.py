################################################################################
#                                metrics                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Callable
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
    mean_pinball_loss
)


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
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_train: list[float] | pd.Series | np.ndarray,
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
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_train: list[float] | pd.Series | np.ndarray,
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


def crps_from_predictions(y_true: float, y_pred: np.ndarray) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a set of
    forecast realizations, for example from bootstrapping. The CRPS compares
    the empirical distribution of a set of forecasted values to a scalar
    observation. The smaller the CRPS, the better.

    Parameters
    ----------
    y_true : float
        The true value of the random variable.
    y_pred : np.ndarray
        The predicted values of the random variable. These are the multiple
        forecasted values for a single observation.

    Returns
    -------
    crps : float
        The CRPS score.
        
    """
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise TypeError("`y_pred` must be a 1D numpy array.")
    
    if not isinstance(y_true, (float, int)):
        raise TypeError("`y_true` must be a float or integer.")

    y_pred = np.sort(y_pred)
    # Define the grid for integration including the true value
    grid = np.concatenate(([y_true], y_pred))
    grid = np.sort(grid)
    cdf_values = np.searchsorted(y_pred, grid, side='right') / len(y_pred)
    indicator = grid >= y_true
    diffs = np.diff(grid)
    crps = np.sum(diffs * (cdf_values[:-1] - indicator[:-1])**2)

    return crps


def crps_from_quantiles(
    y_true: float,
    pred_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a given true value
    and predicted quantiles. The empirical cdf is approximated using linear interpolation
    between the predicted quantiles.

    Parameters
    ----------
    y_true : float
        The true value of the random variable.
    pred_quantiles : numpy ndarray
        The predicted quantile values.
    quantile_levels : numpy ndarray
        The quantile levels corresponding to the predicted quantiles.

    Returns
    -------
    crps : float
        The CRPS score.

    """
    if not isinstance(y_true, (float, int)):
        raise TypeError("`y_true` must be a float or integer.")

    if not isinstance(pred_quantiles, np.ndarray) or pred_quantiles.ndim != 1:
        raise TypeError("`pred_quantiles` must be a 1D numpy array.")
    
    if not isinstance(quantile_levels, np.ndarray) or quantile_levels.ndim != 1:
        raise TypeError("`quantile_levels` must be a 1D numpy array.")
    
    if len(pred_quantiles) != len(quantile_levels):
        raise ValueError(
            "The number of predicted quantiles and quantile levels must be equal."
        )

    sorted_indices = np.argsort(pred_quantiles)
    pred_quantiles = pred_quantiles[sorted_indices]
    quantile_levels = quantile_levels[sorted_indices]

    # Define the empirical CDF function using interpolation
    def empirical_cdf(x):
        return np.interp(x, pred_quantiles, quantile_levels, left=0.0, right=1.0)

    # Define the CRPS integrand
    def crps_integrand(x):
        return (empirical_cdf(x) - (x >= y_true)) ** 2

    # Integration bounds: Extend slightly beyond predicted quantiles
    xmin = np.min(pred_quantiles) * 0.9
    xmax = np.max(pred_quantiles) * 1.1

    # Create a fine grid of x values for integration
    x_values = np.linspace(xmin, xmax, 1000)

    # Compute the integrand values and integrate using the trapezoidal rule
    integrand_values = crps_integrand(x_values)
    crps = np.trapz(integrand_values, x_values)

    return crps


def calculate_coverage(
    y_true: np.ndarray | pd.Series,
    lower_bound: np.ndarray | pd.Series,
    upper_bound: np.ndarray | pd.Series,
) -> float:
    """
    Calculate coverage of a given interval as the proportion of true values
    that fall within the interval.

    Parameters
    ----------
    y_true : numpy ndarray, pandas Series
        True values of the target variable.
    lower_bound : numpy ndarray, pandas Series
        Lower bound of the interval.
    upper_bound : numpy ndarray, pandas Series
        Upper bound of the interval.

    Returns
    -------
    coverage : float
        Coverage of the interval.

    """
    if not isinstance(y_true, (np.ndarray, pd.Series)) or y_true.ndim != 1:
        raise TypeError("`y_true` must be a 1D numpy array or pandas Series.")

    if not isinstance(lower_bound, (np.ndarray, pd.Series)) or lower_bound.ndim != 1:
        raise TypeError("`lower_bound` must be a 1D numpy array or pandas Series.")

    if not isinstance(upper_bound, (np.ndarray, pd.Series)) or upper_bound.ndim != 1:
        raise TypeError("`upper_bound` must be a 1D numpy array or pandas Series.")

    y_true = np.asarray(y_true)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    if y_true.shape != lower_bound.shape or y_true.shape != upper_bound.shape:
        raise TypeError(
            "`y_true`, `lower_bound` and `upper_bound` must have the same shape."
        )

    coverage = np.mean(np.logical_and(y_true >= lower_bound, y_true <= upper_bound))

    return coverage


def create_mean_pinball_loss(alpha: float) -> callable:
    """
    Create pinball loss, also known as quantile loss, for a given quantile.
    Internally, it uses the `mean_pinball_loss` function from scikit-learn.

    Parameters
    ----------
    alpha: float
        Quantile for which the Pinball loss is calculated. Must be between 0 and 1, inclusive.

    Returns
    -------
    mean_pinball_loss_q: callable
        Mean Pinball loss for the given quantile.

    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1, both inclusive.")

    def mean_pinball_loss_q(y_true, y_pred):
        return mean_pinball_loss(y_true, y_pred, alpha=alpha)
    return mean_pinball_loss_q