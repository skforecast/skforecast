################################################################################
#                            Seasonal strength                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
# Seasonal strength measures for time series analysis.
# This module implements seasonal strength heuristics based on STL-like
# decomposition, following Wang, Smith & Hyndman (2006).
# References
# ----------
# - Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based
#   clustering for time series data. Data Mining and Knowledge Discovery,
#   13(3), 335-364.

import numpy as np
from numba import njit


@njit(cache=True)
def _moving_average_jit(x: np.ndarray, window: int) -> np.ndarray:  # pragma: no cover
    """
    JIT-compiled centered moving average.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window : int
        Window size for moving average.

    Returns
    -------
    np.ndarray
        Smoothed array with centered moving average.
    """
    n = len(x)
    result = np.zeros(n)
    half_window = window // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        total = 0.0
        count = 0
        for j in range(start, end):
            total += x[j]
            count += 1
        result[i] = total / count

    return result


@njit(cache=True)
def _seasonal_component_jit(detrended: np.ndarray, period: int) -> np.ndarray:  # pragma: no cover
    """
    JIT-compiled seasonal component extraction.

    Parameters
    ----------
    detrended : np.ndarray
        Detrended time series.
    period : int
        Seasonal period.

    Returns
    -------
    np.ndarray
        Seasonal component array.
    """
    n = len(detrended)
    seasonal = np.zeros(n)

    for i in range(period):
        total = 0.0
        count = 0
        j = i
        while j < n:
            total += detrended[j]
            count += 1
            j += period
        mean_val = total / count if count > 0 else 0.0
        j = i
        while j < n:
            seasonal[j] = mean_val
            j += period

    return seasonal


@njit(cache=True)
def _seas_heuristic_jit(x: np.ndarray, period: int) -> float:  # pragma: no cover
    """
    JIT-compiled seasonal strength computation.

    Parameters
    ----------
    x : np.ndarray
        Time series data.
    period : int
        Seasonal period.

    Returns
    -------
    float
        Seasonal strength measure Fs in [0, 1].
    """
    n = len(x)
    if n < 2 * period:
        return 0.0

    x_valid = x.copy()
    total = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(x_valid[i]):
            total += x_valid[i]
            count += 1
    mean_val = total / count if count > 0 else 0.0
    for i in range(n):
        if np.isnan(x_valid[i]):
            x_valid[i] = mean_val

    trend = _moving_average_jit(x_valid, period)
    detrended = x_valid - trend
    seasonal = _seasonal_component_jit(detrended, period)
    remainder = detrended - seasonal

    var_remainder = np.var(remainder)
    var_seasonal_remainder = np.var(seasonal + remainder)

    if var_seasonal_remainder < 1e-10:
        return 0.0

    Fs = 1.0 - var_remainder / var_seasonal_remainder
    if Fs < 0.0:
        Fs = 0.0
    return Fs


def seas_heuristic(x: np.ndarray, period: int) -> float:
    """
    Compute seasonal strength measure (Wang, Smith & Hyndman, 2006).

    Uses STL-like decomposition to measure how strong the seasonal
    component is relative to the remainder. This is the main entry point
    for the seasonal strength heuristic.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    period : int
        Seasonal period.

    Returns
    -------
    float
        Seasonal strength in [0, 1]. Values > 0.64 suggest seasonal differencing.

    Examples
    --------
    >>> import numpy as np
    >>> # Create seasonal data
    >>> t = np.arange(100)
    >>> y = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(100)
    >>> strength = seas_heuristic(y, period=12)
    >>> print(f"Seasonal strength: {strength:.3f}")

    References
    ----------
    Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based
    clustering for time series data. Data Mining and Knowledge Discovery,
    13(3), 335-364.
    """
    return _seas_heuristic_jit(x, period)
