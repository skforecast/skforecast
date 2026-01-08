"""
Differencing utilities for time series stationarity.

This module provides functions to determine the number of differences
(regular and seasonal) required to achieve stationarity.

References
----------
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting:
  the forecast package for R. Journal of Statistical Software, 27(3), 1-22.
- Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing
  the null hypothesis of stationarity against the alternative of a unit root.
  Journal of Econometrics, 54(1-3), 159-178.
"""

import numpy as np
import math
import warnings
from numba import njit
from statsmodels.tsa.stattools import adfuller, kpss

from ._seasonal_strength import seas_heuristic


@njit(cache=True)
def _is_constant_jit(x: np.ndarray, tol: float) -> bool:
    """
    JIT-compiled constant check for arrays without NaN.

    Parameters
    ----------
    x : np.ndarray
        Input array (must not contain NaN).
    tol : float
        Tolerance for equality check.

    Returns
    -------
    bool
        True if all values are within tolerance of first value.
    """
    if len(x) == 0:
        return True
    first_val = x[0]
    for i in range(1, len(x)):
        if np.abs(x[i] - first_val) >= tol:
            return False
    return True


def is_constant(x: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if series is constant (all values equal within tolerance).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    tol : float
        Tolerance for equality check.

    Returns
    -------
    bool
        True if constant.

    Examples
    --------
    >>> import numpy as np
    >>> is_constant(np.array([1.0, 1.0, 1.0]))
    True
    >>> is_constant(np.array([1.0, 2.0, 3.0]))
    False
    """
    x_valid = x[~np.isnan(x)]
    if len(x_valid) == 0:
        return True
    return _is_constant_jit(x_valid, tol)


def diff(x: np.ndarray, lag: int = 1, differences: int = 1) -> np.ndarray:
    """
    Compute lagged differences of a time series.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    lag : int
        Lag for differencing (default 1).
    differences : int
        Number of times to difference (default 1).

    Returns
    -------
    np.ndarray
        Differenced array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    >>> diff(x, lag=1, differences=1)
    array([2., 3., 4., 5.])
    """
    result = x.copy()
    for _ in range(differences):
        result = result[lag:] - result[:-lag]
    return result


def ndiffs(
    x: np.ndarray,
    alpha: float = 0.05,
    test: str = "kpss",
    kind: str = "level",
    max_d: int = 2,
    **kwargs
) -> int:
    """
    Determine the number of differences required for stationarity.

    Uses KPSS test by default (null = stationarity). For ADF (null = unit root),
    the interpretation is inverted.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    alpha : float
        Significance level (clamped to [0.01, 0.1]).
    test : str
        Unit root test: "kpss", "adf", or "pp".
    kind : str
        Type of stationarity: "level" or "trend".
    max_d : int
        Maximum differencing order.
    **kwargs
        Additional test arguments.

    Returns
    -------
    int
        Number of differences needed for stationarity.

    Examples
    --------
    >>> import numpy as np
    >>> # Random walk (needs differencing)
    >>> y = np.cumsum(np.random.randn(100))
    >>> d = ndiffs(y)
    >>> print(f"Differences needed: {d}")

    References
    ----------
    Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
    Testing the null hypothesis of stationarity against the alternative
    of a unit root.
    """
    x = x[~np.isnan(x)]
    d = 0

    if alpha < 0.01:
        warnings.warn(
            "Specified alpha value is less than the minimum, setting alpha=0.01"
        )
        alpha = 0.01
    elif alpha > 0.1:
        warnings.warn(
            "Specified alpha value is larger than the maximum, setting alpha=0.1"
        )
        alpha = 0.1

    if is_constant(x):
        return d

    def run_kpss_test(x: np.ndarray, alpha: float) -> bool:
        """Run KPSS test. Returns True if differencing needed."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nlags = math.floor(3 * math.sqrt(len(x)) / 13)
                nlags = max(1, nlags)
                stat, pval, _, _ = kpss(x, 'c', nlags=nlags)
                return pval < alpha
        except Exception as e:
            warnings.warn(
                f"The chosen unit root test encountered an error when testing for "
                f"difference {d}.\nFrom {test}(): {e}\n"
                f"{d} differences will be used. Consider using a different unit root test."
            )
            return False

    def run_adf_test(x: np.ndarray, alpha: float) -> bool:
        """Run ADF test. Returns True if differencing needed (unit root not rejected)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = adfuller(x, autolag='AIC')
                pval = result[1]
                return pval >= alpha
        except Exception as e:
            warnings.warn(
                f"The chosen unit root test encountered an error when testing for "
                f"difference {d}.\nFrom {test}(): {e}\n"
                f"{d} differences will be used."
            )
            return False

    if test == "kpss":
        run_test = run_kpss_test
    else:
        run_test = run_adf_test

    dodiff = run_test(x, alpha)

    if math.isnan(dodiff) if isinstance(dodiff, float) else False:
        return d

    while dodiff and d < max_d:
        d += 1
        x = diff(x, 1, 1)
        if len(x) > 0:
            x = x[~np.isnan(x)]
        if len(x) < 3 or is_constant(x):
            return d
        dodiff = run_test(x, alpha)
        if math.isnan(dodiff) if isinstance(dodiff, float) else False:
            return d - 1

    return d


def nsdiffs(
    x: np.ndarray,
    period: int = 1,
    test: str = "seas",
    alpha: float = 0.05,
    max_D: int = 1,
    **kwargs
) -> int:
    """
    Determine the number of seasonal differences required.

    Uses seasonal strength heuristic by default. For seasonal strength > 0.64,
    one seasonal difference is recommended.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    period : int
        Seasonal period.
    test : str
        Seasonal test: "seas", "ocsb", "hegy", "ch".
    alpha : float
        Significance level (clamped to [0.01, 0.1]).
    max_D : int
        Maximum seasonal differencing order.
    **kwargs
        Additional test arguments.

    Returns
    -------
    int
        Number of seasonal differences needed.

    Examples
    --------
    >>> import numpy as np
    >>> # Monthly data with strong seasonality
    >>> t = np.arange(120)
    >>> y = 10 * np.sin(2 * np.pi * t / 12) + np.cumsum(np.random.randn(120))
    >>> D = nsdiffs(y, period=12)
    >>> print(f"Seasonal differences needed: {D}")

    References
    ----------
    Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based
    clustering for time series data.
    """
    D = 0

    if alpha < 0.01:
        warnings.warn(
            "Specified alpha value is less than the minimum, setting alpha=0.01"
        )
        alpha = 0.01
    elif alpha > 0.1:
        warnings.warn(
            "Specified alpha value is larger than the maximum, setting alpha=0.1"
        )
        alpha = 0.1

    if test == 'ocsb':
        warnings.warn(
            "Significance levels other than 5% are not currently supported by "
            "test='ocsb', defaulting to alpha=0.05."
        )
        alpha = 0.05

    if test in ('hegy', 'ch'):
        raise NotImplementedError(f"Test '{test}' is not yet implemented.")

    if is_constant(x):
        return D

    if period == 1:
        raise ValueError("Non-seasonal data (period=1)")
    elif period < 1:
        warnings.warn(
            "Cannot handle data with frequency less than 1. Seasonality will be ignored."
        )
        return 0

    if period >= len(x):
        return 0

    def run_seas_test(x: np.ndarray, period: int) -> bool:
        """Run seasonal strength test. Returns True if seasonal diff needed."""
        try:
            strength = seas_heuristic(x, period)
            dodiff = strength > 0.64
            if dodiff not in (True, False, 0, 1):
                raise ValueError(f"Unexpected result {dodiff} in seasonal test.")
            return bool(dodiff)
        except Exception as e:
            warnings.warn(
                f"The chosen seasonal unit root test encountered an error when "
                f"testing for difference {D}.\nFrom {test}(): {e}\n"
                f"{D} seasonal differences will be used."
            )
            return False

    dodiff = run_seas_test(x, period)

    if dodiff and not isinstance(period, int):
        warnings.warn(
            "The time series frequency has been rounded to support seasonal differencing."
        )
        period = round(period)

    while dodiff and D < max_D:
        D += 1
        x = diff(x, period, 1)
        x = x[~np.isnan(x)]

        if is_constant(x):
            return D

        if len(x) >= 2 * period and D < max_D:
            dodiff = run_seas_test(x, period)
        else:
            dodiff = False

    return D
