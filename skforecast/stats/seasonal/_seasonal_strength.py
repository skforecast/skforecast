################################################################################
#                            Seasonal strength                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################

# Seasonal strength measures for time series analysis.
# This module implements seasonal strength heuristics based on an STL
# decomposition, following Wang, Smith & Hyndman (2006).
# References
# ----------
# - Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based
#   clustering for time series data. Data Mining and Knowledge Discovery,
#   13(3), 335-364.

import math
import numpy as np

from ...utils import check_optional_dependency

try:
    from statsmodels.tsa.seasonal import STL
except ModuleNotFoundError as error:
    if error.name == "statsmodels":
        check_optional_dependency(package_name="statsmodels")
    raise


def _nextodd(x: float) -> int:
    """
    Round to the nearest integer and increase it by one if it is even, as
    `nextodd` in R's `stats::stl`.
    """
    x = int(round(x))
    return x + 1 if x % 2 == 0 else x


def seas_heuristic(x: np.ndarray, period: int) -> float:
    """
    Compute seasonal strength measure (Wang, Smith & Hyndman, 2006).

    Uses an STL decomposition to measure how strong the seasonal
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

    Notes
    -----
    The decomposition follows `forecast::mstl` (used by R's
    `forecast:::seas.heuristic`): statsmodels' `STL` with `seasonal=11`,
    `seasonal_deg=0`, the default trend and low-pass windows and jumps of
    R's `stats::stl`, 2 inner and 0 outer iterations. For odd periods the
    low-pass window is `period + 2` instead of R's `period`, which
    statsmodels does not accept, so the strength can differ from R's in the
    fourth decimal. Series with `len(x) <= 2 * period` return 0. Missing
    values are filled with the mean of the series, while `mstl` interpolates
    them with `na.interp`, so the result only matches R for series without
    missing values.

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
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 2 * period:
        return 0.0

    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmean(x), x)

    # Same decomposition as `mstl` in R's forecast package (used by
    # `forecast:::seas.heuristic`): `stl` with `s.window = 11` and the
    # default settings of `stats::stl`.
    s_window = 11
    t_window = _nextodd(math.ceil(1.5 * period / (1 - 1.5 / s_window)))
    l_window = _nextodd(period)
    if l_window <= period:
        # statsmodels requires `low_pass > period`, while R uses
        # `l.window = period` for odd periods. The next odd value gives
        # seasonal strengths within about 1e-3 of R's.
        l_window = period + 2
    fit = STL(
        x,
        period=period,
        seasonal=s_window,
        trend=t_window,
        low_pass=l_window,
        seasonal_deg=0,
        trend_deg=1,
        low_pass_deg=1,
        seasonal_jump=math.ceil(s_window / 10),
        trend_jump=math.ceil(t_window / 10),
        low_pass_jump=math.ceil(l_window / 10),
        robust=False,
    ).fit(inner_iter=2, outer_iter=0)
    remainder = fit.resid
    seasonal = fit.seasonal

    var_seasonal_remainder = np.var(seasonal + remainder, ddof=1)
    if var_seasonal_remainder < 1e-10:
        return 0.0

    Fs = 1.0 - np.var(remainder, ddof=1) / var_seasonal_remainder
    return float(min(1.0, max(0.0, Fs)))
