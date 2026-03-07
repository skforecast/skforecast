"""
Box-Cox transformation utilities for ARIMA models.

This module implements Box-Cox transformations with automatic lambda selection
using either the Guerrero (1993) method or the log-likelihood method.

References
----------
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. JRSS B, 26, 211-246.
- Guerrero, V.M. (1993). Time-series analysis supported by power transformations.
  Journal of Forecasting, 12, 37-48.
- Bickel, P. J. and Doksum K. A. (1981). An Analysis of Transformations Revisited.
  JASA, 76, 296-311.
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm
from typing import Union, Tuple, Optional, Dict, Any
import warnings


def _guerrero_cv(lam: float, x: np.ndarray, m: int, nonseasonal_length: int = 2) -> float:
    """
    Compute the coefficient of variation for Guerrero's method.

    Parameters
    ----------
    lam : float
        Box-Cox transformation parameter (lambda).
    x : np.ndarray
        Original time series.
    m : int
        Seasonal period (frequency).
    nonseasonal_length : int
        Length of non-seasonal components (default 2).

    Returns
    -------
    float
        Coefficient of variation (std/mean) of the ratio x_sd / x_mean^(1-lam).
    """
    period = max(nonseasonal_length, m)
    nobsf = len(x)
    nyr = nobsf // period
    nobst = nyr * period

    if nobst == 0 or nyr == 0:
        return np.inf

    x_trimmed = x[-(nobst):]
    x_mat = x_trimmed.reshape((period, nyr), order='F')

    x_mean = np.nanmean(x_mat, axis=0)
    x_sd = np.nanstd(x_mat, axis=0, ddof=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        x_rat = x_sd / np.power(x_mean, 1.0 - lam)
        x_rat = x_rat[np.isfinite(x_rat)]

    if len(x_rat) == 0:
        return np.inf

    mean_rat = np.nanmean(x_rat)
    if mean_rat == 0:
        return np.inf

    return np.nanstd(x_rat, ddof=1) / mean_rat


def guerrero(
    x: np.ndarray,
    m: int,
    lower: float = -1.0,
    upper: float = 2.0,
    nonseasonal_length: int = 2
) -> float:
    """
    Select Box-Cox lambda using Guerrero's (1993) method.

    Guerrero's method selects lambda to minimize the coefficient of variation
    for subseries of x, which tends to make the variance more stable across
    the series.

    Parameters
    ----------
    x : np.ndarray
        Time series data (must be positive).
    m : int
        Seasonal period (frequency).
    lower : float
        Lower bound for lambda search (default -1.0).
    upper : float
        Upper bound for lambda search (default 2.0).
    nonseasonal_length : int
        Minimum period length for non-seasonal data (default 2).

    Returns
    -------
    float
        Optimal Box-Cox transformation parameter.

    References
    ----------
    Guerrero, V.M. (1993). Time-series analysis supported by power
    transformations. Journal of Forecasting, 12, 37-48.
    """
    x_clean = x[~np.isnan(x)]

    if np.any(x_clean <= 0):
        warnings.warn(
            "Guerrero's method for selecting a Box-Cox parameter (lambda) "
            "is given for strictly positive data."
        )

    result = optimize.minimize_scalar(
        lambda lam: _guerrero_cv(lam, x_clean, m, nonseasonal_length),
        bounds=(lower, upper),
        method='bounded'
    )

    return result.x


def bcloglik(
    x: np.ndarray,
    m: int,
    lower: float = -1.0,
    upper: float = 2.0,
    is_ts: bool = True
) -> float:
    """
    Select Box-Cox lambda by maximizing profile log-likelihood.

    For non-seasonal data (m=1), fits a linear time trend.
    For seasonal data (m>1), fits a linear time trend with seasonal dummies.

    Parameters
    ----------
    x : np.ndarray
        Time series data (must be positive).
    m : int
        Seasonal period (frequency).
    lower : float
        Lower bound for lambda search (default -1.0).
    upper : float
        Upper bound for lambda search (default 2.0).
    is_ts : bool
        Whether to include time trend (default True).

    Returns
    -------
    float
        Optimal Box-Cox transformation parameter.

    References
    ----------
    Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations.
    JRSS B, 26, 211-246.
    """
    x_clean = x[~np.isnan(x)]
    n = len(x_clean)

    if np.any(x_clean <= 0):
        raise ValueError("x must be positive for log-likelihood method")

    logx = np.log(x_clean)
    xdot = np.exp(np.mean(logx))

    if not is_ts:
        X = np.ones((n, 1))
    else:
        t = np.arange(1, n + 1)
        if m > 1:
            s = np.mod(np.arange(n), m)
            D = np.zeros((n, m - 1))
            for j in range(1, m):
                D[:, j - 1] = (s == j).astype(float)
            X = np.column_stack([np.ones(n), t, D])
        else:
            X = np.column_stack([np.ones(n), t])

    lambdas = np.arange(lower, upper + 0.05, 0.05)
    loglik = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        if np.abs(lam) > 0.02:
            xt = (np.power(x_clean, lam) - 1.0) / lam
        else:
            xt = logx * (1 + (lam * logx) / 2 * (1 + (lam * logx) / 3 * (1 + (lam * logx) / 4)))

        z = xt / np.power(xdot, lam - 1)

        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
            r = z - X @ beta
            loglik[i] = -n / 2 * np.log(np.sum(r ** 2))
        except np.linalg.LinAlgError:
            loglik[i] = -np.inf

    return lambdas[np.argmax(loglik)]


def box_cox_lambda(
    x: np.ndarray,
    m: int = 1,
    method: str = "guerrero",
    lower: float = 0.0,
    upper: float = 1.0,
    nonseasonal_length: int = 2,
    is_ts: bool = True
) -> float:
    """
    Automatic selection of Box-Cox transformation parameter.

    Parameters
    ----------
    x : np.ndarray
        A numeric vector or time series.
    m : int
        Seasonal period (frequency). Default 1 for non-seasonal.
    method : str
        Method for selecting lambda:
        - "guerrero": Guerrero's (1993) method (default)
        - "loglik": Maximum profile log-likelihood
    lower : float
        Lower limit for possible lambda values (default 0.0).
        Set to -1.0 for full range but note that negative lambda
        can cause issues with prediction intervals.
    upper : float
        Upper limit for possible lambda values (default 1.0).
    nonseasonal_length : int
        Minimum period for Guerrero method (default 2).
    is_ts : bool
        Whether to treat as time series for loglik method (default True).

    Returns
    -------
    float
        Box-Cox transformation parameter (lambda).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.abs(np.random.randn(100)) + 1  # Positive data
    >>> lam = box_cox_lambda(y, m=12, method="guerrero")
    >>> print(f"Selected lambda: {lam:.3f}")

    References
    ----------
    - Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations.
      JRSS B, 26, 211-246.
    - Guerrero, V.M. (1993). Time-series analysis supported by power
      transformations. Journal of Forecasting, 12, 37-48.
    """
    x = np.asarray(x, dtype=np.float64)
    x_clean = x[~np.isnan(x)]

    if np.any(x_clean <= 0):
        lower = max(lower, 0.0)

    if len(x_clean) <= 2 * max(m, 1):
        return 1.0

    method = method.lower()
    if method == "loglik":
        return bcloglik(x_clean, m, lower=lower, upper=upper, is_ts=is_ts)
    elif method == "guerrero":
        return guerrero(x_clean, m, lower, upper, nonseasonal_length)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'guerrero' or 'loglik'.")


def box_cox(
    x: np.ndarray,
    m: int = 1,
    lambda_bc: Union[float, str, None] = "auto"
) -> Tuple[np.ndarray, float]:
    """
    Apply Box-Cox transformation to a time series.

    The Box-Cox transformation is defined as:
    - If lambda = 0: y = log(x)
    - If lambda != 0: y = (x^lambda - 1) / lambda

    Parameters
    ----------
    x : np.ndarray
        A numeric vector or time series (must be positive for lambda < 0).
    m : int
        Seasonal period for auto lambda selection (default 1).
    lambda_bc : float, str, or None
        Transformation parameter:
        - float: Use the specified lambda value
        - "auto": Automatically select lambda using Guerrero's method
        - None: No transformation (returns x unchanged with lambda=1)

    Returns
    -------
    transformed : np.ndarray
        Transformed data.
    lambda_bc : float
        The lambda value used (useful when lambda_bc="auto").

    Examples
    --------
    >>> import numpy as np
    >>> y = np.exp(np.random.randn(100))  # Log-normal data
    >>> y_transformed, lam = box_cox(y, m=1, lambda_bc="auto")
    >>> print(f"Lambda: {lam:.3f}")

    References
    ----------
    - Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations.
    - Bickel, P. J. and Doksum K. A. (1981). An Analysis of Transformations Revisited.
    """
    x = np.asarray(x, dtype=np.float64)

    if lambda_bc == "auto":
        lambda_bc = box_cox_lambda(x, m)
    elif lambda_bc is None:
        return x.copy(), 1.0

    lambda_bc = float(lambda_bc)

    transformed = x.copy()
    if lambda_bc < 0:
        transformed[transformed < 0] = np.nan

    if lambda_bc == 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            transformed = np.log(transformed)
    else:
        with np.errstate(invalid='ignore'):
            transformed = (np.sign(transformed) * np.abs(transformed) ** lambda_bc - 1) / lambda_bc

    return transformed, lambda_bc


def inv_box_cox(
    x: np.ndarray,
    lambda_bc: float,
    biasadj: bool = False,
    fvar: Optional[Union[float, np.ndarray, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Reverse the Box-Cox transformation.

    Parameters
    ----------
    x : np.ndarray
        Transformed data.
    lambda_bc : float
        Transformation parameter used in forward transformation.
    biasadj : bool
        Whether to apply bias adjustment for mean forecasts (default False).
        If True, adjusts back-transformation to produce mean forecasts rather
        than median forecasts.
    fvar : float, np.ndarray, or dict, optional
        Forecast variance(s) required when biasadj=True.
        - float: Single variance value applied to all observations
        - np.ndarray: Array of variances (same length as x)
        - dict: Dictionary with keys 'level', 'upper', 'lower' for
                computing variance from prediction intervals

    Returns
    -------
    np.ndarray
        Back-transformed data.

    Notes
    -----
    When `biasadj=True`, the back-transformation includes a correction term
    that produces mean forecasts instead of median forecasts:

        y_adj = y * (1 + 0.5 * sigma^2 * (1 - lambda) / y^(2*lambda))

    References
    ----------
    - Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations.
    - Bickel, P. J. and Doksum K. A. (1981). An Analysis of Transformations Revisited.
    """
    x = np.asarray(x, dtype=np.float64)
    x_work = x.copy()

    if lambda_bc < 0:
        thresh = -1.0 / lambda_bc
        margin = 0.001 * abs(thresh)
        x_work = np.clip(x_work, -np.inf, thresh - margin)

    if lambda_bc == 0:
        out = np.exp(x_work)
    else:
        xx = x_work * lambda_bc + 1
        xx = np.maximum(xx, 1e-10)
        out = np.sign(xx) * np.abs(xx) ** (1.0 / lambda_bc)

    if biasadj:
        if fvar is None:
            raise ValueError("fvar must be provided when biasadj=True")

        if isinstance(fvar, dict):
            level = max(fvar.get('level', [95]))
            upper = np.asarray(fvar['upper'])
            lower = np.asarray(fvar['lower'])

            if upper.ndim == 2 and upper.shape[1] > 1:
                lvlvec = fvar.get('level', [])
                idx = lvlvec.index(level) if level in lvlvec else -1
                upper = upper[:, idx]
                lower = lower[:, idx]

            if level > 1:
                level = level / 100.0
            level = (level + 1.0) / 2.0

            q = norm.ppf(level)
            fvar_bc = ((upper - lower) / (q * 2)) ** 2
        else:
            fvar_bc = np.asarray(fvar)

        if np.isscalar(fvar_bc):
            fvar_bc = np.full_like(out, fvar_bc)
        elif fvar_bc.shape != out.shape:
            fvar_flat = np.tile(fvar_bc.ravel(), int(np.ceil(out.size / fvar_bc.size)))[:out.size]
            fvar_bc = fvar_flat.reshape(out.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            adjustment = 1 + 0.5 * fvar_bc * (1 - lambda_bc) / (out ** (2 * lambda_bc))
            out = out * adjustment

    return out


def box_cox_biasadj(
    y: np.ndarray,
    lambda_bc: float,
    fvar: Optional[Union[float, np.ndarray]] = None
) -> np.ndarray:
    """
    Compute bias-adjusted back-transformed values.

    This is a convenience wrapper for inv_box_cox with biasadj=True.

    Parameters
    ----------
    y : np.ndarray
        Transformed data (on Box-Cox scale).
    lambda_bc : float
        Transformation parameter.
    fvar : float or np.ndarray, optional
        Forecast variances. If None, uses residual variance.

    Returns
    -------
    np.ndarray
        Bias-adjusted back-transformed values.
    """
    if fvar is None:
        fvar = np.nanvar(y)

    return inv_box_cox(y, lambda_bc, biasadj=True, fvar=fvar)
