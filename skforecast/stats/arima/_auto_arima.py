################################################################################
#                 Automatic ARIMA base implementation                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
# Automatic ARIMA model selection using stepwise search or exhaustive grid search,
# minimizing an information criterion (AIC, AICc, or BIC).
# Also includes arima_rjh for ARIMA fitting with drift and Box-Cox support.

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict as DictType, Any, Union, List
import warnings
from scipy.stats import norm
from numba import njit
from statsmodels.tsa.stattools import adfuller, kpss

from ._arima_base import (
    arima, predict_arima, diff, match_arg
)

from ..seasonal import (
    ndiffs,
    nsdiffs,
    is_constant
)

from ..transformations import (
    box_cox,
    inv_box_cox
)


@njit(cache=True)
def _newmodel_jit(
    p: int, d: int, q: int,
    P: int, D: int, Q: int,
    c_int: int,
    results: np.ndarray,
    k: int
) -> bool:  # pragma: no cover
    """JIT-compiled check if model configuration has already been tried."""
    for i in range(k):
        if np.isnan(results[i, 0]):
            continue
        if (results[i, 0] == p and results[i, 1] == d and results[i, 2] == q and
            results[i, 3] == P and results[i, 4] == D and results[i, 5] == Q and
            results[i, 6] == c_int):
            return False
    return True


@njit(cache=True)
def _time_index_jit(n: int, m: int, start: float) -> np.ndarray:  # pragma: no cover
    """JIT-compiled time index generation."""
    m_safe = m if m > 0 else 1
    result = np.zeros(n)
    for i in range(n):
        result[i] = start + i / m_safe
    return result


def analyze_series(x: np.ndarray) -> Tuple[Optional[int], int, np.ndarray]:
    """
    Analyze series to find first/last non-missing and trim leading missings.

    Parameters
    ----------
    x : np.ndarray
        Input time series (may contain NaN).

    Returns
    -------
    firstnonmiss : int or None
        Index of first non-missing value (0-based).
    serieslength : int
        Number of non-missing values in trimmed span.
    x_trim : np.ndarray
        Series with leading missings removed.
    """
    miss = np.isnan(x)
    nonmiss_idx = np.where(~miss)[0]

    if len(nonmiss_idx) == 0:
        return (None, 0, x)

    first = nonmiss_idx[0]
    last = nonmiss_idx[-1]
    serieslength = np.sum(~miss[first:last+1])
    x_trim = x[first:].copy()

    return (first, serieslength, x_trim)


def mean2(x: np.ndarray, omit_na: bool = True) -> float:
    """
    Compute mean with optional NA removal.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    omit_na : bool
        If True, ignore NaN values.

    Returns
    -------
    float
        Mean value.
    """
    if omit_na:
        return np.nanmean(x)
    return np.mean(x)


def compute_approx_offset(
    approximation: bool,
    x: np.ndarray,
    d: int,
    D: int,
    m: int = 1,
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    truncate: Optional[int] = None,
    **kwargs
) -> float:
    """
    Compute approximation offset for CSS-based model selection.

    When using CSS approximation during model search, this offset is added
    to make IC values comparable to ML-based values.

    Parameters
    ----------
    approximation : bool
        Whether approximation is being used.
    x : np.ndarray
        Time series.
    d : int
        Non-seasonal differencing order.
    D : int
        Seasonal differencing order.
    m : int
        Seasonal period.
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    truncate : int or None
        Truncate series to this length.
    **kwargs
        Additional arguments passed to arima().

    Returns
    -------
    float
        Offset value (0.0 if not using approximation).
    """
    if not approximation:
        return 0.0

    xx = x.copy()
    Xreg = xreg
    N0 = len(xx)

    if truncate is not None and N0 > truncate:
        start_idx = N0 - truncate
        xx = xx[start_idx:]

        if Xreg is not None:
            if isinstance(Xreg, pd.DataFrame):
                nrows = len(Xreg)
                if nrows == N0:
                    Xreg = Xreg.iloc[start_idx:]
            else:
                nrows = Xreg.shape[0]
                if nrows == N0:
                    Xreg = Xreg[start_idx:]

    serieslength = len(xx)

    try:
        if D == 0:
            fit = arima(xx, m, order=(0, d, 0), seasonal=(0, 0, 0),
                       xreg=Xreg, include_mean=False, **kwargs)
        else:
            fit = arima(xx, m, order=(0, d, 0), seasonal=(0, D, 0),
                       xreg=Xreg, include_mean=False, **kwargs)

        loglik = fit['loglik']
        sigma2 = fit['sigma2']
        offset = -2 * loglik - serieslength * np.log(sigma2)
        return offset
    except Exception:
        return 0.0


def newmodel(
    p: int, d: int, q: int,
    P: int, D: int, Q: int,
    constant: bool,
    results: np.ndarray,
    k: int
) -> bool:
    """
    Check if a model configuration has already been tried.

    Parameters
    ----------
    p, d, q : int
        Non-seasonal ARIMA orders.
    P, D, Q : int
        Seasonal ARIMA orders.
    constant : bool
        Whether model includes constant.
    results : np.ndarray
        Matrix of previously tried models.
    k : int
        Number of models tried so far.

    Returns
    -------
    bool
        True if this is a new model configuration.
    """
    c_int = 1 if constant else 0
    return _newmodel_jit(p, d, q, P, D, Q, c_int, results, k)


def get_pdq(x: Union[Tuple, List, DictType]) -> Tuple[int, int, int]:
    """
    Extract (p, d, q) from various input formats.

    Parameters
    ----------
    x : tuple, list, or dict
        Order specification.

    Returns
    -------
    tuple
        (p, d, q) tuple.
    """
    if isinstance(x, dict):
        return (x.get('p', 0), x.get('d', 0), x.get('q', 0))
    return (x[0], x[1], x[2])


def get_sum(x: Union[Tuple, List, DictType]) -> int:
    """
    Get sum of order specification.

    Parameters
    ----------
    x : tuple, list, or dict
        Order specification.

    Returns
    -------
    int
        Sum of orders.
    """
    if isinstance(x, dict):
        return x.get('p', 0) + x.get('d', 0) + x.get('q', 0)
    return sum(x)


def arima_trace_str(
    order: Tuple[int, int, int],
    seasonal: Tuple[int, int, int],
    m: int,
    constant: bool,
    ic_value: float = np.inf
) -> str:
    """
    Generate trace string for model display.

    Parameters
    ----------
    order : tuple
        (p, d, q) non-seasonal orders.
    seasonal : tuple
        (P, D, Q) seasonal orders.
    m : int
        Seasonal period.
    constant : bool
        Whether model has constant.
    ic_value : float
        Information criterion value.

    Returns
    -------
    str
        Formatted trace string.
    """
    p, d, q = get_pdq(order)
    P, D, Q = get_pdq(seasonal)

    seasonal_part = f"({P},{D},{Q})[{m}]" if get_sum(seasonal) > 0 and m > 1 else ""

    if constant and (d + D == 0):
        mean_str = " with non-zero mean"
    elif constant and (d + D == 1):
        mean_str = " with drift        "
    elif not constant and (d + D == 0):
        mean_str = " with zero mean    "
    else:
        mean_str = "                   "

    ic_str = f"{ic_value:.4f}" if np.isfinite(ic_value) else "Inf"
    s = f" ARIMA({p},{d},{q}){seasonal_part}{mean_str} : {ic_str}"

    return s


def fit_custom_arima(
    x: np.ndarray,
    m: int,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Tuple[int, int, int] = (0, 0, 0),
    constant: bool = True,
    ic: str = "aic",
    trace: bool = False,
    approximation: bool = False,
    offset: float = 0.0,
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    method: Optional[str] = None,
    nstar: Optional[int] = None,
    **kwargs
) -> DictType[str, Any]:
    """
    Fit ARIMA model with specific parameters and compute information criteria.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    m : int
        Seasonal period.
    order : tuple
        (p, d, q) non-seasonal orders.
    seasonal : tuple
        (P, D, Q) seasonal orders.
    constant : bool
        Include constant/drift term.
    ic : str
        Information criterion: "aic", "aicc", or "bic".
    trace : bool
        Print trace information.
    approximation : bool
        Use CSS approximation.
    offset : float
        Approximation offset.
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    method : str or None
        Estimation method.
    nstar : int or None
        Pre-computed series length.
    **kwargs
        Additional arguments for arima().

    Returns
    -------
    dict
        Fitted model dictionary with IC values.
    """
    if nstar is None:
        valid_idx = ~np.isnan(x)
        if np.any(valid_idx):
            first = np.argmax(valid_idx)
            last = len(x) - 1 - np.argmax(valid_idx[::-1])
            n = np.sum(valid_idx[first:last+1])
        else:
            n = 0
    else:
        n = nstar + order[1] + seasonal[1] * m

    use_season = get_sum(seasonal) > 0 and m > 1
    diffs = order[1] + seasonal[1]

    if method is None:
        method = "CSS" if approximation else "CSS-ML"

    drift_case = (diffs == 1) and constant

    xreg_use = xreg
    if drift_case:
        drift = np.arange(1, len(x) + 1, dtype=np.float64)
        if xreg_use is None:
            xreg_use = pd.DataFrame({'drift': drift})
        elif isinstance(xreg_use, pd.DataFrame):
            xreg_use = xreg_use.copy()
            xreg_use.insert(0, 'drift', drift)
        else:
            xreg_use = np.column_stack([drift, xreg_use])

    try:
        if drift_case:
            fit = arima(
                x, m,
                order=order,
                seasonal=seasonal if use_season else (0, 0, 0),
                xreg=xreg_use,
                method=method,
                include_mean=False,
                **kwargs
            )
        else:
            fit = arima(
                x, m,
                order=order,
                seasonal=seasonal if use_season else (0, 0, 0),
                xreg=xreg_use,
                method=method,
                include_mean=constant,
                **kwargs
            )
    except Exception as e:
        if trace:
            print(arima_trace_str(order, seasonal, m, constant, np.inf))
        return _create_error_model(order, seasonal, m)

    if xreg_use is None:
        nxreg = 0
    elif isinstance(xreg_use, pd.DataFrame):
        nxreg = xreg_use.shape[1]
    else:
        nxreg = xreg_use.shape[1] if xreg_use.ndim > 1 else 1

    nstar_adj = n - order[1] - seasonal[1] * m
    npar = np.sum(fit['mask']) + 1

    if method == "CSS":
        fit['aic'] = offset + nstar_adj * np.log(fit['sigma2']) + 2 * npar

    if not np.isnan(fit['aic']):
        fit['bic'] = fit['aic'] + npar * (np.log(nstar_adj) - 2)
        if nstar_adj - npar - 1 > 0:
            fit['aicc'] = fit['aic'] + 2 * npar * (npar + 1) / (nstar_adj - npar - 1)
        else:
            fit['aicc'] = np.inf

        if ic == "bic":
            fit['ic'] = fit['bic']
        elif ic == "aicc":
            fit['ic'] = fit['aicc']
        else:
            fit['ic'] = fit['aic']
    else:
        fit['aic'] = np.inf
        fit['bic'] = np.inf
        fit['aicc'] = np.inf
        fit['ic'] = np.inf

    resid_valid = fit['residuals'][~np.isnan(fit['residuals'])]
    if len(resid_valid) > npar - 1:
        fit['sigma2'] = np.sum(resid_valid**2) / (nstar_adj - npar + 1)

    minroot = 2.0

    if order[0] + seasonal[0] > 0:
        phi = fit['model']['phi']
        if len(phi) > 0:
            lastnz = len(phi)
            for i in range(len(phi) - 1, -1, -1):
                if abs(phi[i]) > 1e-8:
                    lastnz = i + 1
                    break
            if lastnz > 0:
                try:
                    coeffs = np.concatenate([[1.0], -phi[:lastnz]])
                    proots = np.roots(coeffs[::-1])
                    minroot = min(minroot, np.min(np.abs(proots)))
                except Exception:
                    fit['ic'] = np.inf

    if order[2] + seasonal[2] > 0 and fit['ic'] < np.inf:
        theta = fit['model']['theta']
        if len(theta) > 0:
            lastnz = len(theta)
            for i in range(len(theta) - 1, -1, -1):
                if abs(theta[i]) > 1e-8:
                    lastnz = i + 1
                    break
            if lastnz > 0:
                try:
                    coeffs = np.concatenate([[1.0], theta[:lastnz]])
                    proots = np.roots(coeffs[::-1])
                    minroot = min(minroot, np.min(np.abs(proots)))
                except Exception:
                    fit['ic'] = np.inf

    try:
        var_diag = np.diag(fit['var_coef'])
        bad_variances = np.any(np.isnan(np.sqrt(var_diag[var_diag > 0])))
    except Exception:
        bad_variances = True

    if minroot < 1 + 1e-2 or bad_variances:
        fit['ic'] = np.inf

    fit['xreg'] = xreg_use if drift_case else xreg

    if trace:
        print(arima_trace_str(order, seasonal, m, constant, fit['ic']))

    return fit


def _create_error_model(
    order: Tuple[int, int, int],
    seasonal: Tuple[int, int, int],
    m: int
) -> DictType[str, Any]:
    """Create an error model with infinite IC."""
    return {
        'y': np.array([]),
        'fitted': np.array([]),
        'coef': pd.DataFrame(),
        'sigma2': 0.0,
        'var_coef': np.zeros((0, 0)),
        'mask': np.array([], dtype=bool),
        'loglik': 0.0,
        'aic': np.inf,
        'bic': np.inf,
        'aicc': np.inf,
        'ic': np.inf,
        'arma': [order[0], order[2], seasonal[0], seasonal[2], m, order[1], seasonal[1]],
        'residuals': np.array([]),
        'converged': False,
        'n_cond': 0,
        'nobs': 0,
        'model': {
            'phi': np.array([]),
            'theta': np.array([]),
            'Delta': np.array([]),
            'Z': np.array([]),
            'a': np.array([]),
            'P': np.zeros((1, 1)),
            'T': np.zeros((1, 1)),
            'V': np.zeros((1, 1)),
            'h': 0.0,
            'Pn': np.zeros((1, 1))
        },
        'xreg': None,
        'method': 'Error model',
        'lambda': None,
        'biasadj': None,
        'offset': None,
        'constant': False
    }


def search_arima(
    x: np.ndarray,
    m: int,
    d: int,
    D: int,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    stationary: bool = False,
    ic: str = "aic",
    trace: bool = False,
    approximation: bool = False,
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    offset: float = 0.0,
    allowdrift: bool = True,
    allowmean: bool = True,
    method: Optional[str] = None,
    **kwargs
) -> DictType[str, Any]:
    """
    Exhaustive grid search over ARIMA models.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    m : int
        Seasonal period.
    d, D : int
        Differencing orders.
    max_p, max_q, max_P, max_Q : int
        Maximum orders.
    max_order : int
        Maximum sum of p+q+P+Q.
    stationary : bool
        Restrict to stationary models.
    ic : str
        Information criterion.
    trace : bool
        Print trace.
    approximation : bool
        Use CSS approximation.
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    offset : float
        Approximation offset.
    allowdrift, allowmean : bool
        Allow drift/mean terms.
    method : str or None
        Estimation method.
    **kwargs
        Additional arguments for arima().

    Returns
    -------
    dict
        Best fitted model.
    """
    ic = match_arg(ic, ["aic", "aicc", "bic"])

    allowdrift = allowdrift and (d + D == 1)
    allowmean = allowmean and (d + D == 0)
    maxK = 1 if (allowdrift or allowmean) else 0

    best_ic = np.inf
    bestfit = None
    best_constant = None

    for i in range(max_p + 1):
        for j in range(max_q + 1):
            for I in range(max_P + 1):
                for J in range(max_Q + 1):
                    if i + j + I + J <= max_order:
                        for K in range(maxK + 1):
                            fit = fit_custom_arima(
                                x, m,
                                order=(i, d, j),
                                seasonal=(I, D, J),
                                constant=(K == 1),
                                ic=ic,
                                trace=trace,
                                approximation=approximation,
                                offset=offset,
                                xreg=xreg,
                                method=method,
                                **kwargs
                            )

                            if fit['ic'] < best_ic:
                                best_ic = fit['ic']
                                bestfit = fit
                                best_constant = (K == 1)

    if bestfit is None:
        raise ValueError("No ARIMA model able to be estimated")

    if approximation and bestfit is not None:
        if trace:
            print("\nNow re-fitting the best model(s) without approximations...\n")

        arma = bestfit['arma']
        p, q, P, Q = arma[0], arma[1], arma[2], arma[3]
        d_ = arma[5]
        D_ = arma[6]

        newbestfit = fit_custom_arima(
            x, m,
            order=(p, d_, q),
            seasonal=(P, D_, Q),
            constant=best_constant,
            ic=ic,
            trace=False,
            approximation=False,
            xreg=xreg,
            **kwargs
        )

        if newbestfit['ic'] == np.inf:
            bestfit = search_arima(
                x, m, d=d, D=D,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q,
                max_order=max_order,
                stationary=stationary,
                ic=ic, trace=trace,
                approximation=False,
                xreg=xreg,
                offset=offset,
                allowdrift=allowdrift,
                allowmean=allowmean,
                **kwargs
            )
        else:
            bestfit = newbestfit

    return bestfit


def kpss_test(x: np.ndarray, regression: str = 'c') -> Tuple[float, float]:
    """
    Perform KPSS test for stationarity.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    regression : str
        'c' for constant, 'ct' for constant + trend.

    Returns
    -------
    statistic : float
        Test statistic.
    pvalue : float
        P-value (approximate).
    """
    stat, pval, _, _ = kpss(x[~np.isnan(x)], regression=regression, nlags='auto')
    return stat, pval


def adf_test(x: np.ndarray) -> Tuple[float, float]:
    """
    Perform Augmented Dickey-Fuller test for unit root.

    Parameters
    ----------
    x : np.ndarray
        Time series.

    Returns
    -------
    statistic : float
        Test statistic.
    pvalue : float
        P-value.
    """
    result = adfuller(x[~np.isnan(x)], autolag='AIC')
    return result[0], result[1]


def auto_arima(
    y: np.ndarray,
    m: int = 1,
    d: Optional[int] = None,
    D: Optional[int] = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    max_d: int = 2,
    max_D: int = 1,
    start_p: int = 2,
    start_q: int = 2,
    start_P: int = 1,
    start_Q: int = 1,
    stationary: bool = False,
    seasonal: bool = True,
    ic: str = "aicc",
    stepwise: bool = True,
    nmodels: int = 94,
    trace: bool = False,
    approximation: Optional[bool] = None,
    method: Optional[str] = None,
    truncate: Optional[int] = None,
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    test: str = "kpss",
    test_args: Optional[DictType] = None,
    seasonal_test: str = "seas",
    seasonal_test_args: Optional[DictType] = None,
    allowdrift: bool = True,
    allowmean: bool = True,
    lambda_bc: Union[float, str, None] = None,
    biasadj: bool = False,
    **kwargs
) -> DictType[str, Any]:
    """
    Automatic ARIMA model selection.

    Fit the "best" ARIMA/SARIMA model to a univariate time series by minimizing
    an information criterion (AICc by default). Uses stepwise search by default.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    m : int
        Seasonal period (1 for non-seasonal).
    d, D : int or None
        Non-seasonal/seasonal differencing orders. If None, determined by tests.
    max_p, max_q, max_P, max_Q : int
        Maximum orders for AR/MA components.
    max_order : int
        Maximum sum of p+q+P+Q.
    max_d, max_D : int
        Maximum differencing orders.
    start_p, start_q, start_P, start_Q : int
        Starting orders for stepwise search.
    stationary : bool
        Restrict to stationary models (d=D=0).
    seasonal : bool
        Include seasonal components.
    ic : str
        Information criterion: "aicc", "aic", or "bic".
    stepwise : bool
        Use stepwise search (faster) or exhaustive grid search.
    nmodels : int
        Maximum number of models to try in stepwise search.
    trace : bool
        Print progress.
    approximation : bool or None
        Use CSS approximation during search. If None, auto-determined.
    method : str or None
        Estimation method.
    truncate : int or None
        Truncate series for approximation offset.
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    test : str
        Unit root test: "kpss", "adf", or "pp".
    test_args : dict or None
        Additional arguments for unit root test.
    seasonal_test : str
        Seasonal test: "seas", "ocsb", "hegy", or "ch".
    seasonal_test_args : dict or None
        Additional arguments for seasonal test.
    allowdrift : bool
        Allow drift term when d+D=1.
    allowmean : bool
        Allow mean term when d+D=0.
    lambda_bc : float, str, or None
        Box-Cox transformation parameter:
        - None: No transformation (default)
        - "auto": Automatically select lambda using Guerrero's method
        - float: Use the specified lambda value (0 = log transform)
    biasadj : bool
        Bias adjustment for Box-Cox back-transformation (produces mean
        forecasts instead of median forecasts).
    **kwargs
        Additional arguments passed to arima().

    Returns
    -------
    dict
        Best fitted ARIMA model.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> fit = auto_arima(y, m=1)
    >>> print(fit['coef'])
    """
    if test_args is None:
        test_args = {}
    if seasonal_test_args is None:
        seasonal_test_args = {}

    ic = match_arg(ic, ["aicc", "aic", "bic"])
    test = match_arg(test, ["kpss", "adf", "pp"])
    seasonal_test = match_arg(seasonal_test, ["seas", "ocsb", "hegy", "ch"])

    if approximation is None:
        approximation = len(y) > 150 or m > 12

    y = np.asarray(y, dtype=np.float64)
    firstnm, serieslength, x = analyze_series(y.copy())

    if firstnm is None:
        raise ValueError("All data are missing")

    if xreg is not None:
        if isinstance(xreg, pd.DataFrame):
            xreg = xreg.iloc[firstnm:]
        else:
            xreg = np.asarray(xreg)[firstnm:]

    if is_constant(x):
        if allowmean:
            fit = arima(x, m, order=(0, 0, 0), include_mean=True, **kwargs)
        else:
            fit = arima(x, m, order=(0, 0, 0), include_mean=False, **kwargs)
        fit['constant'] = True
        fit['y'] = y
        return fit

    if not seasonal:
        m = 1
    if m < 1:
        warnings.warn("m < 1 not supported; ignoring seasonality.")
        m = 1

    max_p = min(max_p, serieslength // 3)
    max_q = min(max_q, serieslength // 3)
    max_P = min(max_P, serieslength // (3 * m)) if m > 1 else 0
    max_Q = min(max_Q, serieslength // (3 * m)) if m > 1 else 0

    if serieslength <= 3:
        ic = "aic"

    orig_x = x.copy()
    if lambda_bc is not None:
        if lambda_bc == "auto":
            x, lambda_bc = box_cox(x, m, lambda_bc="auto")
        else:
            x, lambda_bc = box_cox(x, m, lambda_bc=lambda_bc)

    xx = x.copy()
    xregg = xreg

    if xregg is not None:
        if isinstance(xregg, pd.DataFrame):
            xreg_mat = xregg.values
        else:
            xreg_mat = xregg

        nonconstant = [i for i in range(xreg_mat.shape[1])
                      if not is_constant(xreg_mat[:, i])]
        if len(nonconstant) == 0:
            xregg = None
        else:
            xreg_mat = xreg_mat[:, nonconstant]

            valid = ~np.isnan(xx) & np.all(np.isfinite(xreg_mat), axis=1)
            if np.sum(valid) > xreg_mat.shape[1]:
                X = xreg_mat[valid]
                y_valid = xx[valid]
                beta, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
                res = y_valid - X @ beta
                xx[valid] = res

    if stationary:
        d = 0
        D = 0

    if m == 1:
        D = 0
        max_P = 0
        max_Q = 0
    elif D is None:
        if len(xx) <= 2 * m:
            D = 0
        else:
            D = nsdiffs(x=xx, period=m, test=seasonal_test, max_D=max_D, **seasonal_test_args)

    if D is not None and D > 0:
        dx = diff(xx, lag=m, differences=D)
    else:
        D = 0
        dx = xx.copy()

    if d is None:
        d = ndiffs(x=dx, test=test, max_d=max_d, **test_args)

    if D >= 2:
        warnings.warn("More than one seasonal difference is not recommended.")
    elif D + d > 2:
        warnings.warn("3+ differences not recommended. Consider reducing.")

    if d > 0:
        dx = diff(dx, lag=1, differences=d)

    if len(dx) == 0:
        raise ValueError("Not enough data to proceed")

    if is_constant(dx):
        if D > 0:
            fit = arima(x, m, order=(0, d, 0), seasonal=(0, D, 0),
                       xreg=xreg, include_mean=False, method=method, **kwargs)
        else:
            fit = arima(x, m, order=(0, d, 0), xreg=xreg,
                       include_mean=False, method=method, **kwargs)
        fit['y'] = y
        return fit

    if m > 1:
        if max_P > 0:
            max_p = min(max_p, m - 1)
        if max_Q > 0:
            max_q = min(max_q, m - 1)

    offset = compute_approx_offset(
        approximation=approximation,
        x=x, d=d, D=D, m=m,
        xreg=xreg, truncate=truncate,
        **kwargs
    )

    allowdrift = allowdrift and (d + D) == 1
    allowmean = allowmean and (d + D) == 0
    constant = allowdrift or allowmean

    if trace and approximation:
        print("\nFitting models using approximations...\n")

    if not stepwise:
        bestfit = search_arima(
            x, m, d=d, D=D,
            max_p=max_p, max_q=max_q,
            max_P=max_P, max_Q=max_Q,
            max_order=max_order,
            stationary=stationary,
            ic=ic, trace=trace,
            approximation=approximation,
            xreg=xreg,
            offset=offset,
            allowdrift=allowdrift,
            allowmean=allowmean,
            method=method,
            **kwargs
        )
        bestfit['lambda'] = lambda_bc
        bestfit['biasadj'] = biasadj
        bestfit['y'] = y
        if lambda_bc is not None:
            bestfit['y_orig'] = orig_x
        return bestfit

    if len(x) < 10:
        start_p = min(start_p, 1)
        start_q = min(start_q, 1)
        start_P = 0
        start_Q = 0

    p = min(start_p, max_p)
    q = min(start_q, max_q)
    P = min(start_P, max_P)
    Q = min(start_Q, max_Q)

    results = np.full((nmodels, 8), np.nan)

    bestfit = fit_custom_arima(
        x, m, order=(p, d, q), seasonal=(P, D, Q),
        constant=constant, ic=ic, trace=trace,
        approximation=approximation, offset=offset,
        xreg=xreg, method=method, **kwargs
    )
    results[0, :] = [p, d, q, P, D, Q, int(constant), bestfit['ic']]
    k = 1

    # Second initial model: (0,d,0)(0,D,0)
    if k < nmodels:
        fit = fit_custom_arima(
            x, m, order=(0, d, 0), seasonal=(0, D, 0),
            constant=constant, ic=ic, trace=trace,
            approximation=approximation, offset=offset,
            xreg=xreg, method=method, **kwargs
        )
        results[k, :] = [0, d, 0, 0, D, 0, int(constant), fit['ic']]

        if fit['ic'] < bestfit['ic']:
            bestfit = fit
            p, q, P, Q = 0, 0, 0, 0
        k += 1

    if (max_p > 0 or max_P > 0) and k < nmodels:
        pp = 1 if max_p > 0 else 0
        PP = 1 if (m > 1 and max_P > 0) else 0

        fit = fit_custom_arima(
            x, m, order=(pp, d, 0), seasonal=(PP, D, 0),
            constant=constant, ic=ic, trace=trace,
            approximation=approximation, offset=offset,
            xreg=xreg, method=method, **kwargs
        )
        results[k, :] = [pp, d, 0, PP, D, 0, int(constant), fit['ic']]

        if fit['ic'] < bestfit['ic']:
            bestfit = fit
            p = pp
            P = PP
            q, Q = 0, 0
        k += 1

    if (max_q > 0 or max_Q > 0) and k < nmodels:
        qq = 1 if max_q > 0 else 0
        QQ = 1 if (m > 1 and max_Q > 0) else 0

        fit = fit_custom_arima(
            x, m, order=(0, d, qq), seasonal=(0, D, QQ),
            constant=constant, ic=ic, trace=trace,
            approximation=approximation, offset=offset,
            xreg=xreg, method=method, **kwargs
        )
        results[k, :] = [0, d, qq, 0, D, QQ, int(constant), fit['ic']]

        if fit['ic'] < bestfit['ic']:
            bestfit = fit
            p, P = 0, 0
            q = qq
            Q = QQ
        k += 1

    if constant and k < nmodels:
        fit = fit_custom_arima(
            x, m, order=(0, d, 0), seasonal=(0, D, 0),
            constant=False, ic=ic, trace=trace,
            approximation=approximation, offset=offset,
            xreg=xreg, method=method, **kwargs
        )
        results[k, :] = [0, d, 0, 0, D, 0, 0, fit['ic']]

        if fit['ic'] < bestfit['ic']:
            bestfit = fit
            p, P, q, Q = 0, 0, 0, 0
        k += 1

    startk = 0
    while startk < k and k < nmodels:
        startk = k

        neighbors = [
            (p, q, P - 1, Q) if P > 0 else None,
            (p, q, P, Q - 1) if Q > 0 else None,
            (p, q, P + 1, Q) if P < max_P else None,
            (p, q, P, Q + 1) if Q < max_Q else None,
            (p, q, P - 1, Q - 1) if (P > 0 and Q > 0) else None,
            (p, q, P - 1, Q + 1) if (P > 0 and Q < max_Q) else None,
            (p, q, P + 1, Q - 1) if (P < max_P and Q > 0) else None,
            (p, q, P + 1, Q + 1) if (P < max_P and Q < max_Q) else None,
            (p - 1, q, P, Q) if p > 0 else None,
            (p, q - 1, P, Q) if q > 0 else None,
            (p + 1, q, P, Q) if p < max_p else None,
            (p, q + 1, P, Q) if q < max_q else None,
            (p - 1, q - 1, P, Q) if (p > 0 and q > 0) else None,
            (p - 1, q + 1, P, Q) if (p > 0 and q < max_q) else None,
            (p + 1, q - 1, P, Q) if (p < max_p and q > 0) else None,
            (p + 1, q + 1, P, Q) if (p < max_p and q < max_q) else None,
        ]

        for neighbor in neighbors:
            if neighbor is None:
                continue

            np_, nq, nP, nQ = neighbor

            if not newmodel(np_, d, nq, nP, D, nQ, constant, results, k):
                continue

            k += 1
            if k > nmodels:
                break

            fit = fit_custom_arima(
                x, m, order=(np_, d, nq), seasonal=(nP, D, nQ),
                constant=constant, ic=ic, trace=trace,
                approximation=approximation, offset=offset,
                xreg=xreg, method=method, **kwargs
            )
            results[k-1, :] = [np_, d, nq, nP, D, nQ, int(constant), fit['ic']]

            if fit['ic'] < bestfit['ic']:
                bestfit = fit
                p, q, P, Q = np_, nq, nP, nQ
                break

        if (allowdrift or allowmean) and k < nmodels:
            new_constant = not constant
            if newmodel(p, d, q, P, D, Q, new_constant, results, k):
                k += 1
                fit = fit_custom_arima(
                    x, m, order=(p, d, q), seasonal=(P, D, Q),
                    constant=new_constant, ic=ic, trace=trace,
                    approximation=approximation, offset=offset,
                    xreg=xreg, method=method, **kwargs
                )
                results[k-1, :] = [p, d, q, P, D, Q, int(new_constant), fit['ic']]

                if fit['ic'] < bestfit['ic']:
                    bestfit = fit
                    constant = new_constant

    if k >= nmodels:
        warnings.warn(
            f"Stepwise search stopped early due to model limit (`nmodels`): {nmodels}"
        )

    if approximation:
        if trace:
            print("\nNow re-fitting the best model(s) without approximations...")

        valid_mask = ~np.isnan(results[:, 7])
        valid_results = results[valid_mask]
        ic_order = np.argsort(valid_results[:, 7])

        for i in range(min(len(ic_order), 5)):
            idx = ic_order[i]
            mod = valid_results[idx]

            fit = fit_custom_arima(
                x, m,
                order=(int(mod[0]), d, int(mod[2])),
                seasonal=(int(mod[3]), D, int(mod[5])),
                constant=mod[6] > 0,
                ic=ic, trace=trace,
                approximation=False,
                xreg=xreg, method=method,
                nstar=serieslength,
                **kwargs
            )

            if fit['ic'] < np.inf:
                bestfit = fit
                break

    if trace:
        print(
            f"\nBest model found: ARIMA({bestfit['arma'][0]},{bestfit['arma'][5]},{bestfit['arma'][2]})"
            f"({bestfit['arma'][6]},{bestfit['arma'][3]},{bestfit['arma'][4]})[{m}]"
            f" with {ic}: {bestfit['ic']}\n"
        )
    bestfit['lambda'] = lambda_bc
    bestfit['biasadj'] = biasadj
    bestfit['y'] = y
    if lambda_bc is not None:
        bestfit['y_orig'] = orig_x

    return bestfit


def time_index(n: int, m: int, start: float = 1.0) -> np.ndarray:
    """
    Generate time index array.

    Parameters
    ----------
    n : int
        Length of time index.
    m : int
        Seasonal period.
    start : float
        Starting value.

    Returns
    -------
    np.ndarray
        Time index array.
    """
    return _time_index_jit(n, m, start)


def has_coef(fit: DictType[str, Any], name: str) -> bool:
    """
    Check if coefficient exists in fitted model.

    Parameters
    ----------
    fit : dict
        Fitted ARIMA model.
    name : str
        Coefficient name to check.

    Returns
    -------
    bool
        True if coefficient exists.
    """
    coef = fit.get('coef')
    if coef is None:
        return False
    if isinstance(coef, pd.DataFrame):
        return name in coef.columns
    return False


def npar_fit(fit: DictType[str, Any]) -> int:
    """
    Count number of parameters in fitted model.

    Parameters
    ----------
    fit : dict
        Fitted ARIMA model.

    Returns
    -------
    int
        Number of parameters (including sigma2).
    """
    mask = fit.get('mask', np.array([]))
    return int(np.sum(mask)) + 1


def n_and_nstar(fit: DictType[str, Any]) -> Tuple[int, int]:
    """
    Compute n and nstar (effective sample size) from fitted model.

    Parameters
    ----------
    fit : dict
        Fitted ARIMA model.

    Returns
    -------
    n : int
        Observations used in fit (nobs - n_cond).
    nstar : int
        Effective sample size after differencing.
    """
    n = fit.get('nobs', 0) - fit.get('n_cond', 0)
    arma = fit.get('arma', [0, 0, 0, 0, 1, 0, 0])
    d = int(arma[5])
    D = int(arma[6])
    m = int(arma[4])
    nstar = n - d - D * m
    return n, nstar


def prepend_drift(
    xreg: Union[pd.DataFrame, np.ndarray, None],
    drift: np.ndarray
) -> pd.DataFrame:
    """
    Prepend drift column to exogenous regressors.

    Parameters
    ----------
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    drift : np.ndarray
        Drift vector.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift as first column.
    """
    drift_arr = np.asarray(drift, dtype=np.float64).reshape(-1, 1)

    if xreg is None:
        return pd.DataFrame({'drift': drift_arr.flatten()})

    if isinstance(xreg, pd.DataFrame):
        result = pd.DataFrame({'drift': drift_arr.flatten()})
        for col in xreg.columns:
            result[col] = xreg[col].values
        return result
    else:
        xreg_arr = np.asarray(xreg)
        if xreg_arr.ndim == 1:
            xreg_arr = xreg_arr.reshape(-1, 1)
        combined = np.hstack([drift_arr, xreg_arr])
        cols = ['drift'] + [f'x{i}' for i in range(xreg_arr.shape[1])]
        return pd.DataFrame(combined, columns=cols)


def prepare_drift(
    model: DictType[str, Any],
    x: np.ndarray,
    xreg: Union[pd.DataFrame, np.ndarray, None]
) -> pd.DataFrame:
    """
    Prepare drift term for refitting model to new data.

    Reconstructs the drift column by OLS fitting the original drift
    against time indices, then extrapolates to new data length.

    Parameters
    ----------
    model : dict
        Original fitted ARIMA model.
    x : np.ndarray
        New time series data.
    xreg : DataFrame, ndarray, or None
        New exogenous regressors.

    Returns
    -------
    pd.DataFrame
        DataFrame with aligned columns including drift.
    """
    y_train = model.get('y', np.array([]))
    n_train = len(y_train)
    arma = model.get('arma', [0, 0, 0, 0, 1, 0, 0])
    m_train = int(arma[4])

    t_train = time_index(n_train, m_train)

    model_xreg = model.get('xreg')
    if model_xreg is None or not isinstance(model_xreg, pd.DataFrame):
        raise ValueError("Original model has no xreg for drift reconstruction")
    if 'drift' not in model_xreg.columns:
        raise ValueError("Original model has no 'drift' column")

    drift_vec = model_xreg['drift'].values

    X = np.column_stack([np.ones(n_train), t_train])
    coef, _, _, _ = np.linalg.lstsq(X, drift_vec, rcond=None)
    a, b = coef[0], coef[1]

    n_new = len(x)
    m_new = m_train
    t_new = time_index(n_new, m_new)
    newdr = a + b * t_new

    xreg_with_drift = prepend_drift(xreg, newdr)

    target_cols = list(model_xreg.columns)
    aligned = pd.DataFrame(index=range(n_new))
    for col in target_cols:
        if col in xreg_with_drift.columns:
            aligned[col] = xreg_with_drift[col].values
        else:
            aligned[col] = 0.0

    return aligned


def refit_arima_model(
    x: np.ndarray,
    m: int,
    model: DictType[str, Any],
    xreg: Union[pd.DataFrame, np.ndarray, None],
    method: str = "CSS-ML",
    **kwargs
) -> DictType[str, Any]:
    """
    Refit an existing ARIMA model structure to new data.

    Uses fixed coefficients from the original model - no re-estimation.

    Parameters
    ----------
    x : np.ndarray
        New time series data.
    m : int
        Seasonal period (ignored if different from model's period).
    model : dict
        Previously fitted ARIMA model.
    xreg : DataFrame, ndarray, or None
        New exogenous regressors.
    method : str
        Estimation method.
    **kwargs
        Additional arguments for arima().

    Returns
    -------
    dict
        Refitted ARIMA model.
    """
    arma = model.get('arma', [0, 0, 0, 0, 1, 0, 0])
    p, q, P, Q, m_model, d, D = arma[:7]

    if m != m_model:
        warnings.warn(f"Ignoring supplied m={m}; using model's seasonal period m={m_model}")

    order = (int(p), int(d), int(q))
    seasonal = (int(P), int(D), int(Q))

    coef = model.get('coef')
    if coef is not None and isinstance(coef, pd.DataFrame):
        fixed = coef.values.flatten() if coef.size > 0 else None
    else:
        fixed = None

    fit = arima(
        x, int(m_model),
        order=order,
        seasonal=seasonal,
        xreg=xreg,
        include_mean=has_coef(model, 'intercept'),
        method=method,
        fixed=fixed,
        **kwargs
    )

    fit['var_coef'] = np.zeros_like(fit['var_coef'])
    fit['sigma2'] = model.get('sigma2', fit['sigma2'])

    if xreg is not None:
        fit['xreg'] = xreg

    return fit


def arima_rjh(
    y: np.ndarray,
    m: int,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Tuple[int, int, int] = (0, 0, 0),
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    include_mean: bool = True,
    include_drift: bool = False,
    include_constant: Optional[bool] = None,
    lambda_bc: Union[float, str, None] = None,
    biasadj: bool = False,
    method: str = "CSS-ML",
    model: Optional[DictType[str, Any]] = None,
    **kwargs
) -> DictType[str, Any]:
    """
    Fit an ARIMA model with drift and Box-Cox support.

    This is a Python adaptation of Rob J. Hyndman's ARIMA routine with
    two key extensions: support for a drift term and optional Box-Cox
    transformation. You can also pass a previously fitted model and
    re-apply it to new data without re-estimating parameters.

    Parameters
    ----------
    y : np.ndarray
        Univariate time series.
    m : int
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
    order : tuple
        Non-seasonal orders (p, d, q).
    seasonal : tuple
        Seasonal orders (P, D, Q).
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    include_mean : bool
        Include intercept/mean term for undifferenced series.
    include_drift : bool
        Include linear drift term.
    include_constant : bool or None
        If True, sets include_mean=True for undifferenced and
        include_drift=True for single-differenced series.
        If d+D > 1, no constant is included.
    lambda_bc : float, str, or None
        Box-Cox transformation parameter:
        - None: No transformation (default)
        - "auto": Automatically select lambda using Guerrero's method
        - float: Use specified lambda (0 = log transform, 1 = no transform)
    biasadj : bool
        Use bias-adjusted back-transformation for forecasts.
    method : str
        Estimation method: "CSS-ML", "ML", or "CSS".
    model : dict or None
        Previously fitted model to refit without re-estimation.
    **kwargs
        Additional arguments for arima().

    Returns
    -------
    dict
        Fitted ARIMA model dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> fit = arima_rjh(y, 12, order=(1,1,1), seasonal=(0,1,1),
    ...                 include_drift=True)
    >>> print(fit['coef'])
    """
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])

    x2 = np.asarray(y, dtype=np.float64).copy()

    if lambda_bc is not None:
        x2, lambda_bc = box_cox(x2, m, lambda_bc)

    seasonal2 = (0, 0, 0) if m <= 1 else seasonal

    min_len = order[1] + (seasonal2[1] * m if m > 1 else 0)
    if len(x2) <= min_len:
        raise ValueError("Not enough data to fit the model")

    if include_constant is not None:
        if include_constant:
            include_mean = True
            if (order[1] + seasonal2[1]) == 1:
                include_drift = True
        else:
            include_mean = False
            include_drift = False

    if (order[1] + seasonal2[1]) > 1 and include_drift:
        warnings.warn("No drift term fitted as order of difference is 2 or more.")
        include_drift = False

    fit = None

    if model is not None:
        model_xreg = model.get('xreg')
        had_xreg = model_xreg is not None and isinstance(model_xreg, pd.DataFrame)
        use_drift = had_xreg and 'drift' in model_xreg.columns

        if had_xreg and xreg is None:
            raise ValueError("No regressors provided")

        if use_drift:
            xreg2 = prepare_drift(model, x2, xreg)
        elif had_xreg:
            xreg2 = xreg
            if isinstance(xreg2, pd.DataFrame):
                target_cols = list(model_xreg.columns)
                aligned = pd.DataFrame(index=range(len(x2)))
                for col in target_cols:
                    if col in xreg2.columns:
                        aligned[col] = xreg2[col].values
                    else:
                        aligned[col] = 0.0
                xreg2 = aligned
        else:
            xreg2 = xreg

        fit = refit_arima_model(x2, m, model, xreg2, method, **kwargs)
    else:
        xreg2 = prepend_drift(xreg, np.arange(1, len(x2) + 1)) if include_drift else xreg

        fit = arima(
            x2, m,
            order=order,
            seasonal=seasonal2,
            xreg=xreg2,
            include_mean=include_mean,
            method=method,
            **kwargs
        )

    n, nstar = n_and_nstar(fit)
    np_ = npar_fit(fit)

    if fit.get('aic') is not None and np.isfinite(fit['aic']):
        if nstar - np_ - 1 > 0:
            fit['aicc'] = fit['aic'] + 2 * np_ * (nstar / (nstar - np_ - 1) - 1)
        else:
            fit['aicc'] = np.inf
        fit['bic'] = fit['aic'] + np_ * (np.log(nstar) - 2)

    if model is None:
        n_cond = fit.get('n_cond', 0)
        resid = fit['residuals'][n_cond:]
        ss = np.sum(resid[~np.isnan(resid)]**2)
        if nstar - np_ + 1 > 0:
            fit['sigma2'] = ss / (nstar - np_ + 1)

    if model is None and xreg2 is not None:
        fit['xreg'] = xreg2

    fit['lambda'] = lambda_bc
    fit['biasadj'] = biasadj

    return fit


def forecast_arima(
    model: DictType[str, Any],
    h: int = 10,
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    level: Union[List[float], np.ndarray, None] = None,
    fan: bool = False,
    lambda_bc: Optional[float] = None,
    biasadj: Optional[bool] = None,
    bootstrap: bool = False,
    npaths: int = 5000
) -> DictType[str, Any]:
    """
    Generate forecasts from a fitted ARIMA model with Box-Cox support.

    This function produces point forecasts and prediction intervals for ARIMA
    models, properly handling Box-Cox transformations when present.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model from auto_arima() or arima_rjh().
    h : int
        Number of periods to forecast (default 10).
        If xreg is provided, h is derived from xreg rows.
    xreg : DataFrame, ndarray, or None
        Future values of exogenous regressors.
        Column names must match training regressors.
    level : list of float, default None
        Confidence levels for prediction intervals (default [80, 95]).
        Values can be percentages (80, 95) or proportions (0.80, 0.95).
    fan : bool
        If True, sets level to 51:3:99 for fan plots.
    lambda_bc : float or None
        Box-Cox parameter. If None, uses model['lambda'] if present.
    biasadj : bool or None
        Bias adjustment for Box-Cox back-transformation.
        If None, uses model['biasadj'] if present.
    bootstrap : bool
        If True, compute prediction intervals via simulation (not implemented).
    npaths : int
        Number of simulation paths for bootstrap (not implemented).

    Returns
    -------
    result : dict
        Dictionary with:
        - 'mean': Point forecasts (back-transformed if Box-Cox)
        - 'lower': Lower prediction intervals (h x len(level) matrix)
        - 'upper': Upper prediction intervals (h x len(level) matrix)
        - 'level': Confidence levels used
        - 'x': Original series
        - 'fitted': In-sample fitted values
        - 'residuals': Model residuals
        - 'method': Model description

    Notes
    -----
    When Box-Cox transformation is used:
    - Point forecasts are bias-adjusted (if biasadj=True) using forecast variance
    - Prediction intervals are back-transformed without bias adjustment

    Examples
    --------
    >>> fit = auto_arima(y, m=12, lambda_bc="auto")
    >>> fc = forecast_arima(fit, h=12, level=[80, 95])
    >>> print(fc['mean'])  # Point forecasts
    >>> print(fc['lower'])  # Lower bounds for each level
    >>> print(fc['upper'])  # Upper bounds for each level

    References
    ----------
    Hyndman, R.J. & Khandakar, Y. (2008). Automatic time series forecasting:
    the forecast package for R. Journal of Statistical Software, 27(1), 1-22.
    """
    if lambda_bc is None:
        lambda_bc = model.get('lambda')
    if biasadj is None:
        biasadj = model.get('biasadj', False)

    if level is not None:
        if fan:
            levels = np.arange(51, 100, 3).tolist()
        else:
            levels = list(level)
            if min(levels) > 0 and max(levels) < 1:
                levels = [l * 100 for l in levels]
            if min(levels) < 0 or max(levels) > 99.99:
                raise ValueError("Confidence level out of range")
        levels = sorted(levels)
    else:
        levels = []

    n = len(model.get('y', model.get('x', [])))
    model_xreg = model.get('xreg')
    has_drift = (
        model_xreg is not None and
        isinstance(model_xreg, pd.DataFrame) and
        'drift' in model_xreg.columns
    )

    if xreg is not None:
        if isinstance(xreg, np.ndarray):
            xreg = pd.DataFrame(xreg)
        h = len(xreg)

        if has_drift and 'drift' not in xreg.columns:
            xreg = xreg.copy()
            xreg.insert(0, 'drift', np.arange(n + 1, n + h + 1))
    elif has_drift:
        xreg = pd.DataFrame({'drift': np.arange(n + 1, n + h + 1)})

    pred_result = predict_arima(model, n_ahead=h, newxreg=xreg, se_fit=True)
    mean = pred_result['mean']
    se = pred_result['se']

    lower = None
    upper = None
    if levels:
        z_values = [norm.ppf(0.5 + l / 200) for l in levels]
        lower = np.column_stack([mean - z * se for z in z_values])
        upper = np.column_stack([mean + z * se for z in z_values])

    if lambda_bc is not None:
        fvar = se ** 2 if biasadj else None
        mean = inv_box_cox(mean, lambda_bc, biasadj=biasadj, fvar=fvar)

        if levels:
            if lambda_bc < 0:
                asymptote = -1.0 / lambda_bc
                cap_value = asymptote * 0.95
                lower = np.clip(lower, -np.inf, cap_value)
                upper = np.clip(upper, -np.inf, cap_value)

            lower = inv_box_cox(lower, lambda_bc, biasadj=False)
            upper = inv_box_cox(upper, lambda_bc, biasadj=False)

            if lambda_bc < 0:
                stacked = np.stack([lower, upper], axis=-1)
                lower = np.min(stacked, axis=-1)
                upper = np.max(stacked, axis=-1)

    return {
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'level': levels,
        'x': model.get('y', model.get('x')),
        'fitted': model.get('fitted'),
        'residuals': model.get('residuals'),
        'method': model.get('method', 'ARIMA'),
        'lambda': lambda_bc,
        'biasadj': biasadj
    }


if __name__ == "__main__":
    print("Testing auto_arima.py...")
    print("=" * 60)

    print("\n1. Testing analyze_series...")
    x_test = np.array([np.nan, np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, np.nan])
    first, length, trimmed = analyze_series(x_test)
    print(f"   First non-missing: {first}, Series length: {length}")
    print(f"   Trimmed: {trimmed}")

    print("\n2. Testing is_constant...")
    print(f"   [1,1,1] is constant: {is_constant(np.array([1.0, 1.0, 1.0]))}")
    print(f"   [1,2,3] is constant: {is_constant(np.array([1.0, 2.0, 3.0]))}")

    print("\n3. Testing ndiffs...")
    np.random.seed(42)
    y_stationary = np.random.randn(100)
    y_nonstationary = np.cumsum(np.random.randn(100))
    print(f"   Stationary series: ndiffs = {ndiffs(y_stationary)}")
    print(f"   Non-stationary series: ndiffs = {ndiffs(y_nonstationary)}")

    print("\n4. Testing nsdiffs...")
    t = np.arange(120)
    y_seasonal = np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.3
    print(f"   Seasonal series (m=12): nsdiffs = {nsdiffs(y_seasonal, period=12)}")

    print("\n5. Testing fit_custom_arima...")
    np.random.seed(123)
    y_ar = np.zeros(200)
    for t in range(1, 200):
        y_ar[t] = 0.7 * y_ar[t-1] + np.random.randn()

    fit = fit_custom_arima(y_ar, m=1, order=(1, 0, 0), constant=True, ic="aic", trace=True)
    print(f"   AIC: {fit['aic']:.4f}, IC: {fit['ic']:.4f}")

    print("\n6. Testing auto_arima (stepwise)...")
    np.random.seed(456)
    y_test = np.zeros(150)
    for t in range(1, 150):
        y_test[t] = 0.6 * y_test[t-1] + np.random.randn()

    best = auto_arima(y_test, m=1, stepwise=True, trace=False)
    print(f"   Best model: {best['method']}")
    print(f"   Coefficients:\n{best['coef']}")
    print(f"   AICc: {best.get('aicc', 'N/A')}")

    print("\n7. Testing auto_arima on non-stationary data...")
    np.random.seed(789)
    y_rw = np.cumsum(np.random.randn(100))

    best_rw = auto_arima(y_rw, m=1, stepwise=True, trace=False)
    print(f"   Best model: {best_rw['method']}")
    print(f"   Differencing: d={best_rw['arma'][5]}")

    print("\n8. Testing auto_arima on seasonal data...")
    np.random.seed(321)
    t = np.arange(120)
    y_seas = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5

    best_seas = auto_arima(y_seas, m=12, stepwise=True, trace=False, max_P=1, max_Q=1)
    print(f"   Best model: {best_seas['method']}")
    print(f"   Seasonal: D={best_seas['arma'][6]}")

    print("\n9. Testing time_index...")
    ti = time_index(10, 4, start=1.0)
    print(f"   time_index(10, 4): {ti[:5]}...")

    print("\n10. Testing prepend_drift...")
    drift_vec = np.arange(1, 6)
    xreg_test = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [5, 4, 3, 2, 1]})
    result = prepend_drift(xreg_test, drift_vec)
    print(f"   Columns: {list(result.columns)}")
    print(f"   Drift column: {result['drift'].values}")

    print("\n11. Testing box_cox...")
    y_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_bc, lam = box_cox(y_pos, m=1, lambda_bc=0.5)
    print(f"   Lambda: {lam}")
    print(f"   Transformed (first 5): {y_bc[:5]}")

    print("\n12. Testing inv_box_cox...")
    y_back = inv_box_cox(y_bc, lam)
    print(f"   Back-transformed (first 5): {y_back[:5]}")
    print(f"   Max error: {np.max(np.abs(y_back - y_pos)):.2e}")

    print("\n13. Testing arima_rjh (basic)...")
    np.random.seed(999)
    y_drift = np.cumsum(np.random.randn(100)) + np.arange(100) * 0.1
    fit_rjh = arima_rjh(y_drift, m=1, order=(1, 1, 0), include_drift=True)
    print(f"   Coefficients:\n{fit_rjh['coef']}")
    print(f"   Has drift: {'drift' in fit_rjh.get('xreg', pd.DataFrame()).columns}")

    print("\n14. Testing arima_rjh with Box-Cox...")
    np.random.seed(888)
    y_exp = np.exp(np.cumsum(np.random.randn(80) * 0.1) + 2)
    fit_bc = arima_rjh(y_exp, m=1, order=(1, 0, 0), lambda_bc=0.0)
    print(f"   Lambda used: {fit_bc.get('lambda')}")
    print(f"   AIC: {fit_bc.get('aic', 'N/A'):.4f}")

    print("\n" + "=" * 60)
    print("All tests completed!")