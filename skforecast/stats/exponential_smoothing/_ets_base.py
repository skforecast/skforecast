################################################################################
#                                 ETS                                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal, List, Any
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm, jarque_bera, shapiro
import warnings
import math

from ...utils import check_optional_dependency
from ..transformations import box_cox_lambda

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ModuleNotFoundError as error:
    if error.name == "statsmodels":
        check_optional_dependency(package_name="statsmodels")
    raise

# The compiled functions of this module are not compiled with `fastmath`: it
# lets the compiler assume that no NaN or inf values exist, which removes the
# `np.isnan` checks of the absent parameters (NaN) and the guards of the
# objective function, and makes the results depend on the CPU.

# Usual bounds of the smoothing parameters (alpha, beta, gamma, phi), as in
# R's forecast::ets.
PARAM_LOWER = np.array([1e-4, 1e-4, 1e-4, 0.8])
PARAM_UPPER = np.array([0.9999, 0.9999, 0.9999, 0.98])

ERROR_TYPES = {"N": 0, "A": 1, "M": 2}
TREND_TYPES = {"N": 0, "A": 1, "M": 2}
SEASON_TYPES = {"N": 0, "A": 1, "M": 2}


def is_constant(y: NDArray[np.float64]) -> bool:
    """Check if series is constant"""
    return np.all(y == y[0])


def _aicc(aic: float, k: int, n: int) -> float:
    """
    AIC with the small-sample correction, using the same number of
    parameters `k` as the AIC (as in R's forecast::ets). It is infinite when
    the series is too short for the correction to be defined (n <= k + 1).
    """
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


@njit(cache=True)
def _roots_within_radius(coefs: NDArray[np.float64], radius: float) -> bool:  # pragma: no cover
    """
    Check that every root of a polynomial has a modulus <= `radius`.

    Schur-Cohn test applied to q(w) = p(radius * w): the roots of q are all
    inside the unit circle if and only if every reflection coefficient of the
    recursion has modulus below 1. Equivalent to computing the roots, without
    an eigenvalue problem (O(n²) operations). Each step is divided by the
    leading coefficient to avoid underflow.

    Parameters
    ----------
    coefs : NDArray[np.float64]
        Polynomial coefficients in ascending powers.
    radius : float
        Maximum modulus of the roots.

    Returns
    -------
    bool
        True if all the roots have a modulus <= radius.
    """
    n = len(coefs) - 1
    q = np.empty(n + 1)
    tmp = np.empty(n + 1)
    scale = 1.0
    for k in range(n + 1):
        q[k] = coefs[k] * scale
        scale *= radius
    while n > 0:
        if q[n] == 0.0:
            return False
        k_refl = q[0] / q[n]
        if abs(k_refl) >= 1.0:
            return False
        for j in range(1, n + 1):
            tmp[j - 1] = q[j] - k_refl * q[n - j]
        for j in range(n):
            q[j] = tmp[j]
        n -= 1
    return True


@njit(cache=True)
def _admissible_jit(alpha: float, beta: float, gamma: float, phi: float, m: int) -> bool:  # pragma: no cover
    TOL = 1e-8
    if phi < 0.0 or phi > 1.0 + TOL:
        return False

    if np.isnan(gamma):
        if np.isnan(alpha):
            return True

        if alpha < 1.0 - 1.0 / phi or alpha > 1.0 + 1.0 / phi:
            return False

        if not np.isnan(beta):
            if beta < alpha * (phi - 1.0) or beta > (1.0 + phi) * (2.0 - alpha):
                return False

    elif m > 1:
        if np.isnan(alpha):
            return False
        beta_val = 0.0 if np.isnan(beta) else beta
        lower_gamma = max(1.0 - 1.0 / phi - alpha, 0.0)
        upper_gamma = 1.0 + 1.0 / phi - alpha
        if gamma < lower_gamma or gamma > upper_gamma:
            return False

        alpha_lower = 1.0 - 1.0 / phi - gamma * (1.0 - m + phi + phi * m) / (2.0 * phi * m)
        if alpha < alpha_lower:
            return False

        if beta_val < -(1.0 - phi) * (gamma / m + alpha):
            return False

        a = phi * (1.0 - alpha - gamma)
        b = alpha + beta_val - alpha * phi + gamma - 1.0
        c_coef = alpha + beta_val - alpha * phi
        d = alpha + beta_val - phi

        # Characteristic polynomial of the discount matrix, in ascending
        # powers as in R's forecast::ets: a, b, (m - 2) times c, d, 1. The
        # model is admissible when no root has a modulus above 1 + 1e-10.
        if m <= 24:
            P = np.empty(m + 2, dtype=np.float64)
            P[0] = a
            P[1] = b
            for i in range(2, m):
                P[i] = c_coef
            P[m] = d
            P[m + 1] = 1.0
            if not _roots_within_radius(P, 1.0 + 1e-10):
                return False

    return True


def admissible(alpha: Optional[float],
               beta: Optional[float],
               gamma: Optional[float],
               phi: Optional[float],
               m: int) -> bool:
    alpha_val = np.nan if alpha is None else alpha
    beta_val = np.nan if beta is None else beta
    gamma_val = np.nan if gamma is None else gamma
    phi_val = 1.0 if phi is None else phi

    return _admissible_jit(alpha_val, beta_val, gamma_val, phi_val, m)


@njit(cache=True)
def _check_param_jit(alpha: float, beta: float, gamma: float, phi: float,
                     lower: NDArray[np.float64], upper: NDArray[np.float64],
                     check_usual: bool, check_admissible: bool, m: int) -> bool:  # pragma: no cover
    if check_usual:
        if not np.isnan(alpha):
            if alpha < lower[0] or alpha > upper[0]:
                return False

        if not np.isnan(beta):
            if beta < lower[1] or beta > alpha or beta > upper[1]:
                return False

        if not np.isnan(phi):
            if phi < lower[3] or phi > upper[3]:
                return False

        if not np.isnan(gamma):
            if gamma < lower[2] or gamma > 1.0 - alpha or gamma > upper[2]:
                return False

    if check_admissible:
        # A model without damping is checked with phi = 1, as in R
        phi_admissible = 1.0 if np.isnan(phi) else phi
        if not _admissible_jit(alpha, beta, gamma, phi_admissible, m):
            return False

    return True


def check_param(alpha: Optional[float],
                beta: Optional[float],
                gamma: Optional[float],
                phi: Optional[float],
                lower: NDArray[np.float64],
                upper: NDArray[np.float64],
                bounds: str,
                m: int) -> bool:
    alpha_val = np.nan if alpha is None else alpha
    beta_val = np.nan if beta is None else beta
    gamma_val = np.nan if gamma is None else gamma
    phi_val = np.nan if phi is None else phi

    check_usual = bounds != "admissible"
    check_admissible = bounds != "usual"

    return _check_param_jit(alpha_val, beta_val, gamma_val, phi_val,
                            lower, upper, check_usual, check_admissible, m)


@dataclass
class ETSConfig:
    error: Literal["A", "M"] = "A"
    trend: Literal["N", "A", "M"] = "N"
    season: Literal["N", "A", "M"] = "N"
    damped: bool = False
    m: int = 1

    @property
    def error_code(self) -> int:
        return ERROR_TYPES[self.error]

    @property
    def trend_code(self) -> int:
        return TREND_TYPES[self.trend]

    @property
    def season_code(self) -> int:
        return SEASON_TYPES[self.season]

    @property
    def n_states(self) -> int:
        n = 1
        if self.trend != "N":
            n += 1
        if self.season != "N":
            n += self.m - 1
        return n


@dataclass
class ETSParams:
    alpha: float = 0.1
    beta: float = 0.01
    gamma: float = 0.01
    phi: float = 0.98
    init_states: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def to_vector(self, config: ETSConfig) -> NDArray[np.float64]:
        params = [self.alpha]
        if config.trend != "N":
            params.append(self.beta)
        if config.season != "N":
            params.append(self.gamma)
        if config.damped:
            params.append(self.phi)
        return np.concatenate([params, self.init_states])

    @staticmethod
    def from_vector(x: NDArray[np.float64], config: ETSConfig) -> 'ETSParams':
        idx = 0
        alpha = x[idx]
        idx += 1
        beta = x[idx] if config.trend != "N" else 0.0
        if config.trend != "N":
            idx += 1
        gamma = x[idx] if config.season != "N" else 0.0
        if config.season != "N":
            idx += 1
        phi = x[idx] if config.damped else 1.0
        if config.damped:
            idx += 1
        init_states = x[idx:]
        return ETSParams(alpha, beta, gamma, phi, init_states)


@dataclass
class ETSModel:
    """Fitted ETS model"""
    config: ETSConfig
    params: ETSParams
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    states: NDArray[np.float64]
    loglik: float
    aic: float
    aicc: float
    bic: float
    sigma2: float
    y_original: Optional[NDArray[np.float64]] = None
    transform: Optional['BoxCoxTransform'] = None


@dataclass
class BoxCoxTransform:
    lambda_param: float
    shift: float = 0.0

    @staticmethod
    def find_lambda(
        y: NDArray[np.float64],
        m: int = 1,
        lambda_range: Tuple[float, float] = (-0.9, 2.0)
    ) -> float:
        """
        Select the Box-Cox lambda with Guerrero's method, as R's
        `forecast::BoxCox(lambda = "auto")`. Non-positive series are shifted
        to be positive first.
        """
        if np.any(y <= 0):
            y = y + np.abs(np.min(y)) + 1.0

        return box_cox_lambda(
            y, m=m, method="guerrero", lower=lambda_range[0], upper=lambda_range[1]
        )

    def transform(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_shifted = y + self.shift
        if abs(self.lambda_param) < 1e-10:
            return np.log(y_shifted)
        else:
            return (y_shifted ** self.lambda_param - 1) / self.lambda_param

    def inverse_transform(self, y_trans: NDArray[np.float64],
                         bias_adjust: bool = False,
                         variance: Optional[float | NDArray[np.float64]] = None) -> NDArray[np.float64]:
        if abs(self.lambda_param) < 1e-10:
            y_back = np.exp(y_trans)
            if bias_adjust and variance is not None:
                # R's InvBoxCox with lambda = 0
                y_back = y_back * (1 + variance / 2)
        else:
            # As R's InvBoxCox, values outside the range of the transformation
            # (lambda * y + 1 < 0) are NaN when lambda < 0 and keep their sign
            # otherwise, so the inverse is monotonic (interval bounds included).
            xx = self.lambda_param * np.asarray(y_trans, dtype=np.float64) + 1
            if self.lambda_param < 0:
                xx = np.where(xx < 0, np.nan, xx)
            y_back = np.sign(xx) * np.abs(xx) ** (1 / self.lambda_param)
            if bias_adjust and variance is not None:
                y_back = y_back * (
                    1 + (1 - self.lambda_param) * variance
                    / (2 * y_back ** (2 * self.lambda_param))
                )

        return y_back - self.shift


@njit(cache=True, inline="always")
def _ets_step(
    l: float, 
    b: float, 
    s: NDArray[np.float64], 
    y: float,
    m: int, 
    error: int, 
    trend: int, 
    season: int,
    alpha: float, 
    beta: float, 
    gamma: float, 
    phi: float
) -> Tuple:  # pragma: no cover
    """
    Perform one step of the ETS state space model update and forecasting.

    The seasonal states `s` are updated in place (no copy per step).

    Returns
    -------
    l_new, b_new, yhat, e : float
        Updated level and trend, one-step-ahead forecast and error.
    valid : bool
        False when a multiplicative trend has a non-positive level or trend,
        in which case the other values are meaningless.
    """
    TOL = 1e-10

    if trend == 0:
        q = l
        phib = 0.0
    elif trend == 1:
        phib = phi * b
        q = l + phib
    else:
        if b <= 0 or l <= 0:
            return l, b, 0.0, 0.0, False
        phib = b ** phi
        q = l * phib
    if season == 0:
        yhat = q
    elif season == 1:
        yhat = q + s[m - 1]
    else:
        yhat = q * s[m - 1]

    if abs(yhat) < TOL:
        yhat = TOL

    if error == 1:
        e = y - yhat
    else:
        e = (y - yhat) / yhat
    if season == 0:
        p = y
    elif season == 1:
        p = y - s[m - 1]
    else:
        p = y / max(s[m - 1], TOL)
    l_new = q + alpha * (p - q)
    b_new = b
    if trend == 1:
        r = l_new - l
        b_new = phib + (beta / alpha) * (r - phib)
    elif trend == 2:
        r = l_new / max(l, TOL)
        b_new = phib + (beta / alpha) * (r - phib)

    if season > 0:
        if season == 1:
            t = y - q
        else:
            t = y / max(q, TOL)
        new_seasonal = s[m - 1] + gamma * (t - s[m - 1])
        for i in range(m - 1, 0, -1):
            s[i] = s[i - 1]
        s[0] = new_seasonal

    return l_new, b_new, yhat, e, True


@njit(cache=True)
def _ets_likelihood(y: NDArray[np.float64], init_states: NDArray[np.float64],
                    m: int, error: int, trend: int, season: int,
                    alpha: float, beta: float, gamma: float, phi: float) -> Tuple:  # pragma: no cover
    n = len(y)
    n_states = len(init_states)

    l = init_states[0]
    b = init_states[1] if trend > 0 else 0.0
    if season > 0:
        offset_start = 1 + (1 if trend > 0 else 0)
        s = np.zeros(m)
        for j in range(m):
            s[j] = init_states[offset_start + j]
    else:
        s = np.zeros(max(m, 1))

    residuals = np.zeros(n)
    fitted = np.zeros(n)
    sum_e2 = 0.0
    sum_log_yhat = 0.0

    for i in range(n):
        l, b, yhat, e, valid = _ets_step(
            l, b, s, y[i], m, error, trend, season, alpha, beta, gamma, phi
        )

        if not valid:
            return np.inf, residuals, fitted, init_states

        fitted[i] = yhat
        residuals[i] = e
        sum_e2 += e * e
        if error == 2:
            sum_log_yhat += np.log(max(abs(yhat), 1e-10))
    if error == 1:
        loglik = n * np.log(sum_e2 / n)
    else:
        loglik = n * np.log(sum_e2 / n) + 2 * sum_log_yhat
    final_state = np.zeros(n_states)
    final_state[0] = l
    if trend > 0:
        final_state[1] = b
    if season > 0:
        offset = 1 + (1 if trend > 0 else 0)
        for j in range(m):
            final_state[offset + j] = s[j]

    return loglik, residuals, fitted, final_state


@njit(cache=True)
def _fourier_jit(n: int, period: int, K: int, h: int) -> NDArray[np.float64]:  # pragma: no cover
    if h == 0:
        n_times = n
        times = np.arange(1.0, n + 1.0)
    else:
        n_times = h
        times = np.arange(float(n + 1), float(n + h + 1))
    X = np.zeros((n_times, 2 * K), dtype=np.float64)

    TOL = 1e-10
    col_idx = 0

    for k in range(1, K + 1):
        p = float(k) / float(period)
        include_sine = np.abs(2.0 * p - np.round(2.0 * p)) > TOL

        if include_sine:
            for i in range(n_times):
                X[i, col_idx] = np.sin(2.0 * np.pi * p * times[i])
            col_idx += 1

        for i in range(n_times):
            X[i, col_idx] = np.cos(2.0 * np.pi * p * times[i])
        col_idx += 1

    return X[:, :col_idx].copy()


def fourier(x: NDArray[np.float64], period: int, K: int, h: Optional[int] = None) -> NDArray[np.float64]:
    h_val = 0 if h is None else h
    return _fourier_jit(len(x), period, K, h_val)


def init_states(y: NDArray[np.float64], config: ETSConfig) -> NDArray[np.float64]:
    n = len(y)
    m = config.m
    trendtype = config.trend
    seasontype = config.season

    if seasontype != "N":
        if n < 4:
            raise ValueError("Not enough data for seasonal model (need at least 4 observations)")

        if n < 3 * m:
            fouriery = fourier(y, period=m, K=1)
            X_fourier = np.column_stack([
                np.ones(n),
                np.arange(1, n + 1),
                fouriery
            ])
            coefs, *_ = np.linalg.lstsq(X_fourier, y, rcond=None)
            if seasontype == "A":
                seasonal = y - (coefs[0] + coefs[1] * np.arange(1, n + 1))
            else:
                if np.min(y) <= 0:
                    raise ValueError(
                        "Multiplicative seasonality not appropriate for zero/negative values"
                    )
                seasonal = y / (coefs[0] + coefs[1] * np.arange(1, n + 1))
        else:
            decomp = seasonal_decompose(
                y,
                period=m,
                model="additive" if seasontype == "A" else "multiplicative",
                extrapolate_trend='freq'
            )
            seasonal = decomp.seasonal
        init_seas = seasonal[1:m][::-1]
        if seasontype == "A":
            y_sa = y - seasonal
        else:
            init_seas = np.clip(init_seas, a_min=1e-2, a_max=None)
            if init_seas.sum() > m:
                init_seas = init_seas / np.sum(init_seas + 1e-2)
            y_sa = y / np.clip(seasonal, a_min=1e-2, a_max=None)
    else:
        m = 1
        init_seas = np.array([])
        y_sa = y
    maxn = min(max(10, 2 * m), len(y_sa))

    if trendtype == "N":
        l0 = np.mean(y_sa[:maxn])
        return np.concatenate([[l0], init_seas])
    X = np.column_stack([
        np.ones(maxn),
        np.arange(1, maxn + 1)
    ])
    (l, b), *_ = np.linalg.lstsq(X, y_sa[:maxn], rcond=None)

    if trendtype == "A":
        l0 = l
        b0 = b
        if abs(l0 + b0) < 1e-8:
            l0 = l0 * (1 + 1e-3)
            b0 = b0 * (1 - 1e-3)
    else:
        l0 = l + b
        if abs(l0) < 1e-8:
            l0 = 1e-7

        b0 = (l + 2 * b) / l0
        div = b0 if not math.isclose(b0, 0.0, abs_tol=1e-8) else 1e-8
        l0 = l0 / div
        if abs(b0) > 1e10:
            b0 = np.sign(b0) * 1e10
        if l0 < 1e-8 or b0 < 1e-8:
            l0 = max(y_sa[0], 1e-3)
            div = y_sa[0] if not math.isclose(y_sa[0], 0.0, abs_tol=1e-8) else 1e-8
            b0 = max(y_sa[1] / div, 1e-3)

    return np.concatenate([[l0, b0], init_seas])


def initial_smoothing_params(
    config: ETSConfig,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    phi: Optional[float] = None
) -> NDArray[np.float64]:
    """
    Starting values of the smoothing parameters for the optimizer.

    Follows `initparam` of R's forecast::ets: every starting value is inside
    the usual bounds and satisfies beta <= alpha and gamma <= 1 - alpha.
    Values given by the user are kept.

    Parameters
    ----------
    config : ETSConfig
        Model configuration.
    alpha, beta, gamma, phi : float, optional
        Fixed values of the smoothing parameters.

    Returns
    -------
    NDArray[np.float64]
        Values of [alpha, beta, gamma, phi].
    """
    lower, upper = PARAM_LOWER, PARAM_UPPER
    m = config.m

    if alpha is None:
        alpha = lower[0] + 0.2 * (upper[0] - lower[0]) / m
        if alpha > 1 or alpha < 0:
            alpha = lower[0] + 2e-3

    if beta is None:
        upper_beta = min(upper[1], alpha)
        beta = lower[1] + 0.1 * (upper_beta - lower[1])
        if beta < 0 or beta > alpha:
            beta = alpha - 1e-3

    if gamma is None:
        upper_gamma = min(upper[2], 1 - alpha)
        gamma = lower[2] + 0.05 * (upper_gamma - lower[2])
        if gamma < 0 or gamma > 1 - alpha:
            gamma = 1 - alpha - 1e-3

    if phi is None:
        phi = lower[3] + 0.99 * (upper[3] - lower[3])
        if phi < 0 or phi > 1:
            phi = upper[3] - 1e-3

    return np.array([alpha, beta, gamma, phi], dtype=np.float64)


def get_bounds(config: ETSConfig) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = [1e-4]
    upper = [0.9999]

    if config.trend != "N":
        lower.append(1e-4)
        upper.append(0.9999)

    if config.season != "N":
        lower.append(1e-4)
        upper.append(0.9999)

    if config.damped:
        lower.append(0.8)
        upper.append(0.98)
    n_states = config.n_states
    # The initial states are not bounded (as in R): they are in the units of
    # the series, whatever its magnitude.
    lower.extend([-np.inf] * n_states)
    upper.extend([np.inf] * n_states)

    return np.array(lower), np.array(upper)


@njit(cache=True)
def _ets_objective_jit(x: NDArray[np.float64], 
                       y: NDArray[np.float64], 
                       lower: NDArray[np.float64], 
                       upper: NDArray[np.float64],
                       par_values: NDArray[np.float64],
                       par_free: NDArray[np.bool_],
                       m: int, 
                       error_code: int, 
                       trend_code: int, 
                       season_code: int,
                       has_trend: bool, 
                       has_season: bool, 
                       is_damped: bool, 
                       is_mult_season: bool,
                       check_usual: bool, 
                       check_admissible: bool) -> float:  # pragma: no cover
    """
    Module-level JIT-compiled objective function for ETS optimization.
    
    This function is defined at module scope (not inside ets()) to enable
    proper JIT caching. When defined inside a function, numba creates a new
    cache key for each closure, causing recompilation on every call.
    
    This is critical for auto_ets() performance, which calls ets() 6-11 times
    in a loop. With nested JIT, each call recompiles (~0.5-1s overhead each).
    With module-level JIT, only the first call compiles, saving 3-10 seconds.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Vector of the estimated parameters: the free smoothing parameters
        among [alpha, beta?, gamma?, phi?] followed by the initial states.
    y : NDArray[np.float64]
        Time series observations
    lower, upper : NDArray[np.float64]
        Bounds of the elements of `x`.
    par_values : NDArray[np.float64]
        Values of the fixed smoothing parameters [alpha, beta, gamma, phi]
        (entries of the free parameters are ignored).
    par_free : NDArray[np.bool_]
        Whether each of [alpha, beta, gamma, phi] is estimated.
    m : int
        Seasonal period
    error_code, trend_code, season_code : int
        Model component codes (0=N, 1=A, 2=M)
    has_trend, has_season, is_damped : bool
        Model component flags
    is_mult_season : bool
        True if multiplicative seasonality
    check_usual, check_admissible : bool
        Parameter constraint checking flags
        
    Returns
    -------
    float
        Log-likelihood (or penalty if parameters invalid)
    """
    PENALTY = 1e10

    # Bounds checking for parameters
    for i in range(len(x)):
        if x[i] < lower[i] or x[i] > upper[i]:
            return PENALTY

    # Smoothing parameters: estimated ones from x, fixed ones from par_values
    values = par_values.copy()
    idx = 0
    for k in range(4):
        if par_free[k]:
            values[k] = x[idx]
            idx += 1

    alpha = values[0]
    beta = values[1] if has_trend else 0.0
    gamma = values[2] if has_season else np.nan
    phi = values[3] if is_damped else 1.0

    init_states = x[idx:].copy()

    # Check parameter constraints (usual bounds of [alpha, beta, gamma, phi])
    beta_check = beta if has_trend else np.nan
    phi_check = phi if is_damped else np.nan

    if not _check_param_jit(alpha, beta_check, gamma, phi_check,
                            PARAM_LOWER, PARAM_UPPER, check_usual,
                            check_admissible, m):
        return PENALTY

    # Handle seasonal component normalization
    if has_season:
        trend_slots = 1 if has_trend else 0
        seasonal_start = 1 + trend_slots
        seasonal_sum = 0.0
        for i in range(seasonal_start, len(init_states)):
            seasonal_sum += init_states[i]

        # Add extra seasonal component to ensure sum constraint
        if is_mult_season:
            extra = float(m) - seasonal_sum
        else:
            extra = -seasonal_sum

        init_states_full = np.zeros(len(init_states) + 1, dtype=np.float64)
        for i in range(len(init_states)):
            init_states_full[i] = init_states[i]
        init_states_full[len(init_states)] = extra

        # Check non-negativity for multiplicative seasonality
        if is_mult_season:
            for i in range(seasonal_start, len(init_states_full)):
                if init_states_full[i] < 0.0:
                    return PENALTY
    else:
        init_states_full = init_states

    # Compute log-likelihood
    loglik, _, _, _ = _ets_likelihood(
        y, init_states_full, m, error_code, trend_code, season_code,
        alpha, beta, gamma, phi
    )

    if np.isnan(loglik) or np.isinf(loglik):
        return PENALTY

    return loglik


# Optimization of the ETS parameters
# ------------------------------------------------------------------------------
# When the usual bounds apply and alpha is estimated, beta and gamma are
# optimized as fractions of their admissible ranges,
#     beta = 1e-4 + u_beta (alpha - 1e-4),  gamma = 1e-4 + u_gamma (1 - alpha - 1e-4)
# with u_beta, u_gamma in [0, 1], so that the usual bounds (including
# beta <= alpha and gamma <= 1 - alpha) become a box. The likelihood is
# multimodal in the smoothing parameters and the initial states: L-BFGS-B is
# run from several starting points and the best solution is refined with
# Nelder-Mead. Besides R's starting values, the starts set alpha and, when
# they are estimated, the fractions of beta and gamma (None keeps R's value).
# The likelihood often has a local optimum at each end of the range of beta
# (or gamma), so one start begins near the upper end. Known limitations: an
# optimum very close to the corner alpha = beta = 1e-4 can end at the corner,
# and a few series still end at a local optimum (MNA on the fuel consumption
# series: -2 log-likelihood 1.13 above the best of 24 random starts).

ETS_STARTS = ((0.5, None), (0.9, None), (0.5, 0.9))


@njit(cache=True)
def _ets_u_to_x(u: NDArray[np.float64], pos_beta: int, pos_gamma: int) -> NDArray[np.float64]:  # pragma: no cover
    """Map the optimization variables to the model parameters."""
    x = u.copy()
    if pos_beta >= 0:
        x[pos_beta] = 1e-4 + u[pos_beta] * (u[0] - 1e-4)
    if pos_gamma >= 0:
        x[pos_gamma] = 1e-4 + u[pos_gamma] * (1.0 - u[0] - 1e-4)
    return x


def _ets_x_to_u(x: NDArray[np.float64], pos_beta: int, pos_gamma: int) -> NDArray[np.float64]:
    """Inverse of `_ets_u_to_x`."""
    u = x.copy()
    if pos_beta >= 0:
        u[pos_beta] = (x[pos_beta] - 1e-4) / (x[0] - 1e-4)
    if pos_gamma >= 0:
        u[pos_gamma] = (x[pos_gamma] - 1e-4) / (1.0 - x[0] - 1e-4)
    return u


@njit(cache=True)
def _ets_objective_grad(
    u: NDArray[np.float64],
    upper_u: NDArray[np.float64],
    pos_beta: int,
    pos_gamma: int,
    obj_args: Tuple
) -> Tuple[float, NDArray[np.float64]]:  # pragma: no cover
    """
    Objective and forward-difference gradient in the optimization variables.

    The step of a variable goes backwards when the forward point would
    exceed its upper bound.
    """
    N = len(u)
    f0 = _ets_objective_jit(_ets_u_to_x(u, pos_beta, pos_gamma), *obj_args)
    grad = np.empty(N)
    u_step = u.copy()
    for i in range(N):
        h = 1.4901161193847656e-08 * max(1.0, abs(u[i]))
        if u[i] + h > upper_u[i]:
            h = -h
        u_step[i] = u[i] + h
        h = u_step[i] - u[i]
        f_i = _ets_objective_jit(_ets_u_to_x(u_step, pos_beta, pos_gamma), *obj_args)
        grad[i] = (f_i - f0) / h
        u_step[i] = u[i]
    return f0, grad


@njit(cache=True)
def _ets_nelder_mead(
    u0: NDArray[np.float64],
    maxiter: int,
    xtol: float,
    ftol: float,
    pos_beta: int,
    pos_gamma: int,
    obj_args: Tuple
) -> Tuple[NDArray[np.float64], float]:  # pragma: no cover
    """
    Nelder-Mead minimization of the objective in the optimization variables.

    Same algorithm as `scipy.optimize.minimize(method='Nelder-Mead',
    adaptive=True)` (Gao and Han, 2012), compiled to avoid the Python overhead
    of each evaluation. The convergence tolerances are relative: the initial
    states are in the units of the series, so absolute tolerances would never
    be met for series of large magnitude.
    """
    N = len(u0)
    rho = 1.0
    chi = 1.0 + 2.0 / N
    psi = 0.75 - 1.0 / (2.0 * N)
    sigma = 1.0 - 1.0 / N

    sim = np.empty((N + 1, N))
    sim[0] = u0
    for k in range(N):
        vertex = u0.copy()
        if vertex[k] != 0:
            vertex[k] = 1.05 * vertex[k]
        else:
            vertex[k] = 0.00025
        sim[k + 1] = vertex
    fsim = np.empty(N + 1)
    for k in range(N + 1):
        fsim[k] = _ets_objective_jit(_ets_u_to_x(sim[k], pos_beta, pos_gamma), *obj_args)
    order = np.argsort(fsim, kind="mergesort")
    fsim = fsim[order]
    sim = sim[order]

    for _ in range(1, maxiter):
        max_dx = 0.0
        max_df = 0.0
        for k in range(1, N + 1):
            max_df = max(max_df, abs(fsim[0] - fsim[k]))
            for j in range(N):
                max_dx = max(
                    max_dx, abs(sim[k, j] - sim[0, j]) / max(1.0, abs(sim[0, j]))
                )
        if max_dx <= xtol and max_df <= ftol * max(1.0, abs(fsim[0])):
            break

        xbar = np.zeros(N)
        for k in range(N):
            xbar += sim[k]
        xbar /= N

        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = _ets_objective_jit(_ets_u_to_x(xr, pos_beta, pos_gamma), *obj_args)
        shrink = False
        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = _ets_objective_jit(_ets_u_to_x(xe, pos_beta, pos_gamma), *obj_args)
            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        elif fxr < fsim[-2]:
            sim[-1] = xr
            fsim[-1] = fxr
        else:
            if fxr < fsim[-1]:
                xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                fxc = _ets_objective_jit(_ets_u_to_x(xc, pos_beta, pos_gamma), *obj_args)
                if fxc <= fxr:
                    sim[-1] = xc
                    fsim[-1] = fxc
                else:
                    shrink = True
            else:
                xcc = (1 - psi) * xbar + psi * sim[-1]
                fxcc = _ets_objective_jit(_ets_u_to_x(xcc, pos_beta, pos_gamma), *obj_args)
                if fxcc < fsim[-1]:
                    sim[-1] = xcc
                    fsim[-1] = fxcc
                else:
                    shrink = True
            if shrink:
                for j in range(1, N + 1):
                    sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                    fsim[j] = _ets_objective_jit(_ets_u_to_x(sim[j], pos_beta, pos_gamma), *obj_args)
        order = np.argsort(fsim, kind="mergesort")
        sim = sim[order]
        fsim = fsim[order]

    return sim[0].copy(), fsim[0]


def _optimize_ets(
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    pos_beta: int,
    pos_gamma: int,
    alpha_free: bool,
    obj_args: Tuple
) -> Tuple[NDArray[np.float64], float]:
    """
    Minimize the ETS objective.

    L-BFGS-B (with the gradient computed in compiled code) is run from the
    starting values `x0` and, when alpha is estimated, from the starting
    points of `ETS_STARTS`. The best solution is refined with
    Nelder-Mead, restarted while it improves by more than a relative 1e-6.

    Parameters
    ----------
    x0 : NDArray[np.float64]
        Starting values of the estimated parameters and initial states.
    lower, upper : NDArray[np.float64]
        Bounds of the elements of `x0`.
    pos_beta, pos_gamma : int
        Positions of beta and gamma in `x0` when they are optimized as
        fractions of their range (-1 otherwise).
    alpha_free : bool
        Whether alpha is estimated (first element of `x0`).
    obj_args : tuple
        Arguments of `_ets_objective_jit` after the parameter vector.

    Returns
    -------
    x : NDArray[np.float64]
        Estimated parameters and initial states.
    fun : float
        Objective value at `x`.
    """
    u0 = _ets_x_to_u(x0, pos_beta, pos_gamma)
    lower_u = lower.copy()
    upper_u = upper.copy()
    for pos in (pos_beta, pos_gamma):
        if pos >= 0:
            lower_u[pos] = 0.0
            upper_u[pos] = 1.0
    bounds_u = list(zip(lower_u, upper_u))

    starts = [u0]
    if alpha_free:
        for alpha_start, fraction_start in ETS_STARTS:
            u_start = u0.copy()
            u_start[0] = min(max(alpha_start, lower_u[0]), upper_u[0])
            if fraction_start is not None:
                if pos_beta < 0 and pos_gamma < 0:
                    continue
                for pos in (pos_beta, pos_gamma):
                    if pos >= 0:
                        u_start[pos] = fraction_start
            starts.append(u_start)

    best_u, best_f = u0, np.inf
    for u_start in starts:
        result = minimize(
            _ets_objective_grad, u_start,
            args=(upper_u, pos_beta, pos_gamma, obj_args),
            jac=True, method="L-BFGS-B", bounds=bounds_u,
            options={
                "maxiter": 5000, "maxfun": 200000, "maxcor": 20,
                "ftol": 1e-13, "gtol": 1e-9
            }
        )
        if result.fun < best_f - 1e-10 * max(1.0, abs(result.fun)):
            best_u, best_f = result.x, float(result.fun)

    # Nelder-Mead moves along the bounds and the admissibility boundary,
    # where the gradient-based search stops early. It is restarted (with a
    # new simplex) while each run improves the objective noticeably.
    for _ in range(20):
        u_nm, f_nm = _ets_nelder_mead(
            best_u, 2000, 1e-10, 1e-10, pos_beta, pos_gamma, obj_args
        )
        improved = best_f - f_nm > 1e-6 * max(1.0, abs(f_nm))
        if f_nm < best_f:
            best_u, best_f = u_nm, float(f_nm)
        if not improved:
            break

    return _ets_u_to_x(best_u, pos_beta, pos_gamma), best_f


def ets(y: NDArray[np.float64],
        m: int = 1,
        model: str = "ANN",
        damped: bool = False,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        phi: Optional[float] = None,
        lambda_param: Optional[float] = None,
        lambda_auto: bool = False,
        bias_adjust: bool = False,
        bounds: str = "both") -> ETSModel:
    """
    Fit ETS model using scipy optimization

    Parameters
    ----------
    y : array_like
        Time series data
    m : int
        Seasonal period
    model : str
        Three-letter model specification (e.g., "ANN", "AAA", "MAM")
        First letter: Error (A=Additive, M=Multiplicative)
        Second letter: Trend (N=None, A=Additive, M=Multiplicative)
        Third letter: Season (N=None, A=Additive, M=Multiplicative)
    damped : bool
        Whether to use damped trend
    alpha, beta, gamma, phi : float, optional
        Fixed parameter values (if None, will be estimated)
    lambda_param : float, optional
        Box-Cox transformation parameter. If None, no transformation
    lambda_auto : bool
        If True, automatically select optimal lambda
    bias_adjust : bool
        Apply bias adjustment when back-transforming forecasts
    bounds : str
        Parameter bounds type: "usual", "admissible", or "both" (default)

    Returns
    -------
    ETSModel
        Fitted model
    """
    y = np.asarray(y, dtype=np.float64)
    y_original = y.copy()
    n = len(y)

    if model == "ZZZ" and is_constant(y):
        warnings.warn("Series is constant. Fitting simple exponential smoothing with alpha=0.99999")
        config = ETSConfig(error="A", trend="N", season="N", damped=False, m=1)
        alpha_const = 0.99999
        l0 = y[0]

        fitted = np.full(n, y[0])
        residuals = np.zeros(n)

        k_const = 3
        return ETSModel(
            config=config,
            params=ETSParams(alpha=alpha_const, beta=0.0, gamma=0.0, phi=1.0,
                           init_states=np.array([l0])),
            fitted=fitted,
            residuals=residuals,
            states=np.array([l0]),
            loglik=0.0,
            aic=2 * k_const,
            aicc=_aicc(2 * k_const, k_const, n),
            bic=k_const * np.log(n),
            sigma2=0.0,
            y_original=y_original,
            transform=None
        )

    if n < 1:
        raise ValueError(f"Need at least 1 observation to fit ETS model, got {n}")

    if len(model) != 3:
        raise ValueError(f"Model must be 3 characters (e.g., 'AAN', 'MAM'), got '{model}'")

    if model != "ZZZ":
        if (
            model[0] not in ("A", "M", "Z")
            or model[1] not in ("N", "A", "M", "Z")
            or model[2] not in ("N", "A", "M", "Z")
        ):
            raise ValueError(
                f"Invalid model '{model}'. The error component must be 'A' or 'M', "
                f"and the trend and seasonal components must be 'N', 'A' or 'M' "
                f"(uppercase), or use model='ZZZ' for automatic selection."
            )
        if "Z" in model:
            raise ValueError(
                f"Partial automatic model specifications such as '{model}' are not "
                f"supported. Use model='ZZZ' (or None) for automatic selection and "
                f"restrict the search with `seasonal`, `trend`, `damped`, "
                f"`allow_multiplicative` and `allow_multiplicative_trend` (for "
                f"example, model='ZZZ' with seasonal=False instead of 'ZZN'), or "
                f"specify all three components."
            )

    # Handle ZZZ with high frequency by calling auto_ets
    if model == "ZZZ":
        if m > 24:
            warnings.warn(
                f"Frequency too high (m={m} > 24). Using auto_ets to select non-seasonal model. "
                f"Try stlf() if you need seasonal forecasts."
            )
        return auto_ets(
            y_original, m=m, seasonal=m <= 24, trend=None, damped=damped,
            ic="aicc", allow_multiplicative=True,
            allow_multiplicative_trend=False,
            lambda_param=lambda_param, lambda_auto=lambda_auto,
            bias_adjust=bias_adjust, verbose=False
        )

    season_type = model[2]
    if season_type != "N" and m > 24:
        raise ValueError(
            f"Frequency too high (m={m} > 24). "
            f"Seasonal models are not supported for m>24. "
            f"Use model='ZZZ' for automatic non-seasonal model selection."
        )

    if season_type != "N" and m > 1 and n < m:
        raise ValueError(
            f"Cannot fit seasonal model: need at least m={m} observations for seasonal period, but got n={n}. "
            f"R would drop seasonality and fit {model[:2]}N instead. "
            f"Either provide more data or use a non-seasonal model."
        )

    transform = None
    if lambda_auto:
        shift = np.abs(np.min(y)) + 1.0 if np.any(y <= 0) else 0.0
        lambda_opt = BoxCoxTransform.find_lambda(y, m=m)
        transform = BoxCoxTransform(lambda_opt, shift)
        y = transform.transform(y)
    elif lambda_param is not None:
        shift = np.abs(np.min(y)) + 1.0 if np.any(y <= 0) else 0.0
        transform = BoxCoxTransform(lambda_param, shift)
        y = transform.transform(y)

    if len(model) != 3:
        raise ValueError("Model must be 3 characters (e.g., 'ANN', 'AAA')")

    if "M" in model and np.min(y) <= 0:
        raise ValueError(
            f"Inappropriate model '{model}' for data with negative or zero values: "
            f"multiplicative components require a strictly positive series."
        )

    config = ETSConfig(
        error=model[0],
        trend=model[1],
        season=model[2],
        damped=damped,
        m=m
    )

    npars = 2
    if config.trend != "N":
        npars += 2
    if config.season != "N":
        npars += m
    if damped:
        npars += 1

    if n <= npars + 4:
        if damped:
            warnings.warn(
                f"Not enough data ({n} obs) for {npars} parameters with damping. "
                f"Disabling damping."
            )
            damped = False
            config = ETSConfig(
                error=config.error,
                trend=config.trend,
                season=config.season,
                damped=False,
                m=m
            )
            npars -= 1

        if n <= npars + 4 and config.season != "N":
            warnings.warn(
                f"Not enough data ({n} obs) for {npars} parameters. "
                f"Trying simpler model without seasonality."
            )
            config = ETSConfig(
                error=config.error,
                trend=config.trend,
                season="N",
                damped=False,
                m=1
            )
            npars = 2
            if config.trend != "N":
                npars += 2

        if n <= npars + 4 and config.trend != "N":
            warnings.warn(
                f"Not enough data ({n} obs) for {npars} parameters. "
                f"Trying simple exponential smoothing (ANN)."
            )
            config = ETSConfig(
                error="A",
                trend="N",
                season="N",
                damped=False,
                m=1
            )
            npars = 2

        if n <= npars + 4:
            raise ValueError(
                f"Not enough data: {n} observations for {npars} parameters. "
                f"Need at least {npars + 5} observations."
            )

    init_state_vec = init_states(y, config)

    check_usual = (bounds != "admissible")
    check_admissible = (bounds != "usual")
    has_trend = config.trend != "N"
    has_season = config.season != "N"
    is_mult_season = config.season == "M"

    # Smoothing parameters [alpha, beta, gamma, phi]: the ones given by the
    # user are fixed, the rest of the parameters of the model are estimated.
    present = np.array([True, has_trend, has_season, config.damped])
    given = [alpha, beta, gamma, phi]
    par_free = np.array([present[k] and given[k] is None for k in range(4)])
    if any(present[k] and given[k] is not None for k in range(4)):
        # Admissibility involves all the parameters, so it can only be
        # checked here when none is estimated (otherwise the optimizer
        # enforces it).
        if not par_free.any():
            bounds_fixed = bounds
        else:
            bounds_fixed = "usual" if check_usual else None
        fixed_ok = bounds_fixed is None or check_param(
            alpha,
            beta if has_trend else None,
            gamma if has_season else None,
            phi if config.damped else None,
            PARAM_LOWER, PARAM_UPPER, bounds_fixed, config.m
        )
        if not fixed_ok:
            raise ValueError(
                "The fixed smoothing parameters are out of range for the "
                f"'{bounds}' bounds (usual bounds: 1e-4 <= alpha <= 0.9999, "
                "1e-4 <= beta <= alpha, 1e-4 <= gamma <= 1 - alpha, "
                "0.8 <= phi <= 0.98)."
            )
    par_values = initial_smoothing_params(config, alpha, beta, gamma, phi)

    # Bounds of the estimated parameters followed by the initial states
    lower_all, upper_all = get_bounds(config)
    n_smooth = int(np.sum(present))
    keep = np.concatenate([par_free[present], np.ones(len(lower_all) - n_smooth, dtype=bool)])
    lower = lower_all[keep]
    upper = upper_all[keep]
    n_free = int(np.sum(par_free))

    # With the usual bounds, beta <= alpha and gamma <= 1 - alpha are made box
    # constraints: through the reparameterization when alpha is estimated,
    # or directly in their bounds when alpha is fixed.
    pos_beta = pos_gamma = -1
    if check_usual:
        free_names = [name for name, free in zip(("alpha", "beta", "gamma", "phi"), par_free) if free]
        if par_free[0]:
            pos_beta = free_names.index("beta") if "beta" in free_names else -1
            pos_gamma = free_names.index("gamma") if "gamma" in free_names else -1
            # A fixed beta or gamma bounds alpha instead
            if has_trend and not par_free[1]:
                lower[0] = max(lower[0], par_values[1])
            if has_season and not par_free[2]:
                upper[0] = min(upper[0], 1.0 - par_values[2])
        else:
            if "beta" in free_names:
                i = free_names.index("beta")
                upper[i] = min(upper[i], par_values[0])
            if "gamma" in free_names:
                i = free_names.index("gamma")
                upper[i] = min(upper[i], 1.0 - par_values[0])
        if np.any(lower[:n_free] > upper[:n_free]):
            fixed = ", ".join(
                f"{name}={value}"
                for name, value, used in zip(("alpha", "beta", "gamma"), given, present)
                if used and value is not None
            )
            raise ValueError(
                "No value of the estimated smoothing parameters satisfies the "
                f"usual bounds with the fixed {fixed} (1e-4 <= beta <= alpha "
                "and 1e-4 <= gamma <= 1 - alpha)."
            )
        if par_free[0]:
            par_values[0] = min(max(par_values[0], lower[0]), upper[0])

    obj_args = (
        y, lower, upper, par_values, par_free, config.m,
        config.error_code, config.trend_code, config.season_code,
        has_trend, has_season, config.damped, is_mult_season,
        check_usual, check_admissible,
    )
    x0 = np.concatenate([par_values[par_free], init_state_vec])
    x_opt, _ = _optimize_ets(
        x0, lower, upper, pos_beta, pos_gamma, bool(par_free[0]), obj_args
    )

    values = par_values.copy()
    values[par_free] = x_opt[:n_free]
    # With fixed parameters, the bounds may leave no value of the estimated
    # ones (for example, no admissible alpha for the fixed beta and gamma):
    # the optimizer only sees the penalty and returns a point out of range.
    # As R's check.param, the fit is refused.
    if (present & ~par_free).any() and not check_param(
        values[0],
        values[1] if has_trend else None,
        values[2] if has_season else None,
        values[3] if config.damped else None,
        PARAM_LOWER, PARAM_UPPER, bounds, config.m
    ):
        fixed = ", ".join(
            f"{name}={value}"
            for name, value, used in zip(("alpha", "beta", "gamma", "phi"), given, present)
            if used and value is not None
        )
        raise ValueError(
            f"No value of the estimated smoothing parameters satisfies the "
            f"'{bounds}' bounds with the fixed {fixed}."
        )
    full_x = np.concatenate([values[present], x_opt[n_free:]])
    fitted_params = ETSParams.from_vector(full_x, config)

    init_states_final = fitted_params.init_states.copy()
    if config.season != "N":
        trend_slots = 1 if config.trend != "N" else 0
        seasonal_start = 1 + trend_slots
        seasonal_sum = np.sum(init_states_final[seasonal_start:])
        if config.season == "M":
            extra = config.m - seasonal_sum
        else:
            extra = -seasonal_sum
        init_states_final = np.append(init_states_final, extra)

    loglik, residuals, fitted_vals, final_states = _ets_likelihood(
        y, init_states_final,
        config.m, config.error_code, config.trend_code, config.season_code,
        fitted_params.alpha, fitted_params.beta, fitted_params.gamma, fitted_params.phi
    )

    n_params = len(x_opt)
    k = n_params + 1
    aic = loglik + 2 * k
    aicc = _aicc(aic, k, n)
    bic = loglik + k * np.log(n)
    sigma2 = np.sum(residuals ** 2) / (n - n_params)

    fitted_original = fitted_vals
    if transform is not None:
        fitted_original = transform.inverse_transform(fitted_vals, bias_adjust, sigma2)

    return ETSModel(
        config=config,
        params=fitted_params,
        fitted=fitted_original,
        residuals=y_original - fitted_original,
        states=final_states,
        loglik=-0.5 * loglik,
        aic=aic,
        aicc=aicc,
        bic=bic,
        sigma2=sigma2,
        y_original=y_original,
        transform=transform
    )


@njit(cache=True)
def _forecast_ets(
    l: float, 
    b: float, 
    s: NDArray[np.float64],
    h: int, 
    m: int, 
    trend: int, 
    season: int, 
    phi: float
) -> NDArray[np.float64]:  # pragma: no cover
    """Generate h-step ahead forecasts"""
    forecasts = np.zeros(h)
    phi_sum = phi

    for i in range(h):
        if trend == 0:
            fc = l
        elif trend == 1:
            fc = l + phi_sum * b
        else:
            if b <= 0 or l <= 0:
                fc = np.nan
            else:
                fc = l * (b ** phi_sum)

        s_idx = (m - 1 - i) % m if m > 0 else 0
        if season == 1:
            fc += s[s_idx]
        elif season == 2:
            fc *= s[s_idx]

        forecasts[i] = fc

        if i < h - 1:
            phi_sum += phi ** (i + 2)

    return forecasts


def _compute_prediction_variance(model: ETSModel, h: int) -> NDArray[np.float64]:
    """
    Compute analytical prediction variance for ETS models

    Uses analytical formulas for Class 1 and Class 2 models (Hyndman et al. 2008)
    Falls back to simulation for complex models
    """
    sigma = model.sigma2
    m = model.config.m
    error = model.config.error
    trend = model.config.trend
    season = model.config.season
    damped = model.config.damped

    alpha = model.params.alpha
    beta = model.params.beta
    gamma = model.params.gamma
    phi = model.params.phi

    steps = np.arange(1, h + 1)

    if error == "A":
        if trend == "N" and season == "N":
            var = sigma * (1 + alpha**2 * (steps - 1))

        elif trend == "A" and season == "N" and not damped:
            var = sigma * (1 + (steps - 1) * (alpha**2 + alpha * beta * steps +
                          (1 / 6) * beta**2 * steps * (2 * steps - 1)))

        elif trend == "A" and season == "N" and damped:
            exp1 = (beta * phi * steps) / (1 - phi)**2
            exp2 = 2 * alpha * (1 - phi) + beta * phi
            exp3 = (beta * phi * (1 - phi**steps)) / ((1 - phi)**2 * (1 - phi**2))
            exp4 = 2 * alpha * (1 - phi**2) + beta * phi * (1 + 2 * phi - phi**steps)
            var = sigma * (1 + alpha**2 * (steps - 1) + exp1 * exp2 - exp3 * exp4)

        elif trend == "N" and season == "A":
            hm = np.floor((steps - 1) / m)
            var = sigma * (1 + alpha**2 * (steps - 1) + gamma * hm * (2 * alpha + gamma))

        elif trend == "A" and season == "A" and not damped:
            hm = np.floor((steps - 1) / m)
            exp1 = alpha**2 + alpha * beta * steps + (1 / 6) * beta**2 * steps * (2 * steps - 1)
            exp2 = 2 * alpha + gamma + beta * m * (hm + 1)
            var = sigma * (1 + (steps - 1) * exp1 + gamma * hm * exp2)

        else:
            var = None

    elif trend in ("N", "A") and season in ("N", "A"):
        var = _multiplicative_error_variance(model, h)

    else:
        var = None

    return var


def _multiplicative_error_variance(model: ETSModel, h: int) -> NDArray[np.float64]:
    """
    Forecast variance of the ETS models with multiplicative errors and
    additive (or no) trend and seasonality (class 2 in Hyndman et al. 2008,
    Section 6.4), as R's forecast.ets.

    With the state-space form x_t = F x_{t-1} + g e_t and forecast
    mu_h = w F^(h-1) x_n, c_j = w F^(j-1) g, theta_1 = mu_1^2 and
    theta_h = mu_h^2 + sigma2 * sum_{j=1}^{h-1} c_j^2 theta_{h-j}, the
    variance is (1 + sigma2) theta_h - mu_h^2.
    """
    sigma2 = model.sigma2
    m = model.config.m
    has_trend = model.config.trend == "A"
    has_season = model.config.season == "A"
    phi = model.params.phi if model.config.damped else 1.0
    states = np.asarray(model.states, dtype=float)
    p = 1 + has_trend + (m if has_season else 0)
    states = states[:p]

    w = np.zeros(p)
    F = np.zeros((p, p))
    g = np.zeros(p)
    w[0] = 1.0
    F[0, 0] = 1.0
    g[0] = model.params.alpha
    if has_trend:
        w[1] = phi
        F[0, 1] = F[1, 1] = phi
        g[1] = model.params.beta
    if has_season:
        s0 = 1 + has_trend
        w[p - 1] = 1.0
        F[s0, p - 1] = 1.0
        F[s0 + 1:, s0:p - 1] = np.eye(m - 1)
        g[s0] = model.params.gamma

    mu = np.zeros(h)
    c = np.zeros(h)
    Fj = np.eye(p)
    for j in range(h):
        mu[j] = w @ Fj @ states
        c[j] = w @ Fj @ g
        Fj = Fj @ F

    theta = np.zeros(h)
    theta[0] = mu[0]**2
    for j in range(1, h):
        theta[j] = mu[j]**2 + sigma2 * np.sum(c[:j]**2 * theta[j - 1::-1])

    return (1 + sigma2) * theta - mu**2


def forecast_ets(model: ETSModel, h: int = 10, bias_adjust: bool = True,
                level: Optional[List[float]] = None) -> Dict[str, NDArray[np.float64]]:
    """
    Generate forecasts from fitted ETS model with optional prediction intervals

    Parameters
    ----------
    model : ETSModel
        Fitted model
    h : int
        Forecast horizon
    bias_adjust : bool
        Apply bias adjustment if Box-Cox transformation was used
    level : list of float, optional
        Confidence levels for prediction intervals (e.g., [80, 95])
        If None, only return point forecasts

    Returns
    -------
    dict
        Dictionary with:
        - 'mean': Point forecasts
        - 'lower_XX': Lower bounds for XX% intervals (if level provided)
        - 'upper_XX': Upper bounds for XX% intervals (if level provided)
    
    """

    l = model.states[0]
    b = model.states[1] if model.config.trend != "N" else 0.0

    if model.config.season != "N":
        s_start = 1 + (1 if model.config.trend != "N" else 0)
        s = model.states[s_start:]
    else:
        s = np.zeros(1)

    forecasts = _forecast_ets(
        l, b, s, h,
        model.config.m,
        model.config.trend_code,
        model.config.season_code,
        model.params.phi
    )

    # The variance of the forecasts on the scale of the model is needed for
    # the intervals and, with a Box-Cox transformation, for the bias
    # adjustment of the point forecasts. It is analytical when available
    # and estimated from simulated paths otherwise.
    var = None
    simulations = None
    simulation_error = None
    need_var = level is not None or (model.transform is not None and bias_adjust)
    if need_var and model.sigma2 > 0:
        var = _compute_prediction_variance(model, h)
        if var is None:
            try:
                simulations = simulate_ets(model, h=h, n_sim=1000)
            except ValueError as e:
                simulation_error = e

    # As R's forecast.ets, the bias adjustment uses the variance of each
    # horizon (InvBoxCox with the forecast variance). Without an analytical
    # variance, it is the one implied by the 95% simulated interval, as R
    # derives it from the interval bounds (R uses the widest requested level;
    # a fixed level keeps the point forecasts independent of `level`).
    forecasts_model_scale = forecasts
    if model.transform is not None:
        fvar = None
        if bias_adjust:
            if var is not None:
                fvar = var
            elif simulations is not None:
                lv = 95.0
                z = norm.ppf(0.5 + lv / 200)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    width = (
                        np.nanpercentile(simulations, 50 + lv / 2, axis=0)
                        - np.nanpercentile(simulations, 50 - lv / 2, axis=0)
                    )
                fvar = (width / (2 * z)) ** 2
                # Horizons where every simulated path is invalid are not adjusted
                fvar = np.where(np.isfinite(fvar), fvar, 0.0)
        forecasts = model.transform.inverse_transform(forecasts, bias_adjust, fvar)

    # Prediction intervals are computed on the scale of the model and their
    # bounds are back-transformed (quantiles are preserved by the monotonic
    # Box-Cox transformation, so no bias adjustment applies to them).
    def to_original_scale(values):
        if model.transform is None:
            return values
        return model.transform.inverse_transform(values)

    result = {'mean': forecasts}

    if level is not None:
        if model.sigma2 <= 0:
            warnings.warn(
                f"Cannot compute prediction intervals: model has invalid residual variance "
                f"(sigma2={model.sigma2:.2e}). This usually means the model is overfit or "
                f"there is insufficient data. Returning point forecasts only.",
                UserWarning
            )
            return result

        if var is not None:
            for lv in level:
                z = norm.ppf(0.5 + lv / 200)
                std = np.sqrt(var)
                result[f'lower_{int(lv)}'] = to_original_scale(forecasts_model_scale - z * std)
                result[f'upper_{int(lv)}'] = to_original_scale(forecasts_model_scale + z * std)
        elif simulations is not None:
            for lv in level:
                result[f'lower_{int(lv)}'] = to_original_scale(
                    np.nanpercentile(simulations, 50 - lv / 2, axis=0)
                )
                result[f'upper_{int(lv)}'] = to_original_scale(
                    np.nanpercentile(simulations, 50 + lv / 2, axis=0)
                )
        else:
            warnings.warn(
                f"Cannot compute prediction intervals via simulation: {str(simulation_error)}. "
                f"Returning point forecasts only.",
                UserWarning
            )

    return result


@njit(cache=True)
def _simulate_ets_jit(
    l0: float,
    b0: float,
    s0: NDArray[np.float64],
    errors: NDArray[np.float64],
    m: int,
    error: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float
) -> NDArray[np.float64]:  # pragma: no cover
    """
    Simulate future sample paths of an ETS model.

    Parameters
    ----------
    l0, b0 : float
        Final level and trend of the fitted model.
    s0 : NDArray[np.float64]
        Final seasonal states of the fitted model.
    errors : NDArray[np.float64]
        Innovations of shape (n_sim, h).
    m, error, trend, season : int
        Seasonal period and component codes (0=N, 1=A, 2=M).
    alpha, beta, gamma, phi : float
        Smoothing parameters.

    Returns
    -------
    NDArray[np.float64]
        Simulated paths of shape (n_sim, h). A path whose multiplicative trend
        becomes non-positive is NaN from that step on.
    """
    n_sim, h = errors.shape
    simulations = np.full((n_sim, h), np.nan)
    s = np.empty(len(s0))
    for i in range(n_sim):
        l = l0
        b = b0
        s[:] = s0
        for t in range(h):
            fc = _forecast_ets(l, b, s, 1, m, trend, season, phi)[0]
            if np.isnan(fc):
                break
            if error == 1:
                y_new = fc + errors[i, t]
            else:
                y_new = fc * (1.0 + errors[i, t])
            simulations[i, t] = y_new
            l, b, _, _, valid = _ets_step(
                l, b, s, y_new, m, error, trend, season, alpha, beta, gamma, phi
            )
            if not valid:
                break

    return simulations


def simulate_ets(
    model: ETSModel,
    h: int = 10,
    n_sim: int = 1000,
    random_state: int = 123
) -> NDArray[np.float64]:
    """
    Simulate future sample paths from a fitted ETS model.

    The paths are on the scale on which the model was estimated (after the
    Box-Cox transformation, if any).

    Parameters
    ----------
    model : ETSModel
        Fitted model.
    h : int, default 10
        Forecast horizon.
    n_sim : int, default 1000
        Number of simulated paths.
    random_state : int, default 123
        Seed of the random number generator, so the simulations are
        reproducible.

    Returns
    -------
    NDArray[np.float64]
        Simulated paths of shape (n_sim, h).
    """
    if model.sigma2 <= 0:
        raise ValueError(
            f"Cannot simulate: model has invalid residual variance (sigma2={model.sigma2:.2e}). "
            f"This usually means the model is overfit or there is insufficient data."
        )

    rng = np.random.default_rng(random_state)
    errors = rng.normal(loc=0.0, scale=np.sqrt(model.sigma2), size=(n_sim, h))

    l = model.states[0]
    b = model.states[1] if model.config.trend != "N" else 0.0
    if model.config.season != "N":
        s_start = 1 + (1 if model.config.trend != "N" else 0)
        s = model.states[s_start:].astype(np.float64)
    else:
        s = np.zeros(max(model.config.m, 1))

    return _simulate_ets_jit(
        l, b, s, errors,
        model.config.m,
        model.config.error_code,
        model.config.trend_code,
        model.config.season_code,
        model.params.alpha,
        model.params.beta,
        model.params.gamma,
        model.params.phi
    )


def auto_ets(
    y: NDArray[np.float64],
    m: int = 1,
    seasonal: bool = True,
    trend: Optional[bool] = None,
    damped: Optional[bool] = None,
    ic: Literal["aic", "aicc", "bic"] = "aicc",
    allow_multiplicative: bool = True,
    allow_multiplicative_trend: bool = False,
    lambda_auto: bool = False,
    max_models: Optional[int] = None,
    verbose: bool = False,
    lambda_param: Optional[float] = None,
    bias_adjust: bool = False
) -> ETSModel:
    """
    Automatic ETS model selection

    Parameters
    ----------
    y : array_like
        Time series data
    m : int
        Seasonal period
    seasonal : bool
        Allow seasonal models
    trend : bool, optional
        If None, try both with and without trend. If True, only trending models. If False, only non-trending models
    damped : bool, optional
        If None, try both damped and non-damped. If True/False, only try that variant
    ic : str
        Information criterion for model selection ("aic", "aicc", "bic")
    allow_multiplicative : bool
        Allow multiplicative error and season models (default True)
    allow_multiplicative_trend : bool
        Allow multiplicative trend models (default False, matches Julia/R)
        More conservative as multiplicative trend can be unstable
    lambda_auto : bool
        Automatically select Box-Cox transformation
    max_models : int, optional
        Maximum number of models to try (None = try all)
    verbose : bool
        Print progress
    lambda_param : float, optional
        Box-Cox transformation parameter. If None, no transformation unless
        `lambda_auto` is True.
    bias_adjust : bool
        Apply bias adjustment when back-transforming the fitted values.

    Returns
    -------
    ETSModel
        Best model according to information criterion
    """
    n = len(y)
    if n < 1:
        raise ValueError(f"Need at least 1 observation, got {n}")

    # Multiplicative components need a positive series, and only additive
    # models are considered on the Box-Cox scale (as in R)
    if np.min(y) <= 0 or lambda_auto or lambda_param is not None:
        allow_multiplicative = False
        allow_multiplicative_trend = False

    has_trend = False
    if trend is None:
        mid = len(y) // 2
        first_half_mean = np.mean(y[:mid])
        second_half_mean = np.mean(y[mid:])
        pct_change = abs(second_half_mean - first_half_mean) / first_half_mean
        has_trend = pct_change > 0.10
        if verbose and has_trend:
            print(f"Trend detected: {pct_change:.1%} change from first to second half")

    error_types = ["A", "M"] if allow_multiplicative else ["A"]

    if trend is None:
        if has_trend:
            trend_types = ["A"]
            if allow_multiplicative_trend:
                trend_types.append("M")
            trend_types.append("N")
        else:
            trend_types = ["N", "A"]
            if allow_multiplicative_trend:
                trend_types.append("M")
    elif trend:
        trend_types = ["A"]
        if allow_multiplicative_trend:
            trend_types.append("M")
    else:
        trend_types = ["N"]

    if m == 1:
        season_types = ["N"]
    elif not seasonal:
        season_types = ["N"]
    elif m > 24:
        season_types = ["N"]
        if verbose:
            print(f"Frequency too high (m={m} > 24), trying non-seasonal models only")
    elif n < m:
        season_types = ["N"]
        if verbose:
            print(f"Insufficient data for seasonality (n={n} < m={m}), trying non-seasonal models only")
    else:
        if allow_multiplicative:
            season_types = ["A", "M"]
        else:
            season_types = ["A"]

    damped_opts = [True, False] if damped is None else [damped]

    models_to_try = []
    for e in error_types:
        for t in trend_types:
            for s in season_types:
                for d in damped_opts:
                    if t == "N" and d:
                        continue

                    if e == "A" and (t == "M" or s == "M"):
                        continue
                    if e == "M" and t == "M" and s == "A":
                        continue

                    models_to_try.append((f"{e}{t}{s}", d))

    if max_models is not None and len(models_to_try) > max_models:
        if m > 1:
            models_to_try = sorted(models_to_try, key=lambda x: (x[0][2] == 'N', x[1], x[0].count('M')))
        else:
            models_to_try = sorted(models_to_try, key=lambda x: (x[1], x[0].count('M')))
        models_to_try = models_to_try[:max_models]

    if verbose:
        print(f"Trying {len(models_to_try)} models...")

    def format_model_name(model_spec: str, damped: bool) -> str:
        """Format model name with proper ETS notation (e.g., MAdM instead of MAMd)"""
        if damped and model_spec[1] != "N":
            return f"{model_spec[0]}{model_spec[1]}d{model_spec[2]}"
        return model_spec

    best_model = None
    best_ic_value = np.inf
    best_ic_original = np.inf
    results = []

    for model_spec, damped_flag in models_to_try:
        try:
            model = ets(
                y, m=m, model=model_spec, damped=damped_flag,
                lambda_param=lambda_param, lambda_auto=lambda_auto,
                bias_adjust=bias_adjust, bounds="both"
            )

            if ic == "aic":
                ic_value = model.aic
            elif ic == "aicc":
                ic_value = model.aicc
            else:
                ic_value = model.bic

            ic_value_adj = ic_value
            model_name = format_model_name(model_spec, damped_flag)

            if has_trend and model.config.trend == "N":
                ic_value_adj = ic_value + 5.0
                if verbose:
                    print(f"  {model_name:5s}: {ic.upper()}={ic_value:.2f} (penalized: {ic_value_adj:.2f})")
            elif verbose:
                print(f"  {model_name:5s}: {ic.upper()}={ic_value:.2f}")

            results.append((model_spec, damped_flag, ic_value, model))

            if ic_value_adj < best_ic_value:
                best_ic_value = ic_value_adj
                best_ic_original = ic_value
                best_model = model

        except Exception as e:
            if verbose:
                model_name = format_model_name(model_spec, damped_flag)
                print(f"  {model_name:5s}: Failed ({str(e)})")
            continue

    if best_model is None:
        raise ValueError("No model could be fitted successfully")

    if verbose:
        best_model_name = format_model_name(
            f"{best_model.config.error}{best_model.config.trend}{best_model.config.season}",
            best_model.config.damped
        )
        print(f"\nBest model: {best_model_name} ({ic.upper()}={best_ic_original:.2f})")

    return best_model


def residual_diagnostics(model: ETSModel) -> Dict[str, Any]:
    """
    Compute residual diagnostics for ETS model

    Parameters
    ----------
    model : ETSModel
        Fitted ETS model

    Returns
    -------
    dict
        Dictionary with diagnostic statistics:
        - mean: mean of residuals (should be ~0)
        - std: standard deviation of residuals
        - ljung_box_p: Ljung-Box test p-value (>0.05 suggests no autocorrelation)
        - jarque_bera_p: Jarque-Bera test p-value (>0.05 suggests normality)
        - shapiro_p: Shapiro-Wilk test p-value (>0.05 suggests normality)
        - acf: Autocorrelation function (first 10 lags)

    """

    residuals = model.residuals
    n = len(residuals)

    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals, ddof=1)

    try:
        jb_stat, jb_p = jarque_bera(residuals)
    except:
        jb_stat, jb_p = np.nan, np.nan

    try:
        if n >= 3:
            shapiro_stat, shapiro_p = shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
    except:
        shapiro_stat, shapiro_p = np.nan, np.nan

    max_lag = min(10, n // 4)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    residuals_centered = residuals - mean_resid
    c0 = np.sum(residuals_centered ** 2) / n

    for lag in range(1, max_lag + 1):
        c_lag = np.sum(residuals_centered[:-lag] * residuals_centered[lag:]) / n
        acf[lag] = c_lag / c0

    lb_stat = n * (n + 2) * np.sum(acf[1:max_lag + 1] ** 2 / (n - np.arange(1, max_lag + 1)))
    from scipy.stats import chi2
    lb_p = 1 - chi2.cdf(lb_stat, max_lag)

    return {
        "mean": mean_resid,
        "std": std_resid,
        "mae": np.mean(np.abs(residuals)),
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "mape": np.mean(np.abs(residuals / model.y_original)) * 100 if model.y_original is not None else np.nan,
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_p,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "acf": acf,
    }

