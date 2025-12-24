################################################################################
#                                 ETS                                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal, List
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, jarque_bera, shapiro
import warnings
import math
from statsmodels.tsa.seasonal import seasonal_decompose

ERROR_TYPES = {"N": 0, "A": 1, "M": 2}
TREND_TYPES = {"N": 0, "A": 1, "M": 2}
SEASON_TYPES = {"N": 0, "A": 1, "M": 2}


def is_constant(y: NDArray[np.float64]) -> bool:
    """Check if series is constant"""
    return np.all(y == y[0])


@njit(cache=True, fastmath=True)
def _admissible_jit(alpha: float, beta: float, gamma: float, phi: float, m: int) -> bool:
    TOL = 1e-8
    if phi < 0.0 or phi > 1.0 + TOL:
        return False

    if np.isnan(gamma):
        if np.isnan(alpha):
            return True

        if alpha < 1.0 - 1.0/phi or alpha > 1.0 + 1.0/phi:
            return False

        if not np.isnan(beta):
            if beta < alpha * (phi - 1.0) or beta > (1.0 + phi) * (2.0 - alpha):
                return False

    elif m > 1:
        if np.isnan(alpha):
            return False
        beta_val = 0.0 if np.isnan(beta) else beta
        lower_gamma = max(1.0 - 1.0/phi - alpha, 0.0)
        upper_gamma = 1.0 + 1.0/phi - alpha
        if gamma < lower_gamma or gamma > upper_gamma:
            return False

        alpha_lower = 1.0 - 1.0/phi - gamma * (1.0 - m + phi + phi * m) / (2.0 * phi * m)
        if alpha < alpha_lower:
            return False

        if beta_val < -(1.0 - phi) * (gamma / m + alpha):
            return False

        a = phi * (1.0 - alpha - gamma)
        b = alpha + beta_val - alpha * phi + gamma - 1.0
        c_coef = alpha + beta_val - alpha * phi
        d = alpha + beta_val - phi

        n_coef = m + 1
        P = np.zeros(n_coef, dtype=np.float64)
        P[0] = a
        P[1] = b
        for i in range(2, m - 1):
            P[i] = c_coef
        P[m - 1] = d
        P[m] = 1.0

        if m <= 24:
            C = np.zeros((n_coef - 1, n_coef - 1), dtype=np.float64)
            for j in range(n_coef - 1):
                C[0, j] = -P[j + 1] / P[0]
            for i in range(1, n_coef - 1):
                C[i, i - 1] = 1.0
            try:
                eigvals = np.linalg.eigvals(C)
                max_abs_root = np.max(np.abs(eigvals))
                if max_abs_root > 1.0 + 1e-10:
                    return False
            except:
                return False
        else:
            pass

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


@njit(cache=True, fastmath=True)
def _check_param_jit(alpha: float, beta: float, gamma: float, phi: float,
                     lower: NDArray[np.float64], upper: NDArray[np.float64],
                     check_usual: bool, check_admissible: bool, m: int) -> bool:
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
        if not _admissible_jit(alpha, beta, gamma, phi, m):
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
        alpha = x[idx]; idx += 1
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
    bic: float
    sigma2: float
    y_original: Optional[NDArray[np.float64]] = None
    transform: Optional['BoxCoxTransform'] = None


@dataclass
class BoxCoxTransform:
    lambda_param: float
    shift: float = 0.0

    @staticmethod
    def find_lambda(y: NDArray[np.float64], lambda_range: Tuple[float, float] = (-1, 2)) -> float:
        if np.any(y <= 0):
            shift = np.abs(np.min(y)) + 1.0
            y_shifted = y + shift
        else:
            shift = 0.0
            y_shifted = y

        def neg_log_likelihood(lam):
            if abs(lam) < 1e-10:
                y_trans = np.log(y_shifted)
            else:
                y_trans = (y_shifted ** lam - 1) / lam
            return np.var(y_trans)

        result = minimize_scalar(neg_log_likelihood, bounds=lambda_range, method='bounded')
        return result.x

    def transform(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_shifted = y + self.shift
        if abs(self.lambda_param) < 1e-10:
            return np.log(y_shifted)
        else:
            return (y_shifted ** self.lambda_param - 1) / self.lambda_param

    def inverse_transform(self, y_trans: NDArray[np.float64],
                         bias_adjust: bool = False,
                         variance: Optional[float] = None) -> NDArray[np.float64]:
        if abs(self.lambda_param) < 1e-10:
            y_back = np.exp(y_trans)
            if bias_adjust and variance is not None:
                y_back *= np.exp(variance / 2)
        else:
            y_back = (self.lambda_param * y_trans + 1) ** (1 / self.lambda_param)
            if bias_adjust and variance is not None:
                correction = (1 - self.lambda_param) * variance / (2 * y_back ** (2 * self.lambda_param))
                y_back += correction

        return y_back - self.shift


@njit(cache=True, fastmath=True)
def _ets_step(l: float, b: float, s: NDArray[np.float64], y: float,
              m: int, error: int, trend: int, season: int,
              alpha: float, beta: float, gamma: float, phi: float) -> Tuple:
    TOL = 1e-10

    if trend == 0:
        q = l
        phib = 0.0
    elif trend == 1:
        phib = phi * b
        q = l + phib
    else:
        if b <= 0 or l <= 0:
            return l, b, s, -99999.0, 0.0
        phib = b ** phi
        q = l * phib
    if season == 0:
        yhat = q
    elif season == 1:
        yhat = q + s[m-1]
    else:
        yhat = q * s[m-1]

    if abs(yhat) < TOL:
        yhat = TOL

    if error == 1:
        e = y - yhat
    else:
        e = (y - yhat) / yhat
    if season == 0:
        p = y
    elif season == 1:
        p = y - s[m-1]
    else:
        p = y / max(s[m-1], TOL)
    l_new = q + alpha * (p - q)
    b_new = b
    if trend == 1:
        r = l_new - l
        b_new = phib + (beta / alpha) * (r - phib)
    elif trend == 2:
        r = l_new / max(l, TOL)
        b_new = phib + (beta / alpha) * (r - phib)
    
    # Only copy seasonal array if model has seasonality
    # This avoids 10-15% overhead for non-seasonal models (*NN, *AN, etc.)
    if season > 0:
        s_new = s.copy()
        if season == 1:
            t = y - q
        else:
            t = y / max(q, TOL)
        new_seasonal = s[m-1] + gamma * (t - s[m-1])
        s_new[0] = new_seasonal

        for i in range(1, m):
            s_new[i] = s[i-1]
    else:
        s_new = s  # No copy needed for non-seasonal models

    return l_new, b_new, s_new, yhat, e


@njit(cache=True, fastmath=True)
def _ets_likelihood(y: NDArray[np.float64], init_states: NDArray[np.float64],
                    m: int, error: int, trend: int, season: int,
                    alpha: float, beta: float, gamma: float, phi: float) -> Tuple:
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
        l, b, s, yhat, e = _ets_step(l, b, s, y[i], m, error, trend, season,
                                      alpha, beta, gamma, phi)

        if yhat < -99998:
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


@njit(cache=True, fastmath=True)
def _fourier_jit(n: int, period: int, K: int, h: int) -> NDArray[np.float64]:
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
    lower.extend([-1e6] * n_states)
    upper.extend([1e6] * n_states)

    return np.array(lower), np.array(upper)


@njit(cache=True, fastmath=True)
def _ets_objective_jit(x: NDArray[np.float64], 
                       y: NDArray[np.float64], 
                       lower: NDArray[np.float64], 
                       upper: NDArray[np.float64],
                       m: int, 
                       error_code: int, 
                       trend_code: int, 
                       season_code: int,
                       has_trend: bool, 
                       has_season: bool, 
                       is_damped: bool, 
                       is_mult_season: bool,
                       check_usual: bool, 
                       check_admissible: bool) -> float:
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
        Parameter vector [alpha, beta?, gamma?, phi?, init_states...]
    y : NDArray[np.float64]
        Time series observations
    lower, upper : NDArray[np.float64]
        Parameter bounds
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

    # Extract smoothing parameters from x
    idx = 0
    alpha = x[idx]
    idx += 1

    if has_trend:
        beta = x[idx]
        idx += 1
    else:
        beta = 0.0

    if has_season:
        gamma = x[idx]
        idx += 1
    else:
        gamma = np.nan

    if is_damped:
        phi = x[idx]
        idx += 1
    else:
        phi = 1.0

    init_states = x[idx:].copy()

    # Check parameter constraints
    beta_check = beta if has_trend else np.nan
    phi_check = phi if is_damped else np.nan

    if not _check_param_jit(alpha, beta_check, gamma, phi_check,
                            lower, upper, check_usual, check_admissible, m):
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
            bic=k_const * np.log(n),
            sigma2=0.0,
            y_original=y_original,
            transform=None
        )

    if n < 1:
        raise ValueError(f"Need at least 1 observation to fit ETS model, got {n}")

    if len(model) != 3:
        raise ValueError(f"Model must be 3 characters (e.g., 'AAN', 'MAM'), got '{model}'")

    # Handle ZZZ with high frequency by calling auto_ets
    if model == "ZZZ" and m > 24:
        warnings.warn(
            f"Frequency too high (m={m} > 24). Using auto_ets to select non-seasonal model. "
            f"Try stlf() if you need seasonal forecasts."
        )
        return auto_ets(
            y_original, m=m, seasonal=False, trend=None, damped=damped,
            ic="aicc", allow_multiplicative=True, 
            allow_multiplicative_trend=False,
            lambda_auto=lambda_auto, verbose=False
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
        lambda_opt = BoxCoxTransform.find_lambda(y)
        transform = BoxCoxTransform(lambda_opt, shift)
        y = transform.transform(y)
    elif lambda_param is not None:
        shift = np.abs(np.min(y)) + 1.0 if np.any(y <= 0) else 0.0
        transform = BoxCoxTransform(lambda_param, shift)
        y = transform.transform(y)

    if len(model) != 3:
        raise ValueError("Model must be 3 characters (e.g., 'ANN', 'AAA')")

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
    init_params = ETSParams(
        alpha=alpha if alpha is not None else 0.1,
        beta=beta if beta is not None else 0.01,
        gamma=gamma if gamma is not None else 0.01,
        phi=phi if phi is not None else 0.98,
        init_states=init_state_vec
    )

    lower, upper = get_bounds(config)

    check_usual = (bounds != "admissible")
    check_admissible = (bounds != "usual")
    has_trend = config.trend != "N"
    has_season = config.season != "N"
    is_mult_season = config.season == "M"

    def objective(x):
        """Wrapper for scipy.optimize.minimize that calls module-level JIT function"""
        return _ets_objective_jit(
            x,
            y,
            lower,
            upper,
            config.m,
            config.error_code,
            config.trend_code,
            config.season_code,
            has_trend,
            has_season,
            damped,
            is_mult_season,
            check_usual,
            check_admissible,
        )

    x0 = init_params.to_vector(config)

    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
    )

    fitted_params = ETSParams.from_vector(result.x, config)

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

    n_params = len(result.x)
    k = n_params + 1
    aic = loglik + 2 * k
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
        bic=bic,
        sigma2=sigma2,
        y_original=y_original,
        transform=transform
    )


@njit(cache=True, fastmath=True)
def _forecast_ets(l: float, b: float, s: NDArray[np.float64],
                  h: int, m: int, trend: int, season: int, phi: float) -> NDArray[np.float64]:
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
                fc = -99999.0
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
                          (1/6) * beta**2 * steps * (2 * steps - 1)))

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
            exp1 = alpha**2 + alpha * beta * steps + (1/6) * beta**2 * steps * (2 * steps - 1)
            exp2 = 2 * alpha + gamma + beta * m * (hm + 1)
            var = sigma * (1 + (steps - 1) * exp1 + gamma * hm * exp2)

        else:
            var = None

    else:
        var = None

    return var


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

    if model.transform is not None:
        forecasts = model.transform.inverse_transform(forecasts, bias_adjust, model.sigma2)

    result = {'mean': forecasts}

    if level is not None:
        if model.sigma2 <= 0:
            import warnings
            warnings.warn(
                f"Cannot compute prediction intervals: model has invalid residual variance "
                f"(sigma2={model.sigma2:.2e}). This usually means the model is overfit or "
                f"there is insufficient data. Returning point forecasts only.",
                UserWarning
            )
            return result

        var = _compute_prediction_variance(model, h)

        if var is not None:
            for lv in level:
                z = norm.ppf(0.5 + lv / 200)
                std = np.sqrt(var)
                result[f'lower_{int(lv)}'] = forecasts - z * std
                result[f'upper_{int(lv)}'] = forecasts + z * std
        else:
            try:
                simulations = simulate_ets(model, h=h, n_sim=1000)
                for lv in level:
                    result[f'lower_{int(lv)}'] = np.percentile(simulations, 50 - lv/2, axis=0)
                    result[f'upper_{int(lv)}'] = np.percentile(simulations, 50 + lv/2, axis=0)
            except ValueError as e:
                import warnings
                warnings.warn(
                    f"Cannot compute prediction intervals via simulation: {str(e)}. "
                    f"Returning point forecasts only.",
                    UserWarning
                )

    return result


def simulate_ets(model: ETSModel, h: int = 10, n_sim: int = 1000) -> NDArray[np.float64]:
    """Simulate future paths from ETS model"""
    if model.sigma2 <= 0:
        raise ValueError(
            f"Cannot simulate: model has invalid residual variance (sigma2={model.sigma2:.2e}). "
            f"This usually means the model is overfit or there is insufficient data."
        )

    simulations = np.zeros((n_sim, h))

    for i in range(n_sim):
        if model.config.error == "A":
            errors = norm.rvs(loc=0, scale=np.sqrt(model.sigma2), size=h)
        else:
            errors = norm.rvs(loc=0, scale=np.sqrt(model.sigma2), size=h)

        l = model.states[0]
        b = model.states[1] if model.config.trend != "N" else 0.0

        if model.config.season != "N":
            s_start = 1 + (1 if model.config.trend != "N" else 0)
            s = model.states[s_start:].copy()
        else:
            s = np.zeros(max(model.config.m, 1))

        for t in range(h):
            fc = _forecast_ets(l, b, s, 1, model.config.m,
                              model.config.trend_code, model.config.season_code,
                              model.params.phi)[0]

            if model.config.error == "A":
                y_new = fc + errors[t]
            else:
                y_new = fc * (1 + errors[t])

            simulations[i, t] = y_new

            l, b, s, _, _ = _ets_step(
                l, b, s, y_new,
                model.config.m,
                model.config.error_code,
                model.config.trend_code,
                model.config.season_code,
                model.params.alpha,
                model.params.beta,
                model.params.gamma,
                model.params.phi
            )

    return simulations


def auto_ets(y: NDArray[np.float64],
             m: int = 1,
             seasonal: bool = True,
             trend: Optional[bool] = None,
             damped: Optional[bool] = None,
             ic: Literal["aic", "aicc", "bic"] = "aicc",
             allow_multiplicative: bool = True,
             allow_multiplicative_trend: bool = False,
             lambda_auto: bool = False,
             max_models: Optional[int] = None,
             verbose: bool = False) -> ETSModel:
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

    Returns
    -------
    ETSModel
        Best model according to information criterion
    """
    n = len(y)
    if n < 1:
        raise ValueError(f"Need at least 1 observation, got {n}")

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
            model = ets(y, m=m, model=model_spec, damped=damped_flag, lambda_auto=lambda_auto, bounds="both")

            if ic == "aic":
                ic_value = model.aic
            elif ic == "aicc":
                n = len(y)
                k = (1 + (model.config.trend != "N") + (model.config.season != "N") +
                     damped_flag + model.config.n_states)
                ic_value = model.aic + (2 * k * (k + 1)) / (n - k - 1)
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


def residual_diagnostics(model: ETSModel) -> Dict[str, any]:
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

    lb_stat = n * (n + 2) * np.sum(acf[1:max_lag+1] ** 2 / (n - np.arange(1, max_lag+1)))
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

