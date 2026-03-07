################################################################################
#                               ARIMA base implementation                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import njit
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm
import scipy.optimize as opt
from typing import ClassVar, Tuple, Optional, Dict as DictType, Any, Union, List
import warnings


# =============================================================================
# Named constants (replacing bare magic literals)
# =============================================================================

# Observations with innovation variance F above this are treated as
# diffuse (uninformative) and excluded from the likelihood.
# Durbin & Koopman (2012), §5.2.
_DIFFUSE_INNOVATION_VARIANCE_THRESHOLD = 1e4

# Prior variance for non-stationary (diffuse) state components in P₀.
# Durbin & Koopman (2012), §5.2.
_DIFFUSE_STATE_PRIOR_VARIANCE = 1e6

# Step size for numerical Hessian; larger than sqrt(eps) for stability
# in transformed ARIMA parameter space.
_HESSIAN_STEP_SIZE = 1e-2

# Step size for numerical Jacobian of AR parameter transformation.
_JACOBIAN_STEP_SIZE = 1e-3

# Penalty returned by objective functions for invalid parameters.
_OBJECTIVE_PENALTY = 1e10

# OLS standard-error multiplier for initial parameter scaling.
_REGRESSOR_SCALE_FACTOR = 10.0


# =============================================================================
# Section 0: Dataclass Definitions
# =============================================================================

@dataclass(frozen=True)
class SARIMAOrder:
    """
    Immutable specification of SARIMA model orders.

    Parameters
    ----------
    p : int
        Non-seasonal AR order.
    d : int
        Non-seasonal differencing order.
    q : int
        Non-seasonal MA order.
    P : int
        Seasonal AR order.
    D : int
        Seasonal differencing order.
    Q : int
        Seasonal MA order.
    s : int
        Seasonal period (1 = non-seasonal).
    """
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    s: int

    @property
    def n_arma_params(self) -> int:
        """Total number of ARMA parameters (p + q + P + Q)."""
        return self.p + self.q + self.P + self.Q

    @property
    def total_ar_order(self) -> int:
        """Total AR order including seasonal expansion (p + s * P)."""
        return self.p + self.s * self.P

    @property
    def total_ma_order(self) -> int:
        """Total MA order including seasonal expansion (q + s * Q)."""
        return self.q + self.s * self.Q

    def to_arma_list(self) -> list:
        """
        Convert to the legacy ``[p, q, P, Q, s, d, D]`` list for backward
        compatibility with public API return values.

        Returns
        -------
        list
            List of ints ``[p, q, P, Q, s, d, D]``.
        """
        return [self.p, self.q, self.P, self.Q, self.s, self.d, self.D]


@dataclass
class StateSpaceArrays:
    """
    State-space representation arrays for an ARIMA model.

    Parameters
    ----------
    ar_coefs : np.ndarray
        AR coefficients.
    ma_coefs : np.ndarray
        MA coefficients.
    differencing_poly : np.ndarray
        Differencing polynomial coefficients.
    observation_vector : np.ndarray
        Observation vector Z.
    filtered_state : np.ndarray
        Filtered state vector.
    filtered_covariance : np.ndarray
        Filtered state covariance.
    transition_matrix : np.ndarray
        State transition matrix.
    innovation_covariance : np.ndarray
        Innovation covariance R @ R'.
    observation_variance : float
        Observation noise variance (h). Always 0.0 for ARIMA models — the
        standard ARIMA state-space form places all noise in the state
        equation, so there is no separate measurement noise. The field is
        retained as a forward-compatibility placeholder for structural
        time-series models that add a measurement-error term. Because it is
        always zero, ``kalman_forecast_core`` adds a harmless ``+ h`` to
        its forecast-variance computation.
    predicted_covariance : np.ndarray
        Predicted state covariance.
    """
    ar_coefs: np.ndarray
    ma_coefs: np.ndarray
    differencing_poly: np.ndarray
    observation_vector: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    transition_matrix: np.ndarray
    innovation_covariance: np.ndarray
    observation_variance: float
    predicted_covariance: np.ndarray



@dataclass
class ArimaResult:
    """
    Result container for a fitted ARIMA model.

    Parameters
    ----------
    y : np.ndarray
        Original time series.
    fitted_values : np.ndarray
        In-sample fitted values.
    coefficients : pd.DataFrame
        Coefficient estimates as a single-row DataFrame.
    sigma2 : float
        Estimated innovation variance.
    param_covariance : np.ndarray
        Variance-covariance matrix of parameter estimates.
    param_mask : np.ndarray
        Boolean mask indicating free (True) vs fixed (False) parameters.
    loglik : float
        Log-likelihood value.
    aic : float
        Akaike information criterion.
    bic : Optional[float]
        Bayesian information criterion.
    aicc : Optional[float]
        Corrected AIC.
    ic : Optional[float]
        Information criterion used for model selection.
    order : SARIMAOrder
        SARIMA order specification.
    residuals : np.ndarray
        Model residuals.
    converged : bool
        Whether optimization converged.
    n_cond : int
        Number of conditioning observations.
    nobs : int
        Number of observations used.
    state_space : StateSpaceArrays
        State-space representation arrays.
    exog : Any
        Exogenous regressors (if any).
    method : str
        Estimation method description string.
    lambda_bc : Optional[float]
        Box-Cox transformation parameter.
    biasadj : Optional[bool]
        Whether bias adjustment is used for Box-Cox.
    offset : Optional[float]
        Approximation offset.
    constant : Optional[bool]
        Whether model includes a constant (auto_arima only).
    """
    y: np.ndarray
    fitted_values: np.ndarray
    coefficients: pd.DataFrame
    sigma2: float
    param_covariance: np.ndarray
    param_mask: np.ndarray
    loglik: float
    aic: float
    bic: Optional[float]
    aicc: Optional[float]
    ic: Optional[float]
    order: SARIMAOrder
    residuals: np.ndarray
    converged: bool
    n_cond: int
    nobs: int
    state_space: StateSpaceArrays
    exog: Any
    method: str
    lambda_bc: Optional[float]
    biasadj: Optional[bool]
    offset: Optional[float]
    constant: Optional[bool] = None

    # Mapping from legacy dict keys to dataclass field names
    _KEY_MAP: ClassVar[dict] = {
        'y': 'y',
        'fitted': 'fitted_values',
        'coef': 'coefficients',
        'sigma2': 'sigma2',
        'var_coef': 'param_covariance',
        'mask': 'param_mask',
        'loglik': 'loglik',
        'aic': 'aic',
        'bic': 'bic',
        'aicc': 'aicc',
        'ic': 'ic',
        'order_spec': 'order',
        'residuals': 'residuals',
        'converged': 'converged',
        'n_cond': 'n_cond',
        'nobs': 'nobs',
        'state_space': 'state_space',
        'exog': 'exog',
        'method': 'method',
        'lambda': 'lambda_bc',
        'biasadj': 'biasadj',
        'offset': 'offset',
        'constant': 'constant',
    }

    def __post_init__(self):
        # Storage for extra keys not in the dataclass fields
        object.__setattr__(self, '_extra', {})

    def __getitem__(self, key):
        if key == 'arma':
            warnings.warn(
                "Accessing 'arma' via dict-style indexing is deprecated. "
                "Use result.order instead.",
                FutureWarning,
                stacklevel=2,
            )
            return self.order.to_arma_list()
        if key == 'model':
            warnings.warn(
                "Accessing 'model' via dict-style indexing is deprecated. "
                "Use result.state_space instead.",
                FutureWarning,
                stacklevel=2,
            )
            ss = self.state_space
            return {
                'phi': ss.ar_coefs,
                'theta': ss.ma_coefs,
                'Delta': ss.differencing_poly,
                'Z': ss.observation_vector,
                'a': ss.filtered_state,
                'P': ss.filtered_covariance,
                'T': ss.transition_matrix,
                'V': ss.innovation_covariance,
                'h': ss.observation_variance,
                'Pn': ss.predicted_covariance,
            }
        field = self._KEY_MAP.get(key)
        if field is not None:
            return getattr(self, field)
        extra = object.__getattribute__(self, '_extra')
        if key in extra:
            return extra[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key == 'arma':
            return  # read-only computed property
        if key == 'model':
            return  # read-only computed property
        field = self._KEY_MAP.get(key)
        if field is not None:
            object.__setattr__(self, field, value)
        else:
            object.__getattribute__(self, '_extra')[key] = value

    def __contains__(self, key):
        if key in ('arma', 'model'):
            return True
        if key in self._KEY_MAP:
            return True
        return key in object.__getattribute__(self, '_extra')

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# =============================================================================
# Section 1: Utility Functions
# =============================================================================

def _validate_choice(value, choices, param_name="argument"):
    """
    Validate that a value is one of the allowed choices.

    Parameters
    ----------
    value : str
        Value to validate.
    choices : list
        Allowed values.
    param_name : str
        Parameter name for error message.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If value is not in choices.
    """
    if value in choices:
        return value
    raise ValueError(
        f"Invalid `{param_name}`: '{value}'. Must be one of {choices}."
    )

def _ensure_float64_pair(x: np.ndarray, exog: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values by converting to NaN.

    Parameters
    ----------
    x : np.ndarray
        Target variable.
    exog : np.ndarray
        Regressor matrix.

    Returns
    -------
    x, exog : tuple
        Arrays with missing converted to NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    exog = np.asarray(exog, dtype=np.float64)
    return x, exog

def diff(x: np.ndarray, lag: int = 1, differences: int = 1) -> np.ndarray:
    """
    Compute lagged differences of a time series.

    Parameters
    ----------
    x : np.ndarray
        Input array (1D or 2D).
    lag : int
        Lag for differencing.
    differences : int
        Order of differencing.

    Returns
    -------
    np.ndarray
        Differenced array.
    """
    result = x
    for _ in range(differences):
        if result.ndim == 1:
            result = result[lag:] - result[:-lag]
        else:
            result = result[lag:, :] - result[:-lag, :]
    return result


# =============================================================================
# Section 2: Polynomial Utilities
# =============================================================================

def _companion_matrix_roots(poly_ascending: np.ndarray) -> np.ndarray:
    """
    Find roots of a polynomial via companion matrix eigenvalues.

    Given polynomial c₀ + c₁z + ··· + cₙzⁿ in ascending-power form,
    the roots are the eigenvalues of the companion matrix of the monic
    polynomial (divided by cₙ). This is equivalent to what
    `numpy.polynomial.polynomial.polyroots` does internally, but avoids
    the overhead of the polynomial module dispatch.

    Parameters
    ----------
    poly_ascending : np.ndarray
        Polynomial coefficients in ascending power order [c₀, c₁, ..., cₙ].
        Must have length >= 2 (degree >= 1).

    Returns
    -------
    np.ndarray
        Complex array of roots.
    """
    n = len(poly_ascending) - 1
    if n == 1:
        return np.array([-poly_ascending[0] / poly_ascending[1]], dtype=np.complex128)
    # Build companion matrix for monic polynomial
    # (divide through by leading coefficient cₙ)
    companion = np.zeros((n, n), dtype=np.float64)
    companion[1:, :-1] = np.eye(n - 1)
    companion[:, -1] = -poly_ascending[:-1] / poly_ascending[-1]
    return np.linalg.eigvals(companion)


def ar_check(ar: np.ndarray) -> bool:
    """
    Check stationarity of an AR polynomial.

    An AR process yₜ = φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ is stationary if and
    only if all roots of the characteristic polynomial
    φ(z) = 1 − φ₁z − φ₂z² − ··· − φₚzᵖ  lie strictly outside the unit
    circle (Hamilton, 1994, Proposition 3.1).

    Parameters
    ----------
    ar : np.ndarray
        AR coefficients [φ₁, φ₂, ..., φₚ].

    Returns
    -------
    bool
        True if stationary (all roots outside the unit circle).
    """
    ar = np.asarray(ar, dtype=np.float64)
    if len(ar) == 0:
        return True

    # Build characteristic polynomial φ(z) = 1 − φ₁z − φ₂z² − ···
    char_poly = np.concatenate([[1.0], -ar])

    # Trim trailing zeros (degenerate higher-order terms)
    last_nonzero = np.max(np.nonzero(char_poly)[0]) if np.any(char_poly != 0) else 0
    if last_nonzero == 0:
        return True
    char_poly = char_poly[:last_nonzero + 1]

    roots = _companion_matrix_roots(char_poly)
    return bool(np.all(np.abs(roots) > 1.0))

def _poly_from_roots(roots: np.ndarray) -> np.ndarray:
    """
    Build polynomial coefficients (ascending power) from its roots.

    Given roots r₁, ..., rₙ, computes the coefficients of
    (z − r₁)(z − r₂)···(z − rₙ) = c₀ + c₁z + ··· + zⁿ
    via sequential convolution.

    Parameters
    ----------
    roots : np.ndarray
        Complex array of polynomial roots.

    Returns
    -------
    np.ndarray
        Real polynomial coefficients in ascending power order.
    """
    poly = np.array([1.0], dtype=np.complex128)
    for r in roots:
        # Multiply by (z - r) = -r + z  in ascending-power form
        poly = np.convolve(poly, np.array([-r, 1.0], dtype=np.complex128))
    return np.real(poly)


def ma_invert(ma: np.ndarray) -> np.ndarray:
    """
    Enforce invertibility of an MA polynomial by reflecting roots.

    An MA polynomial θ(z) = 1 + θ₁z + θ₂z² + ... is invertible iff all
    its roots lie outside the unit circle. For any root rₖ with |rₖ| < 1,
    replace it with 1/rₖ and reconstruct the polynomial from the modified
    root set.

    Reference: Jones (1980), "Maximum Likelihood Fitting of ARMA Models
    to Time Series", Technometrics 22(3), pp. 389-395.

    Parameters
    ----------
    ma : np.ndarray
        MA coefficients [θ₁, θ₂, ..., θ_q] (without the leading 1).

    Returns
    -------
    np.ndarray
        MA coefficients with all roots reflected outside the unit circle.
        Length matches input.
    """
    ma = np.asarray(ma, dtype=np.float64)
    q = len(ma)
    if q == 0:
        return ma

    # Build MA polynomial θ(z) = 1 + θ₁z + θ₂z² + ... in ascending-power form
    ma_poly = np.concatenate([[1.0], ma])

    # Trim trailing zeros to find effective polynomial degree
    nonzero_idx = np.nonzero(ma_poly)[0]
    if len(nonzero_idx) == 0 or nonzero_idx[-1] == 0:
        return ma
    effective_degree = nonzero_idx[-1]
    ma_poly_trimmed = ma_poly[:effective_degree + 1]

    # Find roots via companion matrix eigenvalues
    roots = _companion_matrix_roots(ma_poly_trimmed)

    # Check if any roots lie inside the unit circle
    inside = np.abs(roots) < 1.0
    if not np.any(inside):
        return ma

    # Reflect roots inside the unit circle to 1/r
    roots[inside] = 1.0 / roots[inside]

    # Reconstruct polynomial from modified roots, normalize so c₀ = 1
    new_poly = _poly_from_roots(roots)
    new_poly = new_poly / new_poly[0]

    # Extract coefficients (skip leading 1), pad to original length
    result = new_poly[1:effective_degree + 1]
    if len(result) < q:
        result = np.concatenate([result, np.zeros(q - len(result))])

    return result[:q]


# =============================================================================
# Section 3: Parameter Transformations
# Jones (1980), "Maximum Likelihood Fitting of ARMA Models to Time Series",
# Technometrics 22(3), pp. 389-395, Equations 3-5.
# =============================================================================

@njit(cache=True)
def transform_unconstrained_to_ar_params(p: int, raw: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Map unconstrained reals to stationary AR coefficients.

    Uses the Jones (1980) parameterization (Eqs. 3-5):

    1. Map to partial autocorrelations via  wⱼ = tanh(rawⱼ),  j = 1..p
       This maps ℝ → (−1, 1), guaranteeing |wⱼ| < 1.

    2. Apply the Durbin-Levinson recursion to recover AR coefficients:
       - φ₁⁽¹⁾ = w₁
       - For j = 2, ..., p:
           φⱼ⁽ʲ⁾ = wⱼ
           φᵢ⁽ʲ⁾ = φᵢ⁽ʲ⁻¹⁾ − wⱼ · φⱼ₋ᵢ⁽ʲ⁻¹⁾   for i = 1, ..., j−1

    The resulting AR polynomial is guaranteed to be stationary.

    Reference: Jones (1980), Technometrics 22(3), pp. 389-395.

    Parameters
    ----------
    p : int
        Number of AR parameters.
    raw : np.ndarray
        Unconstrained parameter vector (length >= p).

    Returns
    -------
    ar_coefs : np.ndarray
        Stationary AR coefficients [φ₁, ..., φₚ], length p.
    """
    if p > 100:
        raise ValueError("Can only transform up to 100 parameters")

    # Step 1: partial autocorrelations via tanh
    partial_corr = np.tanh(raw[:p])

    # Step 2: Durbin-Levinson recursion
    ar_coefs = partial_corr.copy()
    ar_coefs_prev = np.empty(p)

    for j in range(1, p):
        w_j = ar_coefs[j]
        for i in range(j):
            ar_coefs_prev[i] = ar_coefs[i] - w_j * ar_coefs[j - 1 - i]
        ar_coefs[:j] = ar_coefs_prev[:j]

    return ar_coefs

@njit(cache=True)
def inverse_ar_parameter_transform(phi: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Recover unconstrained parameters from AR coefficients.

    Inverts the Jones (1980) transformation by running the Durbin-Levinson
    recursion in reverse:

    For j = p, p−1, ..., 2:
      1. Extract partial autocorrelation: wⱼ = φⱼ⁽ʲ⁾
      2. Recover previous-order coefficients:
         φᵢ⁽ʲ⁻¹⁾ = (φᵢ⁽ʲ⁾ + wⱼ · φⱼ₋ᵢ⁽ʲ⁾) / (1 − wⱼ²)   for i = 1, ..., j−1

    Then map partial autocorrelations back to ℝ via arctanh:
      rawⱼ = arctanh(wⱼ)

    Reference: Jones (1980), Technometrics 22(3), pp. 389-395.

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients [φ₁, ..., φₚ].

    Returns
    -------
    unconstrained : np.ndarray
        Unconstrained parameters [raw₁, ..., rawₚ].
    """
    p = len(phi)
    ar_coefs = phi.copy()
    ar_coefs_prev = np.zeros(p)

    # Reverse Durbin-Levinson to extract partial autocorrelations
    for j in range(p - 1, 0, -1):
        w_j = ar_coefs[j]
        denom = 1.0 - w_j * w_j
        if denom == 0.0:
            return np.full(p, np.nan)
        for i in range(j):
            ar_coefs_prev[i] = (ar_coefs[i] + w_j * ar_coefs[j - 1 - i]) / denom
        ar_coefs[:j] = ar_coefs_prev[:j].copy()

    # Map partial autocorrelations to unconstrained space via arctanh
    unconstrained = np.zeros(p)
    for i in range(p):
        if abs(ar_coefs[i]) <= 1.0:
            unconstrained[i] = np.arctanh(ar_coefs[i])
        else:
            unconstrained[i] = np.nan

    return unconstrained

@njit(cache=True)
def inverse_arima_parameter_transform(
    theta: np.ndarray,
    n_ar: int,
    n_ma: int,
    n_seasonal_ar: int
) -> np.ndarray:  # pragma: no cover
    """
    Map ARIMA AR coefficients back to unconstrained space.

    Applies the inverse Jones (1980) transform separately to the
    non-seasonal AR block and the seasonal AR block of the parameter
    vector. MA and exogenous parameters pass through unchanged.

    Reference: Jones (1980), Technometrics 22(3), pp. 389-395.

    Parameters
    ----------
    theta : np.ndarray
        ARIMA parameter vector [AR | MA | SAR | SMA | exog].
    n_ar : int
        Number of non-seasonal AR parameters.
    n_ma : int
        Number of non-seasonal MA parameters.
    n_seasonal_ar : int
        Number of seasonal AR parameters.

    Returns
    -------
    unconstrained : np.ndarray
        Parameter vector with AR blocks mapped to unconstrained space.
    """
    unconstrained = theta.copy()
    seasonal_ar_offset = n_ar + n_ma

    if n_ar > 0:
        unconstrained[:n_ar] = inverse_ar_parameter_transform(theta[:n_ar])

    if n_seasonal_ar > 0:
        unconstrained[seasonal_ar_offset:seasonal_ar_offset + n_seasonal_ar] = (
            inverse_ar_parameter_transform(theta[seasonal_ar_offset:seasonal_ar_offset + n_seasonal_ar])
        )

    return unconstrained

@njit(cache=True)
def undo_arima_parameter_transform(
    x: np.ndarray,
    n_ar: int,
    n_ma: int,
    n_seasonal_ar: int
) -> np.ndarray:  # pragma: no cover
    """
    Apply the Jones (1980) forward transform to the AR blocks.

    Maps unconstrained parameters in the AR and seasonal-AR positions
    to stationary AR coefficients via the tanh + Durbin-Levinson recursion.
    MA and exogenous parameters pass through unchanged.

    This is the forward direction: unconstrained → constrained.
    See `inverse_arima_parameter_transform` for the reverse.

    Reference: Jones (1980), Technometrics 22(3), pp. 389-395.

    Parameters
    ----------
    x : np.ndarray
        Parameter vector with unconstrained AR values.
    n_ar : int
        Number of non-seasonal AR parameters.
    n_ma : int
        Number of non-seasonal MA parameters.
    n_seasonal_ar : int
        Number of seasonal AR parameters.

    Returns
    -------
    constrained : np.ndarray
        Parameter vector with stationary AR coefficients.
    """
    constrained = x.copy()

    if n_ar > 0:
        constrained[:n_ar] = transform_unconstrained_to_ar_params(n_ar, x)

    seasonal_ar_offset = n_ar + n_ma
    if n_seasonal_ar > 0:
        constrained[seasonal_ar_offset:seasonal_ar_offset + n_seasonal_ar] = (
            transform_unconstrained_to_ar_params(n_seasonal_ar, x[seasonal_ar_offset:])
        )

    return constrained

@njit(cache=True)
def compute_arima_transform_gradient(
    x: np.ndarray,
    n_ar: int,
    n_ma: int,
    n_seasonal_ar: int
) -> np.ndarray:  # pragma: no cover
    """
    Numerical Jacobian of the Jones (1980) AR parameter transformation.

    Computes ∂fᵢ/∂xⱼ using forward finite differences:
        ∂fᵢ/∂xⱼ ≈ (f(x + h·eⱼ) − f(x)) / h

    where f maps unconstrained parameters to AR coefficients via
    `transform_unconstrained_to_ar_params`. The Jacobian is identity
    for MA and exogenous parameter positions (those are not transformed).

    This Jacobian is used in the delta method to convert the Hessian
    from unconstrained to constrained parameter space for variance
    estimation.

    Parameters
    ----------
    x : np.ndarray
        Full parameter vector in unconstrained space.
    n_ar : int
        Number of non-seasonal AR parameters.
    n_ma : int
        Number of non-seasonal MA parameters.
    n_seasonal_ar : int
        Number of seasonal AR parameters.

    Returns
    -------
    jacobian : np.ndarray
        Jacobian matrix of shape (n, n).
    """
    h = 1e-3  # _JACOBIAN_STEP_SIZE — finite-difference step size
    n = len(x)
    jacobian = np.eye(n)
    workspace = np.zeros(100)

    # Jacobian for non-seasonal AR block
    if n_ar > 0:
        for i in range(n_ar):
            workspace[i] = x[i]
        f_base = transform_unconstrained_to_ar_params(n_ar, workspace)

        for j in range(n_ar):
            workspace[j] += h
            f_pert = transform_unconstrained_to_ar_params(n_ar, workspace)
            for i in range(n_ar):
                jacobian[j, i] = (f_pert[i] - f_base[i]) / h
            workspace[j] -= h

    # Jacobian for seasonal AR block
    if n_seasonal_ar > 0:
        seasonal_ar_offset = n_ar + n_ma
        for i in range(n_seasonal_ar):
            workspace[i] = x[seasonal_ar_offset + i]
        f_base = transform_unconstrained_to_ar_params(n_seasonal_ar, workspace)

        for j in range(n_seasonal_ar):
            workspace[j] += h
            f_pert = transform_unconstrained_to_ar_params(n_seasonal_ar, workspace)
            for i in range(n_seasonal_ar):
                jacobian[seasonal_ar_offset + j, seasonal_ar_offset + i] = (
                    (f_pert[i] - f_base[i]) / h
                )
            workspace[j] -= h

    return jacobian

@njit(cache=True)
def transform_arima_parameters(
    params_in: np.ndarray,
    n_ar: int,
    n_ma: int,
    n_seasonal_ar: int,
    n_seasonal_ma: int,
    seasonal_period: int,
    trans: bool
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Expand ARIMA parameters into full AR and MA coefficient vectors.

    Implements the Box-Jenkins multiplicative seasonal model
    (Box, Jenkins & Reinsel, 2015, Ch. 9):

    Full AR polynomial:  φ(B) · Φ(Bˢ)
      where φ(B) = 1 − φ₁B − ··· − φₚBᵖ       (non-seasonal)
            Φ(Bˢ) = 1 − Φ₁Bˢ − ··· − ΦₚBᴾˢ   (seasonal)

    Full MA polynomial:  θ(B) · Θ(Bˢ)
      where θ(B) = 1 + θ₁B + ··· + θ_qB^q       (non-seasonal)
            Θ(Bˢ) = 1 + Θ₁Bˢ + ··· + Θ_QB^Qs   (seasonal)

    The products are computed via polynomial multiplication (convolution).

    If ``trans=True``, the Jones (1980) stationarity transform is applied
    to AR parameters before expansion.

    Parameters
    ----------
    params_in : np.ndarray
        Parameter vector [φ₁..φₚ, θ₁..θ_q, Φ₁..Φ_P, Θ₁..Θ_Q, ...].
    n_ar : int
        Non-seasonal AR order p.
    n_ma : int
        Non-seasonal MA order q.
    n_seasonal_ar : int
        Seasonal AR order P.
    n_seasonal_ma : int
        Seasonal MA order Q.
    seasonal_period : int
        Seasonal period s (1 = non-seasonal).
    trans : bool
        Whether to apply the Jones (1980) stationarity transformation
        to AR parameters.

    Returns
    -------
    phi : np.ndarray
        Expanded AR coefficients from φ(B)·Φ(Bˢ).
    theta : np.ndarray
        Expanded MA coefficients from θ(B)·Θ(Bˢ).
    """
    params = params_in.copy()

    # Apply Jones (1980) stationarity transform to AR blocks if requested
    if trans:
        if n_ar > 0:
            params[:n_ar] = transform_unconstrained_to_ar_params(n_ar, params_in)
        seasonal_ar_offset = n_ar + n_ma
        if n_seasonal_ar > 0:
            params[seasonal_ar_offset:seasonal_ar_offset + n_seasonal_ar] = (
                transform_unconstrained_to_ar_params(n_seasonal_ar, params_in[seasonal_ar_offset:])
            )

    # Extract individual parameter groups
    ar_params = params[:n_ar]
    ma_params = params[n_ar:n_ar + n_ma]
    sar_params = params[n_ar + n_ma:n_ar + n_ma + n_seasonal_ar]
    sma_params = params[n_ar + n_ma + n_seasonal_ar:n_ar + n_ma + n_seasonal_ar + n_seasonal_ma]

    # --- AR side: φ(B) · Φ(Bˢ) ---
    # Non-seasonal AR polynomial: 1 − φ₁B − φ₂B² − ...
    ar_poly = np.ones(n_ar + 1)
    for i in range(n_ar):
        ar_poly[i + 1] = -ar_params[i]

    # Seasonal AR polynomial: 1 − Φ₁Bˢ − Φ₂B²ˢ − ...
    if n_seasonal_ar > 0 and seasonal_period > 0:
        sar_poly = np.zeros(n_seasonal_ar * seasonal_period + 1)
        sar_poly[0] = 1.0
        for j in range(n_seasonal_ar):
            sar_poly[(j + 1) * seasonal_period] = -sar_params[j]
    else:
        sar_poly = np.ones(1)

    # Multiply: full AR polynomial = φ(B) · Φ(Bˢ)
    full_ar_poly = np.convolve(ar_poly, sar_poly)

    # Extract AR coefficients (negate, skip leading 1)
    total_ar = n_ar + seasonal_period * n_seasonal_ar
    if total_ar > 0:
        phi = -full_ar_poly[1:total_ar + 1]
    else:
        phi = np.zeros(0)

    # --- MA side: θ(B) · Θ(Bˢ) ---
    # Non-seasonal MA polynomial: 1 + θ₁B + θ₂B² + ...
    ma_poly = np.ones(n_ma + 1)
    for i in range(n_ma):
        ma_poly[i + 1] = ma_params[i]

    # Seasonal MA polynomial: 1 + Θ₁Bˢ + Θ₂B²ˢ + ...
    if n_seasonal_ma > 0 and seasonal_period > 0:
        sma_poly = np.zeros(n_seasonal_ma * seasonal_period + 1)
        sma_poly[0] = 1.0
        for j in range(n_seasonal_ma):
            sma_poly[(j + 1) * seasonal_period] = sma_params[j]
    else:
        sma_poly = np.ones(1)

    # Multiply: full MA polynomial = θ(B) · Θ(Bˢ)
    full_ma_poly = np.convolve(ma_poly, sma_poly)

    # Extract MA coefficients (skip leading 1)
    total_ma = n_ma + seasonal_period * n_seasonal_ma
    if total_ma > 0:
        theta = full_ma_poly[1:total_ma + 1]
    else:
        theta = np.zeros(0)

    return phi, theta


# =============================================================================
# Section 4: Initial Covariance
# Gardner, Harvey & Phillips (1980), JRSS Series C 29(3)
# =============================================================================

@njit(cache=True)
def _solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Solve the discrete Lyapunov equation  X = A X A' + Q  for X.

    Uses the Smith (1968) doubling algorithm which converges quadratically
    when the spectral radius of A < 1 (guaranteed for stationary ARMA).

    Reference: Smith (1968), SIAM J. Applied Math. 16(1).
    """
    X = Q.copy()
    Ak = A.copy()
    for _ in range(64):
        X = Ak @ X @ Ak.T + X
        Ak = Ak @ Ak
        max_val = 0.0
        for i in range(Ak.shape[0]):
            for j in range(Ak.shape[1]):
                v = abs(Ak[i, j])
                if v > max_val:
                    max_val = v
        if max_val < 1e-15:
            break
    return X


@njit(cache=True)
def _compute_q0_njit(
    phi: np.ndarray, theta: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """
    Compute P0 via Smith doubling (fast @njit path for small state dims).
    """
    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    if r == 1:
        if p > 0:
            return np.array([[1.0 / (1.0 - phi[0] ** 2)]])
        return np.array([[1.0]])

    T_arma = np.zeros((r, r))
    for i in range(p):
        T_arma[i, 0] = phi[i]
    for i in range(1, r):
        T_arma[i - 1, i] = 1.0

    R_arma = np.zeros(r)
    R_arma[0] = 1.0
    mq = min(q, r - 1)
    for i in range(mq):
        R_arma[i + 1] = theta[i]
    V_arma = np.outer(R_arma, R_arma)

    return _solve_discrete_lyapunov(T_arma, V_arma)


# Threshold for switching Lyapunov solvers. For state dimension r > this
# value, scipy's Kronecker-product solver is used instead of Smith
# doubling. The Kronecker method has higher precision for large matrices,
# which keeps BFGS finite-difference gradients accurate and avoids
# unnecessary optimizer iterations in seasonal ARIMA models.
_LYAPUNOV_SCIPY_THRESHOLD = 6


def compute_q0_covariance_matrix(
    phi: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """
    Compute initial state covariance P0 via the discrete Lyapunov equation.

    P0 satisfies: P0 = T_arma @ P0 @ T_arma' + V_arma, where T_arma is the
    ARMA companion matrix and V_arma = R @ R' is the innovation covariance.

    For small state dimensions (r <= 6), uses a fast @njit Smith doubling
    solver. For larger dimensions (seasonal models), uses scipy's
    Kronecker-product solver which has better numerical precision for
    BFGS optimizer convergence.

    Gardner, Harvey & Phillips (1980), JRSS Series C 29(3).

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.

    Returns
    -------
    Q0 : np.ndarray
        Symmetric covariance matrix of size r x r.
    """
    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    if r <= _LYAPUNOV_SCIPY_THRESHOLD:
        return _compute_q0_njit(phi, theta)

    # Large state dimension: use scipy for higher Lyapunov precision
    T_arma = np.zeros((r, r))
    T_arma[:p, 0] = phi
    for i in range(1, r):
        T_arma[i - 1, i] = 1.0

    R_arma = np.zeros(r)
    R_arma[0] = 1.0
    mq = min(q, r - 1)
    R_arma[1:mq + 1] = theta[:mq]
    V_arma = np.outer(R_arma, R_arma)

    return solve_discrete_lyapunov(T_arma, V_arma)

# =============================================================================
# Section 5: Kalman Filter Core
# Harvey (1989); Durbin & Koopman (2012)
# =============================================================================

@njit(cache=True)
def _arima_kalman_core(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    delta: np.ndarray,
    a_init: np.ndarray,
    P_init: np.ndarray,
    Pn_init: np.ndarray,
    update_start: int,
    give_resid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Companion-structure Kalman filter for ARIMA likelihood computation.

    Exploits the known sparsity of the ARIMA state transition matrix
    (companion form: first column = AR coefficients, subdiagonal = identity)
    to achieve O(rd²) per-step cost instead of O(rd³) generic matrix operations.

    The ARIMA state-space form follows Durbin & Koopman (2012, §8.4):
      State transition: T has companion structure
      Observation: Z = [1, 0, ..., 0, δ₁, ..., δ_d]
      Innovation: V = R R' where R = [1, θ₁, ..., θ_{r-1}, 0, ..., 0]

    Parameters
    ----------
    y : np.ndarray
        Observed time series.
    phi : np.ndarray
        Expanded AR coefficients [φ₁, ..., φₚ].
    theta : np.ndarray
        Expanded MA coefficients [θ₁, ..., θ_q].
    delta : np.ndarray
        Differencing polynomial coefficients.
    a_init, P_init, Pn_init : np.ndarray
        Initial state vector, filtered covariance, predicted covariance.
    update_start : int
        Time index to begin updating covariance.
    give_resid : bool
        Whether to compute standardized residuals.

    Returns
    -------
    stats : np.ndarray
        [ssq, sumlog, nu] — sum of squares, log-determinant sum, count.
    residuals : np.ndarray
        Standardized residuals (if give_resid=True, else empty).
    a_final : np.ndarray
        Final filtered state vector.
    P_final : np.ndarray
        Final filtered state covariance.
    """
    n = len(y)
    p = len(phi)
    q = len(theta)
    d = len(delta)
    r = max(p, q + 1)
    rd = r + d

    ssq = 0.0
    sumlog = 0.0
    n_valid = 0

    a = a_init.copy()
    P = P_init.copy()
    Pnew = Pn_init.copy()
    anew = np.empty(rd)
    M = np.empty(rd)
    mm = np.empty((rd, rd))

    if give_resid:
        std_residuals = np.zeros(n)
    else:
        std_residuals = np.empty(0)

    for t in range(n):
        # --- State prediction: anew = T @ a ---
        # Companion structure: T[i,0] = phi[i], T[i-1,i] = 1 for ARMA block
        for i in range(r):
            tmp = a[i + 1] if i < r - 1 else 0.0
            if i < p:
                tmp += phi[i] * a[0]
            anew[i] = tmp
        if d > 0:
            tmp = a[0]
            for j in range(d):
                tmp += delta[j] * a[r + j]
            anew[r] = tmp
            for i in range(1, d):
                anew[r + i] = a[r + i - 1]

        if not np.isnan(y[t]):
            # --- Innovation: v_t = y_t - Z' @ anew ---
            # Z = [1, 0, ..., 0, δ₁, ..., δ_d]
            innovation = y[t] - anew[0]
            for j in range(d):
                innovation -= delta[j] * anew[r + j]

            # --- M = Pnew @ Z ---
            for i in range(rd):
                tmp = Pnew[i, 0]
                for j in range(d):
                    tmp += Pnew[i, r + j] * delta[j]
                M[i] = tmp

            # --- F = Z' @ M ---
            F = M[0]
            for j in range(d):
                F += delta[j] * M[r + j]

            if F < 1e4:
                n_valid += 1
                ssq += innovation * innovation / F
                sumlog += np.log(F)

            # --- State update: a = anew + M * v/F ---
            gain = innovation / F
            for i in range(rd):
                a[i] = anew[i] + M[i] * gain

            # --- Covariance update: P = Pnew - M M'/F ---
            inv_F = 1.0 / F
            for i in range(rd):
                for j in range(rd):
                    P[i, j] = Pnew[i, j] - M[i] * M[j] * inv_F

            if give_resid:
                std_residuals[t] = innovation / np.sqrt(F) if F > 0 else np.nan
        else:
            for i in range(rd):
                a[i] = anew[i]
            if give_resid:
                std_residuals[t] = np.nan

        if t > update_start:
            # --- Covariance prediction: Pnew = T @ P @ T' + V ---
            # Step 1: mm = T @ P (companion structure)
            for j in range(rd):
                for i in range(r):
                    tmp = 0.0
                    if i < p:
                        tmp += phi[i] * P[0, j]
                    if i < r - 1:
                        tmp += P[i + 1, j]
                    mm[i, j] = tmp
            if d > 0:
                for j in range(rd):
                    tmp = P[0, j]
                    for k in range(d):
                        tmp += delta[k] * P[r + k, j]
                    mm[r, j] = tmp
                for i in range(1, d):
                    for j in range(rd):
                        mm[r + i, j] = P[r + i - 1, j]

            # Step 2: Pnew = mm @ T' + V (companion structure, transposed)
            for i in range(rd):
                for j in range(r):
                    tmp = 0.0
                    if j < p:
                        tmp += phi[j] * mm[i, 0]
                    if j < r - 1:
                        tmp += mm[i, j + 1]
                    Pnew[i, j] = tmp
            if d > 0:
                for i in range(rd):
                    tmp = mm[i, 0]
                    for k in range(d):
                        tmp += delta[k] * mm[i, r + k]
                    Pnew[i, r] = tmp
                for j in range(1, d):
                    for i in range(rd):
                        Pnew[i, r + j] = mm[i, r + j - 1]

            # Step 3: Add V = R @ R' where R = [1, θ₁, ..., θ_{r-1}, 0, ..., 0]
            for i in range(min(q + 1, r)):
                vi = 1.0 if i == 0 else theta[i - 1]
                for j in range(min(q + 1, r)):
                    vj = 1.0 if j == 0 else theta[j - 1]
                    Pnew[i, j] += vi * vj

            # P = Pnew
            for i in range(rd):
                for j in range(rd):
                    P[i, j] = Pnew[i, j]

    stats = np.array([ssq, sumlog, float(n_valid)])
    return stats, std_residuals, a, P


# =============================================================================
# Section 6: State-Space Construction
# Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*,
# §8.4: State-space form of ARIMA models.
# =============================================================================

def initialize_arima_state(
    phi: np.ndarray,
    theta: np.ndarray,
    Delta: np.ndarray,
    kappa: float = _DIFFUSE_STATE_PRIOR_VARIANCE,
) -> 'StateSpaceArrays':
    """
    Build the state-space representation of an ARIMA model.

    Following Durbin & Koopman (2012, §8.4), the ARMA(p,q) component
    yₜ = φ₁yₜ₋₁ + ··· + φₚyₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ··· + θ_qεₜ₋ᵧ
    is put in state-space form with state dimension r = max(p, q+1):

      αₜ₊₁ = T · αₜ + R · εₜ        (state equation)
      yₜ   = Z' · αₜ + εₜ            (observation equation)

    where:
      T = companion matrix with [φ₁,...,φᵣ] in first column, identity
          subdiagonal (zero-padded if p < r)
      Z = [1, θ₁, θ₂, ..., θᵣ₋₁]   (zero-padded if q < r-1)
      R = [1, 0, ..., 0]'             (first unit vector)
      V = R·R' = e₁·e₁'              (innovation covariance)

    Differencing states are appended for the integrated (I) component,
    with diffuse prior variance κ on their diagonal entries in P₀.

    The initial covariance P₀ for the stationary ARMA block is obtained
    from the discrete Lyapunov equation P₀ = T·P₀·T' + V
    (computed via `compute_q0_covariance_matrix`).

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients [φ₁, ..., φₚ].
    theta : np.ndarray
        MA coefficients [θ₁, ..., θ_q].
    Delta : np.ndarray
        Differencing polynomial coefficients (without leading 1).
    kappa : float
        Prior variance for diffuse (non-stationary) state components.

    Returns
    -------
    StateSpaceArrays
        Complete state-space representation.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)

    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)   # ARMA state dimension
    d = len(Delta)       # differencing state dimension
    state_dim = r + d

    # --- Observation vector Z ---
    # Z = [1, θ₁, ..., θᵣ₋₁, Δ₁, ..., Δ_d]
    Z = np.zeros(state_dim)
    Z[0] = 1.0
    Z[r:state_dim] = Delta

    # --- Transition matrix T ---
    # ARMA block: companion form with φ in first column, identity subdiagonal
    T = np.zeros((state_dim, state_dim))
    if p > 0:
        T[:p, 0] = phi
    if r > 1:
        for i in range(1, r):
            T[i - 1, i] = 1.0

    # Differencing block: integrates the ARMA output
    if d > 0:
        T[r, :] = Z
        for i in range(1, d):
            T[r + i, r + i - 1] = 1.0

    # --- Innovation vector and covariance V = R·R' ---
    # R = [1, θ₁, ..., θᵣ₋₁, 0, ..., 0]
    theta_padded = np.zeros(r - 1)
    theta_padded[:min(q, r - 1)] = theta[:min(q, r - 1)]
    R_vec = np.concatenate([[1.0], theta_padded, np.zeros(d)])
    V = np.outer(R_vec, R_vec)

    # --- Initial state and covariance ---
    h = 0.0
    a0 = np.zeros(state_dim)
    P0 = np.zeros((state_dim, state_dim))
    Pn = np.zeros((state_dim, state_dim))

    # Stationary ARMA block: P₀ from discrete Lyapunov equation
    if r > 1:
        Pn[:r, :r] = compute_q0_covariance_matrix(phi, theta)
    else:
        if p > 0:
            Pn[0, 0] = 1.0 / (1.0 - phi[0] ** 2)
        else:
            Pn[0, 0] = 1.0

    # Diffuse prior for non-stationary differencing states
    for i in range(r, state_dim):
        Pn[i, i] = kappa

    return StateSpaceArrays(
        ar_coefs=phi,
        ma_coefs=theta,
        differencing_poly=Delta,
        observation_vector=Z,
        filtered_state=a0,
        filtered_covariance=P0,
        transition_matrix=T,
        innovation_covariance=V,
        observation_variance=h,
        predicted_covariance=Pn,
    )

def _update_state_space(
    ss: 'StateSpaceArrays',
    phi: np.ndarray,
    theta: np.ndarray,
) -> 'StateSpaceArrays':
    """
    Update state-space arrays for new AR/MA coefficients.

    Re-computes the coefficient-dependent arrays (T first column, V, P₀)
    in an existing StateSpaceArrays, following the same Durbin & Koopman
    (2012, §8.4) construction as `initialize_arima_state`. This avoids
    re-allocating all arrays during iterative optimization.

    Parameters
    ----------
    ss : StateSpaceArrays
        Existing state-space representation to update in-place.
    phi : np.ndarray
        New AR coefficients [φ₁, ..., φₚ].
    theta : np.ndarray
        New MA coefficients [θ₁, ..., θ_q].

    Returns
    -------
    ss : StateSpaceArrays
        The same object, updated with new coefficient values.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    ss.ar_coefs = phi
    ss.ma_coefs = theta

    # Update first column of transition matrix T (companion form)
    if p > 0:
        ss.transition_matrix[:p, 0] = phi

    # V matrix (innovation_covariance = R @ R') is intentionally NOT rebuilt here.
    #
    # Safety rationale:
    #   - During the ML optimization loop, the likelihood is evaluated via
    #     compute_arima_likelihood() → _arima_kalman_core() (a @njit function).
    #     That function receives (phi, theta, delta) directly as arguments and
    #     inlines V = R @ R' on-the-fly inside the covariance-prediction step
    #     ("Step 3: Add V = R @ R'"). It never reads ss.innovation_covariance.
    #   - kalman_forecast() / kalman_forecast_core() DO read
    #     ss.innovation_covariance, but those functions are never called on the
    #     intermediate ss_holder[0] that is mutated here — only on the fully
    #     rebuilt ss_final created at the end of _fit_ml after optimization.
    #
    # Skipping the V rebuild avoids repeated O(r·d²) matrix outer-product
    # allocations across the hundreds of objective evaluations during
    # numerical optimization.
    #
    # INVARIANT: This shortcut is valid only because _fit_ml always calls
    # initialize_arima_state(phi_final, ...) after the optimizer converges,
    # producing a fully consistent StateSpaceArrays (with correct V) before
    # any call to kalman_forecast(). Do NOT pass ss_holder[0] to
    # kalman_forecast() without first rebuilding innovation_covariance.

    # Recompute stationary initial covariance P₀ via Lyapunov equation
    if r > 1:
        ss.predicted_covariance[:r, :r] = compute_q0_covariance_matrix(phi, theta)
    else:
        if p > 0:
            ss.predicted_covariance[0, 0] = 1.0 / (1.0 - phi[0] ** 2)
        else:
            ss.predicted_covariance[0, 0] = 1.0

    # Reset filtered state to zero for new optimization iteration
    ss.filtered_state[:] = 0.0

    return ss


# =============================================================================
# Section 7: Likelihood Wrappers and Forecasting
# =============================================================================

def compute_arima_likelihood(
    y: np.ndarray,
    model: 'StateSpaceArrays',
    update_start: int = 0,
    give_resid: bool = False
) -> DictType[str, Any]:
    """
    Compute the Gaussian log-likelihood for a univariate ARIMA model using the Kalman filter.

    Uses the companion-structure Kalman filter (`_arima_kalman_core`) which
    exploits the known sparsity of the ARIMA transition matrix for O(rd²)
    per-step cost instead of O(rd³) generic matrix operations.

    Parameters
    ----------
    y : np.ndarray
        Observed time series (may contain NaN for missing).
    model : StateSpaceArrays
        State-space representation.
    update_start : int
        Time index at which to begin updating likelihood.
    give_resid : bool
        If True, also return standardized residuals.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'ssq': Sum of squared innovations.
        - 'sumlog': Accumulated log-determinants.
        - 'nu': Number of innovations.
        - 'resid': Standardized residuals (only if give_resid=True).
        - 'a': Final filtered state vector.
        - 'P': Final filtered state covariance.
    """
    y_f64 = np.asarray(y, dtype=np.float64)
    phi = np.asarray(model.ar_coefs, dtype=np.float64)
    theta = np.asarray(model.ma_coefs, dtype=np.float64)
    delta = np.asarray(model.differencing_poly, dtype=np.float64)
    a = model.filtered_state.astype(np.float64)
    P = model.filtered_covariance.astype(np.float64)
    Pn = model.predicted_covariance.astype(np.float64)

    stats, residuals, a_final, P_final = _arima_kalman_core(
        y_f64, phi, theta, delta, a, P, Pn, update_start, give_resid
    )

    result = {
        'ssq': stats[0],
        'sumlog': stats[1],
        'nu': int(stats[2]),
        'a': a_final,
        'P': P_final
    }

    if give_resid:
        result['resid'] = residuals

    return result

@njit(cache=True)
def compute_css_residuals(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    n_cond: int,
    diff_order: int,
    seasonal_period: int,
    seasonal_diff_order: int
) -> Tuple[float, np.ndarray]:  # pragma: no cover
    """
    Conditional sum-of-squares (CSS) residuals for ARIMA estimation.

    Computes one-step-ahead prediction errors from the ARMA recursion
    applied to the differenced series (Hamilton, 1994, §5.2):

      eₜ = wₜ − φ₁wₜ₋₁ − ··· − φₚwₜ₋ₚ − θ₁eₜ₋₁ − ··· − θ_qeₜ₋ᵧ

    where wₜ = Δᵈ Δₛᴰ yₜ is the differenced series and the recursion
    is initialized with e₁ = ··· = e_{n_cond} = 0 (conditional on the
    first n_cond observations being known).

    The CSS estimate of σ² is  Σₜ eₜ² / n_eff  where the sum runs
    over t = n_cond+1, ..., n.

    References
    ----------
    Hamilton (1994), *Time Series Analysis*, §5.2.
    Box, Jenkins & Reinsel (2015), *Time Series Analysis*, Ch. 7.

    Parameters
    ----------
    y : np.ndarray
        Observed time series.
    phi : np.ndarray
        Expanded AR coefficients [φ₁, ..., φₚ].
    theta : np.ndarray
        Expanded MA coefficients [θ₁, ..., θ_q].
    n_cond : int
        Number of conditioning observations (recursion starts after these).
    diff_order : int
        Non-seasonal differencing order d.
    seasonal_period : int
        Seasonal period s.
    seasonal_diff_order : int
        Seasonal differencing order D.

    Returns
    -------
    sigma2 : float
        Estimated innovation variance (CSS).
    resid : np.ndarray
        Residual vector (zero for t < n_cond).
    """
    n = len(y)
    p = len(phi)
    q = len(theta)

    # Apply differencing operators to obtain stationary series wₜ
    # Non-seasonal: Δᵈ = (1 − B)ᵈ applied in-place via forward scan
    w = y.copy()
    for _ in range(diff_order):
        prev = w[0]
        for t in range(1, n):
            current = w[t]
            w[t] = current - prev
            prev = current

    # Seasonal: Δₛᴰ = (1 − Bˢ)ᴰ applied in-place
    for _ in range(seasonal_diff_order):
        lag_buffer = w[:seasonal_period].copy()
        for t in range(seasonal_period, n):
            lagged_val = lag_buffer[t % seasonal_period]
            lag_buffer[t % seasonal_period] = w[t]
            w[t] = w[t] - lagged_val

    # ARMA recursion for one-step-ahead prediction errors
    resid = np.zeros(n)
    sum_sq = 0.0
    n_valid = 0

    for t in range(n_cond, n):
        # eₜ = wₜ − Σⱼ φⱼ wₜ₋ⱼ − Σⱼ θⱼ eₜ₋ⱼ
        e_t = w[t]

        for j in range(p):
            if t - j - 1 >= 0:
                e_t -= phi[j] * w[t - j - 1]

        available_ma_lags = min(t - n_cond, q)
        for j in range(available_ma_lags):
            if t - j - 1 >= 0:
                e_t -= theta[j] * resid[t - j - 1]

        resid[t] = e_t

        if not np.isnan(e_t):
            n_valid += 1
            sum_sq += e_t ** 2

    sigma2 = sum_sq / n_valid if n_valid > 0 else np.nan
    return sigma2, resid

@njit(cache=True, fastmath=True)
def kalman_forecast_core(
    n_ahead: int,
    T: np.ndarray,
    V: np.ndarray,
    Z: np.ndarray,
    a: np.ndarray,
    P: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Core Kalman forecast computation (Numba-compatible).

    Parameters
    ----------
    n_ahead : int
        Number of steps to forecast.
    T : np.ndarray
        State transition matrix.
    V : np.ndarray
        Innovation covariance matrix (R @ R').
    Z : np.ndarray
        Observation vector.
    a : np.ndarray
        Current state vector.
    P : np.ndarray
        Current state covariance.
    h : float
        Observation noise variance.

    Returns
    -------
    forecasts : np.ndarray
        Point forecasts.
    variances : np.ndarray
        Forecast variances.
    a_final : np.ndarray
        Final state vector.
    P_final : np.ndarray
        Final state covariance.
    """
    forecasts = np.zeros(n_ahead)
    variances = np.zeros(n_ahead)

    a_curr = a
    P_curr = P

    for t in range(n_ahead):
        a_curr = T @ a_curr
        forecasts[t] = np.dot(Z, a_curr)

        P_curr = T @ P_curr @ T.T + V
        variances[t] = h + np.dot(Z, P_curr @ Z)

    return forecasts, variances, a_curr, P_curr

def kalman_forecast(
    n_ahead: int,
    ss: 'StateSpaceArrays',
    update: bool = False
) -> DictType[str, Any]:
    """
    Forecast n steps ahead from current state.

    Parameters
    ----------
    n_ahead : int
        Number of forecast steps.
    ss : StateSpaceArrays
        State-space representation.
    update : bool
        If True, also return updated model.

    Returns
    -------
    result : dict
        Dictionary with 'pred', 'var', and optionally 'mod'.
    """
    T = ss.transition_matrix.astype(np.float64)
    V = ss.innovation_covariance.astype(np.float64)
    Z = ss.observation_vector.astype(np.float64)
    a = ss.filtered_state.astype(np.float64)
    P = ss.filtered_covariance.astype(np.float64)
    h = float(ss.observation_variance)

    forecasts, variances, a_final, P_final = kalman_forecast_core(
        n_ahead, T, V, Z, a, P, h
    )

    result = {'pred': forecasts, 'var': variances}

    if update:
        from copy import copy
        updated_ss = copy(ss)
        updated_ss.filtered_state = a_final
        updated_ss.filtered_covariance = P_final
        result['mod'] = updated_ss

    return result


# =============================================================================
# Section 8: Regressor/Coefficient Utilities
# =============================================================================

def _process_exogenous(
    exog: Union[pd.DataFrame, np.ndarray, None],
    n: int
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Process exogenous regressors.

    Parameters
    ----------
    exog : DataFrame, ndarray, or None
        Exogenous regressors.
    n : int
        Expected number of rows.

    Returns
    -------
    exog_matrix : np.ndarray
        Regressor matrix (n x n_exog).
    n_exog : int
        Number of regressors.
    exog_names : list
        Regressor names.

    Raises
    ------
    ValueError
        If row count doesn't match n.
    """
    if exog is None:
        exog_matrix = np.zeros((n, 0))
        n_exog = 0
        exog_names = []
    elif isinstance(exog, pd.DataFrame):
        if len(exog) != n:
            raise ValueError("Lengths of x and exog do not match!")
        exog_matrix = exog.values.astype(np.float64)
        n_exog = exog_matrix.shape[1]
        exog_names = list(exog.columns)
    else:
        exog = np.asarray(exog, dtype=np.float64)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.shape[0] != n:
            raise ValueError("Lengths of x and exog do not match!")
        exog_matrix = exog
        n_exog = exog_matrix.shape[1]
        exog_names = [f"exog{i+1}" for i in range(n_exog)]

    return exog_matrix, n_exog, exog_names

def add_drift_term(
    exog: Union[pd.DataFrame, np.ndarray, None],
    drift: np.ndarray,
    name: str = "intercept"
) -> pd.DataFrame:
    """
    Add a drift/intercept term to exog.

    Parameters
    ----------
    exog : DataFrame, ndarray, or None
        Existing regressors.
    drift : np.ndarray
        Drift term to add.
    name : str
        Name for the drift column.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift term added.
    """
    if exog is None or (isinstance(exog, np.ndarray) and exog.size == 0):
        return pd.DataFrame({name: drift})

    if isinstance(exog, pd.DataFrame):
        df = exog.copy()
        df.insert(0, name, drift)
        return df
    else:
        exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        combined = np.column_stack([drift, exog])
        cols = [name] + [f"exog{i+1}" for i in range(exog.shape[1])]
        return pd.DataFrame(combined, columns=cols)

def _initialize_regressor_params(
    x: np.ndarray,
    exog: np.ndarray,
    mask: np.ndarray,
    narma: int,
    n_exog: int,
    order_d: int,
    seasonal_d: int,
    m: int,
    Delta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, bool, Optional[Any]]:
    """
    Estimate initial exogenous regression coefficients via OLS.

    Provides starting values for the exogenous regressor parameters
    by fitting OLS on the differenced series. For multiple free
    regressors, an SVD rotation is optionally applied for numerical
    conditioning.

    The OLS fit uses `numpy.linalg.lstsq` (minimum-norm least squares)
    which is a standard, well-conditioned approach.

    Parameters
    ----------
    x : np.ndarray
        Target variable (may contain NaN).
    exog : np.ndarray
        Regressor matrix (n × n_exog).
    mask : np.ndarray
        Boolean mask for free parameters.
    narma : int
        Number of ARMA parameters.
    n_exog : int
        Number of exogenous regressors.
    order_d : int
        Non-seasonal differencing order d.
    seasonal_d : int
        Seasonal differencing order D.
    m : int
        Seasonal period.
    Delta : np.ndarray
        Differencing polynomial coefficients.

    Returns
    -------
    init_params : np.ndarray
        Initial parameter values [zeros(narma), beta_hat].
    param_scale : np.ndarray
        Parameter scaling factors for optimizer.
    n_used : int
        Effective number of observations after differencing.
    use_orig_exog : bool
        Whether original (un-rotated) regressors are used.
    svd_rotation : dict or None
        SVD rotation matrices if applied, else None.
    """
    init_arma = np.zeros(narma)
    scale_arma = np.ones(narma)

    x, exog = _ensure_float64_pair(x, exog)

    # Decide whether to apply SVD rotation for conditioning
    use_orig_exog = (n_exog == 1) or np.any(~mask[narma:narma + n_exog])
    svd_rotation = None

    if not use_orig_exog:
        rows_good = np.array([np.all(np.isfinite(row)) for row in exog])
        if np.sum(rows_good) > 0:
            _, _, Vt = np.linalg.svd(exog[rows_good, :], full_matrices=False)
            svd_rotation = {'V': Vt.T}
            exog = exog @ svd_rotation['V']

    # Apply differencing to both series and regressors
    dx, dexog = x, exog
    if order_d > 0:
        dx = diff(dx, lag=1, differences=order_d)
        dexog = diff(dexog, lag=1, differences=order_d)
        dx, dexog = _ensure_float64_pair(dx, dexog)
    if m > 1 and seasonal_d > 0:
        dx = diff(dx, lag=m, differences=seasonal_d)
        dexog = diff(dexog, lag=m, differences=seasonal_d)
        dx, dexog = _ensure_float64_pair(dx, dexog)

    # OLS on differenced data via np.linalg.lstsq
    ols_coef = None
    ols_rank = 0

    if len(dx) > dexog.shape[1] and dexog.shape[1] > 0:
        try:
            valid = ~np.isnan(dx) & np.all(np.isfinite(dexog), axis=1)
            if np.sum(valid) > dexog.shape[1]:
                beta, _, rank, _ = np.linalg.lstsq(dexog[valid], dx[valid], rcond=None)
                ols_coef = beta
                ols_rank = rank if rank is not None else len(beta)
        except Exception as e:
            warnings.warn(f"Fitting OLS to difference data failed: {e}")

    # Fallback: OLS on undifferenced data
    if ols_rank == 0 and n_exog > 0:
        x, exog = _ensure_float64_pair(x, exog)
        valid = ~np.isnan(x) & np.all(np.isfinite(exog), axis=1)
        if np.sum(valid) > exog.shape[1]:
            beta, _, _, _ = np.linalg.lstsq(exog[valid], x[valid], rcond=None)
            ols_coef = beta

    # Effective sample size
    isna = np.isnan(x) | np.array([np.any(np.isnan(row)) for row in exog])
    n_used = int(np.sum(~isna)) - len(Delta)

    if ols_coef is not None:
        init_params = np.concatenate([init_arma, ols_coef])

        # Approximate standard errors for parameter scaling
        valid = ~np.isnan(x) & np.all(np.isfinite(exog), axis=1)
        X_valid, y_valid = exog[valid], x[valid]
        resid = y_valid - X_valid @ ols_coef
        mse = np.sum(resid**2) / max(len(resid) - len(ols_coef), 1)
        try:
            XtX_inv = np.linalg.inv(X_valid.T @ X_valid)
            ses = np.sqrt(np.diag(XtX_inv) * mse)
        except np.linalg.LinAlgError:
            ses = np.ones(n_exog)
        param_scale = np.concatenate([scale_arma, _REGRESSOR_SCALE_FACTOR * ses])
    else:
        init_params = np.concatenate([init_arma, np.zeros(n_exog)])
        param_scale = np.concatenate([scale_arma, np.ones(n_exog)])

    return init_params, param_scale, n_used, use_orig_exog, svd_rotation

def _build_coefficient_dataframe(
    order_spec: SARIMAOrder,
    coef: np.ndarray,
    cn: List[str],
    n_exog: int
) -> pd.DataFrame:
    """
    Construct a DataFrame representing model coefficients.

    Parameters
    ----------
    order_spec : SARIMAOrder
        SARIMA order specification.
    coef : np.ndarray
        Coefficient values.
    cn : list
        Exogenous regressor names.
    n_exog : int
        Number of exogenous regressors.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with named coefficients.
    """
    names = []

    if order_spec.p > 0:
        names.extend([f"ar{i+1}" for i in range(order_spec.p)])
    if order_spec.q > 0:
        names.extend([f"ma{i+1}" for i in range(order_spec.q)])
    if order_spec.P > 0:
        names.extend([f"sar{i+1}" for i in range(order_spec.P)])
    if order_spec.Q > 0:
        names.extend([f"sma{i+1}" for i in range(order_spec.Q)])
    if n_exog > 0:
        names.extend(cn)

    return pd.DataFrame([coef], columns=names)


# =============================================================================
# Section 9: Numerical Optimization Utilities
# =============================================================================

def optim_hessian(func, x, eps=None):
    """
    Compute numerical Hessian matrix using direct finite differences.

    Uses central differences for diagonal elements and forward differences
    for off-diagonal elements, with shared function evaluations to minimize
    the total number of objective function calls.

    Total evaluations: 1 + 2n + n(n-1)/2 vs 2n(n+1) for gradient-of-gradient.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> scalar.
    x : np.ndarray
        Point at which to compute Hessian.
    eps : float or None
        Step size for finite differences. If None, uses 1e-2 which works well
        for ARIMA parameters in transformed space. The default sqrt(machine epsilon)
        (~1.49e-8) is too small and causes numerical instability.

    Returns
    -------
    H : np.ndarray
        Hessian matrix (symmetric).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if eps is None:
        eps = _HESSIAN_STEP_SIZE

    eps_sq = eps * eps

    # 1 evaluation at the center point
    f0 = func(x)

    # Reuse a single perturbation array — mutate in place and restore
    x_pert = x.copy()

    # 2n evaluations for single-parameter perturbations
    f_plus = np.empty(n)
    f_minus = np.empty(n)
    for i in range(n):
        x_pert[i] += eps
        f_plus[i] = func(x_pert)
        x_pert[i] = x[i] - eps
        f_minus[i] = func(x_pert)
        x_pert[i] = x[i]  # restore

    H = np.empty((n, n))

    # Diagonal: central differences  H[i,i] = (f(x+h*e_i) - 2*f(x) + f(x-h*e_i)) / h^2
    for i in range(n):
        H[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / eps_sq

    # Off-diagonal: forward differences with shared evaluations
    # H[i,j] = (f(x+h*e_i+h*e_j) - f(x+h*e_i) - f(x+h*e_j) + f(x)) / h^2
    # n*(n-1)/2 additional evaluations
    for i in range(n):
        x_pert[i] += eps
        for j in range(i + 1, n):
            x_pert[j] += eps
            f_ij = func(x_pert)
            H[i, j] = (f_ij - f_plus[i] - f_plus[j] + f0) / eps_sq
            H[j, i] = H[i, j]
            x_pert[j] = x[j]  # restore j
        x_pert[i] = x[i]  # restore i

    return H


# =============================================================================
# Section 10: Model Fitting
# Harvey (1989), *Forecasting, Structural Time Series Models and the
#   Kalman Filter*, Ch. 3 (exact ML via Kalman filter).
# Hamilton (1994), *Time Series Analysis*, §5.2 (CSS estimation).
# Hyndman & Khandakar (2008), *Automatic Time Series Forecasting*,
#   J. Stat. Software 27(3) (CSS-ML warm-start strategy).
# =============================================================================

@dataclass
class _ArimaConfig:
    """Internal configuration bundle for ARIMA fitting."""
    x: np.ndarray
    y: np.ndarray
    order_spec: SARIMAOrder
    Delta: np.ndarray
    exog_matrix: np.ndarray
    exog_names: list
    n_exog: int
    n_arma_params: int
    free_param_mask: np.ndarray
    fixed: np.ndarray
    init: np.ndarray
    param_scale: np.ndarray
    enforce_stationarity: bool
    n_conditioning_obs: int
    n_used: int
    kappa: float
    method: str
    optim_method: str
    opt_options: dict
    use_original_regressors: bool
    svd_transform: Any
    exog_original: Any
    m: int
    order: tuple
    seasonal: tuple


@dataclass
class _FitResult:
    """Internal result container from ARIMA optimization."""
    params: np.ndarray
    param_covariance: np.ndarray
    converged: bool
    state_space: StateSpaceArrays
    resid: np.ndarray
    sigma2: float
    optim_fun: float
    n_conditioning_obs: int


def _prepare_arima_config(
    x, m, order, seasonal, exog, fit_intercept, enforce_stationarity,
    fixed, init, method, n_cond, optim_method, opt_options, kappa
) -> _ArimaConfig:
    """
    Validate inputs and build the configuration for ARIMA estimation.

    Constructs the differencing polynomial Δ(B) = (1−B)ᵈ · (1−Bˢ)ᴰ,
    processes exogenous regressors, computes initial parameter estimates
    via OLS, and determines the conditioning period for CSS methods.

    Parameters
    ----------
    x : array-like
        Time series data.
    m : int
        Seasonal period.
    order : tuple
        (p, d, q) non-seasonal orders.
    seasonal : tuple
        (P, D, Q) seasonal orders.
    exog : array-like or None
        Exogenous regressors.
    fit_intercept : bool
        Whether to include an intercept (only for d+D=0).
    enforce_stationarity : bool
        Whether to apply the Jones (1980) stationarity transform.
    fixed : array or None
        Fixed parameter values (NaN = free).
    init : array or None
        User-supplied initial parameter values.
    method : str or None
        Estimation method: 'CSS-ML', 'ML', or 'CSS'.
    n_cond : int or None
        Number of conditioning observations for CSS.
    optim_method : str
        scipy.optimize.minimize method.
    opt_options : dict
        Optimizer options.
    kappa : float
        Diffuse state prior variance.

    Returns
    -------
    _ArimaConfig
        Complete configuration for the fitting routines.
    """
    if method is None:
        method = "CSS-ML"
    method = _validate_choice(method, ["CSS-ML", "ML", "CSS"], "method")

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    y = x.copy()

    order_spec = SARIMAOrder(
        p=order[0], d=order[1], q=order[2],
        P=seasonal[0], D=seasonal[1], Q=seasonal[2],
        s=m,
    )
    n_arma_params = order_spec.n_arma_params

    # Build differencing polynomial Δ(B) = (1−B)ᵈ · (1−Bˢ)ᴰ
    # via repeated polynomial multiplication (Cauchy product).
    diff_poly = np.array([1.0])
    for _ in range(order[1]):
        diff_poly = np.convolve(diff_poly, np.array([1.0, -1.0]))
    for _ in range(seasonal[1]):
        seasonal_diff = np.zeros(m + 1)
        seasonal_diff[0] = 1.0
        seasonal_diff[m] = -1.0
        diff_poly = np.convolve(diff_poly, seasonal_diff)
    # Store as negated coefficients (without leading 1), matching
    # the convention used by the state-space observation vector Z.
    Delta = -diff_poly[1:]

    total_diff_order = order[1] + seasonal[1]
    n_used = int(np.sum(~np.isnan(x))) - len(Delta)

    exog_original = exog

    if fit_intercept and total_diff_order == 0:
        exog = add_drift_term(exog, np.ones(n), "intercept")

    exog_matrix, n_exog, exog_names = _process_exogenous(exog, n)

    # CSS-ML falls back to ML if data contain missing values
    if method == "CSS-ML":
        any_missing = np.any(np.isnan(x))
        if n_exog > 0:
            any_missing = any_missing or np.any(np.isnan(exog_matrix))
        if any_missing:
            method = "ML"

    # Conditioning period for CSS: accounts for differencing and AR lags
    if method in ["CSS", "CSS-ML"]:
        n_conditioning_obs = order[1] + seasonal[1] * m
        n_ar_conditioning = order[0] + seasonal[0] * m
        if n_cond is None:
            n_conditioning_obs += n_ar_conditioning
        else:
            n_conditioning_obs += max(n_cond, n_ar_conditioning)
    else:
        n_conditioning_obs = 0

    # Fixed parameter handling
    if fixed is None:
        fixed = np.full(n_arma_params + n_exog, np.nan)
    elif len(fixed) != n_arma_params + n_exog:
        raise ValueError("Wrong length for 'fixed'")

    free_param_mask = np.isnan(fixed)
    all_params_fixed = not np.any(free_param_mask)

    if all_params_fixed:
        enforce_stationarity = False

    if enforce_stationarity:
        sar_indices = np.arange(
            order_spec.p + order_spec.q,
            order_spec.p + order_spec.q + order_spec.P
        )
        ar_fixed = np.any(~free_param_mask[:order_spec.p])
        sar_fixed = len(sar_indices) > 0 and np.any(~free_param_mask[sar_indices])
        if ar_fixed or sar_fixed:
            warnings.warn("Some AR parameters were fixed: Setting enforce_stationarity = False")
            enforce_stationarity = False

    # Initial parameter estimates
    if n_exog > 0:
        init0, param_scale, n_used, use_original_regressors, svd_transform = (
            _initialize_regressor_params(
                x, exog_matrix, free_param_mask, n_arma_params, n_exog,
                order[1], seasonal[1], m, Delta
            )
        )
    else:
        init0 = np.zeros(n_arma_params)
        param_scale = np.ones(n_arma_params)
        use_original_regressors = True
        svd_transform = None

    if n_used <= 0:
        raise ValueError("Too few non-missing observations")

    # Merge user-supplied initial values
    if init is not None:
        if len(init) != len(init0):
            raise ValueError("'init' is of the wrong length")
        missing = np.isnan(init)
        init[missing] = init0[missing]

        # Validate stationarity of user-supplied AR initial values
        if method == "ML":
            if order_spec.p > 0 and not ar_check(init[:order_spec.p]):
                raise ValueError("non-stationary AR part")
            if order_spec.P > 0:
                sar_start = order_spec.p + order_spec.q
                sar_end = sar_start + order_spec.P
                if not ar_check(init[sar_start:sar_end]):
                    raise ValueError("non-stationary seasonal AR part")
    else:
        init = init0.copy()

    return _ArimaConfig(
        x=x, y=y, order_spec=order_spec, Delta=Delta,
        exog_matrix=exog_matrix, exog_names=exog_names, n_exog=n_exog,
        n_arma_params=n_arma_params, free_param_mask=free_param_mask,
        fixed=fixed, init=init, param_scale=param_scale,
        enforce_stationarity=enforce_stationarity,
        n_conditioning_obs=n_conditioning_obs, n_used=n_used,
        kappa=kappa, method=method,
        optim_method=optim_method, opt_options=opt_options,
        use_original_regressors=use_original_regressors,
        svd_transform=svd_transform, exog_original=exog_original,
        m=m, order=order, seasonal=seasonal,
    )


def _fit_css(config: _ArimaConfig) -> _FitResult:
    """
    Fit ARIMA via conditional sum of squares (CSS).

    Minimizes  ½ log(σ²_CSS)  where σ²_CSS = (1/n_eff) Σ eₜ² and eₜ
    are the ARMA recursion residuals on the differenced series.

    The variance-covariance matrix of parameter estimates is obtained
    from the inverse of the observed information (numerical Hessian
    of the objective, scaled by n_eff).

    Reference: Hamilton (1994), *Time Series Analysis*, §5.2.
    """
    c = config
    params = c.fixed.astype(np.float64).copy()
    all_params_fixed = not np.any(c.free_param_mask)

    # Persistent scratch array for objective — avoids copy per call
    _par = c.fixed.astype(np.float64).copy()

    def _css_objective(free_params):
        """CSS objective: ½ log(σ²) where σ² = Σeₜ² / n_eff."""
        _par[c.free_param_mask] = free_params
        try:
            phi_exp, theta_exp = transform_arima_parameters(
                _par, c.order_spec.p, c.order_spec.q, c.order_spec.P,
                c.order_spec.Q, c.order_spec.s, False
            )
        except Exception:
            return _OBJECTIVE_PENALTY
        adjusted = (c.x - c.exog_matrix @ _par[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
        try:
            sigma2, _ = compute_css_residuals(
                adjusted, phi_exp, theta_exp, c.n_conditioning_obs,
                c.order_spec.d, c.order_spec.s, c.order_spec.D
            )
        except Exception:
            return _OBJECTIVE_PENALTY
        if sigma2 <= 0 or np.isnan(sigma2):
            return _OBJECTIVE_PENALTY
        return 0.5 * np.log(sigma2)

    # Optimize
    if all_params_fixed:
        optim_result = {'converged': True, 'x': np.zeros(0), 'fun': _css_objective(np.zeros(0))}
    else:
        result = opt.minimize(
            _css_objective, c.init[c.free_param_mask],
            method=c.optim_method, options=c.opt_options
        )
        optim_result = {'converged': result.success, 'x': result.x, 'fun': result.fun}

    if not optim_result['converged']:
        warnings.warn(
            "CSS optimization convergence issue. Try to increase 'maxiter' or change the optimization method."
        )

    params[c.free_param_mask] = optim_result['x']

    # Build state-space and compute final residuals
    phi_final, theta_final = transform_arima_parameters(
        params, c.order_spec.p, c.order_spec.q, c.order_spec.P,
        c.order_spec.Q, c.order_spec.s, False
    )
    state_space = initialize_arima_state(phi_final, theta_final, c.Delta, kappa=c.kappa)

    adjusted_series = (c.x - c.exog_matrix @ params[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
    compute_arima_likelihood(adjusted_series, state_space, update_start=0, give_resid=True)
    sigma2, resid = compute_css_residuals(
        adjusted_series, phi_final, theta_final, c.n_conditioning_obs,
        c.order_spec.d, c.order_spec.s, c.order_spec.D
    )

    # Variance-covariance from inverse observed information
    if all_params_fixed:
        param_covariance = np.zeros((0, 0))
    else:
        hessian = optim_hessian(_css_objective, optim_result['x'])
        try:
            param_covariance = np.linalg.inv(hessian * c.n_used)
        except np.linalg.LinAlgError:
            n_free = int(np.sum(c.free_param_mask))
            param_covariance = np.zeros((n_free, n_free))

    return _FitResult(
        params=params, param_covariance=param_covariance,
        converged=optim_result['converged'], state_space=state_space,
        resid=resid, sigma2=sigma2, optim_fun=optim_result['fun'],
        n_conditioning_obs=c.n_conditioning_obs,
    )


def _fit_ml(config: _ArimaConfig, warm_start: np.ndarray = None) -> _FitResult:
    """
    Fit ARIMA via exact maximum likelihood using the Kalman filter.

    The exact Gaussian log-likelihood is computed via the prediction error
    decomposition (Harvey, 1989, Ch. 3):

      −2ℓ/n = log(σ²) + (1/n) Σ log(Fₜ)

    where Fₜ is the innovation variance at time t from the Kalman filter
    and σ² = (1/n) Σ vₜ²/Fₜ with vₜ the innovation.

    When `enforce_stationarity=True`, optimization is performed in the
    unconstrained Jones (1980) parameterization, and the variance-covariance
    matrix is obtained via the delta method: Var(θ) = J' H⁻¹ J where J is
    the Jacobian of the transform and H the Hessian of the objective.

    References
    ----------
    Harvey (1989), *Forecasting, Structural Time Series Models and the
    Kalman Filter*, Ch. 3.
    Jones (1980), Technometrics 22(3), pp. 389-395.
    """
    c = config
    params = c.fixed.astype(np.float64).copy()
    all_params_fixed = not np.any(c.free_param_mask)
    init = warm_start if warm_start is not None else c.init.copy()

    # Mutable container for state-space (updated by closure during optimization)
    ss_holder = [None]

    # Persistent scratch array for objective — avoids copy per call
    _par = c.fixed.astype(np.float64).copy()

    def _ml_objective(free_params, use_transform):
        """Negative concentrated log-likelihood via Kalman filter."""
        _par[c.free_param_mask] = free_params
        try:
            phi_exp, theta_exp = transform_arima_parameters(
                _par, c.order_spec.p, c.order_spec.q, c.order_spec.P,
                c.order_spec.Q, c.order_spec.s, use_transform
            )
        except Exception:
            return _OBJECTIVE_PENALTY
        try:
            ss_holder[0] = _update_state_space(ss_holder[0], phi_exp, theta_exp)
        except Exception:
            return _OBJECTIVE_PENALTY
        adjusted = (c.x - c.exog_matrix @ _par[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
        try:
            kf = compute_arima_likelihood(adjusted, ss_holder[0], update_start=0, give_resid=False)
        except Exception:
            return _OBJECTIVE_PENALTY
        sigma2 = kf['ssq'] / kf['nu'] if kf['nu'] > 0 else _OBJECTIVE_PENALTY
        if sigma2 <= 0:
            return _OBJECTIVE_PENALTY
        # Concentrated log-likelihood: ½[log(σ²) + (1/n)Σlog(Fₜ)]
        return 0.5 * (np.log(sigma2) + kf['sumlog'] / kf['nu'])

    # If stationarity enforced, map initial params to unconstrained space
    if c.enforce_stationarity:
        init = inverse_arima_parameter_transform(
            init, c.order_spec.p, c.order_spec.q, c.order_spec.P
        )
        if c.order_spec.q > 0:
            ma_slice = slice(c.order_spec.p, c.order_spec.p + c.order_spec.q)
            init[ma_slice] = ma_invert(init[ma_slice])
        if c.order_spec.Q > 0:
            sma_start = c.order_spec.p + c.order_spec.q + c.order_spec.P
            sma_slice = slice(sma_start, sma_start + c.order_spec.Q)
            init[sma_slice] = ma_invert(init[sma_slice])

    # Initialize state-space for first objective evaluation
    phi_init, theta_init = transform_arima_parameters(
        init, c.order_spec.p, c.order_spec.q, c.order_spec.P,
        c.order_spec.Q, c.order_spec.s, c.enforce_stationarity
    )
    ss_holder[0] = initialize_arima_state(phi_init, theta_init, c.Delta, kappa=c.kappa)

    # Optimize
    if all_params_fixed:
        optim_result = {
            'converged': True, 'x': np.zeros(0),
            'fun': _ml_objective(np.zeros(0), c.enforce_stationarity)
        }
    else:
        obj_fn = lambda p: _ml_objective(p, c.enforce_stationarity)
        minimize_options = dict(c.opt_options)
        # Seasonal ML with exogenous regressors is prone to very long BFGS runs
        # due to finite-difference gradient noise. A modest gtol dramatically
        # reduces iterations while preserving solution quality in practice.
        if (
            c.optim_method == "BFGS"
            and c.exog_original is not None
            and c.n_exog > 0
            and (c.order_spec.P > 0 or c.order_spec.Q > 0)
            and "gtol" not in minimize_options
        ):
            minimize_options["gtol"] = 1e-3
        result = opt.minimize(
            obj_fn, init[c.free_param_mask],
            method=c.optim_method, options=minimize_options
        )
        optim_result = {'converged': result.success, 'x': result.x, 'fun': result.fun}

    if not optim_result['converged']:
        warnings.warn(
            "Possible convergence problem. "
            "Try to increase 'maxiter' or change the optimization method."
        )

    params[c.free_param_mask] = optim_result['x']

    # Compute variance-covariance matrix of parameter estimates
    if c.enforce_stationarity:
        # Ensure MA invertibility on optimized params
        if c.order_spec.q > 0:
            ma_slice = slice(c.order_spec.p, c.order_spec.p + c.order_spec.q)
            if np.all(c.free_param_mask[ma_slice]):
                params[ma_slice] = ma_invert(params[ma_slice])
        if c.order_spec.Q > 0:
            sma_start = c.order_spec.p + c.order_spec.q + c.order_spec.P
            sma_slice = slice(sma_start, sma_start + c.order_spec.Q)
            if np.all(c.free_param_mask[sma_slice]):
                params[sma_slice] = ma_invert(params[sma_slice])

        # Delta method: Var(θ) = J' · (n·H)⁻¹ · J
        hessian = optim_hessian(
            lambda p: _ml_objective(p, True), params[c.free_param_mask]
        )
        J = compute_arima_transform_gradient(
            params, c.order_spec.p, c.order_spec.q, c.order_spec.P
        )
        J_free = J[np.ix_(c.free_param_mask, c.free_param_mask)]
        try:
            param_covariance = J_free.T @ np.linalg.solve(hessian * c.n_used, J_free)
        except np.linalg.LinAlgError:
            n_free = int(np.sum(c.free_param_mask))
            param_covariance = np.zeros((n_free, n_free))

        # Map back to constrained parameter space
        params = undo_arima_parameter_transform(
            params, c.order_spec.p, c.order_spec.q, c.order_spec.P
        )
    else:
        if all_params_fixed:
            param_covariance = np.zeros((0, 0))
        else:
            hessian = optim_hessian(
                lambda p: _ml_objective(p, c.enforce_stationarity),
                optim_result['x']
            )
            try:
                param_covariance = np.linalg.inv(hessian * c.n_used)
            except np.linalg.LinAlgError:
                n_free = int(np.sum(c.free_param_mask))
                param_covariance = np.zeros((n_free, n_free))

    # Final state-space and residuals at optimum
    phi_final, theta_final = transform_arima_parameters(
        params, c.order_spec.p, c.order_spec.q, c.order_spec.P,
        c.order_spec.Q, c.order_spec.s, False
    )
    # Build a fully consistent state-space at the optimum. Unlike the
    # intermediate ss_holder[0] used during the optimization loop (which
    # skips rebuilding innovation_covariance to save allocations — see
    # _update_state_space and OPT-1 in the deep-analysis document), this
    # object has correct innovation_covariance and can safely be passed to
    # kalman_forecast() for predictions.
    ss_final = initialize_arima_state(phi_final, theta_final, c.Delta, kappa=c.kappa)

    adjusted_series = (c.x - c.exog_matrix @ params[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
    kf_final = compute_arima_likelihood(adjusted_series, ss_final, update_start=0, give_resid=True)
    sigma2 = kf_final['ssq'] / c.n_used
    resid = kf_final['resid']

    ss_final.filtered_state = kf_final['a']
    ss_final.filtered_covariance = kf_final['P']

    return _FitResult(
        params=params, param_covariance=param_covariance,
        converged=optim_result['converged'], state_space=ss_final,
        resid=resid, sigma2=sigma2, optim_fun=optim_result['fun'],
        n_conditioning_obs=0,
    )


def _fit_css_ml(config: _ArimaConfig) -> _FitResult:
    """
    Two-stage CSS-ML estimation: CSS warm start followed by exact ML.

    First fits via CSS to obtain good initial parameter estimates,
    then uses those as a warm start for exact ML via the Kalman filter.
    If the CSS stage produces non-stationary AR parameters, the ML
    stage falls back to zero starting values for the AR block.

    Reference: Hyndman & Khandakar (2008), "Automatic Time Series
    Forecasting: The forecast Package for R", J. Stat. Software 27(3).
    """
    c = config
    all_params_fixed = not np.any(c.free_param_mask)

    # Persistent scratch array for objective — avoids copy per call
    _par = c.fixed.astype(np.float64).copy()

    def _css_objective(free_params):
        """CSS objective for warm-start stage."""
        _par[c.free_param_mask] = free_params
        try:
            phi_exp, theta_exp = transform_arima_parameters(
                _par, c.order_spec.p, c.order_spec.q, c.order_spec.P,
                c.order_spec.Q, c.order_spec.s, False
            )
        except Exception:
            return _OBJECTIVE_PENALTY
        adjusted = (c.x - c.exog_matrix @ _par[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
        try:
            sigma2, _ = compute_css_residuals(
                adjusted, phi_exp, theta_exp, c.n_conditioning_obs,
                c.order_spec.d, c.order_spec.s, c.order_spec.D
            )
        except Exception:
            return _OBJECTIVE_PENALTY
        if sigma2 <= 0 or np.isnan(sigma2):
            return _OBJECTIVE_PENALTY
        return 0.5 * np.log(sigma2)

    warm_start = c.init.copy()

    if not all_params_fixed:
        result = opt.minimize(
            _css_objective, c.init[c.free_param_mask],
            method=c.optim_method, options=c.opt_options
        )
        if result.success:
            css_params = c.init.copy()
            css_params[c.free_param_mask] = result.x

            # Check stationarity of CSS estimates before passing to ML
            sar_start = c.order_spec.p + c.order_spec.q
            sar_end = sar_start + c.order_spec.P
            ar_nonstationary = c.order_spec.p > 0 and not ar_check(css_params[:c.order_spec.p])
            sar_nonstationary = c.order_spec.P > 0 and not ar_check(css_params[sar_start:sar_end])

            if ar_nonstationary or sar_nonstationary:
                warnings.warn(
                    "CSS optimization produced non-stationary parameters. "
                    "Falling back to ML estimation with zero starting values."
                )
                if c.order_spec.p > 0:
                    warm_start[:c.order_spec.p] = 0.0
                if c.order_spec.P > 0:
                    warm_start[sar_start:sar_end] = 0.0
            else:
                warm_start[c.free_param_mask] = result.x

    return _fit_ml(config, warm_start=warm_start)


def _build_arima_result(config: _ArimaConfig, fit: _FitResult) -> ArimaResult:
    """
    Assemble the final ArimaResult from configuration and fit output.

    Computes log-likelihood and AIC from the concentrated log-likelihood
    returned by the optimizer:

      −2ℓ = 2n·f(θ̂) + n + n·log(2π)

    where f(θ̂) is the minimized objective value and n is the effective
    sample size. The AIC follows the standard definition:

      AIC = −2ℓ + 2k

    where k = number of free parameters + 1 (for σ²).

    If SVD rotation was applied to regressors during initialization,
    the coefficients and their covariance are rotated back to the
    original regressor basis.

    Reference: Harvey (1989), Ch. 3; Akaike (1974).
    """
    c = config

    # Log-likelihood from concentrated form
    neg_twice_loglik = 2 * c.n_used * fit.optim_fun + c.n_used + c.n_used * np.log(2 * np.pi)
    loglik = -0.5 * neg_twice_loglik

    # AIC = -2ℓ + 2(k + 1) where +1 accounts for σ² (not computed for CSS)
    n_free = int(np.sum(c.free_param_mask))
    aic = neg_twice_loglik + 2 * n_free + 2 if c.method != "CSS" else np.nan

    params = fit.params
    param_covariance = fit.param_covariance

    # Undo SVD rotation on exogenous coefficients if applied
    if c.n_exog > 0 and not c.use_original_regressors and c.svd_transform is not None:
        exog_slice = slice(c.n_arma_params, c.n_arma_params + c.n_exog)
        params[exog_slice] = c.svd_transform['V'] @ params[exog_slice]
        rotation = np.eye(c.n_arma_params + c.n_exog)
        exog_indices = range(c.n_arma_params, c.n_arma_params + c.n_exog)
        rotation[np.ix_(exog_indices, exog_indices)] = c.svd_transform['V']
        rotation_free = rotation[np.ix_(c.free_param_mask, c.free_param_mask)]
        param_covariance = rotation_free @ param_covariance @ rotation_free.T

    coef_df = _build_coefficient_dataframe(c.order_spec, params, c.exog_names, c.n_exog)
    fitted_vals = c.y - fit.resid

    # Model description string
    order_str = f"({c.order[0]},{c.order[1]},{c.order[2]})"
    seasonal_str = f"({c.seasonal[0]},{c.seasonal[1]},{c.seasonal[2]})[{c.m}]"
    if c.n_exog > 0:
        method_desc = f"Regression with ARIMA{order_str}{seasonal_str} errors"
    else:
        method_desc = f"ARIMA{order_str}{seasonal_str}"

    return ArimaResult(
        y=c.y,
        fitted_values=fitted_vals,
        coefficients=coef_df,
        sigma2=float(np.sum(fit.resid**2) / c.n_used),
        param_covariance=param_covariance,
        param_mask=c.free_param_mask,
        loglik=loglik,
        aic=aic,
        bic=None,
        aicc=None,
        ic=None,
        order=c.order_spec,
        residuals=fit.resid,
        converged=fit.converged,
        n_cond=fit.n_conditioning_obs,
        nobs=c.n_used,
        state_space=fit.state_space,
        exog=c.exog_original,
        method=method_desc,
        lambda_bc=None,
        biasadj=None,
        offset=None,
    )


def arima(
    x: np.ndarray,
    m: int = 1,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Tuple[int, int, int] = (0, 0, 0),
    exog: Union[pd.DataFrame, np.ndarray, None] = None,
    fit_intercept: bool = True,
    enforce_stationarity: bool = True,
    fixed: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    method: Optional[str] = "CSS-ML",
    n_cond: Optional[int] = None,
    optim_method: str = "BFGS",
    opt_options: DictType = {'maxiter': 1000},
    kappa: float = _DIFFUSE_STATE_PRIOR_VARIANCE
) -> ArimaResult:
    """
    Fit a seasonal ARIMA model.

    Supports three estimation methods:
    - **CSS**: Conditional sum of squares (Hamilton, 1994, §5.2)
    - **ML**: Exact maximum likelihood via the Kalman filter
      (Harvey, 1989, Ch. 3; Durbin & Koopman, 2012)
    - **CSS-ML**: CSS warm start followed by ML refinement
      (Hyndman & Khandakar, 2008)

    The multiplicative seasonal ARIMA(p,d,q)(P,D,Q)[s] model is:
      φ(B)·Φ(Bˢ)·Δᵈ·Δₛᴰ yₜ = θ(B)·Θ(Bˢ) εₜ

    When ``enforce_stationarity=True``, the Jones (1980) parameterization
    is used to ensure stationarity of AR polynomials during optimization.

    Parameters
    ----------
    x : array-like
        Time series data.
    m : int
        Seasonal period (1 for non-seasonal).
    order : tuple of int
        (p, d, q) — non-seasonal AR order, differencing, MA order.
    seasonal : tuple of int
        (P, D, Q) — seasonal AR, differencing, MA orders.
    exog : DataFrame, ndarray, or None
        Exogenous regressors.
    fit_intercept : bool
        Include intercept/mean term (only when d + D = 0).
    enforce_stationarity : bool
        Apply Jones (1980) transform to ensure stationarity.
    fixed : ndarray or None
        Fixed parameter values (NaN entries are free).
    init : ndarray or None
        Initial parameter values for optimization.
    method : str
        Estimation method: 'CSS-ML', 'ML', or 'CSS'.
    n_cond : int or None
        Number of conditioning observations for CSS.
    optim_method : str
        Method for `scipy.optimize.minimize`.
    opt_options : dict
        Options for the optimizer.
    kappa : float
        Prior variance for diffuse state components.

    Returns
    -------
    ArimaResult
        Fitted model results.
    """
    # Pipeline: validate → configure → fit → assemble result
    config = _prepare_arima_config(
        x, m, order, seasonal, exog, fit_intercept, enforce_stationarity,
        fixed, init, method, n_cond, optim_method, opt_options, kappa
    )

    if config.method == "CSS":
        fit = _fit_css(config)
    elif config.method == "ML":
        fit = _fit_ml(config)
    else:
        fit = _fit_css_ml(config)

    return _build_arima_result(config, fit)


# =============================================================================
# Section 11: Prediction and Diagnostics
# =============================================================================
def predict_arima(
    model: DictType[str, Any],
    n_ahead: int = 1,
    new_exog: Union[pd.DataFrame, np.ndarray, None] = None,
    se_fit: bool = True,
    level: Union[List[float], np.ndarray, None] = None
) -> DictType[str, Any]:
    """
    Generate forecasts from a fitted ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model from arima().
    n_ahead : int
        Number of periods to forecast.
    new_exog : DataFrame, ndarray, or None
        New exogenous regressors for forecast period.
    se_fit : bool
        Whether to compute standard errors.
    level : list of float, default None
        Confidence levels for prediction intervals (default [80, 95]).
        Values can be percentages (80, 95) or proportions (0.80, 0.95).

    Returns
    -------
    result : dict
        Dictionary with:
        - 'mean': Point forecasts
        - 'lower': Lower bounds of prediction intervals
        - 'upper': Upper bounds of prediction intervals
        - 'level': Confidence levels used
        - 'se': Standard errors
        - 'y': Original data
        - 'fitted': In-sample fitted values
        - 'residuals': Model residuals
        - 'method': Model description
    """
    if model.get('method') == 'Error model':
        raise ValueError(
            "Cannot generate forecasts from an error model: the original fit "
            "failed. Check that the series has sufficient observations and that "
            "the ARIMA order is identifiable."
        )

    order_spec = model['order_spec']
    coef_df = model['coef']
    coefs = coef_df.values.flatten()
    coef_names = list(coef_df.columns)
    narma = order_spec.n_arma_params
    ncoefs = len(coefs)

    if level is not None:
        levels = list(level)
        if min(levels) > 0 and max(levels) < 1:
            levels = [l * 100 for l in levels]
        if min(levels) < 0 or max(levels) > 99.99:
            raise ValueError("Confidence level out of range")
        levels = sorted(levels)
    else:
        levels = []

    # Check for intercept
    intercept_idx = coef_names.index("intercept") if "intercept" in coef_names else None
    has_intercept = intercept_idx is not None

    # Handle exog
    if model['exog'] is not None:
        if isinstance(model['exog'], pd.DataFrame):
            n_exog = model['exog'].shape[1]
        else:
            n_exog = model['exog'].shape[1] if model['exog'].ndim > 1 else 1
    else:
        n_exog = 0

    if new_exog is not None:
        if isinstance(new_exog, pd.DataFrame):
            new_exog = new_exog.values
        new_exog = np.asarray(new_exog, dtype=np.float64)
        if new_exog.ndim == 1:
            new_exog = new_exog.reshape(-1, 1)

    # Compute exog contribution to forecasts
    xm = np.zeros(n_ahead)

    if ncoefs > narma:
        if has_intercept and coef_names[narma] == "intercept":
            intercept_col = np.ones((n_ahead, 1))
            if new_exog is None:
                forecast_exog = intercept_col
            else:
                forecast_exog = np.column_stack([intercept_col, new_exog])
            reg_coef_inds = slice(narma, ncoefs)
        else:
            forecast_exog = new_exog
            reg_coef_inds = slice(narma, ncoefs)

        if forecast_exog is not None:
            if narma == 0:
                xm = forecast_exog @ coefs
            else:
                xm = forecast_exog @ coefs[reg_coef_inds]

    # Kalman forecast
    forecast_result = kalman_forecast(n_ahead, model['state_space'], update=False)
    pred = forecast_result['pred'] + xm

    if se_fit:
        se = np.sqrt(forecast_result['var'] * model['sigma2'])
    else:
        se = np.full(len(pred), np.nan)

    lower = None
    upper = None
    if levels:
        alpha_lvls = 1.0 - np.asarray(levels, dtype=np.float64) / 100.0
        z_scores = norm.ppf(1.0 - alpha_lvls / 2.0)
        se_expanded = se[:, np.newaxis]
        mean_expanded = pred[:, np.newaxis]
        lower = mean_expanded - z_scores * se_expanded  # lower bounds
        upper = mean_expanded + z_scores * se_expanded  # upper bounds

    return {
        'mean': pred,
        'lower': lower,
        'upper': upper,
        'level': levels,
        'se': se,
        'y': model['y'],
        'fitted': model['fitted'],
        'residuals': model['residuals'],
        'method': model['method']
    }

def fitted_values(model: DictType[str, Any]) -> np.ndarray:
    """
    Extract fitted values from ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model.

    Returns
    -------
    np.ndarray
        Fitted values.
    """
    return model['fitted']

def residuals_arima(model: DictType[str, Any]) -> np.ndarray:
    """
    Extract residuals from ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model.

    Returns
    -------
    np.ndarray
        Model residuals.
    """
    return model['residuals']

