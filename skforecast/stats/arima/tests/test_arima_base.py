# Unit test _arima_base
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.stats.arima._arima_base import (
    compute_arima_likelihood,
    transform_unconstrained_to_ar_params,
    inverse_ar_parameter_transform,
    compute_q0_covariance_matrix,
    transform_arima_parameters,
    compute_css_residuals,
    initialize_arima_state,
    _update_state_space,
    ar_check,
    ma_invert,
    kalman_forecast,
    _validate_choice,
    diff,
    _process_exogenous,
    add_drift_term,
    arima,
    predict_arima,
    fitted_values,
    residuals_arima
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def simple_ar1_series():
    """Generate a simple AR(1) series for testing."""
    np.random.seed(42)
    n = 100
    phi = 0.7
    y = np.zeros(n)
    y[0] = np.random.randn()
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.randn()
    return y


# =============================================================================
# Tests for transition matrix (T @ a replaces state_prediction)
# =============================================================================
def test_transition_matrix_ar1():
    """Test transition matrix for AR(1) model gives correct state prediction."""
    ss = initialize_arima_state(
        phi=np.array([0.5]), theta=np.array([]), Delta=np.array([])
    )
    T = ss.transition_matrix
    a = np.array([1.0])

    anew = T @ a

    # For AR(1) with a[0]=1: anew[0] = phi[0] * a[0] = 0.5
    assert anew.shape == (1,)
    np.testing.assert_allclose(anew[0], 0.5, rtol=1e-10)


def test_transition_matrix_ar2():
    """Test transition matrix for AR(2) model gives correct state prediction."""
    ss = initialize_arima_state(
        phi=np.array([0.5, 0.3]), theta=np.array([]), Delta=np.array([])
    )
    T = ss.transition_matrix
    a = np.array([1.0, 0.5])

    anew = T @ a

    # anew[0] = a[1] + phi[0]*a[0] = 0.5 + 0.5*1.0 = 1.0
    # anew[1] = phi[1]*a[0] = 0.3*1.0 = 0.3
    assert anew.shape == (2,)
    np.testing.assert_allclose(anew[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(anew[1], 0.3, rtol=1e-10)


def test_transition_matrix_with_differencing():
    """Test transition matrix with differencing component."""
    ss = initialize_arima_state(
        phi=np.array([0.5]), theta=np.array([]), Delta=np.array([1.0])
    )
    T = ss.transition_matrix
    a = np.array([1.0, 2.0])

    anew = T @ a

    # anew[0] = phi[0]*a[0] = 0.5
    # anew[1] = a[0] + delta[0]*a[1] = 1.0 + 1.0*2.0 = 3.0
    assert anew.shape == (2,)
    np.testing.assert_allclose(anew[0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(anew[1], 3.0, rtol=1e-10)


# =============================================================================
# Tests for covariance prediction (T @ P @ T.T + V)
# =============================================================================
def test_covariance_prediction_ar1():
    """Test covariance prediction for AR(1) without differencing."""
    ss = initialize_arima_state(
        phi=np.array([0.5]), theta=np.array([]), Delta=np.array([])
    )
    T = ss.transition_matrix
    V = ss.innovation_covariance
    P = np.array([[1.0]])

    Pnew = T @ P @ T.T + V

    # For AR(1): Pnew = phi^2 * P + 1
    expected = 0.5**2 * 1.0 + 1.0
    assert Pnew.shape == (1, 1)
    np.testing.assert_allclose(Pnew[0, 0], expected, rtol=1e-10)


def test_covariance_prediction_is_symmetric():
    """Test that covariance prediction produces symmetric matrix."""
    ss = initialize_arima_state(
        phi=np.array([0.5, 0.2]), theta=np.array([0.3]), Delta=np.array([])
    )
    T = ss.transition_matrix
    V = ss.innovation_covariance
    P = np.eye(2) * 0.5

    Pnew = T @ P @ T.T + V

    assert Pnew.shape == (2, 2)
    np.testing.assert_allclose(Pnew, Pnew.T, rtol=1e-10)


def test_covariance_prediction_with_diff_shape():
    """Test covariance prediction with differencing has correct shape."""
    ss = initialize_arima_state(
        phi=np.array([0.5]), theta=np.array([]), Delta=np.array([1.0])
    )
    T = ss.transition_matrix
    V = ss.innovation_covariance
    rd = T.shape[0]
    P = np.eye(rd) * 0.5

    Pnew = T @ P @ T.T + V

    assert Pnew.shape == (rd, rd)


# =============================================================================
# Tests for parameter transformations
# =============================================================================
def test_transform_unconstrained_to_ar_params_single():
    """Test transformation for single AR parameter."""
    raw = np.array([0.5])
    p = 1
    
    result = transform_unconstrained_to_ar_params(p, raw)
    
    # tanh(0.5) ≈ 0.4621
    expected = np.tanh(0.5)
    np.testing.assert_allclose(result[0], expected, rtol=1e-10)


def test_transform_unconstrained_to_ar_params_produces_stationary():
    """Test that transformation produces stationary AR coefficients."""
    np.random.seed(42)
    raw = np.random.randn(3)
    p = 3
    
    result = transform_unconstrained_to_ar_params(p, raw)
    
    # All transformed values should be in (-1, 1)
    assert np.all(np.abs(result) < 1.0)
    # AR polynomial should be stationary
    assert ar_check(result)


def test_inverse_ar_parameter_transform_roundtrip():
    """Test that inverse transformation is the inverse of forward transformation."""
    np.random.seed(42)
    raw = np.random.randn(2) * 0.5  # Small values for stability
    p = 2
    
    forward = transform_unconstrained_to_ar_params(p, raw)
    inverse = inverse_ar_parameter_transform(forward)
    
    np.testing.assert_allclose(inverse, raw[:p], rtol=1e-6)


# =============================================================================
# Tests for Q0 covariance matrix computation
# =============================================================================
def test_compute_q0_covariance_matrix_pure_ma():
    """Test Q0 computation for pure MA model."""
    phi = np.array([])
    theta = np.array([0.5])
    
    Q0 = compute_q0_covariance_matrix(phi, theta)
    
    # For MA(1): r = max(p, q+1) = max(0, 2) = 2
    # Q0 is r x r matrix
    assert Q0.shape == (2, 2)
    # Q0[0,0] = 1.0 for pure MA in Gardner method (variance)
    assert Q0[0, 0] >= 1.0
    # Matrix should be positive semi-definite
    assert np.all(np.linalg.eigvalsh(Q0) >= -1e-10)


def test_compute_q0_covariance_matrix_ar1():
    """Test Q0 computation for AR(1) model."""
    phi = np.array([0.5])
    theta = np.array([])
    
    Q0 = compute_q0_covariance_matrix(phi, theta)
    
    # For AR(1): Q0 = 1/(1-phi^2)
    expected = 1.0 / (1.0 - 0.5**2)
    assert Q0.shape == (1, 1)
    np.testing.assert_allclose(Q0[0, 0], expected, rtol=1e-6)


def test_compute_q0_covariance_matrix_is_symmetric():
    """Test that Q0 is symmetric."""
    phi = np.array([0.5, 0.2])
    theta = np.array([0.3])
    
    Q0 = compute_q0_covariance_matrix(phi, theta)
    
    np.testing.assert_allclose(Q0, Q0.T, rtol=1e-10)


# =============================================================================
# Tests for ar_check and ma_invert
# =============================================================================
def test_ar_check_stationary():
    """Test ar_check returns True for stationary AR coefficients."""
    # AR(1) with |phi| < 1 is stationary
    assert ar_check(np.array([0.5])) == True
    assert ar_check(np.array([0.9])) == True
    assert ar_check(np.array([-0.5])) == True


def test_ar_check_nonstationary():
    """Test ar_check returns False for non-stationary AR coefficients."""
    # AR(1) with |phi| >= 1 is not stationary
    assert ar_check(np.array([1.0])) == False
    assert ar_check(np.array([1.1])) == False
    assert ar_check(np.array([-1.2])) == False


def test_ar_check_empty():
    """Test ar_check with empty array."""
    assert ar_check(np.array([])) == True


def test_ma_invert_invertible():
    """Test ma_invert with already invertible MA coefficients."""
    ma = np.array([0.3])
    result = ma_invert(ma)
    # Should return same coefficients if already invertible
    np.testing.assert_allclose(result, ma, rtol=1e-6)


def test_ma_invert_empty():
    """Test ma_invert with empty array."""
    ma = np.array([])
    result = ma_invert(ma)
    assert len(result) == 0


def test_ma_invert_non_invertible():
    """Test ma_invert reflects roots inside unit circle outside."""
    # MA coefficient with root inside unit circle (non-invertible)
    # For MA(1): root is at -1/theta. If theta=2, root is at -0.5 (inside unit circle)
    ma = np.array([2.0])
    result = ma_invert(ma)
    # The inverted coefficient should make the polynomial invertible
    # Inverted root: 1/(-0.5) = -2, so theta becomes 1/2 = 0.5
    np.testing.assert_allclose(result[0], 0.5, rtol=1e-6)


def test_ma_invert_all_zeros_after_first():
    """Test ma_invert with zeros returns original."""
    ma = np.array([0.0, 0.0, 0.0])
    result = ma_invert(ma)
    # q0=0 case
    np.testing.assert_array_equal(result, ma)


# =============================================================================
# Tests for initialize_arima_state and _update_state_space
# =============================================================================
def test_initialize_arima_state_ar1():
    """Test state-space initialization for AR(1) model."""
    from skforecast.stats.arima._arima_base import StateSpaceArrays
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])

    ss = initialize_arima_state(phi, theta, Delta)

    assert isinstance(ss, StateSpaceArrays)

    # Check dimensions
    assert ss.filtered_state.shape == (1,)
    assert ss.filtered_covariance.shape == (1, 1)
    assert ss.transition_matrix.shape == (1, 1)

    # T[0,0] should equal phi[0]
    np.testing.assert_allclose(ss.transition_matrix[0, 0], 0.5, rtol=1e-10)


def test_initialize_arima_state_with_differencing():
    """Test state-space initialization with differencing."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([1.0])  # d=1

    ss = initialize_arima_state(phi, theta, Delta)

    rd = 2  # r=1, d=1
    assert ss.filtered_state.shape == (rd,)
    assert ss.filtered_covariance.shape == (rd, rd)
    assert ss.transition_matrix.shape == (rd, rd)

    # Diffuse prior for differencing state
    assert ss.predicted_covariance[1, 1] > 1e5


def test_update_arima_changes_coefficients():
    """Test that _update_state_space correctly updates AR/MA coefficients."""
    phi_init = np.array([0.5])
    theta_init = np.array([])
    Delta = np.array([])

    ss = initialize_arima_state(phi_init, theta_init, Delta)

    phi_new = np.array([0.7])
    theta_new = np.array([])

    updated = _update_state_space(ss, phi_new, theta_new)

    np.testing.assert_allclose(updated.ar_coefs, phi_new, rtol=1e-10)
    np.testing.assert_allclose(updated.transition_matrix[0, 0], 0.7, rtol=1e-10)


# =============================================================================
# Tests for compute_arima_likelihood
# =============================================================================
def test_compute_arima_likelihood_returns_dict(simple_ar1_series):
    """Test that compute_arima_likelihood returns expected structure."""
    y = simple_ar1_series
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta)
    result = compute_arima_likelihood(y, model, give_resid=True)
    
    assert 'ssq' in result
    assert 'sumlog' in result
    assert 'nu' in result
    assert 'resid' in result
    assert 'a' in result
    assert 'P' in result
    
    assert result['nu'] == len(y)
    assert len(result['resid']) == len(y)


def test_compute_arima_likelihood_handles_missing():
    """Test likelihood computation with missing values."""
    y = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta)
    result = compute_arima_likelihood(y, model, give_resid=True)
    
    # nu should be less than total length due to NaN
    assert result['nu'] < len(y)
    # Residuals at NaN positions should be NaN
    assert np.isnan(result['resid'][1])
    assert np.isnan(result['resid'][4])


# =============================================================================
# Tests for kalman_forecast
# =============================================================================
def test_kalman_forecast_returns_correct_length():
    """Test that forecast returns correct number of predictions."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])

    ss = initialize_arima_state(phi, theta, Delta)
    ss.filtered_state = np.array([1.0])  # Set initial state

    n_ahead = 10
    result = kalman_forecast(n_ahead, ss)

    assert 'pred' in result
    assert 'var' in result
    assert len(result['pred']) == n_ahead
    assert len(result['var']) == n_ahead


def test_kalman_forecast_variances_increase():
    """Test that forecast variances generally increase with horizon."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])

    ss = initialize_arima_state(phi, theta, Delta)

    result = kalman_forecast(10, ss)

    # For most models, variance should not decrease
    # (allowing for numerical tolerance)
    for i in range(1, len(result['var'])):
        assert result['var'][i] >= result['var'][i-1] - 1e-10


def test_kalman_forecast_with_update():
    """Test that kalman_forecast with update=True returns complete result."""
    from skforecast.stats.arima._arima_base import StateSpaceArrays
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])

    ss = initialize_arima_state(phi, theta, Delta)
    ss.filtered_state = np.array([1.0])

    result = kalman_forecast(5, ss, update=True)

    # Should have predictions and variances
    assert len(result['pred']) == 5
    assert len(result['var']) == 5
    # Result should contain updated state-space model (since update=True)
    assert 'mod' in result
    assert isinstance(result['mod'], StateSpaceArrays)


# =============================================================================
# Tests for utility functions
# =============================================================================
def test_validate_choice_exact():
    """Test _validate_choice with exact match."""
    result = _validate_choice("ML", ["CSS-ML", "ML", "CSS"], "method")
    assert result == "ML"

    result = _validate_choice("CSS", ["CSS", "CSS-ML", "ML"], "method")
    assert result == "CSS"


def test_validate_choice_raises_on_invalid():
    """Test _validate_choice raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Invalid `method`"):
        _validate_choice("invalid", ["CSS-ML", "ML", "CSS"], "method")


def test_diff_simple():
    """Test diff with simple differencing."""
    x = np.array([1.0, 3.0, 6.0, 10.0])
    result = diff(x)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result, expected)


def test_diff_second_order():
    """Test diff with second-order differencing."""
    x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    result = diff(x, differences=2)
    expected = np.array([1.0, 1.0, 1.0])  # d^2 of quadratic is constant
    np.testing.assert_array_equal(result, expected)


def test_diff_with_lag():
    """Test diff with lag > 1 (seasonal differencing)."""
    x = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
    result = diff(x, lag=3)
    expected = np.array([4.0, 5.0, 6.0])  # x[3:] - x[:-3]
    np.testing.assert_array_equal(result, expected)


def test_diff_2d_array():
    """Test diff with 2D array."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0], [10.0, 11.0]])
    result = diff(x, lag=1, differences=1)
    expected = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    np.testing.assert_array_equal(result, expected)




def test_process_exogenous_none():
    """Test _process_exogenous with None input."""
    exog, n_exog, exog_names = _process_exogenous(None, 10)
    assert exog.shape == (10, 0)
    assert n_exog == 0
    assert exog_names == []


def test_process_exogenous_dataframe():
    """Test _process_exogenous with DataFrame input."""
    df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
    exog, n_exog, exog_names = _process_exogenous(df, 3)

    assert exog.shape == (3, 2)
    assert n_exog == 2
    assert exog_names == ['x1', 'x2']


def test_process_exogenous_raises_on_length_mismatch():
    """Test _process_exogenous raises error on length mismatch."""
    df = pd.DataFrame({'x1': [1, 2, 3]})
    with pytest.raises(ValueError, match="do not match"):
        _process_exogenous(df, 5)


def test_add_drift_term_to_none():
    """Test add_drift_term with None xreg."""
    drift = np.array([1, 1, 1])
    result = add_drift_term(None, drift)
    
    assert isinstance(result, pd.DataFrame)
    assert 'intercept' in result.columns
    assert len(result) == 3


def test_add_drift_term_to_dataframe():
    """Test add_drift_term with existing DataFrame."""
    df = pd.DataFrame({'x1': [1, 2, 3]})
    drift = np.array([1, 1, 1])
    result = add_drift_term(df, drift)
    
    assert list(result.columns) == ['intercept', 'x1']
    assert len(result) == 3


def test_add_drift_term_to_ndarray():
    """Test add_drift_term with numpy array xreg."""
    xreg = np.array([[1, 4], [2, 5], [3, 6]])
    drift = np.array([1, 1, 1])
    result = add_drift_term(xreg, drift, name="drift")
    
    assert isinstance(result, pd.DataFrame)
    assert 'drift' in result.columns
    assert result.shape == (3, 3)


# =============================================================================
# Tests for arima function (integration tests)
# =============================================================================
def test_arima_ar1_fit(simple_ar1_series):
    """Test fitting AR(1) model."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    assert result['converged'] == True
    assert 'coef' in result
    assert 'sigma2' in result
    assert 'residuals' in result
    assert 'fitted' in result
    
    # Check coefficient is close to true value (0.7)
    ar_coef = result['coef']['ar1'].values[0]
    assert 0.4 < ar_coef < 1.0  # Reasonable range


def test_arima_with_differencing(simple_ar1_series):
    """Test ARIMA with differencing."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 1, 0), fit_intercept=False)
    
    assert result['converged'] == True
    assert result['order_spec'].d == 1


def test_arima_with_exog(simple_ar1_series):
    """Test ARIMA with exogenous regressors."""
    y = simple_ar1_series
    xreg = pd.DataFrame({'x1': np.random.randn(len(y))})

    result = arima(y, order=(1, 0, 0), exog=xreg, fit_intercept=False)
    
    assert result['converged'] == True
    assert 'x1' in result['coef'].columns


def test_arima_css_method(simple_ar1_series):
    """Test ARIMA with CSS estimation method."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), method="CSS", fit_intercept=False)
    
    assert result['converged'] == True


def test_arima_ml_method(simple_ar1_series):
    """Test ARIMA with ML estimation method."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), method="ML", fit_intercept=False)
    
    assert result['converged'] == True


def test_arima_with_missing_values():
    """Test ARIMA handles missing values in series."""
    np.random.seed(42)
    y = np.random.randn(50)
    y[10] = np.nan  # Add a missing value
    
    # ML method should be used automatically with missing values
    result = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    # Should converge (ML method handles missing)
    assert 'residuals' in result
    assert len(result['residuals']) == len(y)


def test_arima_residuals_length(simple_ar1_series):
    """Test that residuals have same length as input."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    assert len(result['residuals']) == len(y)
    assert len(result['fitted']) == len(y)


# =============================================================================
# Tests for predict_arima
# =============================================================================
def test_predict_arima_basic(simple_ar1_series):
    """Test basic prediction from fitted ARIMA model."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    n_ahead = 5
    result = predict_arima(model, n_ahead=n_ahead)
    
    assert 'mean' in result
    assert 'se' in result
    assert len(result['mean']) == n_ahead
    assert len(result['se']) == n_ahead


def test_predict_arima_with_intervals(simple_ar1_series):
    """Test prediction with confidence intervals."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    result = predict_arima(model, n_ahead=5, level=[80, 95])
    
    assert result['lower'] is not None
    assert result['upper'] is not None
    assert result['lower'].shape == (5, 2)
    assert result['upper'].shape == (5, 2)
    # Lower bounds should be less than upper bounds
    assert np.all(result['lower'] < result['upper'])


def test_predict_arima_with_exog(simple_ar1_series):
    """Test prediction with exogenous regressors."""
    y = simple_ar1_series
    n = len(y)
    xreg = pd.DataFrame({'x1': np.random.randn(n)})

    model = arima(y, order=(1, 0, 0), exog=xreg, fit_intercept=False)

    newxreg = pd.DataFrame({'x1': np.random.randn(5)})
    result = predict_arima(model, n_ahead=5, new_exog=newxreg)
    
    assert len(result['mean']) == 5


# =============================================================================
# Tests for fitted_values and residuals_arima
# =============================================================================
def test_fitted_values_extraction(simple_ar1_series):
    """Test fitted_values function."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    fitted = fitted_values(model)
    
    assert len(fitted) == len(y)
    assert isinstance(fitted, np.ndarray)


def test_residuals_arima_extraction(simple_ar1_series):
    """Test residuals_arima function."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    resid = residuals_arima(model)
    
    assert len(resid) == len(y)
    assert isinstance(resid, np.ndarray)


def test_fitted_plus_residuals_equals_y(simple_ar1_series):
    """Test that fitted + residuals = y."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    fitted = fitted_values(model)
    resid = residuals_arima(model)
    
    reconstructed = fitted + resid
    np.testing.assert_allclose(reconstructed, y, rtol=1e-10)


# =============================================================================
# Tests for transform_arima_parameters
# =============================================================================
def test_transform_arima_parameters_no_transform():
    """Test parameter transformation without stability transform."""
    params = np.array([0.5, 0.3])  # AR(1), MA(1)

    phi, theta = transform_arima_parameters(
        params, n_ar=1, n_ma=1, n_seasonal_ar=0,
        n_seasonal_ma=0, seasonal_period=1, trans=False
    )

    np.testing.assert_allclose(phi[0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(theta[0], 0.3, rtol=1e-10)


def test_transform_arima_parameters_with_transform():
    """Test parameter transformation with stability transform."""
    params = np.array([0.5, 0.3])

    phi, theta = transform_arima_parameters(
        params, n_ar=1, n_ma=1, n_seasonal_ar=0,
        n_seasonal_ma=0, seasonal_period=1, trans=True
    )

    # Transformed AR coefficient should be tanh(0.5) ≈ 0.4621
    expected_phi = np.tanh(0.5)
    np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-6)


# =============================================================================
# Tests for compute_css_residuals
# =============================================================================
def test_compute_css_residuals_ar1():
    """Test CSS residual computation for AR(1)."""
    np.random.seed(42)
    n = 50
    phi = np.array([0.5])
    theta = np.array([])
    eps = np.random.randn(n)

    # Generate AR(1) series
    y = np.zeros(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = phi[0] * y[t-1] + eps[t]

    ncond = 1

    sigma2, resid = compute_css_residuals(
        y, phi, theta, ncond,
        diff_order=0, seasonal_period=1, seasonal_diff_order=0
    )

    assert sigma2 > 0
    assert len(resid) == n
    # Residuals at conditioning positions should be zero
    assert resid[0] == 0.0


def test_compute_css_residuals_with_differencing():
    """Test CSS residuals with differencing."""
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0])
    phi = np.array([])
    theta = np.array([])
    ncond = 1

    sigma2, resid = compute_css_residuals(
        y, phi, theta, ncond,
        diff_order=1, seasonal_period=1, seasonal_diff_order=0
    )

    assert sigma2 > 0
    assert len(resid) == len(y)


def test_predict_arima_no_se(simple_ar1_series):
    """Test predict_arima with se_fit=False."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), fit_intercept=False)
    
    result = predict_arima(model, n_ahead=5, se_fit=False)
    
    assert len(result['mean']) == 5
    # se should be NaN when se_fit=False
    assert np.all(np.isnan(result['se']))


def test_arima_seasonal_model(simple_ar1_series):
    """Test ARIMA with seasonal component."""
    # Create a longer series for seasonal model
    np.random.seed(42)
    n = 60
    y = np.random.randn(n) + np.sin(np.arange(n) * 2 * np.pi / 12) * 2
    
    result = arima(y, m=12, order=(1, 0, 0), seasonal=(1, 0, 0), fit_intercept=True)
    
    assert 'sar1' in result['coef'].columns


def test_process_exogenous_1d_array():
    """Test _process_exogenous with 1D numpy array."""
    xreg = np.array([1.0, 2.0, 3.0])
    exog_mat, n_exog, exog_names = _process_exogenous(xreg, 3)

    assert exog_mat.shape == (3, 1)
    assert n_exog == 1
    assert exog_names == ['exog1']


# =============================================================================
# Tests for _ensure_float64_pair
# =============================================================================
def test_ensure_float64_pair_basic():
    """Test _ensure_float64_pair converts arrays to float64."""
    from skforecast.stats.arima._arima_base import _ensure_float64_pair
    
    x = np.array([1, 2, 3])
    xreg = np.array([[1, 2], [3, 4], [5, 6]])
    
    x_out, xreg_out = _ensure_float64_pair(x, xreg)
    
    assert x_out.dtype == np.float64
    assert xreg_out.dtype == np.float64
    np.testing.assert_array_equal(x_out, x.astype(np.float64))
    np.testing.assert_array_equal(xreg_out, xreg.astype(np.float64))




# =============================================================================
# Tests for _build_coefficient_dataframe
# =============================================================================
@pytest.mark.parametrize(
    'order_args, coef, cn, ncxreg, expected_columns',
    [
        # AR(1)
        (dict(p=1, d=0, q=0, P=0, D=0, Q=0, s=1), np.array([0.5]), [], 0, ['ar1']),
        # ARMA(1,1) with xreg
        (dict(p=1, d=0, q=1, P=0, D=0, Q=0, s=1), np.array([0.5, 0.3, 1.2]), ['exog1'], 1, ['ar1', 'ma1', 'exog1']),
        # Seasonal model p=1, P=1, Q=1, m=12
        (dict(p=1, d=0, q=0, P=1, D=0, Q=1, s=12), np.array([0.5, 0.2, -0.3]), [], 0, ['ar1', 'sar1', 'sma1']),
    ]
)
def test_prep_coefs(order_args, coef, cn, ncxreg, expected_columns):
    """Test _build_coefficient_dataframe creates correct coefficient DataFrame."""
    from skforecast.stats.arima._arima_base import _build_coefficient_dataframe, SARIMAOrder

    order_spec = SARIMAOrder(**order_args)
    result = _build_coefficient_dataframe(order_spec, coef, cn, ncxreg)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_columns
    np.testing.assert_allclose(result.values[0], coef, rtol=1e-10)


# =============================================================================
# Tests for inverse_arima_parameter_transform
# =============================================================================
def test_inverse_arima_parameter_transform_ar():
    """Test inverse_arima_parameter_transform for AR parameters."""
    from skforecast.stats.arima._arima_base import (
        inverse_arima_parameter_transform,
        transform_unconstrained_to_ar_params
    )

    # Start with unconstrained parameters
    raw = np.array([0.3, 0.2])
    p = 2

    # Transform to constrained
    constrained = transform_unconstrained_to_ar_params(p, raw)

    # Now invert back
    theta = constrained.copy()
    inverted = inverse_arima_parameter_transform(theta, n_ar=2, n_ma=0, n_seasonal_ar=0)

    np.testing.assert_allclose(inverted[:2], raw, rtol=1e-5)


# =============================================================================
# Tests for compute_arima_transform_gradient
# =============================================================================
def test_compute_arima_transform_gradient_shape():
    """Test compute_arima_transform_gradient returns correct shape."""
    from skforecast.stats.arima._arima_base import compute_arima_transform_gradient

    x = np.array([0.5, 0.3, 0.2, 0.1])  # 2 AR + 1 seasonal AR + 1 param

    result = compute_arima_transform_gradient(x, n_ar=2, n_ma=1, n_seasonal_ar=1)

    assert result.shape == (4, 4)


# =============================================================================
# Tests for undo_arima_parameter_transform
# =============================================================================
def test_undo_arima_parameter_transform():
    """Test undo_arima_parameter_transform applies transformation."""
    from skforecast.stats.arima._arima_base import undo_arima_parameter_transform

    x = np.array([0.5, 0.3])  # p=1, q=1

    result = undo_arima_parameter_transform(x, n_ar=1, n_ma=1, n_seasonal_ar=0)

    # AR param should be transformed by tanh
    expected_ar = np.tanh(0.5)
    np.testing.assert_allclose(result[0], expected_ar, rtol=1e-10)
    # MA param should remain unchanged
    np.testing.assert_allclose(result[1], 0.3, rtol=1e-10)


# =============================================================================
# Tests for optim_hessian
# =============================================================================
def test_optim_hessian():
    """Test optim_hessian computes correct and symmetric Hessian."""
    from skforecast.stats.arima._arima_base import optim_hessian
    
    # f(x) = x[0]^2 + x[1]^2, Hessian = [[2, 0], [0, 2]]
    def f1(x):
        return x[0]**2 + x[1]**2
    
    x1 = np.array([1.0, 1.0])
    H1 = optim_hessian(f1, x1)
    
    assert H1.shape == (2, 2)
    np.testing.assert_allclose(H1, np.array([[2.0, 0.0], [0.0, 2.0]]), rtol=0.1, atol=1e-10)
    np.testing.assert_allclose(H1, H1.T, rtol=1e-10)  # symmetry
    
    # f(x) = x[0]^2 + 2*x[0]*x[1] + 3*x[1]^2, Hessian = [[2, 2], [2, 6]]
    def f2(x):
        return x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2
    
    H2 = optim_hessian(f2, np.array([0.5, 0.5]))
    np.testing.assert_allclose(H2, H2.T, rtol=1e-10)  # symmetry


# =============================================================================
# Tests for arima() fixed= parameter (issue 3)
# =============================================================================
def test_arima_fixed_ar_parameter(simple_ar1_series):
    """Test arima() with a fixed AR parameter value."""
    y = simple_ar1_series
    fixed = np.array([0.5, np.nan])  # fix AR param, free MA param

    result = arima(y, order=(1, 0, 1), fit_intercept=False, fixed=fixed)

    assert result['converged'] is True
    # Fixed AR coefficient should remain at the specified value
    np.testing.assert_allclose(result['coef']['ar1'].values[0], 0.5, rtol=1e-10)


def test_arima_fixed_all_parameters(simple_ar1_series):
    """Test arima() with all parameters fixed (no optimization needed)."""
    y = simple_ar1_series
    # Fix both AR and MA to specific values
    fixed = np.array([0.5, 0.2])

    result = arima(y, order=(1, 0, 1), fit_intercept=False, fixed=fixed)

    assert result['converged'] is True
    np.testing.assert_allclose(result['coef']['ar1'].values[0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(result['coef']['ma1'].values[0], 0.2, rtol=1e-10)


def test_arima_fixed_wrong_length_raises(simple_ar1_series):
    """Test arima() raises ValueError when fixed= has wrong length."""
    y = simple_ar1_series
    fixed = np.array([0.5])  # AR(1)+MA(1) has 2 params, not 1

    with pytest.raises(ValueError, match="Wrong length for 'fixed'"):
        arima(y, order=(1, 0, 1), fit_intercept=False, fixed=fixed)


# =============================================================================
# Tests for arima() init= parameter (issue 4)
# =============================================================================
def test_arima_init_parameter(simple_ar1_series):
    """Test arima() accepts and uses user-supplied initial parameter values."""
    y = simple_ar1_series
    # Provide a good starting point near the true AR(1) coefficient (~0.7)
    init = np.array([0.6])

    result = arima(y, order=(1, 0, 0), fit_intercept=False, init=init)

    assert result['converged'] is True
    # Result should converge to something reasonable
    ar_coef = result['coef']['ar1'].values[0]
    assert 0.3 < ar_coef < 1.0


def test_arima_init_wrong_length_raises(simple_ar1_series):
    """Test arima() raises ValueError when init= has wrong length."""
    y = simple_ar1_series
    init = np.array([0.5, 0.3, 0.1])  # AR(1) has 1 param, not 3

    with pytest.raises(ValueError, match="'init' is of the wrong length"):
        arima(y, order=(1, 0, 0), fit_intercept=False, init=init)


def test_arima_init_partial_nan(simple_ar1_series):
    """Test arima() fills NaN entries in init= from automatic initialization."""
    y = simple_ar1_series
    # For ARMA(1,1): supply AR init, leave MA as NaN to be auto-initialized
    init = np.array([0.6, np.nan])

    result = arima(y, order=(1, 0, 1), fit_intercept=False, init=init)


# =============================================================================
# Tests for enforce_stationarity=False (TEST-1)
# =============================================================================

def test_arima_enforce_stationarity_false_stationary_data():
    """
    Test arima() with enforce_stationarity=False on stationary data.

    Both paths (True/False) should reach the same optimum. With
    enforce_stationarity=False the Jones (1980) transform is skipped: the
    optimizer works directly in the raw parameter space and the covariance
    is the inverse Hessian (no delta-method Jacobian).
    """
    rng = np.random.default_rng(42)
    n = 100
    phi = 0.5
    e = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]

    result_true = arima(y, order=(1, 0, 0), fit_intercept=False,
                        enforce_stationarity=True)
    result_false = arima(y, order=(1, 0, 0), fit_intercept=False,
                         enforce_stationarity=False)

    assert result_true['converged'] is True
    assert result_false['converged'] is True
    # Both paths must converge to the same coefficient and variance
    np.testing.assert_allclose(
        result_true['coef'].values.flatten(),
        result_false['coef'].values.flatten(),
        rtol=1e-3,
    )
    np.testing.assert_allclose(result_true['sigma2'], result_false['sigma2'], rtol=1e-3)


def test_arima_enforce_stationarity_false_near_unit_root():
    """
    Test arima() with enforce_stationarity=False can estimate near-unit-root models.

    With enforce_stationarity=True the Jones transform keeps |phi| < 1.
    With enforce_stationarity=False the optimizer is unconstrained and correctly
    recovers a near-unit-root coefficient.
    """
    rng = np.random.default_rng(99)
    n = 120
    phi_true = 0.98
    e = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi_true * y[t - 1] + e[t]

    result = arima(y, order=(1, 0, 0), fit_intercept=False,
                   enforce_stationarity=False)

    assert result['converged'] is True
    coef = result['coef'].values.flatten()[0]
    np.testing.assert_allclose(coef, 0.9772565333232445, rtol=1e-4)
    np.testing.assert_allclose(result['sigma2'], 0.8358454427268373, rtol=1e-4)


def test_arima_enforce_stationarity_false_covariance_is_positive_definite():
    """
    Test arima() with enforce_stationarity=False produces a positive-definite
    parameter covariance matrix.

    The direct inverse-Hessian path (used when enforce_stationarity=False) must
    return a valid covariance for a well-identified model.
    """
    rng = np.random.default_rng(42)
    n = 150
    y = rng.standard_normal(n)  # white noise → small but well-identified AR coef

    result = arima(y, order=(1, 0, 0), fit_intercept=False,
                   enforce_stationarity=False)

    assert result['var_coef'] is not None
    var_coef = result['var_coef']
    assert var_coef.shape == (1, 1)
    assert float(np.diag(var_coef)[0]) > 0


# =============================================================================
# Tests for _fit_css_ml fallbacks (TEST-3)
# =============================================================================

def test_arima_css_ml_all_params_fixed_skips_css_stage():
    """
    Test that CSS-ML with all parameters fixed skips the CSS warm-start.

    When every parameter is pinned via fixed=, there is nothing to optimise.
    _fit_css_ml must short-circuit the CSS objective and fall through directly
    to _fit_ml with the fixed values, returning converged=True with the exact
    fixed coefficient.
    """
    rng = np.random.default_rng(42)
    n = 100
    phi = 0.5
    e = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]

    result = arima(y, order=(1, 0, 0), fit_intercept=False,
                   fixed=np.array([phi]), method='CSS-ML')

    assert result['converged'] is True
    np.testing.assert_allclose(
        result['coef'].values.flatten()[0], phi, rtol=1e-10
    )


def test_arima_css_ml_result_consistent_with_ml():
    """
    Test that CSS-ML and ML produce consistent estimates on well-behaved data.

    For stationary data the CSS warm start should land close to the ML
    optimum, so both methods should converge to the same coefficient and
    sigma2 within numerical tolerance.
    """
    rng = np.random.default_rng(7)
    n = 200
    phi = 0.6
    e = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]

    result_css_ml = arima(y, order=(1, 0, 0), fit_intercept=False, method='CSS-ML')
    result_ml = arima(y, order=(1, 0, 0), fit_intercept=False, method='ML')

    np.testing.assert_allclose(
        result_css_ml['coef'].values.flatten(),
        result_ml['coef'].values.flatten(),
        rtol=1e-3,
    )
    np.testing.assert_allclose(result_css_ml['sigma2'], result_ml['sigma2'], rtol=1e-3)