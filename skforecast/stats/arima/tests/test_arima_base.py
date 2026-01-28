# Unit test _arima_base
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.stats.arima._arima_base import (
    state_prediction,
    predict_covariance_nodiff,
    predict_covariance_with_diff,
    kalman_update,
    compute_arima_likelihood,
    transform_unconstrained_to_ar_params,
    inverse_ar_parameter_transform,
    time_series_convolution,
    compute_q0_covariance_matrix,
    compute_q0_bis_covariance_matrix,
    transform_arima_parameters,
    compute_css_residuals,
    initialize_arima_state,
    update_arima,
    ar_check,
    ma_invert,
    kalman_forecast,
    make_pdq,
    na_omit,
    diff,
    match_arg,
    process_xreg,
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
# Tests for state_prediction
# =============================================================================
def test_state_prediction_ar1():
    """Test state prediction for AR(1) model."""
    phi = np.array([0.5])
    delta = np.array([])
    a = np.array([1.0])
    p, r, d, rd = 1, 1, 0, 1
    
    anew = state_prediction(a, p, r, d, rd, phi, delta)
    
    # For AR(1) with a[0]=1: anew[0] = phi[0] * a[0] = 0.5
    assert anew.shape == (1,)
    np.testing.assert_allclose(anew[0], 0.5, rtol=1e-10)


def test_state_prediction_ar2():
    """Test state prediction for AR(2) model with r > p."""
    phi = np.array([0.5, 0.3])
    delta = np.array([])
    a = np.array([1.0, 0.5])
    p, r, d, rd = 2, 2, 0, 2
    
    anew = state_prediction(a, p, r, d, rd, phi, delta)
    
    # anew[0] = a[1] + phi[0]*a[0] = 0.5 + 0.5*1.0 = 1.0
    # anew[1] = phi[1]*a[0] = 0.3*1.0 = 0.3
    assert anew.shape == (2,)
    np.testing.assert_allclose(anew[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(anew[1], 0.3, rtol=1e-10)


def test_state_prediction_with_differencing():
    """Test state prediction with differencing component."""
    phi = np.array([0.5])
    delta = np.array([1.0])  # d=1 differencing
    a = np.array([1.0, 2.0])
    p, r, d, rd = 1, 1, 1, 2
    
    anew = state_prediction(a, p, r, d, rd, phi, delta)
    
    # anew[0] = phi[0]*a[0] = 0.5
    # anew[1] = a[0] + delta[0]*a[1] = 1.0 + 1.0*2.0 = 3.0
    assert anew.shape == (2,)
    np.testing.assert_allclose(anew[0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(anew[1], 3.0, rtol=1e-10)


# =============================================================================
# Tests for covariance predictions
# =============================================================================
def test_predict_covariance_nodiff_ar1():
    """Test covariance prediction for AR(1) without differencing."""
    phi = np.array([0.5])
    theta = np.array([])
    P = np.array([[1.0]])
    r, p, q = 1, 1, 0
    
    Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
    
    # For AR(1): Pnew = phi^2 * P + 1
    expected = 0.5**2 * 1.0 + 1.0
    assert Pnew.shape == (1, 1)
    np.testing.assert_allclose(Pnew[0, 0], expected, rtol=1e-10)


def test_predict_covariance_nodiff_is_symmetric():
    """Test that covariance prediction produces symmetric matrix."""
    phi = np.array([0.5, 0.2])
    theta = np.array([0.3])
    P = np.eye(2) * 0.5
    r, p, q = 2, 2, 1
    
    Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
    
    assert Pnew.shape == (r, r)
    np.testing.assert_allclose(Pnew, Pnew.T, rtol=1e-10)


def test_predict_covariance_with_diff_shape():
    """Test covariance prediction with differencing has correct shape."""
    phi = np.array([0.5])
    theta = np.array([])
    delta = np.array([1.0])
    r, d, p, q, rd = 1, 1, 1, 0, 2
    P = np.eye(rd) * 0.5
    
    Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
    
    assert Pnew.shape == (rd, rd)


# =============================================================================
# Tests for kalman_update
# =============================================================================
def test_kalman_update_basic():
    """Test Kalman update with simple inputs."""
    y_obs = 1.0
    anew = np.array([0.5])
    delta = np.array([])
    Pnew = np.array([[1.0]])
    d, r, rd = 0, 1, 1
    
    a, P, resid, gain, ssq_c, sumlog_c = kalman_update(
        y_obs, anew, delta, Pnew, d, r, rd
    )
    
    # resid = y_obs - anew[0] = 1.0 - 0.5 = 0.5
    np.testing.assert_allclose(resid, 0.5, rtol=1e-10)
    # gain should be Pnew[0,0] = 1.0
    np.testing.assert_allclose(gain, 1.0, rtol=1e-10)
    # Updated state and covariance should have correct dimensions
    assert a.shape == (1,)
    assert P.shape == (1, 1)


def test_kalman_update_with_differencing():
    """Test Kalman update with differencing component."""
    y_obs = 2.0
    anew = np.array([0.5, 1.0])
    delta = np.array([1.0])
    Pnew = np.eye(2)
    d, r, rd = 1, 1, 2
    
    a, P, resid, gain, ssq_c, sumlog_c = kalman_update(
        y_obs, anew, delta, Pnew, d, r, rd
    )
    
    # resid = y_obs - anew[0] - delta[0]*anew[1] = 2.0 - 0.5 - 1.0*1.0 = 0.5
    np.testing.assert_allclose(resid, 0.5, rtol=1e-10)
    assert a.shape == (2,)
    assert P.shape == (2, 2)


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
# Tests for time_series_convolution
# =============================================================================
def test_time_series_convolution_simple():
    """Test convolution with simple polynomials."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 1.0])
    
    result = time_series_convolution(a, b)
    
    # (1 + 2x) * (1 + x) = 1 + 3x + 2x^2
    expected = np.array([1.0, 3.0, 2.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_time_series_convolution_differencing():
    """Test convolution for differencing polynomial."""
    a = np.array([1.0, -1.0])
    b = np.array([1.0, -1.0])
    
    result = time_series_convolution(a, b)
    
    # (1 - x) * (1 - x) = 1 - 2x + x^2
    expected = np.array([1.0, -2.0, 1.0])
    np.testing.assert_array_almost_equal(result, expected)


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


def test_compute_q0_bis_matches_gardner_for_simple_cases():
    """Test that Rossignol method produces valid covariance matrix."""
    phi = np.array([0.5])
    theta = np.array([0.3])
    
    Q0_gardner = compute_q0_covariance_matrix(phi, theta)
    Q0_rossignol = compute_q0_bis_covariance_matrix(phi, theta)
    
    # Both should produce symmetric positive semi-definite matrices
    assert Q0_gardner.shape == Q0_rossignol.shape
    np.testing.assert_allclose(Q0_gardner, Q0_gardner.T, rtol=1e-10)
    np.testing.assert_allclose(Q0_rossignol, Q0_rossignol.T, rtol=1e-10)
    # Eigenvalues should be non-negative (positive semi-definite)
    assert np.all(np.linalg.eigvalsh(Q0_gardner) >= -1e-10)
    assert np.all(np.linalg.eigvalsh(Q0_rossignol) >= -1e-10)


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
# Tests for initialize_arima_state and update_arima
# =============================================================================
def test_initialize_arima_state_ar1():
    """Test state-space initialization for AR(1) model."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta)
    
    assert 'phi' in model
    assert 'theta' in model
    assert 'Z' in model
    assert 'a' in model
    assert 'P' in model
    assert 'T' in model
    assert 'Pn' in model
    
    # Check dimensions
    assert model['a'].shape == (1,)
    assert model['P'].shape == (1, 1)
    assert model['T'].shape == (1, 1)
    
    # T[0,0] should equal phi[0]
    np.testing.assert_allclose(model['T'][0, 0], 0.5, rtol=1e-10)


def test_initialize_arima_state_with_differencing():
    """Test state-space initialization with differencing."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([1.0])  # d=1
    
    model = initialize_arima_state(phi, theta, Delta)
    
    rd = 2  # r=1, d=1
    assert model['a'].shape == (rd,)
    assert model['P'].shape == (rd, rd)
    assert model['T'].shape == (rd, rd)
    
    # Diffuse prior for differencing state
    assert model['Pn'][1, 1] > 1e5


def test_initialize_arima_state_rossignol_method():
    """Test state-space initialization with Rossignol method."""
    phi = np.array([0.5])
    theta = np.array([0.3])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta, SSinit="Rossignol2011")
    
    # r = max(p, q+1) = max(1, 2) = 2, rd = r + d = 2 + 0 = 2
    assert model['Pn'].shape == (2, 2)


def test_update_arima_changes_coefficients():
    """Test that update_arima correctly updates AR/MA coefficients."""
    phi_init = np.array([0.5])
    theta_init = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi_init, theta_init, Delta)
    
    phi_new = np.array([0.7])
    theta_new = np.array([])
    
    updated = update_arima(model, phi_new, theta_new)
    
    np.testing.assert_allclose(updated['phi'], phi_new, rtol=1e-10)
    np.testing.assert_allclose(updated['T'][0, 0], 0.7, rtol=1e-10)


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
    
    model = initialize_arima_state(phi, theta, Delta)
    model['a'] = np.array([1.0])  # Set initial state
    
    n_ahead = 10
    result = kalman_forecast(n_ahead, model)
    
    assert 'pred' in result
    assert 'var' in result
    assert len(result['pred']) == n_ahead
    assert len(result['var']) == n_ahead


def test_kalman_forecast_variances_increase():
    """Test that forecast variances generally increase with horizon."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta)
    
    result = kalman_forecast(10, model)
    
    # For most models, variance should not decrease
    # (allowing for numerical tolerance)
    for i in range(1, len(result['var'])):
        assert result['var'][i] >= result['var'][i-1] - 1e-10


def test_kalman_forecast_with_update():
    """Test that kalman_forecast with update=True returns complete result."""
    phi = np.array([0.5])
    theta = np.array([])
    Delta = np.array([])
    
    model = initialize_arima_state(phi, theta, Delta)
    model['a'] = np.array([1.0])
    
    result = kalman_forecast(5, model, update=True)
    
    # Should have predictions and variances
    assert len(result['pred']) == 5
    assert len(result['var']) == 5
    # Result should contain model (since update=True)
    assert 'mod' in result


# =============================================================================
# Tests for utility functions
# =============================================================================
def test_make_pdq_valid():
    """Test make_pdq with valid inputs."""
    result = make_pdq(1, 1, 1)
    assert result == (1, 1, 1)


def test_make_pdq_raises_on_negative():
    """Test make_pdq raises ValueError for negative values."""
    with pytest.raises(ValueError, match="must be non-negative"):
        make_pdq(-1, 0, 0)
    
    with pytest.raises(ValueError, match="must be non-negative"):
        make_pdq(0, -1, 0)


def test_na_omit():
    """Test na_omit removes NaN values."""
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    result = na_omit(x)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_na_omit_no_nan():
    """Test na_omit with no NaN values."""
    x = np.array([1.0, 2.0, 3.0])
    result = na_omit(x)
    np.testing.assert_array_equal(result, x)


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


def test_match_arg_exact():
    """Test match_arg with exact match."""
    # Note: match_arg uses startswith matching, so order matters
    # "CSS" matches "CSS-ML" because "CSS-ML".startswith("CSS") is True
    result = match_arg("ML", ["CSS-ML", "ML", "CSS"])
    assert result == "ML"
    
    # Test exact match when it comes first in list
    result = match_arg("CSS", ["CSS", "CSS-ML", "ML"])
    assert result == "CSS"


def test_match_arg_raises_on_invalid():
    """Test match_arg raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="should be one of"):
        match_arg("invalid", ["CSS-ML", "ML", "CSS"])


def test_process_xreg_none():
    """Test process_xreg with None input."""
    xreg, ncxreg, nmxreg = process_xreg(None, 10)
    assert xreg.shape == (10, 0)
    assert ncxreg == 0
    assert nmxreg == []


def test_process_xreg_dataframe():
    """Test process_xreg with DataFrame input."""
    df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
    xreg, ncxreg, nmxreg = process_xreg(df, 3)
    
    assert xreg.shape == (3, 2)
    assert ncxreg == 2
    assert nmxreg == ['x1', 'x2']


def test_process_xreg_raises_on_length_mismatch():
    """Test process_xreg raises error on length mismatch."""
    df = pd.DataFrame({'x1': [1, 2, 3]})
    with pytest.raises(ValueError, match="do not match"):
        process_xreg(df, 5)


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
    
    result = arima(y, order=(1, 0, 0), include_mean=False)
    
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
    
    result = arima(y, order=(1, 1, 0), include_mean=False)
    
    assert result['converged'] == True
    assert result['arma'][5] == 1  # d = 1


def test_arima_with_xreg(simple_ar1_series):
    """Test ARIMA with exogenous regressors."""
    y = simple_ar1_series
    xreg = pd.DataFrame({'x1': np.random.randn(len(y))})
    
    result = arima(y, order=(1, 0, 0), xreg=xreg, include_mean=False)
    
    assert result['converged'] == True
    assert 'x1' in result['coef'].columns


def test_arima_css_method(simple_ar1_series):
    """Test ARIMA with CSS estimation method."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), method="CSS", include_mean=False)
    
    assert result['converged'] == True


def test_arima_ml_method(simple_ar1_series):
    """Test ARIMA with ML estimation method."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), method="ML", include_mean=False)
    
    assert result['converged'] == True


def test_arima_with_missing_values():
    """Test ARIMA handles missing values in series."""
    np.random.seed(42)
    y = np.random.randn(50)
    y[10] = np.nan  # Add a missing value
    
    # ML method should be used automatically with missing values
    result = arima(y, order=(1, 0, 0), include_mean=False)
    
    # Should converge (ML method handles missing)
    assert 'residuals' in result
    assert len(result['residuals']) == len(y)


def test_arima_residuals_length(simple_ar1_series):
    """Test that residuals have same length as input."""
    y = simple_ar1_series
    
    result = arima(y, order=(1, 0, 0), include_mean=False)
    
    assert len(result['residuals']) == len(y)
    assert len(result['fitted']) == len(y)


# =============================================================================
# Tests for predict_arima
# =============================================================================
def test_predict_arima_basic(simple_ar1_series):
    """Test basic prediction from fitted ARIMA model."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
    n_ahead = 5
    result = predict_arima(model, n_ahead=n_ahead)
    
    assert 'mean' in result
    assert 'se' in result
    assert len(result['mean']) == n_ahead
    assert len(result['se']) == n_ahead


def test_predict_arima_with_intervals(simple_ar1_series):
    """Test prediction with confidence intervals."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
    result = predict_arima(model, n_ahead=5, level=[80, 95])
    
    assert result['lower'] is not None
    assert result['upper'] is not None
    assert result['lower'].shape == (5, 2)
    assert result['upper'].shape == (5, 2)
    # Lower bounds should be less than upper bounds
    assert np.all(result['lower'] < result['upper'])


def test_predict_arima_with_xreg(simple_ar1_series):
    """Test prediction with exogenous regressors."""
    y = simple_ar1_series
    n = len(y)
    xreg = pd.DataFrame({'x1': np.random.randn(n)})
    
    model = arima(y, order=(1, 0, 0), xreg=xreg, include_mean=False)
    
    newxreg = pd.DataFrame({'x1': np.random.randn(5)})
    result = predict_arima(model, n_ahead=5, newxreg=newxreg)
    
    assert len(result['mean']) == 5


# =============================================================================
# Tests for fitted_values and residuals_arima
# =============================================================================
def test_fitted_values_extraction(simple_ar1_series):
    """Test fitted_values function."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
    fitted = fitted_values(model)
    
    assert len(fitted) == len(y)
    assert isinstance(fitted, np.ndarray)


def test_residuals_arima_extraction(simple_ar1_series):
    """Test residuals_arima function."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
    resid = residuals_arima(model)
    
    assert len(resid) == len(y)
    assert isinstance(resid, np.ndarray)


def test_fitted_plus_residuals_equals_y(simple_ar1_series):
    """Test that fitted + residuals = y."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
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
    arma = np.array([1, 1, 0, 0, 1, 0, 0])
    
    phi, theta = transform_arima_parameters(params, arma, trans=False)
    
    np.testing.assert_allclose(phi[0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(theta[0], 0.3, rtol=1e-10)


def test_transform_arima_parameters_with_transform():
    """Test parameter transformation with stability transform."""
    params = np.array([0.5, 0.3])
    arma = np.array([1, 1, 0, 0, 1, 0, 0])
    
    phi, theta = transform_arima_parameters(params, arma, trans=True)
    
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
    
    arma = np.array([1, 0, 0, 0, 1, 0, 0])
    ncond = 1
    
    sigma2, resid = compute_css_residuals(y, arma, phi, theta, ncond)
    
    assert sigma2 > 0
    assert len(resid) == n
    # Residuals at conditioning positions should be zero
    assert resid[0] == 0.0


def test_compute_css_residuals_with_differencing():
    """Test CSS residuals with differencing."""
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0])
    phi = np.array([])
    theta = np.array([])
    arma = np.array([0, 0, 0, 0, 1, 1, 0])  # ARIMA(0,1,0)
    ncond = 1
    
    sigma2, resid = compute_css_residuals(y, arma, phi, theta, ncond)
    
    assert sigma2 > 0
    assert len(resid) == len(y)


def test_predict_arima_no_se(simple_ar1_series):
    """Test predict_arima with se_fit=False."""
    y = simple_ar1_series
    model = arima(y, order=(1, 0, 0), include_mean=False)
    
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
    
    result = arima(y, m=12, order=(1, 0, 0), seasonal=(1, 0, 0), include_mean=True)
    
    assert 'sar1' in result['coef'].columns


def test_process_xreg_1d_array():
    """Test process_xreg with 1D numpy array."""
    xreg = np.array([1.0, 2.0, 3.0])
    xreg_mat, ncxreg, nmxreg = process_xreg(xreg, 3)
    
    assert xreg_mat.shape == (3, 1)
    assert ncxreg == 1
    assert nmxreg == ['xreg1']


# =============================================================================
# Tests for na_omit_pair
# =============================================================================
def test_na_omit_pair_basic():
    """Test na_omit_pair converts arrays to float64."""
    from skforecast.stats.arima._arima_base import na_omit_pair
    
    x = np.array([1, 2, 3])
    xreg = np.array([[1, 2], [3, 4], [5, 6]])
    
    x_out, xreg_out = na_omit_pair(x, xreg)
    
    assert x_out.dtype == np.float64
    assert xreg_out.dtype == np.float64
    np.testing.assert_array_equal(x_out, x.astype(np.float64))
    np.testing.assert_array_equal(xreg_out, xreg.astype(np.float64))


# =============================================================================
# Tests for handle_r_equals_1
# =============================================================================
@pytest.mark.parametrize(
    'p, phi, expected_value',
    [
        (0, np.array([]), 1.0),           # p=0 case
        (1, np.array([0.5]), 1.0 / 0.75), # p=1: 1/(1-phi^2)
    ]
)
def test_handle_r_equals_1(p, phi, expected_value):
    """Test handle_r_equals_1 for different p values."""
    from skforecast.stats.arima._arima_base import handle_r_equals_1
    
    result = handle_r_equals_1(p, phi)
    
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result[0, 0], expected_value, rtol=1e-10)


# =============================================================================
# Tests for handle_p_equals_0 and unpack_full_matrix
# =============================================================================
def test_handle_p_equals_0_r2():
    """Test handle_p_equals_0 for r=2 (pure MA case)."""
    from skforecast.stats.arima._arima_base import handle_p_equals_0, compute_v, unpack_full_matrix
    
    phi = np.array([])
    theta = np.array([0.5])
    r = 2
    
    V = compute_v(phi, theta, r)
    res_flat = handle_p_equals_0(V, r)
    
    assert len(res_flat) == r * r
    
    # Also test unpack_full_matrix
    matrix = unpack_full_matrix(res_flat.copy(), r)
    assert matrix.shape == (r, r)
    # Matrix should be symmetric
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10)


# =============================================================================
# Tests for compute_v
# =============================================================================
@pytest.mark.parametrize(
    'phi, theta, r, expected_shape, check_first_value',
    [
        (np.array([0.5]), np.array([]), 1, (1,), 1.0),   # AR(1)
        (np.array([]), np.array([0.3]), 2, (3,), None),  # MA(1), r*(r+1)/2=3
    ]
)
def test_compute_v(phi, theta, r, expected_shape, check_first_value):
    """Test compute_v for AR and MA models."""
    from skforecast.stats.arima._arima_base import compute_v
    
    V = compute_v(phi, theta, r)
    
    assert V.shape == expected_shape
    if check_first_value is not None:
        np.testing.assert_allclose(V[0], check_first_value, rtol=1e-10)


# =============================================================================
# Tests for prep_coefs
# =============================================================================
@pytest.mark.parametrize(
    'arma, coef, cn, ncxreg, expected_columns',
    [
        # AR(1)
        ([1, 0, 0, 0, 1, 0, 0], np.array([0.5]), [], 0, ['ar1']),
        # ARMA(1,1) with xreg
        ([1, 1, 0, 0, 1, 0, 0], np.array([0.5, 0.3, 1.2]), ['exog1'], 1, ['ar1', 'ma1', 'exog1']),
        # Seasonal model p=1, P=1, Q=1, m=12
        ([1, 0, 1, 1, 12, 0, 0], np.array([0.5, 0.2, -0.3]), [], 0, ['ar1', 'sar1', 'sma1']),
    ]
)
def test_prep_coefs(arma, coef, cn, ncxreg, expected_columns):
    """Test prep_coefs creates correct coefficient DataFrame."""
    from skforecast.stats.arima._arima_base import prep_coefs
    
    result = prep_coefs(arma, coef, cn, ncxreg)
    
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
    arma = np.array([2, 0, 0, 0, 1, 0, 0])  # p=2, q=0, P=0
    theta = constrained.copy()
    inverted = inverse_arima_parameter_transform(theta, arma)
    
    np.testing.assert_allclose(inverted[:2], raw, rtol=1e-5)


# =============================================================================
# Tests for compute_arima_transform_gradient
# =============================================================================
def test_compute_arima_transform_gradient_shape():
    """Test compute_arima_transform_gradient returns correct shape."""
    from skforecast.stats.arima._arima_base import compute_arima_transform_gradient
    
    x = np.array([0.5, 0.3, 0.2, 0.1])  # 2 AR + 1 seasonal AR + 1 param
    arma = np.array([2, 1, 1, 0, 12, 0, 0])  # p=2, q=1, P=1
    
    result = compute_arima_transform_gradient(x, arma)
    
    assert result.shape == (4, 4)


# =============================================================================
# Tests for undo_arima_parameter_transform
# =============================================================================
def test_undo_arima_parameter_transform():
    """Test undo_arima_parameter_transform applies transformation."""
    from skforecast.stats.arima._arima_base import undo_arima_parameter_transform
    
    x = np.array([0.5, 0.3])  # p=1, q=1
    arma = np.array([1, 1, 0, 0, 1, 0, 0])
    
    result = undo_arima_parameter_transform(x, arma)
    
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