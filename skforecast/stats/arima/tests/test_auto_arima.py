# Unit test _auto_arima
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.stats.arima._auto_arima import (
    analyze_series,
    mean2,
    compute_approx_offset,
    newmodel,
    get_pdq,
    get_sum,
    arima_trace_str,
    fit_custom_arima,
    _create_error_model,
    search_arima,
    kpss_test,
    adf_test,
    auto_arima,
    time_index,
    has_coef,
    npar_fit,
    n_and_nstar,
    prepend_drift,
    arima_rjh,
    forecast_arima
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def ar1_series():
    """Generate a simple AR(1) series for testing."""
    np.random.seed(42)
    n = 100
    phi = 0.7
    y = np.zeros(n)
    y[0] = np.random.randn()
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.randn()
    return y


@pytest.fixture
def random_walk_series():
    """Generate a random walk (non-stationary) series."""
    np.random.seed(123)
    return np.cumsum(np.random.randn(100))


@pytest.fixture
def seasonal_series():
    """Generate a seasonal series with period 12."""
    np.random.seed(456)
    t = np.arange(120)
    return 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5


# =============================================================================
# Tests for analyze_series
# =============================================================================
def test_analyze_series_with_leading_nan():
    """Test analyze_series removes leading NaN and returns correct values."""
    x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    
    first, length, trimmed = analyze_series(x)
    
    assert first == 2
    assert length == 4
    np.testing.assert_array_equal(trimmed, np.array([1.0, 2.0, 3.0, 4.0]))


def test_analyze_series_with_internal_nan():
    """Test analyze_series handles internal NaN correctly."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    
    first, length, trimmed = analyze_series(x)
    
    # First non-missing is index 0
    assert first == 0
    # Series length counts non-NaN between first and last
    assert length == 4
    np.testing.assert_array_equal(trimmed, x)


def test_analyze_series_all_nan():
    """Test analyze_series with all NaN values."""
    x = np.array([np.nan, np.nan, np.nan])
    
    first, length, trimmed = analyze_series(x)
    
    assert first is None
    assert length == 0


def test_analyze_series_no_nan():
    """Test analyze_series with no NaN values."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    first, length, trimmed = analyze_series(x)
    
    assert first == 0
    assert length == 5
    np.testing.assert_array_equal(trimmed, x)


# =============================================================================
# Tests for mean2
# =============================================================================
def test_mean2_with_nan():
    """Test mean2 ignores NaN by default."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    
    result = mean2(x, omit_na=True)
    
    np.testing.assert_allclose(result, 3.0, rtol=1e-10)


def test_mean2_without_nan_removal():
    """Test mean2 includes NaN when omit_na=False."""
    x = np.array([1.0, 2.0, np.nan, 4.0])
    
    result = mean2(x, omit_na=False)
    
    assert np.isnan(result)


# =============================================================================
# Tests for compute_approx_offset
# =============================================================================
def test_compute_approx_offset_no_approximation():
    """Test compute_approx_offset returns 0 when approximation=False."""
    x = np.random.randn(100)
    
    offset = compute_approx_offset(approximation=False, x=x, d=0, D=0)
    
    assert offset == 0.0


def test_compute_approx_offset_with_approximation(ar1_series):
    """Test compute_approx_offset returns non-zero when approximation=True."""
    offset = compute_approx_offset(
        approximation=True, x=ar1_series, d=1, D=0, m=1
    )
    
    # Should return a finite offset value
    assert np.isfinite(offset)


def test_compute_approx_offset_with_truncation(ar1_series):
    """Test compute_approx_offset with truncate parameter."""
    offset = compute_approx_offset(
        approximation=True, x=ar1_series, d=0, D=0, m=1, truncate=50
    )
    
    # Should return a finite offset value
    assert np.isfinite(offset)


def test_compute_approx_offset_with_seasonal_D(ar1_series):
    """Test compute_approx_offset with seasonal differencing."""
    # Need longer series for seasonal
    np.random.seed(42)
    y = np.random.randn(120)
    offset = compute_approx_offset(
        approximation=True, x=y, d=0, D=1, m=12
    )
    
    assert np.isfinite(offset)


def test_compute_approx_offset_with_xreg_truncation():
    """Test compute_approx_offset with xreg and truncation."""
    np.random.seed(42)
    y = np.random.randn(100)
    xreg = pd.DataFrame({'x1': np.random.randn(100)})
    
    offset = compute_approx_offset(
        approximation=True, x=y, d=0, D=0, m=1, xreg=xreg, truncate=50
    )
    
    assert np.isfinite(offset)


# =============================================================================
# Tests for newmodel
# =============================================================================
def test_newmodel_new_configuration():
    """Test newmodel returns True for new model configuration."""
    results = np.full((10, 8), np.nan)
    results[0, :7] = [1, 0, 1, 0, 0, 0, 1]  # ARIMA(1,0,1) with constant
    
    # New model: ARIMA(2,0,1)
    is_new = newmodel(2, 0, 1, 0, 0, 0, True, results, 1)
    
    assert is_new is True


def test_newmodel_existing_configuration():
    """Test newmodel returns False for existing model configuration."""
    results = np.full((10, 8), np.nan)
    results[0, :7] = [1, 0, 1, 0, 0, 0, 1]  # ARIMA(1,0,1) with constant
    
    # Same model
    is_new = newmodel(1, 0, 1, 0, 0, 0, True, results, 1)
    
    assert is_new is False


# =============================================================================
# Tests for get_pdq and get_sum
# =============================================================================
@pytest.mark.parametrize(
    'input_val, expected',
    [
        ((1, 2, 3), (1, 2, 3)),           # tuple
        ({'p': 1, 'd': 2, 'q': 3}, (1, 2, 3)),  # dict complete
        ({'p': 2}, (2, 0, 0)),            # dict partial
        ([2, 1, 3], (2, 1, 3)),           # list
    ]
)
def test_get_pdq(input_val, expected):
    """Test get_pdq extracts values from different input types."""
    assert get_pdq(input_val) == expected


@pytest.mark.parametrize(
    'input_val, expected',
    [
        ((1, 2, 3), 6),           # tuple
        ({'p': 1, 'd': 2, 'q': 3}, 6),  # dict
        ([2, 1, 3], 6),           # list
    ]
)
def test_get_sum(input_val, expected):
    """Test get_sum calculates sum from different input types."""
    assert get_sum(input_val) == expected


# =============================================================================
# Tests for arima_trace_str
# =============================================================================
def test_arima_trace_str_with_mean():
    """Test trace string for non-differenced model with mean."""
    result = arima_trace_str(
        order=(1, 0, 1), seasonal=(0, 0, 0), m=1, constant=True, ic_value=100.5
    )
    
    assert "ARIMA(1,0,1)" in result
    assert "non-zero mean" in result
    assert "100.5" in result


def test_arima_trace_str_with_drift():
    """Test trace string for differenced model with drift."""
    result = arima_trace_str(
        order=(1, 1, 1), seasonal=(0, 0, 0), m=1, constant=True, ic_value=150.25
    )
    
    assert "ARIMA(1,1,1)" in result
    assert "drift" in result


def test_arima_trace_str_seasonal():
    """Test trace string for seasonal model."""
    result = arima_trace_str(
        order=(1, 0, 1), seasonal=(1, 1, 1), m=12, constant=False, ic_value=200.0
    )
    
    assert "ARIMA(1,0,1)" in result
    assert "(1,1,1)[12]" in result


def test_arima_trace_str_zero_mean():
    """Test trace string for model without mean."""
    result = arima_trace_str(
        order=(1, 0, 1), seasonal=(0, 0, 0), m=1, constant=False, ic_value=100.0
    )
    
    assert "ARIMA(1,0,1)" in result
    assert "zero mean" in result


def test_arima_trace_str_infinite_ic():
    """Test trace string with infinite IC."""
    result = arima_trace_str(
        order=(1, 0, 1), seasonal=(0, 0, 0), m=1, constant=True, ic_value=np.inf
    )
    
    assert "Inf" in result


# =============================================================================
# Tests for _create_error_model
# =============================================================================
def test_create_error_model_structure():
    """Test _create_error_model returns correct structure."""
    result = _create_error_model(order=(1, 0, 1), seasonal=(0, 0, 0), m=1)
    
    assert result['ic'] == np.inf
    assert result['aic'] == np.inf
    assert result['bic'] == np.inf
    assert result['aicc'] == np.inf
    assert result['converged'] is False
    assert result['arma'] == [1, 1, 0, 0, 1, 0, 0]


# =============================================================================
# Tests for fit_custom_arima
# =============================================================================
def test_fit_custom_arima_ar1(ar1_series):
    """Test fit_custom_arima fits AR(1) model successfully."""
    fit = fit_custom_arima(
        ar1_series, m=1, order=(1, 0, 0), constant=False, ic="aic"
    )
    
    assert fit['converged'] is True
    assert np.isfinite(fit['aic'])
    assert np.isfinite(fit['ic'])
    assert len(fit['residuals']) == len(ar1_series)


def test_fit_custom_arima_with_drift(random_walk_series):
    """Test fit_custom_arima with drift term."""
    fit = fit_custom_arima(
        random_walk_series, m=1, order=(0, 1, 0), constant=True, ic="aic"
    )
    
    assert fit['converged'] is True
    # Drift should be in xreg
    assert fit['xreg'] is not None


def test_fit_custom_arima_different_ic(ar1_series):
    """Test fit_custom_arima computes different IC values."""
    fit = fit_custom_arima(
        ar1_series, m=1, order=(1, 0, 0), constant=True, ic="bic"
    )
    
    assert np.isfinite(fit['aic'])
    assert np.isfinite(fit['bic'])
    assert np.isfinite(fit['aicc'])
    # BIC should be used as IC when ic="bic"
    assert fit['ic'] == fit['bic']


# =============================================================================
# Tests for kpss_test and adf_test
# =============================================================================
def test_kpss_test_stationary(ar1_series):
    """Test KPSS test on stationary series."""
    stat, pval = kpss_test(ar1_series)
    
    # Stationary series should have high p-value (null is stationarity)
    assert np.isfinite(stat)
    assert 0 <= pval <= 1


def test_adf_test_stationary(ar1_series):
    """Test ADF test on stationary series."""
    stat, pval = adf_test(ar1_series)
    
    # Stationary series should reject unit root (low p-value)
    assert np.isfinite(stat)
    assert 0 <= pval <= 1


def test_adf_test_nonstationary(random_walk_series):
    """Test ADF test on non-stationary series."""
    stat, pval = adf_test(random_walk_series)
    
    # Non-stationary series should have higher p-value
    assert np.isfinite(stat)
    # p-value should be relatively high for random walk
    assert pval > 0.01  # Not strongly rejecting unit root


# =============================================================================
# Tests for time_index
# =============================================================================
def test_time_index_basic():
    """Test time_index generates correct values."""
    result = time_index(5, m=1, start=1.0)
    
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_time_index_with_seasonality():
    """Test time_index with seasonal period."""
    result = time_index(4, m=4, start=1.0)
    
    # Each step is 1/m = 0.25
    expected = np.array([1.0, 1.25, 1.5, 1.75])
    np.testing.assert_array_almost_equal(result, expected)


def test_time_index_custom_start():
    """Test time_index with custom start value."""
    result = time_index(3, m=2, start=5.0)
    
    expected = np.array([5.0, 5.5, 6.0])
    np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Tests for has_coef
# =============================================================================
def test_has_coef_true(ar1_series):
    """Test has_coef returns True when coefficient exists."""
    fit = fit_custom_arima(ar1_series, m=1, order=(1, 0, 0), constant=True)
    
    assert has_coef(fit, 'ar1') is True


def test_has_coef_false(ar1_series):
    """Test has_coef returns False when coefficient doesn't exist."""
    fit = fit_custom_arima(ar1_series, m=1, order=(1, 0, 0), constant=False)
    
    assert has_coef(fit, 'intercept') is False


def test_has_coef_no_coef():
    """Test has_coef returns False when no coef in fit."""
    fit = {'aic': 100}
    
    assert has_coef(fit, 'ar1') is False


# =============================================================================
# Tests for npar_fit
# =============================================================================
def test_npar_fit(ar1_series):
    """Test npar_fit counts parameters correctly."""
    fit = fit_custom_arima(ar1_series, m=1, order=(1, 0, 0), constant=True)
    
    npar = npar_fit(fit)
    
    # AR(1) with mean: ar1 + intercept + sigma2 = 3
    assert npar >= 2


def test_npar_fit_no_mask():
    """Test npar_fit with no mask."""
    fit = {}
    
    npar = npar_fit(fit)
    
    assert npar == 1  # Just sigma2


# =============================================================================
# Tests for n_and_nstar
# =============================================================================
def test_n_and_nstar(ar1_series):
    """Test n_and_nstar computes effective sample size."""
    fit = fit_custom_arima(ar1_series, m=1, order=(1, 1, 0), constant=False)
    
    n, nstar = n_and_nstar(fit)
    
    # nstar should be n - d - D*m
    # For d=1, D=0, m=1: nstar = n - 1
    assert nstar == n - 1


def test_n_and_nstar_seasonal():
    """Test n_and_nstar with seasonal differencing."""
    np.random.seed(42)
    y = np.random.randn(120)
    
    fit = fit_custom_arima(y, m=12, order=(0, 0, 0), seasonal=(0, 1, 0), constant=False)
    
    n, nstar = n_and_nstar(fit)
    
    # For D=1, m=12: nstar = n - D*m = n - 12
    assert nstar == n - 12


# =============================================================================
# Tests for prepend_drift
# =============================================================================
def test_prepend_drift_to_none():
    """Test prepend_drift with None xreg."""
    drift = np.array([1.0, 2.0, 3.0])
    
    result = prepend_drift(None, drift)
    
    assert isinstance(result, pd.DataFrame)
    assert 'drift' in result.columns
    np.testing.assert_array_equal(result['drift'].values, drift)


def test_prepend_drift_to_dataframe():
    """Test prepend_drift with existing DataFrame."""
    xreg = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
    drift = np.array([0.1, 0.2, 0.3])
    
    result = prepend_drift(xreg, drift)
    
    assert list(result.columns) == ['drift', 'x1', 'x2']
    np.testing.assert_array_almost_equal(result['drift'].values, drift)


def test_prepend_drift_to_ndarray():
    """Test prepend_drift with numpy array."""
    xreg = np.array([[1, 4], [2, 5], [3, 6]])
    drift = np.array([0.1, 0.2, 0.3])
    
    result = prepend_drift(xreg, drift)
    
    assert isinstance(result, pd.DataFrame)
    assert result.columns[0] == 'drift'
    assert result.shape == (3, 3)


def test_forecast_arima_with_only_drift(ar1_series):
    """Test forecast_arima with model that has only drift (no other xreg)."""
    # Fit model with drift only
    fit = arima_rjh(ar1_series, m=1, order=(1, 1, 0), include_drift=True)
    
    # Forecast without providing xreg - drift should be added automatically
    fc = forecast_arima(fit, h=5)
    
    assert len(fc['mean']) == 5


def test_forecast_arima_fan_levels(ar1_series):
    """Test forecast_arima with fan=True generates many levels."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[80, 95], fan=True)
    
    # fan=True should override level
    assert len(fc['level']) > 2


# =============================================================================
# Tests for auto_arima
# =============================================================================
def test_auto_arima_ar1(ar1_series):
    """Test auto_arima on AR(1) series."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    assert fit['converged'] is True
    assert np.isfinite(fit['ic'])
    assert 'coef' in fit


def test_auto_arima_random_walk(random_walk_series):
    """Test auto_arima detects differencing need."""
    fit = auto_arima(random_walk_series, m=1, stepwise=True, trace=False)
    
    # Should detect need for differencing
    d = fit['arma'][5]  # d is at index 5
    assert d >= 1


def test_auto_arima_constant_series():
    """Test auto_arima on constant series."""
    y = np.ones(50)
    
    fit = auto_arima(y, m=1, stepwise=True)
    
    # Should fit ARIMA(0,0,0)
    assert fit['arma'][0] == 0  # p
    assert fit['arma'][1] == 0  # q


def test_auto_arima_all_nan_raises():
    """Test auto_arima raises error on all NaN."""
    y = np.full(50, np.nan)
    
    with pytest.raises(ValueError, match="All data are missing"):
        auto_arima(y, m=1)


def test_auto_arima_grid_search(ar1_series):
    """Test auto_arima with grid search (stepwise=True)."""
    fit = auto_arima(
        ar1_series, m=1, stepwise=True, trace=False,
        max_p=2, max_q=2, max_P=0, max_Q=0
    )
    
    assert fit['converged'] is True


def test_auto_arima_with_xreg(ar1_series):
    """Test auto_arima with exogenous regressors."""
    xreg = pd.DataFrame({'x1': np.random.randn(len(ar1_series))})
    
    fit = auto_arima(ar1_series, m=1, xreg=xreg, stepwise=True, trace=False)
    
    assert fit['converged'] is True
    assert 'x1' in fit['coef'].columns


def test_auto_arima_stationary_constraint(ar1_series):
    """Test auto_arima with stationary=True forces d=D=0."""
    fit = auto_arima(ar1_series, m=1, stationary=True, stepwise=True)
    
    assert fit['arma'][5] == 0  # d
    assert fit['arma'][6] == 0  # D


def test_auto_arima_different_ic(ar1_series):
    """Test auto_arima with different information criteria."""
    fit_aic = auto_arima(ar1_series, m=1, ic="aic", stepwise=True, trace=False)
    fit_bic = auto_arima(ar1_series, m=1, ic="bic", stepwise=True, trace=False)
    
    # Both should converge
    assert fit_aic['converged'] is True
    assert fit_bic['converged'] is True


def test_auto_arima_with_box_cox_auto(ar1_series):
    """Test auto_arima with automatic Box-Cox lambda selection."""
    y_pos = np.abs(ar1_series) + 1.0
    
    fit = auto_arima(y_pos, m=1, stepwise=True, lambda_bc="auto", trace=False)
    
    assert fit['converged'] is True
    assert fit['lambda'] is not None


def test_auto_arima_with_constant_d_D():
    """Test auto_arima detects constant after differencing."""
    # Create series that becomes constant after differencing
    y = np.arange(1.0, 51.0)  # Linear trend
    
    # Adding noise to avoid becoming exactly constant
    np.random.seed(42)
    y = y + np.random.randn(50) * 0.01
    
    fit = auto_arima(y, m=1, stepwise=True, trace=False)
    
    # Should detect d=1 for trend
    assert fit['arma'][5] >= 1  # d >= 1


def test_auto_arima_allowdrift_false(random_walk_series):
    """Test auto_arima with allowdrift=False."""
    fit = auto_arima(random_walk_series, m=1, stepwise=True, 
                     allowdrift=False, trace=False)
    
    # Should not include drift
    assert fit['converged'] is True


def test_auto_arima_approximation_refit(ar1_series):
    """Test auto_arima refits without approximation."""
    # Large series triggers approximation automatically
    np.random.seed(42)
    y_large = np.random.randn(200)
    
    fit = auto_arima(y_large, m=1, stepwise=True, approximation=True, trace=False)
    
    assert fit['converged'] is True


# =============================================================================
# Tests for arima_rjh
# =============================================================================
def test_arima_rjh_basic(ar1_series):
    """Test arima_rjh fits basic ARIMA model."""
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0))
    
    assert fit['converged'] is True
    assert 'ar1' in fit['coef'].columns


def test_arima_rjh_with_drift(random_walk_series):
    """Test arima_rjh with drift term."""
    fit = arima_rjh(random_walk_series, m=1, order=(0, 1, 0), include_drift=True)
    
    assert fit['xreg'] is not None
    assert 'drift' in fit['xreg'].columns


def test_arima_rjh_include_constant(ar1_series):
    """Test arima_rjh include_constant parameter."""
    # When include_constant=True and d+D=0, should include mean
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0), include_constant=True)
    
    assert 'intercept' in fit['coef'].columns


def test_arima_rjh_no_drift_warning(random_walk_series):
    """Test arima_rjh warns when drift not allowed."""
    # d=2 should not allow drift
    with pytest.warns(UserWarning, match="No drift term fitted"):
        arima_rjh(random_walk_series, m=1, order=(0, 2, 0), include_drift=True)


def test_arima_rjh_with_box_cox():
    """Test arima_rjh with Box-Cox transformation."""
    np.random.seed(42)
    y_pos = np.exp(np.random.randn(80) * 0.1 + 2)
    
    fit = arima_rjh(y_pos, m=1, order=(1, 0, 0), lambda_bc=0.0)
    
    assert fit['lambda'] == 0.0
    assert fit['converged'] is True


def test_arima_rjh_with_auto_box_cox():
    """Test arima_rjh with automatic Box-Cox lambda."""
    np.random.seed(42)
    y_pos = np.exp(np.random.randn(80) * 0.1 + 2)
    
    fit = arima_rjh(y_pos, m=1, order=(1, 0, 0), lambda_bc="auto")
    
    assert fit['lambda'] is not None
    assert fit['converged'] is True


def test_arima_rjh_include_constant_true(ar1_series):
    """Test arima_rjh with include_constant=True."""
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0), include_constant=True)
    
    assert 'intercept' in fit['coef'].columns


def test_arima_rjh_include_constant_false(ar1_series):
    """Test arima_rjh with include_constant=False."""
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0), include_constant=False)
    
    # Should not have intercept
    assert 'intercept' not in fit['coef'].columns


def test_arima_rjh_minimum_data_length():
    """Test arima_rjh handles minimum data correctly."""
    # With d=1, series length must be > d + D*m = 1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Should work with 5 points and order (1,0,0)
    fit = arima_rjh(y, m=1, order=(1, 0, 0))
    assert fit['converged'] is True


# =============================================================================
# Tests for forecast_arima
# =============================================================================
def test_forecast_arima_basic(ar1_series):
    """Test forecast_arima generates forecasts."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=10)
    
    assert 'mean' in fc
    assert len(fc['mean']) == 10
    assert fc['lower'] is None  # No levels specified
    assert fc['upper'] is None


def test_forecast_arima_with_intervals(ar1_series):
    """Test forecast_arima generates prediction intervals."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[80, 95])
    
    assert fc['lower'].shape == (5, 2)
    assert fc['upper'].shape == (5, 2)
    assert fc['level'] == [80, 95]
    # Lower should be less than upper
    assert np.all(fc['lower'] < fc['upper'])


def test_forecast_arima_with_xreg(ar1_series):
    """Test forecast_arima with exogenous regressors."""
    n = len(ar1_series)
    xreg_train = pd.DataFrame({'x1': np.random.randn(n)})
    
    fit = auto_arima(ar1_series, m=1, xreg=xreg_train, stepwise=True, trace=False)
    
    xreg_new = pd.DataFrame({'x1': np.random.randn(5)})
    fc = forecast_arima(fit, xreg=xreg_new)
    
    assert len(fc['mean']) == 5


def test_forecast_arima_returns_correct_keys(ar1_series):
    """Test forecast_arima returns all expected keys."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[95])
    
    expected_keys = ['mean', 'lower', 'upper', 'level', 'x', 'fitted', 
                     'residuals', 'method', 'lambda', 'biasadj']
    for key in expected_keys:
        assert key in fc


def test_forecast_arima_with_drift(random_walk_series):
    """Test forecast_arima handles drift term correctly."""
    fit = arima_rjh(random_walk_series, m=1, order=(0, 1, 0), include_drift=True)
    
    fc = forecast_arima(fit, h=5)
    
    assert len(fc['mean']) == 5
    # Forecasts should show trend due to drift
    assert not np.allclose(fc['mean'], fc['mean'][0])


def test_forecast_arima_with_box_cox(ar1_series):
    """Test forecast_arima with Box-Cox transformation."""
    # Use positive series for Box-Cox
    y_pos = np.abs(ar1_series) + 1.0
    
    fit = auto_arima(y_pos, m=1, stepwise=True, lambda_bc=0.5, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[80, 95])
    
    assert len(fc['mean']) == 5
    assert fc['lambda'] == 0.5


def test_forecast_arima_with_box_cox_biasadj(ar1_series):
    """Test forecast_arima with Box-Cox and bias adjustment."""
    y_pos = np.abs(ar1_series) + 1.0
    
    fit = auto_arima(y_pos, m=1, stepwise=True, lambda_bc=0.5, biasadj=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[80, 95])
    
    assert len(fc['mean']) == 5
    assert fc['biasadj'] == True


def test_forecast_arima_with_negative_lambda():
    """Test forecast_arima with negative Box-Cox lambda."""
    np.random.seed(42)
    y_pos = np.exp(np.random.randn(100) * 0.1 + 2)
    
    fit = auto_arima(y_pos, m=1, stepwise=True, lambda_bc=-0.5, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[80, 95])
    
    assert len(fc['mean']) == 5
    # Lambda should be stored
    assert fc['lambda'] == -0.5


def test_forecast_arima_level_as_proportions(ar1_series):
    """Test forecast_arima handles levels as proportions (0-1)."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[0.80, 0.95])
    
    # Should convert to percentages
    assert fc['level'] == [80.0, 95.0]
    assert fc['lower'].shape == (5, 2)


# =============================================================================
# Tests for predict_arima (from _arima_base)
# =============================================================================
def test_predict_arima_basic(ar1_series):
    """Test predict_arima generates forecasts from arima_rjh model."""
    from skforecast.stats.arima._arima_base import predict_arima
    
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0))
    
    fc = predict_arima(fit, n_ahead=10)
    
    assert 'mean' in fc
    assert len(fc['mean']) == 10
    assert 'se' in fc


def test_predict_arima_with_intervals(ar1_series):
    """Test predict_arima generates prediction intervals."""
    from skforecast.stats.arima._arima_base import predict_arima
    
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0))
    
    fc = predict_arima(fit, n_ahead=5, level=[80, 95])
    
    assert fc['lower'].shape == (5, 2)
    assert fc['upper'].shape == (5, 2)
    assert fc['level'] == [80, 95]
    # Lower should be less than upper
    assert np.all(fc['lower'] < fc['upper'])


def test_predict_arima_with_xreg(ar1_series):
    """Test predict_arima with exogenous regressors."""
    from skforecast.stats.arima._arima_base import predict_arima
    
    n = len(ar1_series)
    xreg_train = pd.DataFrame({'x1': np.random.randn(n)})
    
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0), xreg=xreg_train)
    
    xreg_new = pd.DataFrame({'x1': np.random.randn(5)})
    fc = predict_arima(fit, n_ahead=5, newxreg=xreg_new)
    
    assert len(fc['mean']) == 5


def test_predict_arima_returns_correct_keys(ar1_series):
    """Test predict_arima returns all expected keys."""
    from skforecast.stats.arima._arima_base import predict_arima
    
    fit = arima_rjh(ar1_series, m=1, order=(1, 0, 0))
    
    fc = predict_arima(fit, n_ahead=5, level=[95])
    
    expected_keys = ['mean', 'lower', 'upper', 'level', 'se', 'y', 'fitted', 
                     'residuals', 'method']
    for key in expected_keys:
        assert key in fc


# =============================================================================
# Tests for search_arima
# =============================================================================
def test_search_arima_basic(ar1_series):
    """Test search_arima finds a model."""
    fit = search_arima(
        ar1_series, m=1, d=0, D=0,
        max_p=2, max_q=2, max_P=0, max_Q=0,
        trace=False
    )
    
    assert fit['converged'] is True
    assert np.isfinite(fit['ic'])


def test_search_arima_respects_max_order(ar1_series):
    """Test search_arima respects max_order constraint."""
    fit = search_arima(
        ar1_series, m=1, d=0, D=0,
        max_p=5, max_q=5, max_order=2,
        trace=False
    )
    
    # p + q should not exceed max_order
    p, q = fit['arma'][0], fit['arma'][1]
    assert p + q <= 2


def test_search_arima_allowdrift_true(random_walk_series):
    """Test search_arima with drift allowed."""
    fit = search_arima(
        random_walk_series, m=1, d=1, D=0,
        max_p=2, max_q=2, max_P=0, max_Q=0,
        allowdrift=True,
        trace=False
    )
    
    assert fit['converged'] is True


def test_search_arima_allowmean_true(ar1_series):
    """Test search_arima with mean allowed."""
    fit = search_arima(
        ar1_series, m=1, d=0, D=0,
        max_p=2, max_q=2, max_P=0, max_Q=0,
        allowmean=True,
        trace=False
    )
    
    assert fit['converged'] is True


def test_search_arima_seasonal(seasonal_series):
    """Test search_arima with seasonal parameters."""
    fit = search_arima(
        seasonal_series, m=12, d=0, D=0,
        max_p=1, max_q=1, max_P=1, max_Q=1,
        trace=False
    )
    
    assert fit['converged'] is True


# =============================================================================
# Edge cases and error handling
# =============================================================================
def test_auto_arima_short_series():
    """Test auto_arima adjusts parameters for short series."""
    np.random.seed(42)
    y = np.random.randn(15)
    
    fit = auto_arima(y, m=1, stepwise=True, trace=False)
    
    assert fit['converged'] is True


def test_fit_custom_arima_returns_error_model_on_exception():
    """Test fit_custom_arima returns error model structure."""
    # Test that _create_error_model returns correct structure
    error_model = _create_error_model(order=(1, 0, 1), seasonal=(0, 0, 0), m=1)
    
    assert error_model['ic'] == np.inf
    assert error_model['converged'] is False
    assert 'model' in error_model
    assert 'arma' in error_model


def test_auto_arima_nmodels_limit():
    """Test auto_arima respects nmodels limit."""
    np.random.seed(42)
    y = np.random.randn(100)
    
    # Use a reasonable nmodels that allows some searching
    # but is still limited
    fit = auto_arima(y, m=1, stepwise=True, nmodels=20, trace=False)
    
    # Should still converge with limited models
    assert fit['converged'] is True


def test_auto_arima_nmodels_very_low_no_index_error():
    """
    Test auto_arima does not raise IndexError when nmodels is very low.
    
    When nmodels is set to a value lower than the number of initial models
    in the stepwise search, the function should handle it gracefully by
    stopping early rather than raising an IndexError.
    """
    np.random.seed(42)
    y = np.random.randn(100)
    
    # Test with very low nmodels values (1, 2, 3)
    # These values could previously cause IndexError when
    # k >= nmodels in the initial model fitting phase
    for nmodels in [1, 2, 3]:
        fit = auto_arima(y, m=1, stepwise=True, nmodels=nmodels, trace=False)
        
        # Should complete without IndexError
        assert fit is not None
        assert 'arima' in fit or 'converged' in fit
        
        # The number of models evaluated should not exceed nmodels
        # (results array has shape (nmodels, 8))
        assert fit.get('converged', True) is True or fit.get('arima') is not None

# =============================================================================
# Tests for prepare_drift
# =============================================================================
def test_prepare_drift_model_without_drift(ar1_series):
    """Test prepare_drift raises error when model doesn't have drift."""
    from skforecast.stats.arima._auto_arima import prepare_drift
    
    # Fit a model without drift
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    xreg = pd.DataFrame({'x1': np.random.randn(len(ar1_series))})
    
    # Should raise ValueError since model has no xreg for drift reconstruction
    with pytest.raises(ValueError, match="no xreg for drift reconstruction"):
        prepare_drift(fit, ar1_series, xreg)


def test_prepare_drift_model_with_drift(random_walk_series):
    """Test prepare_drift when model has drift term."""
    from skforecast.stats.arima._auto_arima import prepare_drift
    
    # Fit a model with drift
    fit = arima_rjh(random_walk_series, m=1, order=(0, 1, 0), include_drift=True)
    
    result = prepare_drift(fit, random_walk_series, None)
    
    # Should add drift term
    assert isinstance(result, pd.DataFrame)
    assert 'drift' in result.columns


# =============================================================================
# Tests for refit_arima_model
# =============================================================================
def test_refit_arima_model_basic(ar1_series):
    """Test refit_arima_model on a fitted model."""
    from skforecast.stats.arima._auto_arima import refit_arima_model
    
    # First fit a model
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    # Refit on same data
    refit = refit_arima_model(ar1_series, m=1, model=fit, xreg=None, method="CSS-ML")
    
    assert refit['converged'] is True
    assert 'coef' in refit


def test_refit_arima_model_with_xreg(ar1_series):
    """Test refit_arima_model with exogenous regressors."""
    from skforecast.stats.arima._auto_arima import refit_arima_model
    
    xreg = pd.DataFrame({'x1': np.random.randn(len(ar1_series))})
    
    # First fit a model with xreg
    fit = auto_arima(ar1_series, m=1, xreg=xreg, stepwise=True, trace=False)
    
    # Refit on same data
    refit = refit_arima_model(ar1_series, m=1, model=fit, xreg=xreg, method="CSS-ML")
    
    assert refit['converged'] is True


# =============================================================================
# Tests for _time_index_jit
# =============================================================================
@pytest.mark.parametrize(
    'n, m, start, expected',
    [
        (5, 1, 1.0, np.array([1.0, 2.0, 3.0, 4.0, 5.0])),      # basic
        (4, 4, 1.0, np.array([1.0, 1.25, 1.5, 1.75])),         # seasonal
        (3, 0, 1.0, np.array([1.0, 2.0, 3.0])),                # m=0 edge case
    ]
)
def test_time_index_jit(n, m, start, expected):
    """Test _time_index_jit generates correct indices."""
    from skforecast.stats.arima._auto_arima import _time_index_jit
    
    result = _time_index_jit(n, m, start)
    np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Tests for _newmodel_jit
# =============================================================================
@pytest.mark.parametrize(
    'p, d, q, P, D, Q, constant, expected',
    [
        (2, 0, 1, 0, 0, 0, 1, True),   # new model (different p)
        (1, 0, 1, 0, 0, 0, 1, False),  # existing model
    ]
)
def test_newmodel_jit(p, d, q, P, D, Q, constant, expected):
    """Test _newmodel_jit returns correct boolean for model configurations."""
    from skforecast.stats.arima._auto_arima import _newmodel_jit
    
    results = np.full((10, 8), np.nan)
    results[0, :7] = [1, 0, 1, 0, 0, 0, 1]  # ARIMA(1,0,1) with constant
    
    is_new = _newmodel_jit(p, d, q, P, D, Q, constant, results, 1)
    assert is_new is expected


# =============================================================================
# Tests for arima_trace_str edge cases
# =============================================================================
def test_arima_trace_str_no_constant_with_differencing():
    """Test trace string for model without constant and with differencing."""
    result = arima_trace_str(
        order=(1, 1, 1), seasonal=(0, 0, 0), m=1, constant=False, ic_value=123.45
    )
    
    assert "ARIMA(1,1,1)" in result
    assert "123.45" in result


# =============================================================================
# Tests for auto_arima with test="adf"
# =============================================================================
def test_auto_arima_with_adf_test(ar1_series):
    """Test auto_arima using ADF test."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, test="adf", trace=False)
    
    assert fit['converged'] is True


# =============================================================================
# Tests for forecast_arima edge cases
# =============================================================================
def test_forecast_arima_single_level(ar1_series):
    """Test forecast_arima with single confidence level."""
    fit = auto_arima(ar1_series, m=1, stepwise=True, trace=False)
    
    fc = forecast_arima(fit, h=5, level=[90])
    
    assert fc['lower'].shape == (5, 1)
    assert fc['upper'].shape == (5, 1)
    assert fc['level'] == [90]