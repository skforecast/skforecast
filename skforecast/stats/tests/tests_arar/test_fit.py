# Unit test fit method - Arar
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
import warnings
from ..._arar import Arar
from ....exceptions import ExogenousInterpretationWarning


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def ar4_series(n=200, phi=(0.5, 0.3, -0.2, 0.1), sigma=1.0, seed=123):
    """Helper function to generate AR(4) series for testing coefficient recovery."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[:4] = e[:4]
    for t in range(4, n):
        y[t] = phi[0] * y[t-1] + phi[1] * y[t-2] + phi[2] * y[t-3] + phi[3] * y[t-4] + e[t]
    return y


# =============================================================================
# Input and Parameter Validation Tests (MUST BE FIRST)
# =============================================================================

def test_arar_fit_invalid_y_type_raises():
    """
    Test that fit raises TypeError for invalid y type.
    """
    y = [1, 2, 3, 4, 5]  # List, not Series or ndarray
    model = Arar()
    
    with pytest.raises(TypeError, match="must be a pandas Series or numpy ndarray"):
        model.fit(y)


def test_arar_fit_invalid_exog_type_raises():
    """
    Test that fit raises TypeError for invalid exog type.
    """
    y = np.random.randn(50)
    exog = [1, 2, 3]  # List, not valid type
    model = Arar()
    
    with pytest.raises(TypeError, match="must be None, a pandas Series, pandas DataFrame, or numpy ndarray"):
        model.fit(y, exog=exog)


def test_arar_fit_multidimensional_y_raises():
    """
    Test that fit raises error for multidimensional y input.
    """
    y = np.random.randn(50, 2)
    model = Arar()
    
    with pytest.raises(ValueError, match="must be a 1D array-like"):
        model.fit(y)


def test_arar_fit_with_exog_length_mismatch():
    """
    Test that fit raises error when exog length doesn't match y length.
    """
    y = ar1_series(100)
    exog_wrong = np.random.randn(80, 2)  # Wrong length
    
    model = Arar()
    
    with pytest.raises(ValueError, match="Length of exog"):
        model.fit(y, exog=exog_wrong)


def test_arar_fit_with_exog_3d_raises():
    """
    Test that fit raises error for 3D exog input.
    """
    y = ar1_series(50)
    exog_3d = np.random.randn(50, 2, 3)  # 3D array
    
    model = Arar()
    
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        model.fit(y, exog=exog_3d)


def test_arar_fit_very_short_series_safe_true():
    """
    Test that very short series works with safe=True.
    """
    y = np.array([1.0])
    model = Arar(safe=True)
    
    # Should not raise error with safe=True
    model.fit(y)
    assert model.model_ is not None


def test_arar_fit_very_short_series_safe_false_raises():
    """
    Test that fit raises error for very short series when safe=False.
    """
    y = np.array([1.0]) 
    model = Arar(safe=False)
    
    with pytest.raises(ValueError, match="Series too short"):
        model.fit(y)


def test_arar_fit_two_observations_safe_true():
    """
    Test fit with exactly 2 observations and safe=True.
    """
    y = np.array([1.0, 2.0])
    model = Arar(safe=True)
    
    # Should work with safe=True
    model.fit(y)
    assert model.model_ is not None


# =============================================================================
# Basic Fit Functionality Tests
# =============================================================================

def test_arar_fit_basic():
    """
    Test basic fit functionality and resulting attributes with exact values.
    """
    y = ar1_series(100, seed=123)
    model = Arar()
    result = model.fit(y)

    # Test return value
    assert result is model
    
    # Test model attributes exist and check exact values
    assert hasattr(model, "model_")
    np.testing.assert_array_almost_equal(model.y_, y, decimal=10)
    assert model.y_.shape == y.shape
    
    # Check exact coefficient values
    assert model.coef_.shape == (4,)
    expected_coef = np.array([0.6621508, -0.1511758, -0.08748325, -0.09202529])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Check exact lag values
    assert isinstance(model.lags_, tuple) and len(model.lags_) == 4
    assert model.lags_ == (1, 3, 15, 18)
    
    # Check exact sigma2
    assert np.isscalar(model.sigma2_) and model.sigma2_ >= 1e-12
    np.testing.assert_almost_equal(model.sigma2_, 0.7510658616551584, decimal=6)
    
    # Check exact psi
    assert model.psi_.ndim == 1 and model.psi_.size >= 1
    expected_psi = np.array([1.0])
    np.testing.assert_array_almost_equal(model.psi_, expected_psi, decimal=10)
    
    # Check exact sbar
    assert isinstance(model.sbar_, float)
    np.testing.assert_almost_equal(model.sbar_, 0.2555764169122907, decimal=6)
    
    # Check scalar attributes
    assert model.n_features_in_ == 1
    assert model.n_exog_features_in_ == 0
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.memory_reduced_ is False

    # Test fitted values and residuals shapes
    assert model.fitted_values_.shape == y.shape
    assert model.residuals_in_.shape == y.shape
    
    # Verify residuals = y - fitted_values exactly
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=10
    )
    
    # Check some exact fitted values (non-NaN portion)
    # First values will be NaN due to lag structure (lag 18 means first 17 are NaN)
    expected_fitted_first_valid = 0.5800004993187984  # At index 18
    assert np.isnan(model.fitted_values_[0])  # First value should be NaN
    # The first non-NaN fitted value should match expected
    first_valid_idx = np.where(~np.isnan(model.fitted_values_))[0][0]
    assert first_valid_idx == 18  # Should be at index 18 (lag 18)
    np.testing.assert_almost_equal(model.fitted_values_[first_valid_idx], expected_fitted_first_valid, decimal=6)
    
    # Check exact AIC/BIC values
    assert isinstance(model.aic_, float)
    assert isinstance(model.bic_, float)
    np.testing.assert_almost_equal(model.aic_, 223.6333233886359, decimal=6)
    np.testing.assert_almost_equal(model.bic_, 238.0736388722214, decimal=6)
    assert not np.isnan(model.aic_)
    assert not np.isnan(model.bic_)


def test_arar_fit_exact_aic_bic_values():
    """
    Test that AIC and BIC are computed correctly with exact values.
    """
    y = ar1_series(n=100, phi=0.6, seed=123)
    
    model = Arar()
    model.fit(y)
    
    # Manual AIC/BIC calculation
    n = len(y)
    k_arar = 6  # 4 AR coefficients + sbar + sigma2
    residuals = model.residuals_in_
    valid_residuals = residuals[~np.isnan(residuals)]
    n_valid = len(valid_residuals)
    
    # Log-likelihood (assuming normal errors)
    rss = np.sum(valid_residuals**2)
    log_likelihood = -0.5 * n_valid * (np.log(2 * np.pi) + np.log(rss / n_valid) + 1)
    
    # Expected AIC and BIC
    expected_aic = -2 * log_likelihood + 2 * k_arar
    expected_bic = -2 * log_likelihood + k_arar * np.log(n_valid)
    
    # Check they match
    np.testing.assert_almost_equal(model.aic_, expected_aic, decimal=6)
    np.testing.assert_almost_equal(model.bic_, expected_bic, decimal=6)
    
    # BIC should be larger for n > exp(2) ≈ 7.4
    assert model.bic_ > model.aic_


def test_arar_fit_fitted_values_accuracy():
    """
    Test that fitted values are reasonably accurate.
    """
    y = ar1_series(n=100, phi=0.7, sigma=0.5, seed=42)
    
    model = Arar()
    model.fit(y)
    
    # Fitted values should exist and match length
    assert model.fitted_values_.shape == y.shape
    
    # First few values will be NaN due to lag structure
    assert np.isnan(model.fitted_values_[0])
    
    # Later values should be finite
    valid_fitted = model.fitted_values_[~np.isnan(model.fitted_values_)]
    assert len(valid_fitted) > 0
    assert np.all(np.isfinite(valid_fitted))
    
    # Residuals should sum to approximately zero (up to numerical precision)
    valid_residuals = model.residuals_in_[~np.isnan(model.residuals_in_)]
    assert abs(np.mean(valid_residuals)) < 0.1  # Mean close to zero
    
    # Check residuals = y - fitted
    mask = ~np.isnan(model.fitted_values_)
    np.testing.assert_array_almost_equal(
        model.residuals_in_[mask],
        y[mask] - model.fitted_values_[mask],
        decimal=10
    )


def test_arar_fit_with_pandas_datetime_index():
    """
    Test fit with pandas Series with datetime index.
    """
    y_array = ar1_series(100, seed=123)
    date_index = pd.date_range('2020-01-01', periods=100, freq='D')
    y = pd.Series(y_array, index=date_index)
    
    model = Arar()
    model.fit(y)
    
    # Should store values as numpy array
    assert isinstance(model.y_, np.ndarray)
    np.testing.assert_array_almost_equal(model.y_, y.values, decimal=10)
    
    # Verify all learned values are exact
    expected_coef = np.array([0.6621508, -0.1511758, -0.08748325, -0.09202529])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    assert model.lags_ == (1, 3, 15, 18)
    np.testing.assert_almost_equal(model.sigma2_, 0.7510658616551584, decimal=6)


def test_arar_fit_updates_max_ar_depth_and_max_lag():
    """
    Test that fit updates max_ar_depth and max_lag when None.
    """
    y = ar1_series(100)
    model = Arar(max_ar_depth=None, max_lag=None)
    model.fit(y)
    
    # After fitting, these should be set by the underlying arar function
    assert model.max_ar_depth is not None
    assert model.max_lag is not None
    assert isinstance(model.max_ar_depth, int)
    assert isinstance(model.max_lag, int)
    # For n=100, defaults should be 26 and 40
    assert model.max_ar_depth == 26
    assert model.max_lag == 40


def test_arar_fit_preserves_explicit_params():
    """
    Test that explicit parameters are preserved after fitting.
    """
    y = ar1_series(100)
    model = Arar(max_ar_depth=10, max_lag=20, safe=True)
    model.fit(y)
    
    # These should be preserved
    assert model.max_ar_depth == 10
    assert model.max_lag == 20
    assert model.safe is True


def test_arar_fit_model_tuple_structure():
    """
    Test that model_ tuple has correct structure and values.
    """
    y = ar1_series(100, seed=123)
    model = Arar()
    model.fit(y)
    
    # Check model_ tuple structure
    assert isinstance(model.model_, tuple)
    assert len(model.model_) == 8
    
    # Unpack and verify
    Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag = model.model_
    
    # Verify each component matches class attributes
    np.testing.assert_array_almost_equal(Y, model.y_, decimal=10)
    np.testing.assert_array_almost_equal(best_phi, model.coef_, decimal=10)
    assert best_lag == model.lags_
    np.testing.assert_almost_equal(sigma2, model.sigma2_, decimal=10)
    np.testing.assert_array_almost_equal(psi, model.psi_, decimal=10)
    np.testing.assert_almost_equal(sbar, model.sbar_, decimal=10)
    assert max_ar_depth == model.max_ar_depth
    assert max_lag == model.max_lag


def test_arar_fit_resets_memory_reduced_flag():
    """
    Test that refitting resets the memory_reduced flag.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    model.reduce_memory()
    
    assert model.memory_reduced_ is True
    
    # Refit
    model.fit(y)
    
    assert model.memory_reduced_ is False
    assert model.fitted_values_ is not None
    assert model.residuals_in_ is not None


def test_arar_fit_nan_pattern_in_fitted_values():
    """
    Test that NaN pattern in fitted_values matches expected lag structure.
    """
    y = ar1_series(100, seed=123)
    model = Arar()
    model.fit(y)
    
    # Check NaN pattern based on largest lag
    largest_lag = max(model.lags_)  # Should be 18
    
    # First largest_lag values should be NaN (not largest_lag-1)
    nan_count = np.sum(np.isnan(model.fitted_values_))
    assert nan_count == largest_lag  # 18 NaN values for lag 18
    
    # First nan_count values should be NaN
    assert np.all(np.isnan(model.fitted_values_[:nan_count]))
    
    # Remaining values should be finite
    assert np.all(np.isfinite(model.fitted_values_[nan_count:]))
    
    # Same pattern for residuals
    assert np.sum(np.isnan(model.residuals_in_)) == nan_count
    assert np.all(np.isnan(model.residuals_in_[:nan_count]))
    assert np.all(np.isfinite(model.residuals_in_[nan_count:]))


# =============================================================================
# Exogenous variables tests - basic functionality
# =============================================================================

def test_arar_fit_with_exog():
    """
    Test Arar fit with exogenous variables and verify exact learned values.
    """
    np.random.seed(42)
    n = 100
    
    # Create AR(1) series with exogenous effects
    y_ar = ar1_series(n=n, phi=0.6, sigma=0.5, seed=42)
    exog = np.column_stack([
        np.sin(np.linspace(0, 4*np.pi, n)),
        np.cos(np.linspace(0, 4*np.pi, n))
    ])
    y = y_ar + 2.0 * exog[:, 0] + 1.5 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    # Check attributes
    assert model.exog_model_ is not None
    assert model.coef_exog_ is not None
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2
    
    # Check that original y is stored, not residuals
    np.testing.assert_array_almost_equal(model.y_, y, decimal=10)
    
    # Check exact exog coefficients
    assert len(model.coef_exog_) == 2
    expected_coef_exog = np.array([1.96805132, 1.36576084])
    np.testing.assert_array_almost_equal(model.coef_exog_, expected_coef_exog, decimal=6)
    
    # Check exact ARAR coefficients on residuals
    expected_lags = (1, 6, 10, 16)
    assert model.lags_ == expected_lags
    
    expected_coef = np.array([0.5740112, -0.15199523, 0.17431845, 0.1893768])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    np.testing.assert_almost_equal(model.sigma2_, 0.13259783018819143, decimal=6)
    np.testing.assert_almost_equal(model.sbar_, 2.1094237467877975e-17, decimal=6)
    
    # Check exact AIC/BIC with exog
    np.testing.assert_almost_equal(model.aic_, 79.0305393657764, decimal=6)
    np.testing.assert_almost_equal(model.bic_, 100.90789055536621, decimal=6)
    
    # Verify exog_model has intercept
    assert hasattr(model.exog_model_, 'intercept_')
    expected_intercept = -0.049223690270959776
    np.testing.assert_almost_equal(model.exog_model_.intercept_, expected_intercept, decimal=6)
    
    # Verify fitted_values include exog contribution
    # fitted = exog_fitted + arar_fitted
    exog_contribution = model.exog_model_.predict(exog)
    assert model.fitted_values_.shape == y.shape
    # Check residuals are correct
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=10
    )


def test_arar_fit_1d_exog_numpy_and_pandas():
    """
    Test Arar with 1D exogenous variable as numpy array and pandas Series.
    """
    np.random.seed(42)
    n = 100
    y = ar1_series(n=n, seed=42)
    exog_1d = np.sin(np.linspace(0, 4*np.pi, n))
    
    # Test with numpy 1D array
    model1 = Arar()
    model1.fit(y, exog=exog_1d, suppress_warnings=True)
    
    assert model1.n_features_in_ == 1
    assert model1.n_exog_features_in_ == 1
    assert model1.exog_model_ is not None
    assert len(model1.coef_exog_) == 1
    
    # Test with pandas Series
    exog_series = pd.Series(exog_1d)
    model2 = Arar()
    model2.fit(y, exog=exog_series, suppress_warnings=True)
    
    assert model2.n_features_in_ == 1
    assert model2.n_exog_features_in_ == 1
    assert model2.exog_model_ is not None
    assert len(model2.coef_exog_) == 1
    
    # Both should give same results
    np.testing.assert_array_almost_equal(model1.coef_exog_, model2.coef_exog_, decimal=10)


def test_arar_fit_multiple_features_exog():
    """
    Test Arar with multiple exogenous features and verify coefficient storage.
    """
    np.random.seed(42)
    n = 150
    y = ar1_series(n=n, phi=0.5, seed=42)
    exog = np.column_stack([
        np.sin(np.linspace(0, 4*np.pi, n)),
        np.cos(np.linspace(0, 4*np.pi, n)),
        np.linspace(0, 1, n),  # trend
        np.random.randn(n) * 0.1  # small noise feature
    ])
    y = y + 1.0 * exog[:, 0] + 0.5 * exog[:, 1] + 2.0 * exog[:, 2]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 4
    assert len(model.coef_exog_) == 4
    assert model.exog_model_ is not None
    
    # Check coefficients are reasonable
    assert np.all(np.isfinite(model.coef_exog_))


def test_arar_fit_exog_with_custom_params():
    """
    Test Arar with exogenous variables and custom max_ar_depth and max_lag.
    """
    np.random.seed(42)
    n = 100
    y = ar1_series(n=n, seed=42)
    exog = np.random.randn(n, 2) * 0.5
    
    model = Arar(max_ar_depth=10, max_lag=20)
    model.fit(y, exog=exog, suppress_warnings=True)
    
    assert model.max_ar_depth == 10
    assert model.max_lag == 20
    assert model.exog_model_ is not None


def test_arar_fit_exog_zero_variance_feature():
    """
    Test Arar handles exogenous features with zero variance (constant).
    """
    np.random.seed(42)
    n = 100
    y = ar1_series(n=n, seed=42)
    
    # One feature with variance, one constant
    exog = np.column_stack([
        np.random.randn(n) * 0.5,
        np.ones(n) * 5.0  # constant feature
    ])
    
    model = Arar()
    # Should not crash even with constant feature
    model.fit(y, exog=exog, suppress_warnings=True)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2
    assert model.exog_model_ is not None


def test_arar_fit_exog_coefficient_recovery():
    """
    Test that exogenous coefficients are reasonably recovered.
    """
    np.random.seed(42)
    n = 300  # Larger sample for better recovery
    
    # Create data with known linear relationship
    true_coef = [1.5, -0.8, 2.0]
    exog = np.column_stack([
        np.sin(np.linspace(0, 10*np.pi, n)),
        np.cos(np.linspace(0, 10*np.pi, n)),
        np.linspace(0, 1, n)
    ])
    
    # Pure linear relationship + small AR(1) component
    y_linear = 5.0 + exog @ true_coef
    y_ar = ar1_series(n=n, phi=0.3, sigma=0.2, seed=42)
    y = y_linear + y_ar
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    # Coefficients should be close to true values
    assert len(model.coef_exog_) == 3
    for i, true_c in enumerate(true_coef):
        # Allow reasonable deviation since ARAR models residuals
        assert abs(model.coef_exog_[i] - true_c) < 0.5


# =============================================================================
# Warning tests
# =============================================================================

def test_arar_fit_exog_emits_warning():
    """
    Test that fit with exog emits ExogenousInterpretationWarning by default.
    """
    y = ar1_series(50, seed=42)
    exog = np.random.randn(50, 1)
    
    model = Arar()
    
    with pytest.warns(ExogenousInterpretationWarning, match="two-step approach"):
        model.fit(y, exog=exog)


def test_arar_fit_exog_suppress_warnings():
    """
    Test that suppress_warnings=True prevents warning emission.
    """
    y = ar1_series(50, seed=42)
    exog = np.random.randn(50, 1)
    
    model = Arar()
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        # Should not raise any warning
        model.fit(y, exog=exog, suppress_warnings=True)
    
    assert model.exog_model_ is not None


# =============================================================================
# AIC/BIC Tests
# =============================================================================

def test_arar_fit_aic_bic_with_exog():
    """
    Test that AIC/BIC are computed correctly with exogenous variables.
    """
    y = ar1_series(100, seed=42)
    exog = np.random.randn(100, 2)
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    # AIC and BIC should account for both ARAR and exog parameters
    assert isinstance(model.aic_, float)
    assert isinstance(model.bic_, float)
    assert np.isfinite(model.aic_)
    assert np.isfinite(model.bic_)
    # BIC should be larger than AIC for reasonable sample size
    assert model.bic_ > model.aic_
    
    # With exog, should have k_arar=6 + k_exog=3 (intercept + 2 features)
    # Total = 9 parameters
    n = len(y)
    residuals = model.residuals_in_
    valid_residuals = residuals[~np.isnan(residuals)]
    n_valid = len(valid_residuals)
    
    # Verify parameter counting
    k_total = 6 + 3  # ARAR + exog regression
    rss = np.sum(valid_residuals**2)
    log_likelihood = -0.5 * n_valid * (np.log(2 * np.pi) + np.log(rss / n_valid) + 1)
    
    expected_aic = -2 * log_likelihood + 2 * k_total
    expected_bic = -2 * log_likelihood + k_total * np.log(n_valid)
    
    np.testing.assert_almost_equal(model.aic_, expected_aic, decimal=6)
    np.testing.assert_almost_equal(model.bic_, expected_bic, decimal=6)

