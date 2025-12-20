# Unit test fit method - Ets
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from ..._ets import Ets


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Generate AR(1) series for testing"""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def seasonal_series(n=120, m=12, seed=42):
    """Generate series with seasonal pattern"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.1 * t
    seasonal = 3 * np.sin(2 * np.pi * t / m)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise

def test_ets_fit_invalid_y_type_raises():
    """
    Test that fit raises TypeError for invalid y type.
    """
    y = [1, 2, 3, 4, 5]  # List, not Series or ndarray
    model = Ets()
    with pytest.raises((TypeError, ValueError)):
        model.fit(y)


def test_ets_fit_multidimensional_y_raises():
    """
    Test that fit raises error for multidimensional y input.
    """
    y = np.random.randn(50, 2)
    model = Ets()
    msg = "`y` must be a 1D array-like sequence."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_ets_fit_empty_series_raises():
    """
    Test that fit raises error for empty series.
    """
    y = np.array([])
    model = Ets()
    msg = "Series too short to fit ETS model."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_ets_fit_2d_single_column_array():
    """
    Test that fit accepts 2D array with single column and squeezes it.
    """
    y = ar1_series(50).reshape(-1, 1)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.y_.shape == (50,)
    assert model.y_.ndim == 1

def test_estimator_fit_and_attributes():
    """Test Ets estimator fit and attributes"""
    y = ar1_series(100)
    est = Ets(m=1, model="ANN")
    est.fit(y)

    assert hasattr(est, "model_")
    assert hasattr(est, "y_")
    assert hasattr(est, "config_")
    assert hasattr(est, "params_")
    assert hasattr(est, "fitted_values_")
    assert hasattr(est, "residuals_in_")
    assert hasattr(est, "n_features_in_")

    assert est.y_.shape == y.shape
    assert est.fitted_values_.shape == y.shape
    assert est.residuals_in_.shape == y.shape
    assert est.n_features_in_ == 1


def test_estimator_with_pandas_series():
    """Test Ets estimator with pandas Series input"""
    y_array = ar1_series(80)
    y_series = pd.Series(y_array)

    est = Ets(m=1, model="ANN")
    est.fit(y_series)

    assert est.y_.shape == (80,)
    assert isinstance(est.y_, np.ndarray)


def test_estimator_with_fixed_parameters():
    """Test Ets estimator with fixed smoothing parameters"""
    y = ar1_series(80)
    est = Ets(m=1, model="ANN", alpha=0.3)
    est.fit(y)

    # Alpha should be close to the fixed value
    assert abs(est.params_.alpha - 0.3) < 0.1


def test_estimator_auto_selection():
    """Test Ets estimator with automatic model selection"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="ZZZ")
    est.fit(y)

    assert hasattr(est, "model_")
    assert hasattr(est, "config_")

    # Should have selected a model
    assert est.config_.error in ["A", "M"]
    assert est.config_.trend in ["N", "A", "M"]
    assert est.config_.season in ["N", "A", "M"]


def test_estimator_seasonal_model():
    """Test Ets estimator with seasonal model"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="AAA")
    est.fit(y)

    assert est.config_.season == "A"
    assert est.config_.m == 12


# Tests for exact learned parameter values
# ==============================================================================

def test_ets_fit_ann_model_exact_parameters():
    """
    Test that ANN model learns expected parameter values.
    """
    rng = np.random.default_rng(123)
    y = 10 + 0.5 * np.arange(50) + rng.normal(0, 0.5, 50)
    est = Ets(m=1, model='ANN')
    est.fit(y)
    
    # Check exact parameter values
    assert est.params_.alpha == 0.1
    np.testing.assert_almost_equal(
        est.params_.init_states[0], 
        12.294441518810888, 
        decimal=10
    )
    
    # Check exact fitted values
    expected_fitted = np.array([12.29444152, 12.0155413, 11.84559784])
    np.testing.assert_array_almost_equal(
        est.fitted_values_[:3],
        expected_fitted,
        decimal=8
    )


def test_ets_fit_aan_model_exact_parameters():
    """
    Test that AAN model learns expected parameter and trend values.
    """
    rng = np.random.default_rng(123)
    y = 10 + 0.5 * np.arange(50) + rng.normal(0, 0.5, 50)
    est = Ets(m=1, model='AAN')
    est.fit(y)
    
    # Check exact smoothing parameters
    assert est.params_.alpha == 0.1
    assert est.params_.beta == 0.01
    
    # Check exact initial states
    np.testing.assert_almost_equal(
        est.params_.init_states[0],
        9.547864485468777,
        decimal=10
    )
    np.testing.assert_almost_equal(
        est.params_.init_states[1],
        0.49937764242583876,
        decimal=10
    )
    
    # Check exact fitted values
    expected_fitted = np.array([10.04724213, 10.48702146, 10.96218045])
    np.testing.assert_array_almost_equal(
        est.fitted_values_[:3],
        expected_fitted,
        decimal=8
    )


def test_ets_fit_aaa_model_exact_parameters():
    """
    Test that AAA model learns expected seasonal parameter values.
    """
    rng = np.random.default_rng(123)
    t = np.arange(60)
    y = 10 + 0.1*t + 2*np.sin(2*np.pi*t/12) + rng.normal(0, 0.3, 60)
    est = Ets(m=12, model='AAA')
    est.fit(y)
    
    # Check exact smoothing parameters
    assert est.params_.alpha == 0.1
    assert est.params_.beta == 0.01
    assert est.params_.gamma == 0.01
    
    # Check exact initial states
    np.testing.assert_almost_equal(
        est.params_.init_states[0],
        9.809283495511119,
        decimal=10
    )
    np.testing.assert_almost_equal(
        est.params_.init_states[1],
        0.11056198008716075,
        decimal=10
    )
    
    # Check exact first few seasonal states
    expected_seasonal = np.array([-0.9890557, -1.6628666, -2.24873172])
    np.testing.assert_array_almost_equal(
        est.params_.init_states[2:5],
        expected_seasonal,
        decimal=8
    )
    
    # Check exact fitted values
    expected_fitted = np.array([10.12450248, 10.93641956, 12.06930232])
    np.testing.assert_array_almost_equal(
        est.fitted_values_[:3],
        expected_fitted,
        decimal=8
    )


def test_ets_fit_ana_model_exact_parameters():
    """
    Test that ANA model (additive error, no trend, additive seasonal) learns exact values.
    """
    rng = np.random.default_rng(42)
    t = np.arange(48)
    y = 15 + 3*np.sin(2*np.pi*t/12) + rng.normal(0, 0.4, 48)
    est = Ets(m=12, model='ANA')
    est.fit(y)
    
    # Check exact smoothing parameters
    assert est.params_.alpha == 0.1
    assert est.params_.gamma == 0.01
    assert est.params_.beta == 0.0
    
    # Check initial states shape (level + m-1 seasonal)
    assert len(est.params_.init_states) == 12  # level + 11 seasonal (one is implicit)
    
    # Check exact initial level
    np.testing.assert_almost_equal(
        est.params_.init_states[0],
        14.99239297,
        decimal=8
    )
    
    # Check exact first few seasonal states
    expected_seasonal = np.array([-1.34342466, -2.26290369, -3.22831963])
    np.testing.assert_array_almost_equal(
        est.params_.init_states[1:4],
        expected_seasonal,
        decimal=8
    )
    
    # Check exact fitted values
    expected_fitted = np.array([14.95208798, 16.37390343, 17.64039636])
    np.testing.assert_array_almost_equal(
        est.fitted_values_[:3],
        expected_fitted,
        decimal=8
    )


def test_ets_fit_man_model_exact_parameters():
    """
    Test that MAN model (multiplicative error, additive trend, no season) learns exact values.
    """
    rng = np.random.default_rng(456)
    y = 5 + 0.3 * np.arange(50) + rng.normal(0, 0.3, 50)
    # Ensure positive values for multiplicative error
    y = np.abs(y) + 1.0
    
    est = Ets(m=1, model='MAN')
    est.fit(y)
    
    # Check config
    assert est.config_.error == 'M'
    assert est.config_.trend == 'A'
    assert est.config_.season == 'N'
    
    # Check that parameters exist and are valid
    assert 0 < est.params_.alpha <= 1
    assert 0 <= est.params_.beta <= 1
    assert est.params_.gamma == 0.0
    
    # Check initial states
    assert len(est.params_.init_states) == 2  # level + trend
    assert np.all(np.isfinite(est.params_.init_states))


def test_ets_fit_damped_trend_exact_parameters():
    """
    Test that damped trend model includes phi parameter.
    """
    rng = np.random.default_rng(789)
    y = 20 + 2 * np.arange(60) * np.exp(-0.05 * np.arange(60)) + rng.normal(0, 0.5, 60)
    
    est = Ets(m=1, model='AAN', damped=True)
    est.fit(y)
    
    # Check that phi parameter exists
    assert hasattr(est.params_, 'phi')
    assert 0 < est.params_.phi <= 1
    
    # Check config
    assert est.config_.damped is True
    
    # Damped models should have alpha, beta, and phi
    assert 0 < est.params_.alpha <= 1
    assert 0 <= est.params_.beta <= 1


def test_ets_fit_fixed_alpha_exact_value():
    """
    Test that fixed alpha parameter is respected.
    """
    y = ar1_series(80)
    fixed_alpha = 0.3
    
    est = Ets(m=1, model='ANN', alpha=fixed_alpha)
    est.fit(y)
    
    # Alpha should be exactly the fixed value
    assert est.params_.alpha == fixed_alpha


def test_ets_fit_fixed_beta_exact_value():
    """
    Test that fixed beta parameter is respected.
    """
    y = ar1_series(80)
    fixed_beta = 0.2
    
    est = Ets(m=1, model='AAN', beta=fixed_beta)
    est.fit(y)
    
    # Beta should be exactly the fixed value
    assert est.params_.beta == fixed_beta


def test_ets_fit_fixed_gamma_exact_value():
    """
    Test that fixed gamma parameter is respected for seasonal model.
    """
    y = seasonal_series(60, m=12)
    fixed_gamma = 0.15
    
    est = Ets(m=12, model='AAA', gamma=fixed_gamma)
    est.fit(y)
    
    # Gamma should be exactly the fixed value
    assert est.params_.gamma == fixed_gamma

