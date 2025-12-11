# Additional unit tests for comprehensive coverage of ETS implementation

import math
import numpy as np
import pandas as pd
import pytest
import warnings

from skforecast.stats._ets import (
    ets,
    forecast_ets,
    auto_ets,
    simulate_ets,
    residual_diagnostics,
    Ets,
    ETSConfig,
    ETSParams,
    init_states,
    get_bounds,
    admissible,
    check_param,
    is_constant,
)


def seasonal_series(n=120, m=12, seed=42):
    """Generate series with seasonal pattern"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.1 * t
    seasonal = 3 * np.sin(2 * np.pi * t / m)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Generate AR(1) series for testing"""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


# ============================================================================
# Tests for edge cases and error conditions
# ============================================================================

def test_ets_not_enough_data_for_parameters():
    """Test ETS with insufficient observations for number of parameters"""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Only 6 observations
    
    # AAA model with m=4 needs many parameters, should fail
    with pytest.raises(ValueError, match="Not enough data"):
        ets(y, m=4, model="AAA")


def test_ets_empty_series():
    """Test ETS with empty series"""
    y = np.array([])
    
    with pytest.raises(ValueError, match="at least 1 observation"):
        ets(y, m=1, model="ANN")


def test_ets_single_observation():
    """Test ETS with single observation"""
    y = np.array([10.0])
    
    # Should fail - not enough data
    with pytest.raises(ValueError, match="Not enough data"):
        ets(y, m=1, model="ANN")


def test_ets_seasonal_less_than_m_observations():
    """Test seasonal model with n < m"""
    y = np.array([1.0, 2.0, 3.0, 4.0])  # Only 4 observations
    
    # AAA with m=12 needs at least 12 observations
    with pytest.raises(ValueError, match="need at least m=12"):
        ets(y, m=12, model="AAA")


# ============================================================================
# Tests for simulation functionality
# ============================================================================

def test_simulate_ets_basic():
    """Test simulate_ets function"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN")
    
    simulations = simulate_ets(model, h=12, n_sim=100)
    
    assert simulations.shape == (100, 12)
    assert np.all(np.isfinite(simulations))


def test_simulate_ets_seasonal():
    """Test simulate_ets with seasonal model"""
    y = seasonal_series(120, m=12)
    model = ets(y, m=12, model="AAA")
    
    simulations = simulate_ets(model, h=24, n_sim=50)
    
    assert simulations.shape == (50, 24)
    assert np.all(np.isfinite(simulations))


def test_simulate_ets_large_horizon():
    """Test simulate_ets with large horizon"""
    y = ar1_series(80)
    model = ets(y, m=1, model="AAN")
    
    simulations = simulate_ets(model, h=100, n_sim=10)
    
    assert simulations.shape == (10, 100)
    assert np.all(np.isfinite(simulations))


# ============================================================================
# Tests for Box-Cox transformation
# ============================================================================

def test_ets_fixed_lambda():
    """Test ETS with fixed Box-Cox lambda"""
    y = np.exp(ar1_series(80)) + 1.0  # Ensure positive
    
    model = ets(y, m=1, model="ANN", lambda_param=0.5)
    
    assert model.transform is not None
    assert abs(model.transform.lambda_param - 0.5) < 1e-6


def test_ets_lambda_zero_log_transform():
    """Test ETS with lambda=0 (log transformation)"""
    y = np.exp(ar1_series(80)) + 1.0  # Ensure positive
    
    model = ets(y, m=1, model="ANN", lambda_param=0.0)
    
    assert model.transform is not None
    assert model.transform.lambda_param == 0.0
    
    # Forecast should work
    forecasts = forecast_ets(model, h=10)
    assert np.all(forecasts["mean"] > 0)


def test_forecast_ets_bias_adjust():
    """Test forecast with and without bias adjustment"""
    y = np.exp(ar1_series(80)) + 1.0
    model = ets(y, m=1, model="ANN", lambda_param=0.5)
    
    # With bias adjustment
    fc_adj = forecast_ets(model, h=10, bias_adjust=True)
    
    # Without bias adjustment
    fc_no_adj = forecast_ets(model, h=10, bias_adjust=False)
    
    # Results should differ
    assert not np.allclose(fc_adj["mean"], fc_no_adj["mean"])


# ============================================================================
# Tests for parameter bounds and admissibility
# ============================================================================

def test_admissible_with_seasonal_high_m():
    """Test admissibility with high seasonal period"""
    # m > 24 uses different admissibility logic
    result = admissible(alpha=0.3, beta=0.1, gamma=0.05, phi=0.95, m=30)
    assert isinstance(result, (bool, np.bool_))


def test_admissible_extreme_parameters():
    """Test admissibility with extreme parameter values"""
    # Alpha near boundary
    assert not admissible(alpha=0.9999, beta=0.5, gamma=0.1, phi=0.95, m=12)
    
    # Phi > 1
    assert not admissible(alpha=0.3, beta=0.1, gamma=0.05, phi=1.1, m=12)
    
    # Negative phi
    assert not admissible(alpha=0.3, beta=0.1, gamma=0.05, phi=-0.1, m=12)


def test_check_param_with_different_bounds_types():
    """Test check_param with 'usual', 'admissible', and 'both' bounds"""
    lower = np.array([1e-4, 1e-4, 1e-4, 0.8, 0.0, 0.0])
    upper = np.array([0.9999, 0.9999, 0.9999, 0.98, 100.0, 100.0])
    
    # Usual bounds only
    result_usual = check_param(0.3, 0.1, 0.05, 0.95, lower, upper, "usual", 12)
    assert isinstance(result_usual, (bool, np.bool_))
    
    # Admissible only
    result_admissible = check_param(0.3, 0.1, 0.05, 0.95, lower, upper, "admissible", 12)
    assert isinstance(result_admissible, (bool, np.bool_))
    
    # Both
    result_both = check_param(0.3, 0.1, 0.05, 0.95, lower, upper, "both", 12)
    assert isinstance(result_both, (bool, np.bool_))


# ============================================================================
# Tests for model configuration
# ============================================================================

def test_ets_config_all_combinations():
    """Test ETSConfig with various error/trend/season combinations"""
    for error in ["A", "M"]:
        for trend in ["N", "A", "M"]:
            for season in ["N", "A", "M"]:
                config = ETSConfig(error=error, trend=trend, season=season, m=12)
                
                assert config.error == error
                assert config.trend == trend
                assert config.season == season
                assert config.error_code in [1, 2]
                assert config.trend_code in [0, 1, 2]
                assert config.season_code in [0, 1, 2]


def test_ets_params_none_values():
    """Test ETSParams handles None values in to_vector"""
    config = ETSConfig(error="A", trend="N", season="N", m=1)
    params = ETSParams(alpha=0.3, beta=None, gamma=None, phi=None, init_states=np.array([10.0]))
    
    vec = params.to_vector(config)
    # Should only include alpha and init_states
    assert len(vec) == 1 + 1


# ============================================================================
# Tests for auto model selection edge cases
# ============================================================================

def test_auto_ets_no_valid_models():
    """Test auto_ets when no models can be fitted"""
    # Very short series that can't fit most models
    y = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="No model could be fitted"):
        auto_ets(y, m=1, seasonal=False, verbose=False)


def test_auto_ets_with_multiplicative_disabled():
    """Test auto_ets with multiplicative models disabled"""
    y = seasonal_series(120, m=12)
    
    model = auto_ets(y, m=12, allow_multiplicative=False, verbose=False)
    
    # Should only select additive models
    assert model.config.error == "A"
    if model.config.season != "N":
        assert model.config.season == "A"


def test_auto_ets_with_multiplicative_trend_enabled():
    """Test auto_ets with multiplicative trend allowed"""
    y = np.exp(ar1_series(100)) + 10.0  # Positive series
    
    model = auto_ets(y, m=1, allow_multiplicative_trend=True, verbose=False)
    
    assert hasattr(model, "config")
    # Model may or may not select multiplicative trend, but shouldn't error


def test_auto_ets_verbose_output(capsys):
    """Test auto_ets verbose output"""
    y = ar1_series(80)
    
    auto_ets(y, m=1, seasonal=False, verbose=True)
    
    captured = capsys.readouterr().out
    # Should print model selection progress
    assert len(captured) > 0


def test_auto_ets_trend_none_vs_true():
    """Test auto_ets with trend=None vs trend=True"""
    y = ar1_series(100)
    
    # trend=None: auto-detect
    model_auto = auto_ets(y, m=1, trend=None, seasonal=False, verbose=False)
    
    # trend=True: force trend
    model_forced = auto_ets(y, m=1, trend=True, seasonal=False, verbose=False)
    
    assert hasattr(model_auto, "config")
    assert model_forced.config.trend != "N"


# ============================================================================
# Tests for helper functions
# ============================================================================

def test_is_constant_function():
    """Test is_constant helper function"""
    # Constant series
    assert is_constant(np.array([5.0, 5.0, 5.0, 5.0]))
    
    # Non-constant series
    assert not is_constant(np.array([1.0, 2.0, 3.0, 4.0]))
    
    # Series with tiny variation (still not constant)
    y_almost = np.array([5.0, 5.0, 5.000001, 5.0])
    assert not is_constant(y_almost)


def test_residual_diagnostics_short_series():
    """Test residual diagnostics with short series"""
    y = ar1_series(20)  # Short series
    model = ets(y, m=1, model="ANN")
    
    diag = residual_diagnostics(model)
    
    # Should still compute diagnostics
    assert "mean" in diag
    assert "std" in diag
    assert np.isfinite(diag["mean"])


# ============================================================================
# Tests for sklearn compatibility
# ============================================================================

def test_estimator_clone_compatibility():
    """Test Ets estimator can be cloned (sklearn compatibility)"""
    from sklearn.base import clone
    
    est = Ets(m=12, model="AAA", damped=True, alpha=0.3)
    est_cloned = clone(est)
    
    assert est_cloned.m == 12
    assert est_cloned.model == "AAA"
    assert est_cloned.damped == True
    assert est_cloned.alpha == 0.3


def test_estimator_repr():
    """Test Ets estimator string representation"""
    est = Ets(m=12, model="AAA")
    repr_str = repr(est)
    
    assert "Ets" in repr_str
    assert "m=12" in repr_str


# ============================================================================
# Tests for forecast edge cases
# ============================================================================

def test_forecast_ets_large_horizon():
    """Test forecast with very large horizon"""
    y = ar1_series(80)
    model = ets(y, m=1, model="AAN")
    
    forecasts = forecast_ets(model, h=1000)
    
    assert forecasts["mean"].shape == (1000,)
    assert np.all(np.isfinite(forecasts["mean"]))


def test_forecast_ets_multiple_levels():
    """Test forecast with multiple confidence levels"""
    y = ar1_series(80)
    model = ets(y, m=1, model="AAN")
    
    forecasts = forecast_ets(model, h=10, level=[50, 80, 90, 95, 99])
    
    assert "lower_50" in forecasts
    assert "upper_50" in forecasts
    assert "lower_99" in forecasts
    assert "upper_99" in forecasts
    
    # Check that wider intervals contain narrower ones
    assert np.all(forecasts["lower_99"] <= forecasts["lower_50"])
    assert np.all(forecasts["upper_99"] >= forecasts["upper_50"])


# ============================================================================
# Tests for multiplicative models
# ============================================================================

def test_ets_multiplicative_trend():
    """Test ETS with multiplicative trend"""
    y = np.exp(ar1_series(100)) + 10.0
    
    model = ets(y, m=1, model="AMN")
    
    assert model.config.trend == "M"
    assert model.fitted.shape == y.shape


def test_ets_multiplicative_seasonal():
    """Test ETS with multiplicative seasonal"""
    y = np.exp(seasonal_series(120, m=12)) + 5.0
    
    model = ets(y, m=12, model="AAM")
    
    assert model.config.season == "M"
    assert model.fitted.shape == y.shape


def test_ets_fully_multiplicative():
    """Test fully multiplicative model (M,M,M)"""
    y = np.exp(seasonal_series(120, m=12)) + 20.0
    
    model = ets(y, m=12, model="MMM")
    
    assert model.config.error == "M"
    assert model.config.trend == "M"
    assert model.config.season == "M"


# ============================================================================
# Tests for initialization methods
# ============================================================================

def test_init_states_with_insufficient_seasonal_data():
    """Test init_states when there's barely enough data"""
    y = seasonal_series(24, m=12)  # Exactly 2 seasonal periods
    config = ETSConfig(error="A", trend="A", season="A", m=12)
    
    states = init_states(y, config)
    
    assert len(states) == config.n_states
    assert np.all(np.isfinite(states))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
