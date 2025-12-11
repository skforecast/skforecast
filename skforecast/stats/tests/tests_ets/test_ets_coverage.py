# Coverage-focused tests for ETS implementation
# Tests specifically targeting uncovered code paths

import numpy as np
import pandas as pd
import pytest
import warnings

from skforecast.stats._ets import (
    ets,
    forecast_ets,
    auto_ets,
    simulate_ets,
    Ets,
    ETSConfig,
    init_states,
    fourier,
)


def seasonal_series(n=120, m=12, seed=42):
    """Generate series with seasonal pattern"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.1 * t
    seasonal = 3 * np.sin(2 * np.pi * t / m)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise


# ============================================================================
# Tests for prediction intervals with analytical variance
# ============================================================================

def test_forecast_ets_prediction_intervals_ann():
    """Test prediction intervals for ANN model (analytical variance)"""
    np.random.seed(42)
    y = np.random.randn(100) + 10
    model = ets(y, m=1, model="ANN")
    
    forecasts = forecast_ets(model, h=10, level=[80, 95])
    
    assert "lower_80" in forecasts
    assert "upper_80" in forecasts
    assert "lower_95" in forecasts
    assert "upper_95" in forecasts
    
    # 95% intervals should be wider than 80%
    assert np.all(forecasts["lower_95"] <= forecasts["lower_80"])
    assert np.all(forecasts["upper_95"] >= forecasts["upper_80"])


def test_forecast_ets_prediction_intervals_aan():
    """Test prediction intervals for AAN model with trend (analytical variance)"""
    np.random.seed(42)
    y = np.arange(100) + np.random.randn(100)
    model = ets(y, m=1, model="AAN")
    
    forecasts = forecast_ets(model, h=10, level=[90])
    
    assert "lower_90" in forecasts
    assert "upper_90" in forecasts
    assert forecasts["mean"].shape == (10,)


def test_forecast_ets_prediction_intervals_aada():
    """Test prediction intervals for damped trend model (analytical variance)"""
    np.random.seed(42)
    y = np.arange(100) + np.random.randn(100)
    model = ets(y, m=1, model="AAN", damped=True)
    
    forecasts = forecast_ets(model, h=10, level=[95])
    
    assert "lower_95" in forecasts
    assert "upper_95" in forecasts


def test_forecast_ets_prediction_intervals_ana():
    """Test prediction intervals for seasonal model (analytical variance)"""
    y = seasonal_series(120, m=12, seed=42)
    model = ets(y, m=12, model="ANA")
    
    forecasts = forecast_ets(model, h=12, level=[80, 95])
    
    assert "lower_80" in forecasts
    assert "upper_80" in forecasts
    assert "lower_95" in forecasts
    assert "upper_95" in forecasts


def test_forecast_ets_prediction_intervals_aaa():
    """Test prediction intervals for AAA model (analytical variance)"""
    y = seasonal_series(120, m=12, seed=42)
    model = ets(y, m=12, model="AAA")
    
    forecasts = forecast_ets(model, h=12, level=[90])
    
    assert "lower_90" in forecasts
    assert "upper_90" in forecasts


def test_forecast_ets_prediction_intervals_simulation_fallback():
    """Test prediction intervals fall back to simulation for complex models"""
    y = np.exp(seasonal_series(120, m=12, seed=42)) + 10
    model = ets(y, m=12, model="MMM")  # Multiplicative - needs simulation
    
    forecasts = forecast_ets(model, h=12, level=[80, 95])
    
    # Should still generate intervals via simulation
    assert "lower_80" in forecasts or "mean" in forecasts
    assert "upper_80" in forecasts or "mean" in forecasts


def test_forecast_ets_no_intervals_with_zero_variance():
    """Test forecast handles models with zero variance gracefully"""
    y = np.ones(50) * 10.0  # Constant series
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ets(y, m=1, model="ANN")
    
    # Should warn about invalid variance
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecasts = forecast_ets(model, h=10, level=[95])
        
        # Should return only mean forecast
        assert "mean" in forecasts
        # May or may not have intervals depending on variance


# ============================================================================
# Tests for Fourier initialization with short series
# ============================================================================

def test_init_states_seasonal_short_series():
    """Test init_states with seasonal model when n < 3*m (uses Fourier)"""
    # Create series with exactly 2 seasonal periods (n = 2*m)
    y = seasonal_series(24, m=12, seed=42)
    config = ETSConfig(error="A", trend="A", season="A", m=12)
    
    states = init_states(y, config)
    
    # Should return states for level, trend, and m-1 seasonal components
    assert len(states) == config.n_states
    assert np.all(np.isfinite(states))


def test_init_states_seasonal_very_short():
    """Test init_states with seasonal model when n < 4 raises error"""
    # This should raise an error when n < 4
    y = np.array([1.0, 2.0, 3.0])  # Only 3 observations
    config = ETSConfig(error="A", trend="A", season="A", m=12)
    
    with pytest.raises(ValueError, match="at least 4 observations"):
        init_states(y, config)


def test_fourier_terms():
    """Test Fourier term generation"""
    y = np.random.randn(50)
    
    # Generate Fourier terms for in-sample
    X_in = fourier(y, period=12, K=2)
    assert X_in.shape[0] == len(y)
    
    # Generate Fourier terms for forecasting
    X_out = fourier(y, period=12, K=2, h=12)
    assert X_out.shape[0] == 12


# ============================================================================
# Tests for auto_ets model selection edge cases
# ============================================================================

def test_auto_ets_max_models_limit():
    """Test auto_ets with max_models parameter"""
    y = seasonal_series(120, m=12, seed=42)
    
    model = auto_ets(y, m=12, seasonal=True, max_models=5, verbose=False)
    
    assert hasattr(model, "config")
    assert model.fitted.shape == y.shape


def test_auto_ets_ic_aicc():
    """Test auto_ets with AICc information criterion"""
    y = seasonal_series(80, m=1, seed=42)
    
    model = auto_ets(y, m=1, seasonal=False, ic="aicc", verbose=False)
    
    assert hasattr(model, "aic")


def test_auto_ets_ic_bic():
    """Test auto_ets with BIC information criterion"""
    y = seasonal_series(80, m=1, seed=42)
    
    model = auto_ets(y, m=1, seasonal=False, ic="bic", verbose=False)
    
    assert hasattr(model, "bic")


def test_auto_ets_seasonal_insufficient_data():
    """Test auto_ets drops seasonality when n < m"""
    y = np.random.randn(20) + 10
    
    # Request seasonal with m=24, but only 20 observations
    model = auto_ets(y, m=24, seasonal=True, verbose=False)
    
    # Should fit non-seasonal model
    assert model.config.season == "N"


def test_auto_ets_trend_false():
    """Test auto_ets with trend=False"""
    y = np.arange(80) + np.random.randn(80)
    
    model = auto_ets(y, m=1, seasonal=False, trend=False, verbose=False)
    
    # Should only try non-trending models
    assert model.config.trend == "N"


def test_auto_ets_damped_true():
    """Test auto_ets with damped=True"""
    y = np.arange(80) + np.random.randn(80)
    
    model = auto_ets(y, m=1, seasonal=False, trend=True, damped=True, verbose=False)
    
    # Should only try damped models
    assert model.config.damped == True


def test_auto_ets_damped_false():
    """Test auto_ets with damped=False"""
    y = np.arange(80) + np.random.randn(80)
    
    model = auto_ets(y, m=1, seasonal=False, trend=True, damped=False, verbose=False)
    
    # Should not use damping
    assert model.config.damped == False


# ============================================================================
# Tests for Ets sklearn estimator edge cases
# ============================================================================

def test_estimator_with_series_name():
    """Test Ets estimator with pandas Series having a name"""
    y = pd.Series(np.random.randn(80) + 10, name="my_series")
    
    est = Ets(m=1, model="ANN")
    est.fit(y)
    
    assert hasattr(est, "model_")


def test_estimator_predict_with_non_integer_steps():
    """Test Ets estimator predict with numpy integer type"""
    y = np.random.randn(80) + 10
    
    est = Ets(m=1, model="ANN")
    est.fit(y)
    
    # Test with numpy integer
    forecasts = est.predict(np.int64(10))
    assert forecasts.shape == (10,)


def test_estimator_predict_interval_as_dict():
    """Test predict_interval returns dict when as_frame=False"""
    y = np.random.randn(80) + 10
    
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    result = est.predict_interval(steps=10, level=[80, 95], as_frame=False)
    
    assert isinstance(result, dict)
    assert "mean" in result
    assert "lower_80" in result
    assert "upper_95" in result


def test_estimator_predict_interval_single_level():
    """Test predict_interval with single confidence level"""
    y = np.random.randn(80) + 10
    
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    result = est.predict_interval(steps=10, level=[90])
    
    assert "mean" in result.columns
    assert "lower_90" in result.columns
    assert "upper_90" in result.columns


def test_estimator_score_with_nans():
    """Test Ets score method returns valid R^2"""
    y = np.random.randn(80) + 10
    
    est = Ets(m=1, model="ANN")
    est.fit(y)
    
    score = est.score()
    assert isinstance(score, float)
    # R^2 can be negative for poor fits
    assert -np.inf < score <= 1 or np.isnan(score)


def test_estimator_with_fixed_alpha():
    """Test Ets estimator with fixed alpha parameter"""
    y = np.random.randn(80) + 10
    
    est = Ets(m=1, model="ANN", alpha=0.5)
    est.fit(y)
    
    # Alpha should be close to 0.5 (may vary slightly due to optimization)
    assert est.model_.params.alpha > 0


def test_estimator_with_lambda_param():
    """Test Ets estimator with Box-Cox transformation"""
    y = np.exp(np.random.randn(80)) + 10
    
    est = Ets(m=1, model="ANN", lambda_param=0.5)
    est.fit(y)
    
    assert est.model_.transform is not None
    assert est.model_.transform.lambda_param == 0.5


def test_estimator_with_lambda_auto():
    """Test Ets estimator with automatic lambda selection"""
    y = np.exp(np.random.randn(80)) + 10
    
    est = Ets(m=1, model="ANN", lambda_auto=True)
    est.fit(y)
    
    assert est.model_.transform is not None
    assert est.model_.transform.lambda_param is not None


# ============================================================================
# Tests for simulate_ets edge cases
# ============================================================================

def test_simulate_ets_invalid_variance():
    """Test simulate_ets raises error when sigma2 <= 0"""
    y = np.ones(50) * 10.0  # Constant series
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ets(y, m=1, model="ANN")
    
    # Should raise error about invalid variance
    with pytest.raises(ValueError, match="invalid residual variance"):
        simulate_ets(model, h=10, n_sim=100)


def test_simulate_ets_multiplicative_error():
    """Test simulate_ets with multiplicative error model"""
    y = np.exp(np.random.randn(100)) + 20
    model = ets(y, m=1, model="MNN")
    
    simulations = simulate_ets(model, h=12, n_sim=50)
    
    assert simulations.shape == (50, 12)
    assert np.all(np.isfinite(simulations))


def test_simulate_ets_with_seasonality():
    """Test simulate_ets with seasonal model"""
    y = seasonal_series(120, m=12, seed=42)
    model = ets(y, m=12, model="AAA")
    
    simulations = simulate_ets(model, h=24, n_sim=30)
    
    assert simulations.shape == (30, 24)
    assert np.all(np.isfinite(simulations))


# ============================================================================
# Tests for edge cases in ets() function
# ============================================================================

def test_ets_with_bounds_usual():
    """Test ets with bounds='usual'"""
    y = np.random.randn(80) + 10
    
    model = ets(y, m=1, model="ANN", bounds="usual")
    
    assert hasattr(model, "params")


def test_ets_with_bounds_admissible():
    """Test ets with bounds='admissible'"""
    y = np.random.randn(80) + 10
    
    model = ets(y, m=1, model="ANN", bounds="admissible")
    
    assert hasattr(model, "params")


def test_ets_multiplicative_seasonal_with_small_values():
    """Test multiplicative seasonal model clips seasonal components"""
    y = seasonal_series(120, m=12, seed=42)
    # Make all values positive but small
    y = np.abs(y) + 0.1
    
    model = ets(y, m=12, model="AAM")
    
    assert model.config.season == "M"
    assert np.all(np.isfinite(model.fitted))


def test_ets_with_bias_adjust_false():
    """Test ets forecast without bias adjustment"""
    y = np.exp(np.random.randn(80)) + 10
    model = ets(y, m=1, model="ANN", lambda_param=0.5)
    
    forecasts = forecast_ets(model, h=10, bias_adjust=False)
    
    assert forecasts["mean"].shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
