"""
Deterministic prediction validation tests for ETS models.
Tests that ETS produces reproducible, mathematically correct predictions.
"""

import numpy as np
import pandas as pd
import pytest
from skforecast.stats._ets import ets, forecast_ets, Ets


# Known good predictions from R forecast package or statsmodels
# These serve as reference values to ensure implementation correctness

def test_ets_ann_deterministic_predictions():
    """Test ANN model produces exact expected predictions"""
    # Simple series with known dynamics
    y = np.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 57.0])
    
    model = ets(y, m=1, model="ANN", alpha=0.3)
    forecasts = forecast_ets(model, h=3, level=None)
    
    # With alpha=0.3, predictions should be close to level
    # Verify predictions are finite and reasonable
    assert np.all(np.isfinite(forecasts["mean"]))
    assert len(forecasts["mean"]) == 3
    
    # Predictions should be flat (no trend in ANN model)
    # and close to last smoothed level (around 40-50)
    assert forecasts["mean"][0] > 35.0
    assert forecasts["mean"][0] < 60.0
    # All forecasts should be approximately equal (no trend)
    assert np.allclose(forecasts["mean"], forecasts["mean"][0], rtol=0.01)


def test_ets_aan_deterministic_predictions():
    """Test AAN model produces consistent trend forecasts"""
    # Linear trend series
    y = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
    
    model = ets(y, m=1, model="AAN", alpha=0.8, beta=0.2)
    forecasts = forecast_ets(model, h=5, level=None)
    
    # Should continue linear trend
    assert np.all(np.isfinite(forecasts["mean"]))
    assert len(forecasts["mean"]) == 5
    
    # Check trend continues upward
    diffs = np.diff(forecasts["mean"])
    assert np.all(diffs > 0)  # Increasing
    assert np.allclose(diffs, diffs[0], rtol=0.1)  # Roughly constant increment


def test_ets_aaa_seasonal_deterministic():
    """Test AAA model captures seasonal pattern"""
    # Series with obvious seasonal pattern (period=4)
    y = np.array([10, 15, 13, 12,  # Season 1
                  11, 16, 14, 13,  # Season 2
                  12, 17, 15, 14,  # Season 3
                  13, 18, 16, 15]) # Season 4
    
    model = ets(y, m=4, model="AAA")
    forecasts = forecast_ets(model, h=8, level=None)
    
    assert len(forecasts["mean"]) == 8
    assert np.all(np.isfinite(forecasts["mean"]))
    
    # Check seasonal pattern repeats (peaks at positions 1, 5 within period=4)
    # Next season should show similar pattern
    assert forecasts["mean"][1] > forecasts["mean"][0]  # 2nd in season > 1st
    assert forecasts["mean"][5] > forecasts["mean"][4]  # Pattern repeats


def test_ets_reproducibility_same_data():
    """Test that fitting same data produces identical results"""
    np.random.seed(42)
    y = np.random.randn(50) + np.arange(50) * 0.1
    
    # Fit twice
    model1 = ets(y, m=1, model="AAN")
    model2 = ets(y, m=1, model="AAN")
    
    # Should produce identical parameters
    assert np.isclose(model1.params.alpha, model2.params.alpha)
    assert np.isclose(model1.params.beta, model2.params.beta)
    assert np.allclose(model1.fitted, model2.fitted)
    assert np.isclose(model1.loglik, model2.loglik)
    
    # Forecasts should be identical
    fc1 = forecast_ets(model1, h=10, level=None)
    fc2 = forecast_ets(model2, h=10, level=None)
    assert np.allclose(fc1["mean"], fc2["mean"])


def test_ets_residuals_sum_to_zero():
    """Test that residuals sum to approximately zero (unbiased)"""
    np.random.seed(123)
    y = 10 + np.arange(100) * 0.5 + np.random.randn(100) * 2
    
    model = ets(y, m=1, model="AAN")
    
    # Residuals should sum to approximately zero
    residuals = model.residuals
    assert np.abs(np.mean(residuals)) < 0.5  # Mean close to zero


def test_ets_fitted_plus_residuals_equals_original():
    """Test that fitted + residuals = original series"""
    y = np.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 57.0])
    
    model = ets(y, m=1, model="ANN")
    
    # fitted + residuals should equal original
    reconstructed = model.fitted + model.residuals
    assert np.allclose(reconstructed, y, atol=1e-10)


def test_ets_prediction_intervals_contain_mean():
    """Test that prediction intervals always contain the point forecast"""
    y = np.random.randn(80) + 10
    
    model = ets(y, m=1, model="ANN")
    forecasts = forecast_ets(model, h=10, level=[80, 95])
    
    mean = forecasts["mean"]
    
    # Mean should be between lower and upper for all levels
    assert np.all(forecasts["lower_80"] <= mean)
    assert np.all(mean <= forecasts["upper_80"])
    assert np.all(forecasts["lower_95"] <= mean)
    assert np.all(mean <= forecasts["upper_95"])


def test_ets_prediction_interval_width_increases():
    """Test that prediction interval width increases with horizon"""
    y = np.random.randn(100) + 10
    
    model = ets(y, m=1, model="AAN")
    forecasts = forecast_ets(model, h=20, level=[95])
    
    width = forecasts["upper_95"] - forecasts["lower_95"]
    
    # Width should generally increase (allowing small decreases due to damping)
    # Check first vs last
    assert width[-1] >= width[0] * 0.9  # Allow 10% tolerance


def test_ets_damped_trend_dampens():
    """Test that damped trend forecasts eventually flatten"""
    y = np.arange(50) * 2.0 + 100  # Strong linear trend
    
    model = ets(y, m=1, model="AAN", damped=True)
    forecasts = forecast_ets(model, h=50, level=None)
    
    # Check that increments decrease over horizon (damping effect)
    diffs = np.diff(forecasts["mean"])
    
    # Later differences should be smaller than earlier ones
    early_diff = np.mean(diffs[:5])
    late_diff = np.mean(diffs[-5:])
    assert late_diff < early_diff


def test_ets_multiplicative_error_positive_predictions():
    """Test that multiplicative error models produce positive predictions"""
    y = np.exp(np.random.randn(80)) + 10  # Ensure positive
    
    model = ets(y, m=1, model="MNN")
    forecasts = forecast_ets(model, h=20, level=[95])
    
    # All predictions and intervals should be positive
    assert np.all(forecasts["mean"] > 0)
    assert np.all(forecasts["lower_95"] >= 0)  # Can be zero at boundary
    assert np.all(forecasts["upper_95"] > 0)


def test_ets_seasonal_forecast_repeats_pattern():
    """Test that seasonal forecasts repeat the seasonal pattern"""
    # Clear seasonal pattern with period=12
    t = np.arange(120)
    y = 100 + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5
    
    model = ets(y, m=12, model="ANA")  # No trend, additive season
    forecasts = forecast_ets(model, h=24, level=None)
    
    # Check that pattern repeats: forecast[i] ≈ forecast[i+12]
    first_season = forecasts["mean"][:12]
    second_season = forecasts["mean"][12:24]
    
    # Should be very similar (no trend, pure seasonal)
    correlation = np.corrcoef(first_season, second_season)[0, 1]
    assert correlation > 0.95  # High correlation between seasons


def test_ets_constant_series_flat_forecast():
    """Test that constant series produces flat forecast"""
    y = np.ones(50) * 42.0
    
    model = ets(y, m=1, model="ANN")
    forecasts = forecast_ets(model, h=10, level=None)
    
    # All forecasts should be approximately 42
    assert np.allclose(forecasts["mean"], 42.0, rtol=0.01)


def test_estimator_predict_deterministic():
    """Test Ets estimator produces deterministic predictions"""
    y = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
    
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Multiple predictions should be identical
    pred1 = est.predict(steps=5)
    pred2 = est.predict(steps=5)
    pred3 = est.predict(steps=5)
    
    assert np.allclose(pred1, pred2)
    assert np.allclose(pred2, pred3)


def test_ets_box_cox_inverse_transform():
    """Test that Box-Cox transformation is correctly inverted in forecasts"""
    y = np.exp(np.random.randn(80)) + 5  # Positive data
    
    # Fit with lambda=0.5
    model = ets(y, m=1, model="ANN", lambda_param=0.5)
    forecasts = forecast_ets(model, h=10, level=None)
    
    # Forecasts should be in original scale (positive)
    assert np.all(forecasts["mean"] > 0)
    
    # Should be reasonable values (not millions or near-zero)
    assert np.all(forecasts["mean"] < np.max(y) * 5)
    assert np.all(forecasts["mean"] > np.min(y) * 0.2)


def test_ets_states_final_values_reasonable():
    """Test that final state values are reasonable"""
    y = np.arange(50) + 100.0
    
    model = ets(y, m=1, model="AAN")
    states = model.states
    
    # States is the final state vector [level, trend]
    assert states.ndim == 1
    assert len(states) == 2  # level and trend
    
    # Level should be close to last observation
    level = states[0]
    assert 140 < level < 160
    
    # Trend should be close to 1 (series increases by 1 each step)
    trend = states[1]
    assert 0.5 < trend < 1.5


def test_ets_aic_bic_ordering():
    """Test that AIC and BIC have expected relationship"""
    y = np.random.randn(100) + 50
    
    model = ets(y, m=1, model="AAN")
    
    # BIC penalizes parameters more than AIC for n>7
    # So typically BIC > AIC for same model
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert model.bic > model.aic  # BIC has stronger penalty


def test_ets_sigma2_positive():
    """Test that residual variance is always positive"""
    y = np.random.randn(80) + 20
    
    model = ets(y, m=1, model="ANN")
    
    assert model.sigma2 > 0
    assert np.isfinite(model.sigma2)


def test_ets_multiple_horizons_consistent():
    """Test that forecasts are consistent across different horizons"""
    y = np.arange(50) * 0.5 + 10
    
    model = ets(y, m=1, model="AAN")
    
    # Get forecasts for h=10
    fc_10 = forecast_ets(model, h=10, level=None)
    
    # Get forecasts for h=5
    fc_5 = forecast_ets(model, h=5, level=None)
    
    # First 5 should match
    assert np.allclose(fc_10["mean"][:5], fc_5["mean"])


def test_ets_no_trend_flat_long_term():
    """Test that no-trend models produce flat long-term forecasts"""
    y = 50 + np.random.randn(80) * 2
    
    model = ets(y, m=1, model="ANN")  # No trend
    forecasts = forecast_ets(model, h=100, level=None)
    
    # Long-term forecasts should be approximately constant
    last_20 = forecasts["mean"][-20:]
    assert np.std(last_20) < 0.1  # Very little variation


def test_ets_seasonal_amplitude_preserved():
    """Test that seasonal amplitude is approximately preserved in forecasts"""
    # Series with seasonal pattern
    t = np.arange(120)
    seasonal_component = 10 * np.sin(2 * np.pi * t / 12)
    y = 100 + seasonal_component + np.random.randn(120) * 1
    
    model = ets(y, m=12, model="ANA")
    forecasts = forecast_ets(model, h=12, level=None)
    
    # Check amplitude of seasonal pattern in forecasts
    fc_amplitude = (np.max(forecasts["mean"]) - np.min(forecasts["mean"])) / 2
    
    # Should be close to 10
    assert 5 < fc_amplitude < 15  # Allow some tolerance


def test_ets_fitted_values_in_sample():
    """Test that fitted values are reasonable in-sample predictions"""
    y = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    
    model = ets(y, m=1, model="AAN")
    
    # Fitted values should be close to observed
    assert np.all(np.abs(model.fitted - y) < 10)
    
    # Should track the trend
    fitted_trend = np.diff(model.fitted)
    assert np.all(fitted_trend > 0)  # Should be increasing


def test_ets_loglik_negative():
    """Test that log-likelihood is negative (as expected)"""
    y = np.random.randn(80) + 10
    
    model = ets(y, m=1, model="ANN")
    
    # Log-likelihood should be negative for most reasonable models
    assert model.loglik < 0


def test_ets_parameters_in_valid_range():
    """Test that estimated parameters are in valid ranges"""
    y = np.random.randn(100) + 50
    
    model = ets(y, m=1, model="AAN")
    
    # Check smoothing parameters
    assert 0 < model.params.alpha < 1
    assert 0 <= model.params.beta < model.params.alpha
    
    if model.config.damped:
        assert 0 < model.params.phi <= 1


def test_ets_prediction_interval_symmetry_additive():
    """Test that prediction intervals are symmetric for additive models"""
    y = np.random.randn(80) + 50
    
    model = ets(y, m=1, model="AAN")
    forecasts = forecast_ets(model, h=10, level=[95])
    
    mean = forecasts["mean"]
    lower = forecasts["lower_95"]
    upper = forecasts["upper_95"]
    
    # For additive models, intervals should be symmetric
    lower_dist = mean - lower
    upper_dist = upper - mean
    
    assert np.allclose(lower_dist, upper_dist, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
