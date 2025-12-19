# Unit test for Ets
# ==============================================================================
import math
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


def test_estimator_predict():
    """Test Ets estimator predict method"""
    y = ar1_series(120)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    mean = est.predict(steps=8)
    assert mean.shape == (8,)
    assert np.all(np.isfinite(mean))


def test_estimator_predict_interval():
    """Test Ets estimator predict_interval method"""
    y = ar1_series(120)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    # Test with as_frame=True
    df = est.predict_interval(steps=5, level=(80, 95), as_frame=True)
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "lower_80" in df.columns
    assert "upper_80" in df.columns
    assert "lower_95" in df.columns
    assert "upper_95" in df.columns
    assert len(df) == 5

    # Test with as_frame=False
    raw = est.predict_interval(steps=3, level=(90,), as_frame=False)
    assert isinstance(raw, dict)
    assert "mean" in raw
    assert "lower_90" in raw
    assert "upper_90" in raw
    assert raw["mean"].shape == (3,)


def test_estimator_invalid_steps():
    """Test Ets estimator with invalid steps parameter"""
    y = ar1_series(50)
    est = Ets(m=1, model="ANN")
    est.fit(y)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=0)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=-2)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=1.5)


def test_estimator_unfitted():
    """Test Ets estimator before fitting"""
    est = Ets(m=1, model="ANN")

    with pytest.raises(Exception):
        est.predict(steps=1)


def test_estimator_residuals_and_fitted_helpers():
    """Test residuals_() and fitted_() helper methods"""
    y = ar1_series(70)
    est = Ets(m=1, model="AAN").fit(y)

    r = est.residuals_()
    f = est.fitted_()

    assert r.shape == y.shape
    assert f.shape == y.shape

    # Residuals should equal y - fitted (approximately)
    assert np.allclose(r, y - f, atol=1e-10)


def test_estimator_summary(capsys):
    """Test Ets estimator summary output"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    est.summary()
    captured = capsys.readouterr().out

    assert "ETS Model Summary" in captured
    assert "Number of observations:" in captured
    assert "Smoothing parameters:" in captured
    assert "alpha (level):" in captured


def test_estimator_score():
    """Test Ets estimator score method"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    score = est.score()

    # R^2 should be between -inf and 1
    assert score <= 1.0
    assert np.isfinite(score) or np.isnan(score)


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


def test_estimator_seasonal_model():
    """Test Ets estimator with seasonal model"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="AAA")
    est.fit(y)

    assert est.config_.season == "A"
    assert est.config_.m == 12

    # Forecast should capture seasonality
    forecasts = est.predict(steps=24)
    assert len(forecasts) == 24
    
