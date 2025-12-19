# Unit test predict method - Ets
# ==============================================================================
import numpy as np
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


def test_estimator_predict():
    """Test Ets estimator predict method"""
    y = ar1_series(120)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    mean = est.predict(steps=8)
    assert mean.shape == (8,)
    assert np.all(np.isfinite(mean))


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


def test_estimator_seasonal_forecast():
    """Test Ets predict with seasonal model"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="AAA")
    est.fit(y)

    # Forecast should capture seasonality
    forecasts = est.predict(steps=24)
    assert len(forecasts) == 24
    assert np.all(np.isfinite(forecasts))


def test_reduce_memory_preserves_predictions():
    """Test that predictions are the same after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Predict before memory reduction
    pred_before = est.predict(steps=10)
    
    # Reduce memory
    est.reduce_memory()
    
    # Predict after memory reduction
    pred_after = est.predict(steps=10)
    
    # Predictions should be identical
    np.testing.assert_array_equal(pred_before, pred_after)
