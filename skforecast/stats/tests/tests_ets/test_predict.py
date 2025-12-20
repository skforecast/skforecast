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
    
    # Check exact predicted values
    expected_mean = np.array([
        -0.21174572, -0.2703575, -0.32896928, -0.38758106, -0.44619284,
        -0.50480462, -0.5634164, -0.62202818
    ])
    np.testing.assert_array_almost_equal(mean, expected_mean, decimal=8)


def test_estimator_invalid_steps():
    """Test Ets estimator with invalid steps parameter"""
    y = ar1_series(50)
    est = Ets(m=1, model="ANN")
    est.fit(y)

    with pytest.raises(ValueError, match="`steps` must be a positive integer"):
        est.predict(steps=0)

    with pytest.raises(ValueError, match="`steps` must be a positive integer"):
        est.predict(steps=-2)

    with pytest.raises(ValueError, match="`steps` must be a positive integer"):
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
    
    # Check exact forecast values
    expected_first_12 = np.array([
        21.79508244, 23.43787498, 24.89522337, 25.42741021, 24.90993587,
        24.00846331, 22.78025097, 21.16618097, 20.23863544, 19.89626461,
        20.49691337, 21.6713034
    ])
    expected_last_12 = np.array([
        23.12161355, 24.76440609, 26.22175448, 26.75394132, 26.23646698,
        25.33499441, 24.10678208, 22.49271208, 21.56516655, 21.22279572,
        21.82344448, 22.99783451
    ])
    np.testing.assert_array_almost_equal(forecasts[:12], expected_first_12, decimal=6)
    np.testing.assert_array_almost_equal(forecasts[12:24], expected_last_12, decimal=6)


def test_reduce_memory_preserves_predictions():
    """Test that predictions are the same after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Predict before memory reduction
    pred_before = est.predict(steps=10)
    
    # Check exact prediction values
    expected_predictions = np.array([
        0.39943144, 0.39595466, 0.39247788, 0.3890011, 0.38552432,
        0.38204754, 0.37857076, 0.37509398, 0.3716172, 0.36814041
    ])
    np.testing.assert_array_almost_equal(pred_before, expected_predictions, decimal=8)
    
    # Reduce memory
    est.reduce_memory()
    
    # Predict after memory reduction
    pred_after = est.predict(steps=10)
    
    # Predictions should be identical
    np.testing.assert_array_equal(pred_before, pred_after)
