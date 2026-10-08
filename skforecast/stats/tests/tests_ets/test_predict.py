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
    
    expected_mean = np.array([
        0.1647033144, 0.1662691275, 0.1678349406, 0.1694007537, 0.1709665668,
        0.1725323799, 0.174098193, 0.1756640061
    ])
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5, atol=1e-6)


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

    forecasts = est.predict(steps=24)
    assert len(forecasts) == 24
    assert np.all(np.isfinite(forecasts))
    
    expected = np.array([
        21.7119669666, 23.3449832475, 24.7929372905, 25.3181161357, 24.7941342012,
        23.8828471306, 22.6341107929, 21.0085260619, 20.0684274667, 19.7130601543,
        20.30497262, 21.4707654929,
        22.893467927, 24.5264842079, 25.9744382509, 26.4996170961, 25.9756351616,
        25.064348091, 23.8156117533, 22.1900270223, 21.249928427, 20.8945611146,
        21.4864735804, 22.6522664533
    ])
    np.testing.assert_allclose(forecasts, expected, rtol=1e-5, atol=1e-6)


def test_reduce_memory_preserves_predictions():
    """Test that predictions are the same after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    pred_before = est.predict(steps=10)
    
    expected_predictions = np.array([
        0.5896991484, 0.5958411095, 0.6019830706, 0.6081250317, 0.6142669927,
        0.6204089538, 0.6265509149, 0.632692876, 0.6388348371, 0.6449767981
    ])
    np.testing.assert_allclose(pred_before, expected_predictions, rtol=1e-5, atol=1e-6)
    
    est.reduce_memory()
    
    pred_after = est.predict(steps=10)
    
    np.testing.assert_array_equal(pred_before, pred_after)


def test_estimator_ann_single_step():
    """Test ANN model (no trend) single-step prediction"""
    y = ar1_series(80)
    est = Ets(m=1, model='ANN')
    est.fit(y)
    
    pred = est.predict(steps=1)
    assert pred.shape == (1,)
    
    expected = np.array([-0.1165068109])
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)


def test_estimator_ann_no_trend():
    """Test ANN model (no trend) predictions remain constant"""
    y = ar1_series(80)
    est = Ets(m=1, model='ANN')
    est.fit(y)
    
    pred = est.predict(steps=10)
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
    
    expected = np.array([
        -0.1165068109, -0.1165068109, -0.1165068109, -0.1165068109, -0.1165068109,
        -0.1165068109, -0.1165068109, -0.1165068109, -0.1165068109, -0.1165068109
    ])
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)


def test_estimator_damped_trend():
    """Test damped trend model predictions"""
    rng = np.random.default_rng(789)
    y = 20 + 2 * np.arange(60) * np.exp(-0.05 * np.arange(60)) + rng.normal(0, 0.5, 60)
    
    est = Ets(m=1, model='AAN', damped=True)
    est.fit(y)
    
    pred = est.predict(steps=12)
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))
    
    expected = np.array([
        26.0228921872, 25.8828448005, 25.760258682, 25.6529567408, 25.5590333283,
        25.4768203941, 25.4048578625, 25.3418677016, 25.2867312259, 25.238469229,
        25.196224593, 25.1592470665
    ])
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)
    
    # Verify trend is decreasing (damped)
    assert np.all(np.diff(pred) < 0)


def test_estimator_multiplicative_error():
    """Test MAN model (multiplicative error, additive trend) predictions"""
    rng = np.random.default_rng(456)
    y = 5 + 0.3 * np.arange(50) + rng.normal(0, 0.3, 50)
    # Ensure positive values for multiplicative error
    y = np.abs(y) + 1.0
    
    est = Ets(m=1, model='MAN')
    est.fit(y)
    
    # Forecast with multiplicative error
    pred = est.predict(steps=8)
    assert pred.shape == (8,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)  # Must be positive for multiplicative model
    
    expected = np.array([
        21.0777933234, 21.3790225557, 21.6802517881, 21.9814810204, 22.2827102527,
        22.583939485, 22.8851687173, 23.1863979497
    ])
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)
