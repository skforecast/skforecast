# Unit test predict_interval method - Ets
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


def test_predict_interval_values_contain_point_forecast():
    """Test that prediction intervals contain the point forecast"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    # Get point predictions
    pred_point = est.predict(steps=10)
    
    # Get interval predictions
    pred_interval = est.predict_interval(steps=10, level=(80, 95), as_frame=True)
    
    # Point predictions should match interval mean
    np.testing.assert_allclose(pred_point, pred_interval['mean'].values, rtol=1e-10)
    
    # All intervals should contain the mean
    assert np.all(pred_interval['lower_80'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_80'])
    assert np.all(pred_interval['lower_95'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_95'])
    
    # 95% intervals should be wider than 80% intervals
    width_80 = pred_interval['upper_80'] - pred_interval['lower_80']
    width_95 = pred_interval['upper_95'] - pred_interval['lower_95']
    assert np.all(width_95 > width_80)
