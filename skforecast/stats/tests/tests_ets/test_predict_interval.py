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
    df = est.predict_interval(steps=5, level=(0.8, 0.95), as_frame=True)
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "lower_0.8" in df.columns
    assert "upper_0.8" in df.columns
    assert "lower_0.95" in df.columns
    assert "upper_0.95" in df.columns
    assert len(df) == 5
    
    expected_mean = np.array([0.1647033144, 0.1662691275, 0.1678349406, 0.1694007537, 0.1709665668])
    expected_lower_80 = np.array([-1.1934638323, -1.6591995525, -2.027702611, -2.3423248984, -2.621429854])
    expected_upper_80 = np.array([1.5228704611, 1.9917378075, 2.3633724922, 2.6811264058, 2.9633629877])
    expected_lower_95 = np.array([-1.91243409, -2.6255442992, -3.1899499847, -3.6719521418, -4.0996352276])
    expected_upper_95 = np.array([2.2418407188, 2.9580825543, 3.5256198659, 4.0107536493, 4.4415683612])

    tol = {'rtol': 1e-5, 'atol': 1e-6}
    np.testing.assert_allclose(df['mean'].values, expected_mean, **tol)
    np.testing.assert_allclose(df['lower_0.8'].values, expected_lower_80, **tol)
    np.testing.assert_allclose(df['upper_0.8'].values, expected_upper_80, **tol)
    np.testing.assert_allclose(df['lower_0.95'].values, expected_lower_95, **tol)
    np.testing.assert_allclose(df['upper_0.95'].values, expected_upper_95, **tol)


def test_predict_interval_values_contain_point_forecast():
    """Test that prediction intervals contain the point forecast"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    pred_point = est.predict(steps=10)
    pred_interval = est.predict_interval(steps=10, level=(0.8, 0.95), as_frame=True)
    
    np.testing.assert_allclose(pred_point, pred_interval['mean'].values, rtol=1e-10)
    assert np.all(pred_interval['lower_0.8'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_0.8'])
    assert np.all(pred_interval['lower_0.95'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_0.95'])
    
    # 95% intervals should be wider than 80% intervals
    width_80 = pred_interval['upper_0.8'] - pred_interval['lower_0.8']
    width_95 = pred_interval['upper_0.95'] - pred_interval['lower_0.95']
    assert np.all(width_95 > width_80)
