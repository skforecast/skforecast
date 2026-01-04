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
    
    expected_mean = np.array([-0.21174572, -0.2703575, -0.32896928, -0.38758106, -0.44619284])
    expected_lower_80 = np.array([-1.93400505, -2.00300517, -2.07389933, -2.14681648, -2.22187478])
    expected_upper_80 = np.array([1.5105136, 1.46229016, 1.41596077, 1.37165436, 1.3294891])
    expected_lower_95 = np.array([-2.84571406, -2.92021344, -2.9976095, -3.07809946, -3.16186402])
    expected_upper_95 = np.array([2.42222262, 2.37949843, 2.33967094, 2.30293734, 2.26947834])
    
    np.testing.assert_array_almost_equal(df['mean'].values, expected_mean, decimal=8)
    np.testing.assert_array_almost_equal(df['lower_80'].values, expected_lower_80, decimal=6)
    np.testing.assert_array_almost_equal(df['upper_80'].values, expected_upper_80, decimal=6)
    np.testing.assert_array_almost_equal(df['lower_95'].values, expected_lower_95, decimal=6)
    np.testing.assert_array_almost_equal(df['upper_95'].values, expected_upper_95, decimal=6)


def test_predict_interval_values_contain_point_forecast():
    """Test that prediction intervals contain the point forecast"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    pred_point = est.predict(steps=10)
    pred_interval = est.predict_interval(steps=10, level=(80, 95), as_frame=True)
    
    np.testing.assert_allclose(pred_point, pred_interval['mean'].values, rtol=1e-10)
    assert np.all(pred_interval['lower_80'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_80'])
    assert np.all(pred_interval['lower_95'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_95'])
    
    # 95% intervals should be wider than 80% intervals
    width_80 = pred_interval['upper_80'] - pred_interval['lower_80']
    width_95 = pred_interval['upper_95'] - pred_interval['lower_95']
    assert np.all(width_95 > width_80)
