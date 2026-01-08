# Unit test predict_interval method - Arima
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from ..._arima import Arima


def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_predict_interval_raises_error_for_unfitted_model():
    """
    Test that predict_interval raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.predict_interval(steps=1)


def test_predict_interval_raises_error_for_invalid_steps():
    """
    Test that predict_interval raises ValueError for invalid steps parameter.
    """
    y = ar1_series(50)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        model.predict_interval(steps=0)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        model.predict_interval(steps=-1)


def test_predict_interval_level_and_alpha_cannot_both_be_specified():
    """
    Test that specifying both level and alpha raises error.
    """
    y = ar1_series(50)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    msg = "Cannot specify both `level` and `alpha`. Use one or the other."
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, level=(80, 95), alpha=0.05)


def test_predict_interval_alpha_validation():
    """
    Test that alpha parameter is validated correctly.
    """
    y = ar1_series(50)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    msg = "`alpha` must be between 0 and 1."
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, alpha=0)
    
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, alpha=1)
    
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, alpha=1.5)


def test_predict_interval_returns_dataframe_by_default():
    """
    Test that predict_interval returns DataFrame when as_frame=True (default).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 10
    # Default levels are 80 and 95
    assert 'mean' in result.columns
    assert 'lower_80' in result.columns
    assert 'upper_80' in result.columns
    assert 'lower_95' in result.columns
    assert 'upper_95' in result.columns
    
    # Check exact values for first 3 steps
    expected_mean = np.array([-1.610352, -1.107263, -0.775395])
    expected_lower_95 = np.array([-3.506176, -3.150786, -2.879955])
    expected_upper_95 = np.array([0.285473, 0.93626, 1.329165])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=5)


def test_predict_interval_returns_array_when_as_frame_false():
    """
    Test that predict_interval returns ndarray when as_frame=False.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Compare to DataFrame output for consistency
    df = model.predict_interval(steps=10)
    result = model.predict_interval(steps=10, as_frame=False)
    
    assert isinstance(result, np.ndarray)
    # columns: mean, lower_80, upper_80, lower_95, upper_95
    assert result.shape == (10, 5)
    np.testing.assert_array_almost_equal(result[:, 0], df['mean'].values, decimal=12)
    np.testing.assert_array_almost_equal(result[:, 1], df['lower_80'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 2], df['upper_80'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 3], df['lower_95'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 4], df['upper_95'].values, decimal=6)


def test_predict_interval_with_single_level():
    """
    Test predict_interval with a single confidence level.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(90,))
    
    assert 'mean' in result.columns
    assert 'lower_90' in result.columns
    assert 'upper_90' in result.columns
    assert 'lower_80' not in result.columns
    assert 'lower_95' not in result.columns
    
    # Check exact values for first 3 steps
    expected_mean = np.array([-1.610352, -1.107263, -0.775395])
    expected_lower_90 = np.array([-3.201378, -2.822241, -2.541598])
    expected_upper_90 = np.array([-0.019326, 0.607716, 0.990808])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_90'].iloc[:3], expected_lower_90, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_90'].iloc[:3], expected_upper_90, decimal=5)


def test_predict_interval_with_alpha_parameter():
    """
    Test predict_interval with alpha parameter instead of level.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # alpha=0.05 should give 95% interval
    result = model.predict_interval(steps=10, alpha=0.05)
    
    assert 'mean' in result.columns
    assert 'lower_95' in result.columns
    assert 'upper_95' in result.columns
    assert len(result.columns) == 3  # Only mean and one interval
    
    # Check exact values for first 3 steps
    expected_mean = np.array([-1.610352, -1.107263, -0.775395])
    expected_lower_95 = np.array([-3.506176, -3.150786, -2.879955])
    expected_upper_95 = np.array([0.285473, 0.93626, 1.329165])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=5)


def test_predict_interval_with_custom_levels():
    """
    Test predict_interval with custom confidence levels.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(50, 75, 99))
    
    assert 'mean' in result.columns
    assert 'lower_50' in result.columns
    assert 'upper_50' in result.columns
    assert 'lower_75' in result.columns
    assert 'upper_75' in result.columns
    assert 'lower_99' in result.columns
    assert 'upper_99' in result.columns
    
    # Check exact values for first 2 steps
    expected_mean = np.array([-1.610352, -1.107263])
    expected_lower_50 = np.array([-2.262769, -1.810508])
    expected_upper_50 = np.array([-0.957934, -0.404017])
    expected_lower_99 = np.array([-4.101887, -3.792907])
    expected_upper_99 = np.array([0.881184, 1.578382])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:2], expected_mean, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_50'].iloc[:2], expected_lower_50, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_50'].iloc[:2], expected_upper_50, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_99'].iloc[:2], expected_lower_99, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_99'].iloc[:2], expected_upper_99, decimal=5)


def test_predict_interval_bounds_are_symmetric():
    """
    Test that prediction intervals are symmetric around the mean.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(95,))
    
    lower_distance = result['mean'] - result['lower_95']
    upper_distance = result['upper_95'] - result['mean']
    
    np.testing.assert_array_almost_equal(lower_distance, upper_distance, decimal=10)


def test_predict_interval_wider_for_higher_confidence():
    """
    Test that intervals get wider for higher confidence levels.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(80, 95, 99))
    
    # 99% interval should be wider than 95%, which should be wider than 80%
    width_80 = result['upper_80'] - result['lower_80']
    width_95 = result['upper_95'] - result['lower_95']
    width_99 = result['upper_99'] - result['lower_99']
    
    assert np.all(width_80 < width_95)
    assert np.all(width_95 < width_99)


def test_predict_interval_widens_with_horizon():
    """
    Test that prediction intervals generally widen as forecast horizon increases.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=20, level=(95,))
    
    width = result['upper_95'] - result['lower_95']
    
    # Width should generally increase (allowing for small variations)
    assert width.iloc[-1] >= width.iloc[0]


def test_predict_interval_with_exog():
    """
    Test predict_interval with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    exog_pred = np.random.randn(10, 2)
    result = model.predict_interval(steps=10, exog=exog_pred)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 10
    assert 'mean' in result.columns
    
    # Check exact values for first 3 steps
    expected_mean = np.array([-0.708626, -0.284829, -0.093369])
    expected_lower_95 = np.array([-2.814754, -2.524217, -2.387988])
    expected_upper_95 = np.array([1.397502, 1.954559, 2.20125])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=5)


def test_predict_interval_index_starts_at_one():
    """
    Test that DataFrame index starts at 1 (not 0).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=10)
    
    assert result.index[0] == 1
    assert result.index[-1] == 10
    assert result.index.name == "step"


def test_predict_interval_all_values_finite():
    """
    Test that all returned values are finite (not NaN or inf).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.predict_interval(steps=20)
    
    assert np.all(np.isfinite(result.values))


def test_predict_interval_seasonal_model():
    """
    Test predict_interval for seasonal ARIMA model.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    result = model.predict_interval(steps=24, level=(95,))
    
    assert result.shape[0] == 24
    assert np.all(np.isfinite(result.values))
    
    # Check exact values for first and last steps
    expected_mean_first = np.array([2.682367, 2.670684, 2.651222])
    expected_lower_95_first = np.array([-0.327364, -0.918436, -1.386137])
    expected_upper_95_first = np.array([5.692097, 6.259803, 6.68858])
    
    expected_mean_last = np.array([2.555421, 2.552322, 2.54934])
    expected_lower_95_last = np.array([-3.882809, -3.913229, -3.940557])
    expected_upper_95_last = np.array([8.99365, 9.017874, 9.039238])
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean_first, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95_first, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95_first, decimal=5)
    
    np.testing.assert_array_almost_equal(result['mean'].iloc[-3:], expected_mean_last, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[-3:], expected_lower_95_last, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[-3:], expected_upper_95_last, decimal=5)
