# Unit test predict_interval method - Arar
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_estimator_predict_intervals():
    """
    Test basic predict_interval functionality.
    """
    y = ar1_series(120)
    est = Arar()
    est.fit(y)

    df = est.predict_interval(steps=5, level=(50, 80, 95), as_frame=True)
    assert list(df.columns) == ["mean", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"]
    assert df.shape == (5, 1 + 2 * 3)

    raw = est.predict_interval(steps=3, level=(90,), as_frame=False)
    assert raw["mean"].shape == (3,)
    assert raw["upper"].shape == (3, 1)
    assert raw["lower"].shape == (3, 1)
    assert raw["level"] == [90]


def test_arar_predict_interval_with_exog():
    """
    Test Arar predict_interval with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4*np.pi, n))
    ])
    y = y + 0.5 * exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.column_stack([
        np.random.randn(5),
        np.sin(np.linspace(4*np.pi, 4*np.pi + 0.2*np.pi, 5))
    ])
    
    pred_intervals = model.predict_interval(steps=5, exog=exog_future, level=(80, 95))
    
    assert isinstance(pred_intervals, pd.DataFrame)
    assert pred_intervals.shape == (5, 5)  # mean + 2 lower + 2 upper
    assert 'mean' in pred_intervals.columns
    assert 'lower_80' in pred_intervals.columns
    assert 'upper_80' in pred_intervals.columns
    assert 'lower_95' in pred_intervals.columns
    assert 'upper_95' in pred_intervals.columns


def test_arar_predict_interval_without_exog_raises_error_when_fitted_with_exog():
    """
    Test that predict_interval raises error when exog is missing but model was fitted with exog.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog)
    
    with pytest.raises(ValueError, match="Model was fitted with exog"):
        model.predict_interval(steps=5)


def test_arar_predict_interval_with_exog_raises_error_when_fitted_without_exog():
    """
    Test that predict_interval raises error when exog is provided but model was fitted without exog.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    
    model = Arar()
    model.fit(y)
    
    exog_future = np.random.randn(5, 2)
    
    with pytest.raises(ValueError, match="Model was fitted without exog"):
        model.predict_interval(steps=5, exog=exog_future)


def test_arar_exog_interval_feature_count_mismatch():
    """
    Test that predict_interval raises error when exog feature count doesn't match.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future_wrong = np.random.randn(5, 3)  # Wrong number of features
    
    with pytest.raises(ValueError, match="Mismatch in exogenous features"):
        model.predict_interval(steps=5, exog=exog_future_wrong)


def test_arar_exog_interval_length_mismatch():
    """
    Test that predict_interval raises error when exog length doesn't match steps.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future_wrong = np.random.randn(3, 2)  # Wrong length
    
    with pytest.raises(ValueError, match="Length of exog"):
        model.predict_interval(steps=5, exog=exog_future_wrong)


def test_arar_predict_interval_as_frame_false_with_exog():
    """
    Test predict_interval with as_frame=False and exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    result = model.predict_interval(steps=5, exog=exog_future, level=(80, 95), as_frame=False)
    
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'upper' in result
    assert 'lower' in result
    assert 'level' in result
    assert result['mean'].shape == (5,)
    assert result['upper'].shape == (5, 2)
    assert result['lower'].shape == (5, 2)
    assert result['level'] == [80, 95]


def test_arar_exog_prediction_intervals_include_uncertainty():
    """
    Test that prediction intervals with exog properly account for uncertainty.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + 0.5 * exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(10, 2)
    
    intervals = model.predict_interval(steps=10, exog=exog_future, level=(80, 95))
    
    # Check that intervals are wider for 95% than 80%
    width_80 = intervals['upper_80'] - intervals['lower_80']
    width_95 = intervals['upper_95'] - intervals['lower_95']
    
    assert np.all(width_95 > width_80)
    
    # Check that intervals contain the mean
    assert np.all(intervals['lower_80'] < intervals['mean'])
    assert np.all(intervals['mean'] < intervals['upper_80'])
    assert np.all(intervals['lower_95'] < intervals['mean'])
    assert np.all(intervals['mean'] < intervals['upper_95'])
    
    # Check that intervals generally widen over time
    # (not guaranteed but should be true most of the time)
    assert width_95.iloc[-1] >= width_95.iloc[0]


def test_arar_predict_interval_values_contain_point_forecast():
    """
    Test that prediction intervals contain the point forecast and
    have correct width relationships.
    """
    np.random.seed(789)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + 2.0 * exog_train[:, 0] + 3.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(10, 2)
    
    # Get point predictions
    pred_point = model.predict(steps=10, exog=exog_future)
    
    # Get interval predictions
    pred_interval = model.predict_interval(steps=10, exog=exog_future, level=(80, 95))
    
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


def test_arar_predict_interval_deterministic_with_exog():
    """
    Test that prediction intervals are deterministic (same inputs = same outputs).
    """
    np.random.seed(111)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    # Same for intervals
    interval1 = model.predict_interval(steps=5, exog=exog_future, level=(90,))
    interval2 = model.predict_interval(steps=5, exog=exog_future, level=(90,))
    
    pd.testing.assert_frame_equal(interval1, interval2)


def test_arar_predict_interval_exog_affects_mean_not_width():
    """
    Test that changing exog values shifts the intervals but doesn't
    dramatically change their width (since exog is assumed deterministic).
    """
    np.random.seed(333)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 1)
    y = y + 5.0 * exog_train[:, 0]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # Two different exog scenarios
    exog_scenario1 = np.array([[0.0], [0.0], [0.0]])
    exog_scenario2 = np.array([[10.0], [10.0], [10.0]])
    
    interval1 = model.predict_interval(steps=3, exog=exog_scenario1, level=(90,))
    interval2 = model.predict_interval(steps=3, exog=exog_scenario2, level=(90,))
    
    # Means should be very different
    mean_diff = np.abs(interval2['mean'].values - interval1['mean'].values)
    assert np.all(mean_diff > 20.0)  # With coef ~5 and diff of 10, expect ~50
    
    # But widths should be similar (exog doesn't add uncertainty)
    width1 = interval1['upper_90'].values - interval1['lower_90'].values
    width2 = interval2['upper_90'].values - interval2['lower_90'].values
    
    # Widths should be nearly identical
    np.testing.assert_allclose(width1, width2, rtol=0.01)


def test_arar_predict_interval_as_dict_values():
    """
    Test that predict_interval with as_frame=False returns correct
    dict structure with proper values.
    """
    np.random.seed(555)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + exog_train[:, 0] + exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    result = model.predict_interval(
        steps=5, 
        exog=exog_future, 
        level=(80, 95), 
        as_frame=False
    )
    
    # Check structure
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'upper' in result
    assert 'lower' in result
    assert 'level' in result
    
    # Check values
    assert result['mean'].shape == (5,)
    assert result['upper'].shape == (5, 2)
    assert result['lower'].shape == (5, 2)
    assert result['level'] == [80, 95]
    
    # Check value relationships
    for i in range(5):
        # Lower < mean < upper for both levels
        assert result['lower'][i, 0] < result['mean'][i] < result['upper'][i, 0]
        assert result['lower'][i, 1] < result['mean'][i] < result['upper'][i, 1]
        
        # 95% interval wider than 80%
        width_80 = result['upper'][i, 0] - result['lower'][i, 0]
        width_95 = result['upper'][i, 1] - result['lower'][i, 1]
        assert width_95 > width_80
