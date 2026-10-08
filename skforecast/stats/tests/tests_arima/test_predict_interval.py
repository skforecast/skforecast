# Unit test predict_interval method - Arima
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ..._arima import Arima
from .fixtures_arima import air_passengers, multi_seasonal, fuel_consumption, tol_pred


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
        model.predict_interval(steps=5, level=(0.8, 0.95), alpha=0.05)


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
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 10
    # Default levels are 80 and 95
    assert 'mean' in result.columns
    assert 'lower_0.8' in result.columns
    assert 'upper_0.8' in result.columns
    assert 'lower_0.95' in result.columns
    assert 'upper_0.95' in result.columns

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_95 = np.array([-3.14220258, -3.04262281, -2.86490692])
    expected_upper_95 = np.array([-0.07719573,  0.81158072,  1.28818783])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.95'].iloc[:3], expected_upper_95, decimal=4)


def test_predict_interval_returns_array_when_as_frame_false():
    """
    Test that predict_interval returns ndarray when as_frame=False.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Compare to DataFrame output for consistency
    df = model.predict_interval(steps=10)
    result = model.predict_interval(steps=10, as_frame=False)
    
    assert isinstance(result, np.ndarray)
    # columns: mean, lower_0.8, upper_0.8, lower_0.95, upper_0.95
    assert result.shape == (10, 5)
    np.testing.assert_array_almost_equal(result[:, 0], df['mean'].values, decimal=12)
    np.testing.assert_array_almost_equal(result[:, 1], df['lower_0.8'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 2], df['upper_0.8'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 3], df['lower_0.95'].values, decimal=6)
    np.testing.assert_array_almost_equal(result[:, 4], df['upper_0.95'].values, decimal=6)


def test_predict_interval_with_single_level():
    """
    Test predict_interval with a single confidence level.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(0.9,))
    
    assert 'mean' in result.columns
    assert 'lower_0.9' in result.columns
    assert 'upper_0.9' in result.columns
    assert 'lower_0.8' not in result.columns
    assert 'lower_0.95' not in result.columns

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_90 = np.array([-2.89581658, -2.73279583, -2.53105303])
    expected_upper_90 = np.array([-0.32358174,  0.50175374,  0.95433394])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.9'].iloc[:3], expected_lower_90, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.9'].iloc[:3], expected_upper_90, decimal=4)


def test_predict_interval_with_alpha_parameter():
    """
    Test predict_interval with alpha parameter instead of level.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # alpha=0.05 should give 95% interval
    result = model.predict_interval(steps=10, alpha=0.05)
    
    assert 'mean' in result.columns
    assert 'lower_0.95' in result.columns
    assert 'upper_0.95' in result.columns
    assert len(result.columns) == 3  # Only mean and one interval

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_95 = np.array([-3.14220258, -3.04262281, -2.86490692])
    expected_upper_95 = np.array([-0.07719573,  0.81158072,  1.28818783])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.95'].iloc[:3], expected_upper_95, decimal=4)


def test_predict_interval_with_custom_levels():
    """
    Test predict_interval with custom confidence levels.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(0.5, 0.75, 0.99))
    
    assert 'mean' in result.columns
    assert 'lower_0.5' in result.columns
    assert 'upper_0.5' in result.columns
    assert 'lower_0.75' in result.columns
    assert 'upper_0.75' in result.columns
    assert 'lower_0.99' in result.columns
    assert 'upper_0.99' in result.columns
    
    expected_mean = np.array([-1.60969915, -1.11552107])
    expected_lower_50 = np.array([-2.13708531, -1.7787018])
    expected_upper_50 = np.array([-1.08231301, -0.45234029])
    expected_lower_99 = np.array([-3.62375006, -3.64816207])
    expected_upper_99 = np.array([0.40435174, 1.41711998])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:2], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.5'].iloc[:2], expected_lower_50, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.5'].iloc[:2], expected_upper_50, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.99'].iloc[:2], expected_lower_99, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.99'].iloc[:2], expected_upper_99, decimal=4)


def test_predict_interval_bounds_are_symmetric():
    """
    Test that prediction intervals are symmetric around the mean.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(0.95,))
    
    lower_distance = result['mean'] - result['lower_0.95']
    upper_distance = result['upper_0.95'] - result['mean']
    
    np.testing.assert_array_almost_equal(lower_distance, upper_distance, decimal=10)


def test_predict_interval_wider_for_higher_confidence():
    """
    Test that intervals get wider for higher confidence levels.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(0.8, 0.95, 0.99))
    
    # 99% interval should be wider than 95%, which should be wider than 80%
    width_80 = result['upper_0.8'] - result['lower_0.8']
    width_95 = result['upper_0.95'] - result['lower_0.95']
    width_99 = result['upper_0.99'] - result['lower_0.99']
    
    assert np.all(width_80 < width_95)
    assert np.all(width_95 < width_99)


def test_predict_interval_with_exog():
    """
    Test predict_interval with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    exog_pred = np.random.randn(10, 2)
    result = model.predict_interval(steps=10, exog=exog_pred)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 10
    assert 'mean' in result.columns
    
    expected_mean = np.array([-0.06405277, -0.26701348, -0.02394156])
    expected_lower_95 = np.array([-1.87153262, -2.43846264, -2.33849578])
    expected_upper_95 = np.array([1.74342694, 1.90443526, 2.29061227])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.95'].iloc[:3], expected_upper_95, decimal=4)


def test_predict_interval_index_starts_at_one():
    """
    Test that DataFrame index starts at 1 (not 0).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
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
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
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
    
    result = model.predict_interval(steps=24, level=(0.95,))
    
    assert result.shape[0] == 24
    assert np.all(np.isfinite(result.values))
    
    # Check exact values for first and last steps (R-based implementation)
    expected_mean_first = np.array([2.60268909, 2.51255651, 2.4209728 ])
    expected_lower_95_first = np.array([ 0.39209868, -0.52604646, -1.19827238])
    expected_upper_95_first = np.array([4.81328026, 5.55116097, 6.04022012])

    expected_mean_last = np.array([1.5023065 , 1.47665117, 1.45242347])
    expected_lower_95_last = np.array([-4.87278857, -4.92671517, -4.97598378])
    expected_upper_95_last = np.array([7.8774106, 7.88002671, 7.88084008])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean_first, decimal=3)
    np.testing.assert_array_almost_equal(result['lower_0.95'].iloc[:3], expected_lower_95_first, decimal=3)
    np.testing.assert_array_almost_equal(result['upper_0.95'].iloc[:3], expected_upper_95_first, decimal=3)

    np.testing.assert_array_almost_equal(result['mean'].iloc[-3:], expected_mean_last, decimal=3)
    np.testing.assert_array_almost_equal(result['lower_0.95'].iloc[-3:], expected_lower_95_last, decimal=3)
    np.testing.assert_array_almost_equal(result['upper_0.95'].iloc[-3:], expected_upper_95_last, decimal=3)


def test_predict_interval_with_differencing():
    """
    Test predict_interval for ARIMA with differencing (d > 0) returns exact values.
    """
    # Create a random walk (needs differencing)
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 1, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=5, level=(0.95,))
    
    assert result.shape[0] == 5
    assert np.all(np.isfinite(result.values))
    
    # Expected values from skforecast implementation
    expected_mean = np.array([
        -10.38304468, -10.38305569, -10.38305562, -10.38305562,
        -10.38305562
    ])
    expected_lower_95 = np.array([
        -12.18112564, -12.91723107, -13.48326464, -13.96084055,
        -14.38177968
    ])
    expected_upper_95 = np.array([
        -8.58496372, -7.84888031, -7.28284659, -6.80527068, -6.38433156
    ])
    
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_0.95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_0.95'].values, expected_upper_95, decimal=4)


def test_predict_interval_fuel_consumption_data_with_exog():
    """
    Test predict_interval works correctly with auto ARIMA on Fuel Consumption dataset
    """

    model = Arima(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1),
                m=12,
                fit_intercept=True,
                enforce_stationarity=True,
                method="CSS-ML",
                n_cond=None,
                optim_method="BFGS",
                optim_kwargs={"maxiter": 2000},
            )
    model.fit(
        y=fuel_consumption.loc[:'1989-09-01', 'y'],
        exog=fuel_consumption.loc[:'1989-09-01'].drop(columns=['y']),
        suppress_warnings=True
    )
    pred = model.predict_interval(
        steps=5,
        exog=fuel_consumption.loc['1989-09-01':].drop(columns=['y']),
        level=(0.95, 0.99),
    )

    expected = pd.DataFrame({
        'mean': np.array([702178.55298827, 670303.23645305, 639996.22102265,
                          675010.52440616, 614773.51301366]),
        'lower_0.95': np.array([666828.59029521, 634940.27376599, 602227.57665577,
                                636237.10109855, 574640.4192159]),
        'upper_0.95': np.array([737536.59344436, 705664.60384832, 677769.99833367,
                                713778.63943271, 654915.06165959]),
        'lower_0.99': np.array([655719.55786925, 623828.67619292, 590359.00203085,
                                624054.44248357, 562028.35882426]),
        'upper_0.99': np.array([748645.62587032, 716776.20142139, 689638.5729586,
                                725961.29804769, 667527.12205123])
    }, index=[1, 2, 3, 4, 5]).rename_axis('step')

    pd.testing.assert_frame_equal(pred, expected, rtol=1e-4)
    

def test_predict_interval_with_exog_dataframe():
    """
    Test predict_interval with exogenous variables as pandas DataFrame and Series.
    """
    np.random.seed(42)
    y = ar1_series(80, seed=42)
    
    # Create exog as DataFrame
    exog_train = pd.DataFrame({
        'feature1': np.random.randn(80),
        'feature2': np.random.randn(80)
    })
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    assert model.n_exog_features_in_ == 2
    
    # Predict with DataFrame
    np.random.seed(123)
    exog_pred_df = pd.DataFrame({
        'feature1': np.random.randn(5),
        'feature2': np.random.randn(5)
    })
    result = model.predict_interval(steps=5, exog=exog_pred_df, level=(0.95,))
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 5
    assert np.all(np.isfinite(result.values))
    
    # Verify intervals are symmetric
    lower_distance = result['mean'] - result['lower_0.95']
    upper_distance = result['upper_0.95'] - result['mean']
    np.testing.assert_array_almost_equal(lower_distance, upper_distance, decimal=10)
    
    # Check exact predicted values for DataFrame exog
    expected_mean_df = np.array([
        -0.03453459, 0.47554571, 0.18911951, -0.08577119, 0.18902427
    ])
    expected_lower_95_df = np.array([
        -1.5421005, -1.27997536, -1.64660733, -1.9492182, -1.68419208
    ])
    expected_upper_95_df = np.array([
        1.47303138, 2.23106688, 2.02484647, 1.7776759, 2.06224071
    ])
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean_df, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_0.95'].values, expected_lower_95_df, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_0.95'].values, expected_upper_95_df, decimal=5)
    
    # Test with Series (1D exog)
    np.random.seed(42)
    y2 = ar1_series(80, seed=42)
    exog_train_1d = pd.Series(np.random.randn(80), name='single_feature')
    
    model2 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model2.fit(y2, exog=exog_train_1d)
    
    exog_pred_series = pd.Series(np.random.randn(5))
    result2 = model2.predict_interval(steps=5, exog=exog_pred_series, level=(0.95,))
    
    assert result2.shape[0] == 5
    assert np.all(np.isfinite(result2.values))
    
    # Check exact predicted values for Series exog
    expected_mean_series = np.array([
        0.15081644, 0.14658572, 0.17453836, 0.08650628, 0.06935746
    ])
    expected_lower_95_series = np.array([
        -1.37095963, -1.6264518, -1.68007956, -1.79641783, -1.82358292
    ])
    expected_upper_95_series = np.array([
        1.67259257, 1.91962332, 2.02915639, 1.96943045, 1.9622979
    ])
    np.testing.assert_array_almost_equal(result2['mean'].values, expected_mean_series, decimal=5)
    np.testing.assert_array_almost_equal(result2['lower_0.95'].values, expected_lower_95_series, decimal=5)
    np.testing.assert_array_almost_equal(result2['upper_0.95'].values, expected_upper_95_series, decimal=5)


def test_predict_interval_level_as_single_value():
    """
    Test predict_interval with level as a single float (not tuple).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)

    result_85 = model.predict_interval(steps=5, level=0.85)
    assert 'lower_0.85' in result_85.columns
    assert 'upper_0.85' in result_85.columns
    assert len([c for c in result_85.columns if 'lower' in c]) == 1

    # Check exact values for level=0.9
    result_90 = model.predict_interval(steps=5, level=0.9)
    expected_mean = np.array([
        -1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809
    ])
    expected_lower_90 = np.array([
        -2.89581658, -2.73279583, -2.53105303, -2.36667095, -2.24569056
    ])
    expected_upper_90 = np.array([
        -0.32358174,  0.50175374,  0.95433394,  1.22313432,  1.38893438
    ])
    
    np.testing.assert_array_almost_equal(result_90['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result_90['lower_0.9'].values, expected_lower_90, decimal=4)
    np.testing.assert_array_almost_equal(result_90['upper_0.9'].values, expected_upper_90, decimal=4)


def test_predict_interval_exog_errors():
    """
    Test predict_interval raises appropriate errors for exog issues:
    - exog not provided when model was fitted with exog
    - exog with wrong number of features
    - exog with wrong length
    - exog with wrong dimensions (3D)
    """
    import re
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    # Test: exog not provided when needed
    msg = (
        "Model was fitted with 2 exogenous features, "
        "but `exog` was not provided for prediction."
    )
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5)
    
    # Test: exog with wrong number of features
    exog_wrong_features = np.random.randn(5, 3)
    msg = (
        "Number of exogenous features \\(3\\) does not match "
        "the number used during fitting \\(2\\)."
    )
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_wrong_features)
    
    # Test: exog with wrong length
    exog_wrong_length = np.random.randn(3, 2)
    msg = re.escape("Length of `exog` (3) must match `steps` (5).")
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_wrong_length)
    
    # Test: exog with 3D array
    exog_3d = np.random.randn(5, 2, 1)
    msg = "`exog` must be 1- or 2-dimensional."
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_3d)


def test_predict_interval_after_reduce_memory():
    """
    Test that predict_interval still works after reduce_memory() is called.
    reduce_memory removes fitted_values_ and in_sample_residuals_ but
    predict_interval should still work as it uses the model_ object.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Get prediction before reduce_memory
    result_before = model.predict_interval(steps=5, level=(0.95,))
    
    # Call reduce_memory
    model.reduce_memory()
    assert model.is_memory_reduced is True
    
    # predict_interval should still work
    result_after = model.predict_interval(steps=5, level=(0.95,))
    
    # Results should be identical
    pd.testing.assert_frame_equal(result_before, result_after)
    
    # Check exact expected values
    expected_mean = np.array([
        -1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809
    ])
    expected_lower_95 = np.array([
        -3.14220258, -3.04262281, -2.86490692, -2.71052672, -2.59383946
    ])
    expected_upper_95 = np.array([
        -0.07719573,  0.81158072,  1.28818783,  1.56699009,  1.73708328
    ])
    
    np.testing.assert_array_almost_equal(result_after['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result_after['lower_0.95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result_after['upper_0.95'].values, expected_upper_95, decimal=4)


def test_predict_interval_auto_arima_air_passengers_data():
    """
    Test predict_interval works correctly with auto ARIMA on Air Passengers dataset
    """

    model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        max_order=5,
        max_d=2,
        max_D=1,
        ic="aic",
        seasonal=True,
        test="kpss",
        nmodels=94,
        optim_method="BFGS",
        approximation=False,
        optim_kwargs={
            'maxiter': 5000,
            'gtol': 1e-6,
            'ftol': 1e-9
        },
        m=12,
        trace=False,
        stepwise=True,
    )
    model.fit(air_passengers, suppress_warnings=True)
    pred = model.predict_interval(steps=5, level=(0.95, 0.99))

    expected = pd.DataFrame({
        'mean': np.array([451.34858312, 427.10478883, 463.38985401,
                          499.70660932, 514.03811796]),
        'lower_0.95': np.array([428.69996732, 400.25630331, 432.91236432,
                                465.99349743, 477.37054524]),
        'upper_0.95': np.array([473.99260914, 453.94736573, 493.85526354,
                                533.41266196, 550.69619237]),
        'lower_0.99': np.array([421.58397756, 391.82082604, 423.33754091,
                                455.401179, 465.85025113]),
        'upper_0.99': np.array([481.1085989, 462.382843, 503.43008694,
                                544.00498039, 562.21648647])
    }, index=[1, 2, 3, 4, 5]).rename_axis('step')
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (0, 1, 1)
    assert model.best_params_['seasonal_order'] == (2, 1, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(0,1,1)(2,1,0)[12]"
    pd.testing.assert_frame_equal(pred, expected, rtol=tol_pred['rtol'])


def test_predict_interval_auto_arima_multi_seasonal_data():
    """
    Test predict_interval works correctly with auto ARIMA on multi-seasonal dataset
    """   

    expected = pd.DataFrame({
        'mean': np.array([174.22831851, 174.13324908, 174.86422913, 
                          174.85907826, 174.81533986]),
        'lower_0.95': np.array([153.13683798, 153.03928683, 153.71540634,
                                153.65260745, 153.55657799]),
        'upper_0.95': np.array([195.31979904, 195.22721133, 196.01305192,
                                196.06554908, 196.07410173]),
        'lower_0.99': np.array([146.50941453, 146.41108393, 147.06996441,
                                146.98905099, 146.87659144]),
        'upper_0.99': np.array([201.9472220748722, 201.85541414676786, 202.65849422127053,
                                202.72910590891777, 202.75408890610674])
    }, index=[1, 2, 3, 4, 5]).rename_axis('step')
    
    model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        max_order=5,
        max_d=2,
        max_D=1,
        ic="aic",
        seasonal=True,
        test="kpss",
        nmodels=94,
        optim_method="BFGS",
        approximation=False,
        optim_kwargs={
            'maxiter': 5000,
            'gtol': 1e-6,
            'ftol': 1e-9
        },
        m=12,
        trace=False,
        stepwise=True,
    )
    model.fit(multi_seasonal, suppress_warnings=True)
    pred = model.predict_interval(steps=5, level=(0.95, 0.99))
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 0, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(2,1,1)"
    pd.testing.assert_frame_equal(pred, expected, rtol=tol_pred['rtol'])


def test_predict_interval_output_when_css_estimates_are_non_stationary():
    """
    Test that predict_interval returns finite values when the CSS estimates of
    the AR part are not stationary, and that fit warns about it. The
    predictions, the intervals and the fitted values used to be NaN.
    """
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12, method="CSS")

    warn_msg = "CSS estimation produced non-stationary AR parameters"
    with pytest.warns(UserWarning, match=warn_msg):
        model.fit(air_passengers)
    pred = model.predict_interval(steps=3, level=0.95)

    expected = pd.DataFrame(
        {
            'mean': [448.6295512687, 423.9093188721, 455.4486508233],
            'lower_0.95': [426.3090023038, 396.6478326811, 426.0606817600],
            'upper_0.95': [470.9501002336, 451.1708050630, 484.8366198866],
        },
        index=pd.RangeIndex(start=1, stop=4, name='step')
    )

    assert int(np.isnan(model.fitted_values_).sum()) == 13
    pd.testing.assert_frame_equal(pred, expected, rtol=tol_pred['rtol'])

