# Unit test predict_interval method - Arima
# ==============================================================================
import pytest
import platform
import numpy as np
import pandas as pd
from ..._arima import Arima
from .fixtures_arima import air_passengers, multi_seasonal, fuel_consumption


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
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
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

    expected_mean = np.array([-1.6093598390526909, -1.1075455569986326, -0.7762886674209408])
    expected_lower_95 = np.array([-3.503023994440283, -3.148571512838887, -2.87829926709904])
    expected_upper_95 = np.array([0.28430431633490105, 0.9334803988416218, 1.3257219322571583])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=4)


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
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(90,))
    
    assert 'mean' in result.columns
    assert 'lower_90' in result.columns
    assert 'upper_90' in result.columns
    assert 'lower_80' not in result.columns
    assert 'lower_95' not in result.columns

    expected_mean = np.array([-1.6093598390526909, -1.1075455569986326, -0.7762886674209408])
    expected_lower_90 = np.array([-3.1985728954080246, -2.8204285347711915, -2.5403515716596297])
    expected_upper_90 = np.array([-0.020146782697357368, 0.6053374207739266, 0.987774236817748])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_90'].iloc[:3], expected_lower_90, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_90'].iloc[:3], expected_upper_90, decimal=4)


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
    assert 'lower_95' in result.columns
    assert 'upper_95' in result.columns
    assert len(result.columns) == 3  # Only mean and one interval

    expected_mean = np.array([-1.6093598390526909, -1.1075455569986326, -0.7762886674209408])
    expected_lower_95 = np.array([-3.503023994440283, -3.148571512838887, -2.87829926709904])
    expected_upper_95 = np.array([0.28430431633490105, 0.9334803988416218, 1.3257219322571583])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=4)


def test_predict_interval_with_custom_levels():
    """
    Test predict_interval with custom confidence levels.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(50, 75, 99))
    
    assert 'mean' in result.columns
    assert 'lower_50' in result.columns
    assert 'upper_50' in result.columns
    assert 'lower_75' in result.columns
    assert 'upper_75' in result.columns
    assert 'lower_99' in result.columns
    assert 'upper_99' in result.columns
    
    expected_mean = np.array([-1.6093598390526909, -1.1075455569986326])
    expected_lower_50 = np.array([-2.2610335806099293, -1.8099314671261963])
    expected_upper_50 = np.array([-0.9576860974954522, -0.40515964687106876])
    expected_lower_99 = np.array([-4.0980563972561335, -3.7899083492878867])
    expected_upper_99 = np.array([0.8793367191507513, 1.5748172352906216])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:2], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_50'].iloc[:2], expected_lower_50, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_50'].iloc[:2], expected_upper_50, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_99'].iloc[:2], expected_lower_99, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_99'].iloc[:2], expected_upper_99, decimal=4)


def test_predict_interval_bounds_are_symmetric():
    """
    Test that prediction intervals are symmetric around the mean.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
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
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=10, level=(80, 95, 99))
    
    # 99% interval should be wider than 95%, which should be wider than 80%
    width_80 = result['upper_80'] - result['lower_80']
    width_95 = result['upper_95'] - result['lower_95']
    width_99 = result['upper_99'] - result['lower_99']
    
    assert np.all(width_80 < width_95)
    assert np.all(width_95 < width_99)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="ARIMA optimizer converges to different values on macOS"
)
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
    
    expected_mean = np.array([-0.7086257004415824, -0.284828771407133, -0.09336887676023536])
    expected_lower_95 = np.array([-2.81475362569791, -2.524216657859611, -2.38798771222519])
    expected_upper_95 = np.array([1.397502224814745, 1.954559115045345, 2.201249958704719])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95, decimal=4)


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
    
    result = model.predict_interval(steps=24, level=(95,))
    
    assert result.shape[0] == 24
    assert np.all(np.isfinite(result.values))
    
    # Check exact values for first and last steps (R-based implementation)
    expected_mean_first = np.array([2.5914675155579836, 2.4889048143336394, 2.387446263368255])
    expected_lower_95_first = np.array([-0.38728085255965894, -1.0503631477894753, -1.5800038558839398])
    expected_upper_95_first = np.array([5.570215883675626, 6.0281727764567545, 6.354896382620449])

    expected_mean_last = np.array([1.3855544497908163, 1.359369045520463, 1.3347875868963126])
    expected_lower_95_last = np.array([-4.701677560989702, -4.747743231167784, -4.789766090307112])
    expected_upper_95_last = np.array([7.472786460571334, 7.46648132220871, 7.459341264099737])

    np.testing.assert_array_almost_equal(result['mean'].iloc[:3], expected_mean_first, decimal=3)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[:3], expected_lower_95_first, decimal=3)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[:3], expected_upper_95_first, decimal=3)

    np.testing.assert_array_almost_equal(result['mean'].iloc[-3:], expected_mean_last, decimal=3)
    np.testing.assert_array_almost_equal(result['lower_95'].iloc[-3:], expected_lower_95_last, decimal=3)
    np.testing.assert_array_almost_equal(result['upper_95'].iloc[-3:], expected_upper_95_last, decimal=3)


def test_predict_interval_with_differencing():
    """
    Test predict_interval for ARIMA with differencing (d > 0) returns exact values.
    """
    # Create a random walk (needs differencing)
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 1, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=5, level=(95,))
    
    assert result.shape[0] == 5
    assert np.all(np.isfinite(result.values))
    
    # Expected values from skforecast implementation
    expected_mean = np.array([
        -10.38328947357723, -10.38329738436147, -10.383297338422926,
        -10.383297338689694, -10.383297338688145
    ])
    expected_lower_95 = np.array([
        -12.905674243876813, -13.46960987040093, -13.945350587262698,
        -14.364644088332614, -14.743804251212477
    ])
    expected_upper_95 = np.array([
        -7.860904703277647, -7.296984898322008, -6.821244089583153,
        -6.401950589046773, -6.022790426163812
    ])
    
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_95'].values, expected_upper_95, decimal=4)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Arima optimizer converges to different local minima on macOS"
)
def test_predict_interval_fuel_consumption_data_with_exog():
    """
    Test predict_interval works correctly with auto ARIMA on Fuel Consumption dataset
    """

    model = Arima(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1),
                m=12,
                fit_intercept = True,
                enforce_stationarity = True,
                method = "CSS-ML",
                n_cond = None,
                optim_method = "BFGS",
                optim_kwargs = {"maxiter": 2000},
            )
    model.fit(
        y=fuel_consumption.loc[:'1989-09-01', 'y'],
        exog=fuel_consumption.loc[:'1989-09-01'].drop(columns=['y']),
        suppress_warnings=True
    )
    pred = model.predict_interval(
        steps=5,
        exog=fuel_consumption.loc['1989-09-01':].drop(columns=['y']),
        level=(95, 99),
    )

    expected = pd.DataFrame({
        'mean': np.array([1574769.8281793601, 1449368.4439919414, 1509265.1457557934,
                          1484681.4419804534, 1404070.9470290802]),
        'lower_95': np.array([1540695.3715728726, 1412978.6499784153, 1471924.3571476156,
                              1446032.6308096114, 1364297.6520355889]),
        'upper_95': np.array([1608844.2847858476, 1485758.2380054675, 1546605.9343639712,
                              1523330.2531512955, 1443844.2420225716]),
        'lower_99': np.array([1529988.401315492, 1401544.1479796118, 1460191.030989781,
                              1433888.2941744516, 1351799.97697502]),
        'upper_99': np.array([1619551.2550432282, 1497192.740004271, 1558339.2605218058,
                              1535474.5897864553, 1456341.9170831405])
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
    result = model.predict_interval(steps=5, exog=exog_pred_df, level=(95,))
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 5
    assert np.all(np.isfinite(result.values))
    
    # Verify intervals are symmetric
    lower_distance = result['mean'] - result['lower_95']
    upper_distance = result['upper_95'] - result['mean']
    np.testing.assert_array_almost_equal(lower_distance, upper_distance, decimal=10)
    
    # Check exact predicted values for DataFrame exog
    expected_mean_df = np.array([
        -0.2013851394327377, 0.19484881398087814, -0.03919903271683243,
        -0.24332740278243575, -0.02550560210158951
    ])
    expected_lower_95_df = np.array([
        -1.9071715588559717, -1.5896109822633926, -1.8509925234675968,
        -2.0648062455103826, -1.8504391606350956
    ])
    expected_upper_95_df = np.array([
        1.5044012799904962, 1.979308610225149, 1.7725944580339321,
        1.5781514399455112, 1.7994279564319164
    ])
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean_df, decimal=5)
    np.testing.assert_array_almost_equal(result['lower_95'].values, expected_lower_95_df, decimal=5)
    np.testing.assert_array_almost_equal(result['upper_95'].values, expected_upper_95_df, decimal=5)
    
    # Test with Series (1D exog)
    np.random.seed(42)
    y2 = ar1_series(80, seed=42)
    exog_train_1d = pd.Series(np.random.randn(80), name='single_feature')
    
    model2 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model2.fit(y2, exog=exog_train_1d)
    
    exog_pred_series = pd.Series(np.random.randn(5))
    result2 = model2.predict_interval(steps=5, exog=exog_pred_series, level=(95,))
    
    assert result2.shape[0] == 5
    assert np.all(np.isfinite(result2.values))
    
    # Check exact predicted values for Series exog
    expected_mean_series = np.array([
        -0.022408721684020402, 0.00698414790442866, 0.09581769743756172,
        -0.09615035372144291, -0.12792912995651143
    ])
    expected_lower_95_series = np.array([
        -1.7563641540743558, -1.80777749582571, -1.747180975673626,
        -1.9492140108927054, -1.984604615737813
    ])
    expected_upper_95_series = np.array([
        1.7115467107063151, 1.8217457916345672, 1.9388163705487496,
        1.7569133034498194, 1.72874635582479
    ])
    np.testing.assert_array_almost_equal(result2['mean'].values, expected_mean_series, decimal=5)
    np.testing.assert_array_almost_equal(result2['lower_95'].values, expected_lower_95_series, decimal=5)
    np.testing.assert_array_almost_equal(result2['upper_95'].values, expected_upper_95_series, decimal=5)


def test_predict_interval_level_as_single_value():
    """
    Test predict_interval with level as single int or float (not tuple).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Test with single int
    result_int = model.predict_interval(steps=5, level=90)
    assert 'lower_90' in result_int.columns
    assert 'upper_90' in result_int.columns
    assert len([c for c in result_int.columns if 'lower' in c]) == 1
    
    # Test with single float
    result_float = model.predict_interval(steps=5, level=85.0)
    assert 'lower_85' in result_float.columns
    assert 'upper_85' in result_float.columns
    
    # Check exact values for level=90
    expected_mean = np.array([
        -1.6093598390526909, -1.1075455569986326, -0.7762886674209408,
        -0.557619867394906, -0.41327254780923617
    ])
    expected_lower_90 = np.array([
        -3.1985728954080246, -2.8204285347711915, -2.5403515716596297,
        -2.3435260000697435, -2.2086138751658373
    ])
    expected_upper_90 = np.array([
        -0.020146782697357368, 0.6053374207739266, 0.987774236817748,
        1.2282862652799316, 1.3820687795473647
    ])
    
    np.testing.assert_array_almost_equal(result_int['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result_int['lower_90'].values, expected_lower_90, decimal=4)
    np.testing.assert_array_almost_equal(result_int['upper_90'].values, expected_upper_90, decimal=4)


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
    result_before = model.predict_interval(steps=5, level=(95,))
    
    # Call reduce_memory
    model.reduce_memory()
    assert model.is_memory_reduced is True
    
    # predict_interval should still work
    result_after = model.predict_interval(steps=5, level=(95,))
    
    # Results should be identical
    pd.testing.assert_frame_equal(result_before, result_after)
    
    # Check exact expected values
    expected_mean = np.array([
        -1.6093598390526909, -1.1075455569986326, -0.7762886674209408,
        -0.557619867394906, -0.41327254780923617
    ])
    expected_lower_95 = np.array([
        -3.503023994440283, -3.148571512838887, -2.87829926709904,
        -2.685658279116244, -2.552553687430718
    ])
    expected_upper_95 = np.array([
        0.28430431633490105, 0.9334803988416218, 1.3257219322571583,
        1.570418544326432, 1.7260085918122456
    ])
    
    np.testing.assert_array_almost_equal(result_after['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result_after['lower_95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result_after['upper_95'].values, expected_upper_95, decimal=4)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Arima optimizer converges to different local minima on macOS"
)
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
    pred = model.predict_interval(steps=5, level=(95, 99))

    expected = pd.DataFrame({
        'mean': np.array([445.56810006, 420.28762041, 449.04990788,
                          491.65672001, 503.18017215]),
        'lower_95': np.array([419.21246699, 390.75262223, 417.45534898,
                              458.55402781, 468.97127735]),
        'upper_95': np.array([471.92373313, 449.82261858, 480.64446678,
                              524.7594122, 537.38906695]),
        'lower_99': np.array([410.93092688, 381.47205319, 407.52761906,
                              448.15240806, 458.22206355]),
        'upper_99': np.array([480.20527324, 459.10318762, 490.57219671,
                              535.16103195, 548.13828075])
    }, index=[1, 2, 3, 4, 5]).rename_axis('step')

    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 1, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(2,1,1)(0,1,0)[12]"
    pd.testing.assert_frame_equal(pred, expected, rtol=1e-4)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Arima optimizer converges to different local minima on macOS"
)
def test_predict_interval_auto_arima_multi_seasonal_data():
    """
    Test predict_interval works correctly with auto ARIMA on multi-seasonal dataset
    """   

    expected = {
        'Linux':
            pd.DataFrame({
                'mean': np.array([174.22844329118465, 174.1332421021498, 174.864069787337,
                                    174.85893903926353, 174.8152181258244]),
                'lower_95': np.array([153.12242630177764, 153.02473854182287, 153.70067144462485,
                                    153.63785622900835, 153.54180987173586]),
                'upper_95': np.array([195.33446028059166, 195.24174566247675, 196.02746813004916,
                                        196.08002184951872, 196.08862637991294]),
                'lower_99': np.array([146.49043510769576, 146.39196601053024, 147.0506497225653,
                                        146.9697087335747, 146.8572205302209]),
                'upper_99': np.array([201.96645147467353, 201.87451819376938, 202.6774898521087,
                                        202.74816934495237, 202.7732157214279])
            }, index=[1, 2, 3, 4, 5]).rename_axis('step'),
        'Darwin':
            pd.DataFrame({
                'mean': np.array([445.2856573681562, 419.83465616022573, 448.44413493780667,
                                490.92842560605027, 502.3511798022874]),
                'lower_95': np.array([419.01183057292997, 390.374886879868, 416.9267606112706,
                                    457.9051928081597, 468.2289786562149]),
                'upper_95': np.array([471.5594841633824, 449.2944254405835, 479.96150926434274,
                                        523.9516584039409, 536.4733809483598]),
                'lower_99': np.array([410.7559958492004, 381.1179564725308, 407.02328383973776,
                                        447.52854101139263, 457.50700597719066]),
                'upper_99': np.array([479.81531888711197, 458.55135584792066, 489.8649860358756,
                                        534.3283102007078, 547.1953536273841])
            }, index=[1, 2, 3, 4, 5]).rename_axis('step'),
        'Windows':
            pd.DataFrame({
                'mean': np.array([174.22838488, 174.13325775, 174.86414245, 174.85900122,
                                  174.81527385]),
                'lower_95': np.array([153.136897, 153.03928613, 153.71531173, 153.65252434,
                                      153.55650703]),
                'upper_95': np.array([195.31987277, 195.22722936, 196.01297316, 196.06547809,
                                      196.07404066]),
                'lower_99': np.array([146.50947118, 146.41107987, 147.06986749, 146.98896637,
                                      146.87651836]),
                'upper_99': np.array([201.94729859, 201.85543563, 202.6584174, 202.72903607,
                                      202.75402933])
            }, index=[1, 2, 3, 4, 5]).rename_axis('step')
    }
    
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
    pred = model.predict_interval(steps=5, level=(95, 99))
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 0, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(2,1,1)"
    pd.testing.assert_frame_equal(pred, expected[platform.system()], rtol=1e-3)
