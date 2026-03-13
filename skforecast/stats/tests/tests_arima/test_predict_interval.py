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

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_95 = np.array([-3.11903999, -3.01349618, -2.83352155])
    expected_upper_95 = np.array([-0.10035832,  0.78245404,  1.25680241])

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

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_90 = np.array([-2.87637791, -2.708352  , -2.50471361])
    expected_upper_90 = np.array([-0.3430204 ,  0.47730986,  0.92799446])

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

    expected_mean = np.array([-1.60969915, -1.11552107, -0.78835957])
    expected_lower_95 = np.array([-3.11903999, -3.01349618, -2.83352155])
    expected_upper_95 = np.array([-0.10035832,  0.78245404,  1.25680241])

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
    
    expected_mean = np.array([-1.60969915, -1.11552107])
    expected_lower_50 = np.array([-2.12911427, -1.76867836])
    expected_upper_50 = np.array([-1.09028404, -0.46236378])
    expected_lower_99 = np.array([-3.59330925, -3.60988319])
    expected_upper_99 = np.array([0.37391094, 1.37884105])

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
    
    expected_mean = np.array([-0.69037816, -0.28696593, -0.09660924])
    expected_lower_95 = np.array([-2.45209152, -2.40343258, -2.35255749])
    expected_upper_95 = np.array([1.07133521, 1.82950073, 2.15933901])

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
    expected_mean_first = np.array([2.60268909, 2.51255651, 2.4209728 ])
    expected_lower_95_first = np.array([ 0.42550965, -0.48012104, -1.14357126])
    expected_upper_95_first = np.array([4.77986853, 5.50523406, 5.98551686])

    expected_mean_last = np.array([1.5023065 , 1.47665117, 1.45242347])
    expected_lower_95_last = np.array([-4.77643708, -4.82993643, -4.87882661])
    expected_upper_95_last = np.array([7.78105008, 7.78323877, 7.78367355])

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
        -10.38304468, -10.38305569, -10.38305562, -10.38305562,
        -10.38305562
    ])
    expected_lower_95 = np.array([
        -12.17202138, -12.90439972, -13.46756728, -13.94272507,
        -14.36153284
    ])
    expected_upper_95 = np.array([
        -8.59406799, -7.86171166, -7.29854395, -6.82338616, -6.40457839
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
        level=(95, 99),
    )

    expected = {
        'Linux':
            pd.DataFrame({
                'mean': np.array([1574719.88796327, 1449374.80320486, 1509201.84849718,
                                  1484751.10902458, 1403989.16888583]),
                'lower_95': np.array([1540585.1426525 , 1415232.19910491, 1472732.76680002,
                                      1447317.00684114, 1365235.48780817]),
                'upper_95': np.array([1608854.63327405, 1483517.40730482, 1545670.93019435,
                                      1522185.21120802, 1442742.84996349]),
                'lower_99': np.array([1529859.22831125, 1404503.81535314, 1461273.35080536,
                                      1435554.35943289, 1353058.19866004]),
                'upper_99': np.array([1619580.54761529, 1494245.79105659, 1557130.34618901,
                                      1533947.85861628, 1454920.13911162])
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
                'mean': np.array([1574725.01852883, 1449374.78703   , 1509207.29257651,
                                  1484746.32126999, 1403996.4394836 ]),
                'lower_95': np.array([1540590.50961915, 1415232.25129223, 1472739.44949257,
                                      1447312.6000913 , 1365243.11336475]),
                'upper_95': np.array([1608859.52743851, 1483517.32276778, 1545675.13566045,
                                      1522180.04244868, 1442749.76560244]),
                'lower_99': np.array([1529864.66956051, 1404503.8890214 , 1461280.4226984 ,
                                      1435550.07240342, 1353065.93575277]),
                'upper_99': np.array([1619585.36749716, 1494245.68503861, 1557134.16245463,
                                      1533942.57013657, 1454926.94321442])
            }, index=[1, 2, 3, 4, 5]).rename_axis('step')
    }
    
    pd.testing.assert_frame_equal(pred, expected[platform.system()], rtol=1e-4)
    

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
        -0.18187, 0.20608757, -0.02945256, -0.22661237, -0.0109613
    ])
    expected_lower_95_df = np.array([
        -1.65126352, -1.50498276, -1.81869782, -2.04287589, -1.8367468
    ])
    expected_upper_95_df = np.array([
        1.28752353, 1.91715791, 1.7597927 , 1.58965114, 1.81482419
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
        -0.00370962, 0.0191783, 0.09982658, -0.08205884, -0.11285304
    ])
    expected_lower_95_series = np.array([
        -1.49667978, -1.72029709, -1.71968492, -1.92934068, -1.96996154
    ])
    expected_upper_95_series = np.array([
        1.48926054, 1.75865368, 1.91933809, 1.76522299, 1.74425545
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
        -1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809
    ])
    expected_lower_90 = np.array([
        -2.87637791, -2.708352  , -2.50471361, -2.33954242, -2.21822331
    ])
    expected_upper_90 = np.array([
        -0.3430204 ,  0.47730986,  0.92799446,  1.19600576,  1.36147714
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
        -1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809
    ])
    expected_lower_95 = np.array([
        -3.11903999, -3.01349618, -2.83352155, -2.67820108, -2.56111021
    ])
    expected_upper_95 = np.array([
        -0.10035832,  0.78245404,  1.25680241,  1.53466442,  1.70435404
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

    expected = {
        'Linux':
            pd.DataFrame({
                'mean': np.array([451.34858312, 427.10478883, 463.38985401,
                                  499.70660932, 514.03811796]),
                'lower_95': np.array([428.96315079, 400.56937873, 433.27094968,
                                      466.38741355, 477.80016527]),
                'upper_95': np.array([473.73401545, 453.64019893, 493.50875834,
                                      533.02580509, 550.27607065]),
                'lower_99': np.array([421.92913816, 392.23134857, 423.80690403,
                                      455.91776345, 466.41337527]),
                'upper_99': np.array([480.76802809, 461.97822909, 502.97280399,
                                      543.49545519, 561.66286065])
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
                'mean': np.array([451.34628823, 427.10183452, 463.38381393, 499.7030797 ,
                                  514.0333688 ]),
                'lower_95': np.array([428.96077818, 400.56547521, 433.26329483, 466.38172054,
                                      477.79277993]),
                'upper_95': np.array([473.73179828, 453.63819383, 493.50433303, 533.02443886,
                                      550.27395767]),
                'lower_99': np.array([421.92674113, 392.22714679, 423.79874178, 455.91139065,
                                      466.40516159]),
                'upper_99': np.array([480.76583533, 461.97652225, 502.96888608, 543.49476874,
                                      561.66157602])
            }, index=[1, 2, 3, 4, 5]).rename_axis('step')
    }
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (0, 1, 1)
    assert model.best_params_['seasonal_order'] == (2, 1, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(0,1,1)(2,1,0)[12]"
    pd.testing.assert_frame_equal(pred, expected[platform.system()], rtol=1e-4)


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
                'mean': np.array([174.22831851, 174.13324908, 174.86422913, 
                                    174.85907826, 174.81533986]),
                'lower_95': np.array([153.13683798, 153.03928683, 153.71540634,
                                    153.65260745, 153.55657799]),
                'upper_95': np.array([195.31979904, 195.22721133, 196.01305192,
                                        196.06554908, 196.07410173]),
                'lower_99': np.array([146.50941453, 146.41108393, 147.06996441,
                                        146.98905099, 146.87659144]),
                'upper_99': np.array([201.9472220748722, 201.85541414676786, 202.65849422127053,
                                        202.72910590891777, 202.75408890610674])
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
