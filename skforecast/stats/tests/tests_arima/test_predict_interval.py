# Unit test predict_interval method - Arima
# ==============================================================================
import platform
import numpy as np
import pandas as pd
import pytest
from ..._arima import Arima
from .fixtures_arima import air_passengers, multi_seasonal


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

    # Check exact values for first 3 steps (R-based implementation)
    expected_mean = np.array([-1.6103516001266247, -1.1072627028353699, -0.7753950199537363])
    expected_lower_95 = np.array([-3.5061759690202594, -3.15078572725202, -2.8799551358866573])
    expected_upper_95 = np.array([0.2854727687670102, 0.9362603215812801, 1.3291650959791845])

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

    # Check exact values for first 3 steps (R-based implementation)
    expected_mean = np.array([-1.6103516001266247, -1.1072627028353699, -0.7753950199537363])
    expected_lower_90 = np.array([-3.201377564804984, -2.822241286617544, -2.5415975456812863])
    expected_upper_90 = np.array([-0.019325635448265155, 0.6077158809468042, 0.9908075057738135])

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

    # Check exact values for first 3 steps (R-based implementation)
    expected_mean = np.array([-1.6103516001266247, -1.1072627028353699, -0.7753950199537363])
    expected_lower_95 = np.array([-3.5061759690202594, -3.15078572725202, -2.8799551358866573])
    expected_upper_95 = np.array([0.2854727687670102, 0.9362603215812801, 1.3291650959791845])

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
    
    # Check exact values for first 2 steps (R-based implementation)
    expected_mean = np.array([-1.6103516001266247, -1.1072627028353699])
    expected_lower_50 = np.array([-2.262768744052251, -1.8105079385289138])
    expected_upper_50 = np.array([-0.9579344562009986, -0.40401746714182596])
    expected_lower_99 = np.array([-4.10188716011514, -3.792907199514493])
    expected_upper_99 = np.array([0.8811839598618907, 1.5783817938437532])

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


def test_predict_interval_widens_with_horizon():
    """
    Test that prediction intervals generally widen as forecast horizon increases.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    result = model.predict_interval(steps=20, level=(95,))
    
    width = result['upper_95'] - result['lower_95']
    
    # Width should generally increase (allowing for small variations)
    assert width.iloc[-1] >= width.iloc[0]


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
    
    # Check exact values for first 3 steps (R-based implementation)
    expected_mean = np.array([-0.7086254997055901, -0.28482871141306915, -0.09336880032784575])
    expected_lower_95 = np.array([-2.814753398261278, -2.524216549793444, -2.387987574296295])
    expected_upper_95 = np.array([1.3975023988500974, 1.954559126967306, 2.2012499736406035])

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
    expected_mean_first = np.array([2.6823667306424746, 2.6706837793404903, 2.6512215984190295])
    expected_lower_95_first = np.array([-0.3273637082331806, -0.9184356280195636, -1.3861368693058589])
    expected_upper_95_first = np.array([5.69209716951813, 6.259803186700545, 6.688580066143918])

    expected_mean_last = np.array([2.5554206376667588, 2.552322493238509, 2.549340142505205])
    expected_lower_95_last = np.array([-3.8828087372309636, -3.9132293455135105, -3.9405573648502026])
    expected_upper_95_last = np.array([8.993650012564482, 9.017874331990528, 9.039237649860613])

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
        -10.38329029, -10.38329819, -10.38329815, -10.38329815,
        -10.38329815
    ])
    expected_lower_95 = np.array([
        -12.90567947, -13.46961784, -13.94536068, -14.36465596,
        -14.74381768
    ])
    expected_upper_95 = np.array([
        -7.86090112, -7.29697855, -6.82123562, -6.40194034, -6.02277862
    ])
    
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result['lower_95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result['upper_95'].values, expected_upper_95, decimal=4)


def test_predict_interval_auto_arima_air_passengers_data():
    """
    Test predict_interval works correctly with auto ARIMA on Air Passengers dataset
    """
    
    model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
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
        m=12,
        trace=False,
        stepwise=True,
    )
    model.fit(air_passengers, suppress_warnings=True)
    pred = model.predict_interval(steps=5, level=(95, 99))

    expected = pd.DataFrame({
            'mean': {1: 445.2856573681562,
            2: 419.83465616022573,
            3: 448.44413493780667,
            4: 490.92842560605027,
            5: 502.3511798022874},
            'lower_95': {1: 419.01183057292997,
            2: 390.374886879868,
            3: 416.9267606112706,
            4: 457.9051928081597,
            5: 468.2289786562149},
            'upper_95': {1: 471.5594841633824,
            2: 449.2944254405835,
            3: 479.96150926434274,
            4: 523.9516584039409,
            5: 536.4733809483598},
            'lower_99': {1: 410.7559958492004,
            2: 381.1179564725308,
            3: 407.02328383973776,
            4: 447.52854101139263,
            5: 457.50700597719066},
            'upper_99': {1: 479.81531888711197,
            2: 458.55135584792066,
            3: 489.8649860358756,
            4: 534.3283102007078,
            5: 547.1953536273841}
            },
            index=[1, 2, 3, 4, 5],
    ).rename_axis('step')
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 1, 0)
    assert model.best_params_['m'] == 12
    pd.testing.assert_frame_equal(pred, expected)
    
    
def test_predict_interval_auto_arima_multi_seasonal_data():
    """
    Test predict_interval works correctly with auto ARIMA on multi-seasonal dataset
    """
    
    model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
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
        m=12,
        trace=False,
        stepwise=True,
    )
    model.fit(multi_seasonal, suppress_warnings=True)
    pred = model.predict_interval(steps=5, level=(95, 99))

    expected = pd.DataFrame({
        'mean': {1: 174.22831851,
        2: 174.13324908,
        3: 174.86422913,
        4: 174.85907826,
        5: 174.81533986},
        'lower_95': {1: 153.13683798,
        2: 153.03928683,
        3: 153.71540634,
        4: 153.65260745,
        5: 153.55657799},
        'upper_95': {1: 195.31979904,
        2: 195.22721133,
        3: 196.01305192,
        4: 196.06554908,
        5: 196.07410173},
        'lower_99': {1: 146.50941453,
        2: 146.41108393,
        3: 147.06996441,
        4: 146.98905099,
        5: 146.87659144},
        'upper_99': {1: 201.94722249,
        2: 201.85541423,
        3: 202.65849385,
        4: 202.72910554,
        5: 202.75408828}
        },
        index=[1, 2, 3, 4, 5],
        ).rename_axis('step')
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 0, 0)
    assert model.best_params_['m'] == 12
    pd.testing.assert_frame_equal(pred, expected)
    

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
        -0.2013853, 0.19484881, -0.03919908, -0.24332757, -0.0255057
    ])
    expected_lower_95_df = np.array([
        -1.90717175, -1.58961103, -1.85099262, -2.06480647, -1.85043932
    ])
    expected_upper_95_df = np.array([
        1.50440114, 1.97930865, 1.77259447, 1.57815133, 1.79942792
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
        -0.02240879, 0.0069841, 0.09581765, -0.09615039, -0.12792916
    ])
    expected_lower_95_series = np.array([
        -1.75636421, -1.80777753, -1.747181, -1.94921403, -1.98460463
    ])
    expected_upper_95_series = np.array([
        1.71154664, 1.82174572, 1.93881631, 1.75691325, 1.7287463
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
        -1.6103516, -1.10726272, -0.77539506, -0.55647521, -0.41206252
    ])
    expected_lower_90 = np.array([
        -3.20137755, -2.82224127, -2.54159755, -2.3445097, -2.20951444
    ])
    expected_upper_90 = np.array([
        -0.01932566, 0.60771583, 0.99080742, 1.23155927, 1.38538939
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
        -1.6103516, -1.10726272, -0.77539506, -0.55647521, -0.41206252
    ])
    expected_lower_95 = np.array([
        -3.50617595, -3.15078571, -2.87995513, -2.68704971, -2.55385858
    ])
    expected_upper_95 = np.array([
        0.28547275, 0.93626026, 1.329165, 1.57409929, 1.72973354
    ])
    
    np.testing.assert_array_almost_equal(result_after['mean'].values, expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(result_after['lower_95'].values, expected_lower_95, decimal=4)
    np.testing.assert_array_almost_equal(result_after['upper_95'].values, expected_upper_95, decimal=4)
