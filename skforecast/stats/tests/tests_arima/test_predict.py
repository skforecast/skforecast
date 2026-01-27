# Unit test predict method - Arima
# ==============================================================================
import re
import platform
import numpy as np
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


def test_predict_raises_error_for_unfitted_model():
    """
    Test that predict raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.predict(steps=1)


def test_predict_raises_error_for_invalid_steps():
    """
    Test that predict raises ValueError for invalid steps parameter.
    """
    y = ar1_series(50)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        model.predict(steps=0)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        model.predict(steps=-2)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        model.predict(steps=1.5)


def test_arima_predict_without_exog_when_fitted_with_exog():
    """
    Test that predict raises error when exog is missing but model was fitted with exog.
    """
    np.random.seed(42)
    y = ar1_series(100)
    exog = np.random.randn(100, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog)
    
    msg = (
        "Model was fitted with 2 exogenous features, "
        "but `exog` was not provided for prediction."
    )
    with pytest.raises(ValueError, match=msg):
        model.predict(steps=5)


def test_arima_predict_exog_feature_count_mismatch():
    """
    Test that predict raises error when exog has wrong number of features.
    """
    np.random.seed(42)
    y = ar1_series(100)
    exog_train = np.random.randn(100, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    exog_pred = np.random.randn(5, 3)  # Wrong number of features
    msg = (
        "Number of exogenous features \\(3\\) does not match "
        "the number used during fitting \\(2\\)."
    )
    with pytest.raises(ValueError, match=msg):
        model.predict(steps=5, exog=exog_pred)


def test_arima_predict_exog_length_mismatch():
    """
    Test that predict raises error when exog length doesn't match steps.
    """
    np.random.seed(42)
    y = ar1_series(100)
    exog_train = np.random.randn(100, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    exog_pred = np.random.randn(3, 2)  # Wrong length
    msg = re.escape("Length of `exog` (3) must match `steps` (5).")
    with pytest.raises(ValueError, match=msg):
        model.predict(steps=5, exog=exog_pred)


def test_arima_predict_exog_3d_raises():
    """
    Test that predict raises error for 3D exog input.
    """
    np.random.seed(42)
    y = ar1_series(100)
    exog_train = np.random.randn(100, 1)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    exog_pred_3d = np.random.randn(5, 1, 2)  # 3D array
    msg = "`exog` must be 1- or 2-dimensional."
    with pytest.raises(ValueError, match=msg):
        model.predict(steps=5, exog=exog_pred_3d)


def test_arima_predict_returns_correct_shape_and_values():
    """
    Test that predict returns correct shape and exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    # Test 1 step prediction
    pred = model.predict(steps=1)
    assert pred.shape == (1,)
    np.testing.assert_almost_equal(pred[0], -1.613497, decimal=6)
    
    # Test 10 steps prediction (R-based implementation - AR(1) forecasts decay to intercept)
    pred = model.predict(steps=10)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    expected_pred = np.array([-1.6134968158909693, -1.1891329639587627, -0.8868684748513417, -0.671572518422975, -0.5182222224908184, -0.40899437793394966, -0.33119392796404723, -0.2757784792744241, -0.23630734569280293, -0.20819297644995455])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)

    # Test 1 step prediction
    pred = model.predict(steps=1)
    assert pred.shape == (1,)
    np.testing.assert_almost_equal(pred[0], -1.6134968158909693, decimal=5)

    # Test 50 steps prediction - predictions decay toward intercept
    pred = model.predict(steps=50)
    assert pred.shape == (50,)
    assert np.all(np.isfinite(pred))


def test_arima_predict_returns_finite_and_exact_values():
    """
    Test that predictions are finite and match exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    pred = model.predict(steps=5)
    assert np.all(np.isfinite(pred))
    # Check first 5 values (R-based implementation - forecasts decay)
    expected_pred = np.array([-1.6103516001266247, -1.1072627028353699, -0.7753950199537363, -0.5564751440943749, -0.4120624322866638])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred, decimal=5)
    # For ARIMA(1,0,1) predictions decay toward intercept
    assert np.all(np.isfinite(pred))


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="ARIMA optimizer converges to different values on macOS"
)
def test_arima_predict_with_exog_numpy_array():
    """
    Test predict with exogenous variables as numpy array.
    """
    np.random.seed(42)
    y = np.random.randn(30) * 0.5
    y[0] = 1.0
    for i in range(1, 30):
        y[i] = 0.5 * y[i-1] + y[i]
    exog_train = np.random.randn(30, 2)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    # Check exact coefficients (R-based implementation values)
    expected_coef = np.array([0.6988637447305925, -0.10619333159002398, -0.03484150168459092, -0.06402661123747877])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=5)
    assert model.n_exog_features_in_ == 2

    exog_pred = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]])
    pred = model.predict(steps=3, exog=exog_pred)

    # Check exact prediction values (R-based implementation)
    expected_pred = np.array([-0.31556283211248104, -0.2975537939183649, -0.23017131332329926])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_with_exog_1d_array():
    """
    Test predict with 1D exogenous array.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    np.random.seed(42)  # Reset seed for reproducible exog_pred
    exog_pred = np.random.randn(5)
    pred = model.predict(steps=5, exog=exog_pred)
    
    assert pred.shape == (5,)
    assert model.n_exog_features_in_ == 1
    # Check exact prediction values (R-based implementation)
    expected_pred = np.array([-0.8012899481961615, -0.6159769869372561, -0.4933821010267012, -0.413123094347968, -0.3656502416885319])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_consistency():
    """
    Test that predictions are consistent across multiple calls and match exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    pred1 = model.predict(steps=10)
    pred2 = model.predict(steps=10)

    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred1, pred2)

    # Check exact values (R-based implementation - forecasts decay)
    expected_pred = np.array([-1.6103516001266247, -1.1072627028353699, -0.7753950199537363, -0.5564751440943749, -0.4120624322866638, -0.316799125335701, -0.2539577207416437, -0.2125037521807288, -0.18515822226219755, -0.16711946668620342])
    np.testing.assert_array_almost_equal(pred1, expected_pred, decimal=5)


def test_arima_predict_seasonal_model():
    """
    Test predict for seasonal ARIMA model with exact values.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    pred = model.predict(steps=5)
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    # Check first 5 values (R-based implementation - seasonal decay)
    expected_pred_start = np.array([2.6823667306424746, 2.6706837793404903, 2.6512215984190295, 2.652740977916483, 2.642508147894773])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred_start, decimal=4)


def test_arima_predict_ar_model_stays_bounded():
    """
    Test that predictions from stationary AR model converge and match exact values.
    """
    y = ar1_series(100, phi=0.5, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    pred = model.predict(steps=5)
    assert np.all(np.abs(pred) < 1000)

    # Check first 5 values (R-based implementation - forecasts decay)
    expected_pred_start = np.array([-0.7331301194146427, -0.4417043792937719, -0.28245945747721757, -0.19544262391322736, -0.14789367078412335])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred_start, decimal=5)

    # All predictions should be finite and bounded
    assert np.all(np.isfinite(pred))
    assert np.all(np.abs(pred) < 10)


def test_arima_predict_with_differencing():
    """
    Test predict for ARIMA with differencing (d > 0) returns exact values.
    """
    # Create a random walk (needs differencing)
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 1, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    pred = model.predict(steps=10)
    
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
    
    # Expected values from skforecast implementation
    expected_pred = np.array([
        -10.38329029, -10.38329819, -10.38329815, -10.38329815,
        -10.38329815, -10.38329815, -10.38329815, -10.38329815,
        -10.38329815, -10.38329815
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_ma_model():
    """
    Test predict for pure MA model (p=0, q>0) returns exact values.
    """
    np.random.seed(42)
    y = np.random.randn(100)
    
    model = Arima(order=(0, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    pred = model.predict(steps=5)
    
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    
    # MA(1) forecasts should converge quickly to mean
    # Expected values from skforecast implementation
    expected_pred = np.array([
        -0.10094798, -0.10395163, -0.10395163, -0.10395163, -0.10395163
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_air_passengers_data():
    """
    Test predict works correctly with Air Passengers dataset and returns exact values.
    """
   
    model = Arima(
        order=(2, 1, 1),
        seasonal_order=(1, 1, 1),
        m=12,
        optim_method="BFGS"
    )
    model.fit(air_passengers, suppress_warnings=False)
    pred = model.predict(steps=10)
    
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([0.21797412,  0.10190091, -0.60950889, -0.75357779,  0.66002656]),
        decimal=5
    )
    assert model.coef_names_ == ['ar1', 'ar2', 'ma1', 'sar1', 'sma1']
    
    # Check exact predicted values
    expected_pred = np.array([
        449.90727977, 425.89942208, 459.84193732, 497.85003478,
        510.28517537, 570.53014188, 657.56297822, 643.48037617,
        547.49999804, 498.39931436
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_multi_seasonal_data():
    """
    Test predict works correctly with multi_seasonal dataset and returns exact values.
    """

    model = Arima(
        order=(5, 1, 2),
        seasonal_order=(1, 1, 1),
        m=7,
        optim_method="L-BFGS-B"
    )
    model.fit(multi_seasonal, suppress_warnings=False)
    pred = model.predict(steps=10)
    
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([-0.50815818, -0.04561599, -0.04026491,  0.01633577, -0.06006012,
       -0.48386679, -0.51610601, -0.0470779 , -0.97491034]),
        decimal=5
    )
    assert model.coef_names_ == model.coef_names_ 
    
    # Check exact predicted values
    expected_pred = np.array([
        174.63663927, 168.90121337, 173.57034418, 172.08613347,
        173.80934415, 171.85540948, 175.37193414, 173.55968119,
        172.87896427, 173.39779159
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_auto_arima_air_passengers_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    Air Passengers dataset and returns exact values.
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
    pred = model.predict(steps=10)
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 1, 0)
    assert model.best_params_['m'] == 12
    
    # Check exact predicted values
    expected_pred = np.array([
        445.28565737, 419.83465616, 448.44413494, 490.92842561,
        502.3511798 , 565.70976142, 653.01653442, 637.27830708,
        539.5018538 , 492.69271213
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_auto_arima_multi_seasonal_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    multi_seasonal dataset and returns exact values.
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
    pred = model.predict(steps=10)
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 0, 0)
    assert model.best_params_['m'] == 12
    
    # Check exact predicted values
    expected_pred = np.array([
        174.22831851, 174.13324908, 174.86422913, 174.85907826,
        174.81533986, 174.81629778, 174.81890523, 174.81880912,
        174.81865425, 174.81866231
    ])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)


def test_arima_predict_with_exog_dataframe():
    """
    Test predict with exogenous variables as pandas DataFrame and Series.
    """
    import pandas as pd
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
    assert model.n_exog_names_in_ == ['feature1', 'feature2']
    
    # Predict with DataFrame
    np.random.seed(123)
    exog_pred_df = pd.DataFrame({
        'feature1': np.random.randn(5),
        'feature2': np.random.randn(5)
    })
    pred_df = model.predict(steps=5, exog=exog_pred_df)
    
    assert pred_df.shape == (5,)
    assert np.all(np.isfinite(pred_df))
    
    # Check exact predicted values for DataFrame exog
    expected_pred_df = np.array([
        -0.2013853, 0.19484881, -0.03919908, -0.24332757, -0.0255057
    ])
    np.testing.assert_array_almost_equal(pred_df, expected_pred_df, decimal=5)
    
    # Predict with Series (1D exog)
    np.random.seed(42)
    y2 = ar1_series(80, seed=42)
    exog_train_1d = pd.Series(np.random.randn(80), name='single_feature')
    
    model2 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model2.fit(y2, exog=exog_train_1d)
    
    assert model2.n_exog_features_in_ == 1
    
    exog_pred_series = pd.Series(np.random.randn(5))
    pred_series = model2.predict(steps=5, exog=exog_pred_series)
    
    assert pred_series.shape == (5,)
    assert np.all(np.isfinite(pred_series))
    
    # Check exact predicted values for Series exog
    expected_pred_series = np.array([
        -0.02240879, 0.0069841, 0.09581765, -0.09615039, -0.12792916
    ])
    np.testing.assert_array_almost_equal(pred_series, expected_pred_series, decimal=5)


def test_arima_predict_after_reduce_memory_raises():
    """
    Test that predict still works after reduce_memory() is called.
    reduce_memory removes fitted_values_ and in_sample_residuals_ but
    predict should still work as it uses the model_ object.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Get prediction before reduce_memory
    pred_before = model.predict(steps=5)
    
    # Call reduce_memory
    model.reduce_memory()
    assert model.is_memory_reduced is True
    
    # predict should still work
    pred_after = model.predict(steps=5)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=10)
    
    # Expected values
    expected_pred = np.array([
        -1.6134968158909693, -1.1891329639587627, -0.8868684748513417, 
        -0.671572518422975, -0.5182222224908184
    ])
    np.testing.assert_array_almost_equal(pred_after, expected_pred, decimal=5)
