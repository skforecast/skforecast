# Unit test predict method - Arima
# ==============================================================================
import re
import pytest
import platform
import numpy as np
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
    np.testing.assert_almost_equal(pred[0], -1.6134967874292199, decimal=5)

    # Test 10 steps prediction (AR(1) forecasts decay to intercept)
    pred = model.predict(steps=10)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    expected_pred = np.array([-1.61349679, -1.18913291, -0.8868684, -0.67157243, -0.51822213, -0.40899427, -0.33119381, -0.27577836, -0.23630722, -0.20819285])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)

    # Test 1 step prediction
    pred = model.predict(steps=1)
    assert pred.shape == (1,)
    np.testing.assert_almost_equal(pred[0], -1.6134967874292199, decimal=5)

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
    # Check first 5 values (forecasts decay toward intercept)
    expected_pred = np.array([-1.60935984, -1.10754556, -0.77628867, -0.55761987, -0.41327255])
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

    # Check exact values (forecasts decay toward intercept)
    expected_pred = np.array([-1.60935984, -1.10754556, -0.77628867, -0.55761987, -0.41327255, -0.31798621, -0.25508594, -0.21356431, -0.18615511, -0.16806179])
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
    # Check first 5 values (seasonal decay)
    expected_pred_start = np.array([2.59146752, 2.48890481, 2.38744626, 2.30466908, 2.21979508])
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
        -10.38328947, -10.38329738, -10.38329734, -10.38329734,
        -10.38329734, -10.38329734, -10.38329734, -10.38329734,
        -10.38329734, -10.38329734
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
        -0.10100203, -0.1039547, -0.1039547, -0.1039547, -0.1039547
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

    expected_coef = np.array([
        0.57655285, 0.23048522, -0.98090342, -0.93688392, 0.85971439
    ])
    expected_pred = np.array([
        448.09333297, 423.88721629, 458.43899853, 498.09725889,
        510.25520497, 568.95468453, 656.71760162, 642.02925408,
        547.19632004, 497.94294517
    ])

    assert model.coef_names_ == ['ar1', 'ar2', 'ma1', 'sar1', 'sma1']
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=2)
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=2)


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

    expected_coef = np.array([
        -0.50771885, -0.04577332, -0.04034762, 0.01609814, -0.06015838,
        -0.48427446, -0.51572118, -0.04678275, -0.97496519
    ])
    expected_pred = np.array([
        174.62926832, 168.90557295, 173.56344223, 172.08788695,
        173.80911065, 171.86014034, 175.37610935, 173.56315404,
        172.87984319, 173.39731174
    ])

    assert model.coef_names_ == ['ar1', 'ar2', 'ar3', 'ar4', 'ar5', 'ma1', 'ma2', 'sar1', 'sma1']
    np.testing.assert_allclose(model.coef_, expected_coef, atol=1e-3)
    np.testing.assert_allclose(pred, expected_pred, rtol=1e-4)


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


def test_predict_fuel_consumption_data_with_exog():
    """
    Test predict works correctly with auto ARIMA on Fuel Consumption dataset
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
    pred = model.predict(
        steps=5,
        exog=fuel_consumption.loc['1989-09-01':].drop(columns=['y']),
    )

    expected = np.array([
        1574769.82817936, 1449368.44399194, 1509265.14575579,
        1484681.44198045, 1404070.94702908
    ])
    np.testing.assert_allclose(pred, expected, rtol=1e-4)


def test_arima_predict_auto_arima_air_passengers_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    Air Passengers dataset and returns exact values.
    """

    expected_order = (2, 1, 1)
    expected_seasonal_order = (0, 1, 0)
    expected_estimator_name = "AutoArima(2,1,1)(0,1,0)[12]"
    expected_pred = np.array([
        445.56810006, 420.28762041, 449.04990788, 491.65672001,
        503.18017215, 566.62089458, 653.99463684, 638.31090126,
        540.57869761, 493.80541080
    ])

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
    pred = model.predict(steps=10)

    assert model.is_auto is True
    assert model.best_params_['order'] == expected_order
    assert model.best_params_['seasonal_order'] == expected_seasonal_order
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == expected_estimator_name
    np.testing.assert_allclose(pred, expected_pred, rtol=1e-4)


def test_arima_predict_auto_arima_multi_seasonal_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    multi_seasonal dataset and returns exact values.
    """

    expected_pred = np.array([
        174.22844329, 174.1332421, 174.86406979, 174.85893904,
        174.81521813, 174.81617385, 174.81877978, 174.81868387,
        174.81852912, 174.81853717
    ])

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
    pred = model.predict(steps=10)

    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 0, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(2,1,1)"
    np.testing.assert_allclose(pred, expected_pred, rtol=1e-4)


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
        -1.61349679, -1.18913291, -0.8868684,
        -0.67157243, -0.51822213
    ])
    np.testing.assert_array_almost_equal(pred_after, expected_pred, decimal=5)
