# Unit test predict method - Arima
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from ..._arima import Arima
from .fixtures_arima import (
    air_passengers, multi_seasonal, fuel_consumption, tol_pred
)


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
    np.testing.assert_almost_equal(pred[0], -1.61696432, decimal=5)
    
    # Test 10 steps prediction (R-based implementation - AR(1) forecasts decay to intercept)
    pred = model.predict(steps=10)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    expected_pred = np.array([-1.61696432, -1.19545972, -0.89550894, -0.68205819, -0.53016252,
       -0.42207065, -0.34515039, -0.29041245, -0.25145988, -0.22374048])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=5)

    # Test 1 step prediction
    pred = model.predict(steps=1)
    assert pred.shape == (1,)
    np.testing.assert_almost_equal(pred[0], -1.61696432125585, decimal=5)

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
    expected_pred = np.array([-1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred, decimal=5)
    # For ARIMA(1,0,1) predictions decay toward intercept
    assert np.all(np.isfinite(pred))


def test_arima_predict_with_exog_numpy_array():
    """
    Test predict with exogenous variables as numpy array.
    """
    np.random.seed(42)
    y = np.random.randn(30) * 0.5
    y[0] = 1.0
    for i in range(1, 30):
        y[i] = 0.5 * y[i - 1] + y[i]
    exog_train = np.random.randn(30, 2)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog_train)
    
    # Check exact coefficients (R-based implementation values)
    expected_coef = np.array([0.71943317, -0.08436689, -0.05881312, 0.07702519])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=5)
    assert model.n_exog_features_in_ == 2

    exog_pred = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]])
    pred = model.predict(steps=3, exog=exog_pred)

    # Check exact prediction values (R-based implementation)
    expected_pred = np.array([-0.38481775, -0.31047103, -0.13680244])
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
    expected_pred = np.array([-0.39764827, -0.11623889, -0.12015593, -0.17904368, 0.14837879])
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
    expected_pred = np.array([-1.60969915, -1.11552107, -0.78835957, -0.57176833, -0.42837809,
                               -0.33344922, -0.27060331, -0.22899733, -0.20145286, -0.18321755])
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
    expected_pred_start = np.array([2.60268909, 2.51255651, 2.4209728 , 2.35015301, 2.27426289])
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
    expected_pred_start = np.array([-0.73302124, -0.44291341, -0.28513773, -0.19933115, -0.15266508])
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
        -10.38304468, -10.38305569, -10.38305562, -10.38305562,
        -10.38305562, -10.38305562, -10.38305562, -10.38305562,
        -10.38305562, -10.38305562
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
        -0.10097229, -0.10395622, -0.10395622, -0.10395622, -0.10395622
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

    # The coefficients of this model are weakly identified (near-cancelling
    # seasonal AR and MA roots): 1-ULP perturbations of the likelihood move
    # them by up to 0.1 while the log-likelihood and the predictions barely
    # change, so only the latter are compared, the predictions with a looser
    # tolerance than `tol_pred`.
    expected_pred = np.array([
        448.15134639, 423.94754801, 458.77487091, 497.6941214 ,
        510.03106867, 569.15507525, 656.8192311 , 642.51146838,
        547.66285933, 498.20993537
    ])

    assert model.coef_names_ == ['ar1', 'ar2', 'ma1', 'sar1', 'sma1']
    np.testing.assert_allclose(model.loglik_, -503.12458724226894, **tol_pred)
    np.testing.assert_allclose(pred, expected_pred, rtol=1e-2)


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

    # The coefficients of this model are weakly identified (near-cancelling AR
    # and MA roots): 1-ULP perturbations of the likelihood move them by more
    # than 0.1 while the log-likelihood and the predictions barely change, so
    # only the latter are compared, the predictions with a looser tolerance
    # than `tol_pred`.
    expected_pred = np.array([
        174.58287008, 168.87668826, 173.52439196, 172.05375924,
        173.79383314, 171.87072619, 175.36589749, 173.55553196,
        172.87120721, 173.38014803
    ])

    assert model.coef_names_ == ['ar1', 'ar2', 'ar3', 'ar4', 'ar5', 'ma1', 'ma2', 'sar1', 'sma1']
    np.testing.assert_allclose(model.loglik_, -2732.5297717066032, **tol_pred)
    np.testing.assert_allclose(pred, expected_pred, rtol=1e-2)


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
        -0.03453459, 0.47554571, 0.18911951, -0.08577119, 0.18902427
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
        0.15081644, 0.14658572, 0.17453836, 0.08650628, 0.06935746
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
    pred = model.predict(
        steps=5,
        exog=fuel_consumption.loc['1989-09-01':].drop(columns=['y']),
    )

    expected = np.array([702178.55298827, 670303.23645305, 639996.22102265,
                         675010.52440616, 614773.51301366])
    np.testing.assert_allclose(pred, expected, rtol=1e-4)


def test_arima_predict_auto_arima_air_passengers_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    Air Passengers dataset and returns exact values.
    """

    expected_pred = np.array([
        451.34858312, 427.10478883, 463.38985401, 499.70660932,
        514.03811796, 571.85282378, 661.31031948, 648.08486292,
        551.28819333, 501.07050856
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
    assert model.best_params_['order'] == (0, 1, 1)
    assert model.best_params_['seasonal_order'] == (2, 1, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(0,1,1)(2,1,0)[12]"
    np.testing.assert_allclose(pred, expected_pred, **tol_pred)


@pytest.mark.parametrize(
    "arima_kwargs, expected_name",
    [({'order': None, 'seasonal_order': None}, 'AutoArima(0,1,0)'),
     ({'order': (0, 1, 0), 'include_drift': True}, 'Arima(0,1,0)')],
    ids=['auto_arima', 'manual_include_drift']
)
@pytest.mark.parametrize(
    "exog_type, expected_coef_names",
    [('ndarray', ['drift', 'exog1']),
     ('DataFrame', ['drift', 'x'])],
    ids=['exog_ndarray', 'exog_DataFrame']
)
def test_arima_predict_with_drift_and_exog(
    arima_kwargs, expected_name, exog_type, expected_coef_names
):
    """
    Test predict and predict_interval when the model includes a drift term and
    exogenous variables, either selected by auto ARIMA or set manually with
    `include_drift=True`. Both must give the same model. Before the fix, auto
    ARIMA failed to predict because the future drift column was not added.
    Data follow a random walk with drift 0.5 plus 2 * x.
    """
    rng = np.random.RandomState(123)
    n = 100
    steps = 5
    x = rng.normal(0, 1, n + steps)
    y = np.cumsum(0.5 + rng.normal(0, 1, n + steps)) + 2 * x
    if exog_type == 'ndarray':
        exog_train, exog_pred = x[:n], x[n:]
    else:
        exog_train, exog_pred = pd.DataFrame({'x': x[:n]}), pd.DataFrame({'x': x[n:]})

    model = Arima(**arima_kwargs)
    model.fit(y[:n], exog=exog_train)
    pred = model.predict(steps=steps, exog=exog_pred)
    pred_interval = model.predict_interval(steps=steps, exog=exog_pred, level=[0.95])

    expected_pred = np.array([
        50.77783587521633, 46.23795674516272, 51.8946026257248,
        56.007178045793324, 51.461562680362405
    ])
    expected_interval = np.array([
        [50.77783587521633, 48.939290412283285, 52.61638133814938],
        [46.23795674516272, 43.637860816443286, 48.83805267388215]
    ])

    assert model.estimator_name_ == expected_name
    assert model.coef_names_ == expected_coef_names
    np.testing.assert_array_almost_equal(
        model.coef_, np.array([0.4909742755025071, 1.9202151091518056])
    )
    np.testing.assert_array_almost_equal(pred, expected_pred)
    np.testing.assert_array_almost_equal(
        pred_interval.to_numpy()[:2], expected_interval
    )


def test_arima_predict_after_freezing_best_params_with_drift():
    """
    Test that freezing the parameters selected by auto ARIMA (as done by
    `backtesting_stats` with `freeze_params=True`) keeps the drift term. The
    predictions must be the same with the already fitted model (the prediction
    path depends on the fitted model, not on the current parameters) and after
    refitting with the frozen parameters.
    """
    rng = np.random.RandomState(123)
    n = 100
    x = rng.normal(0, 1, n)
    y = np.cumsum(0.5 + rng.normal(0, 1, n)) + 2 * x

    model = Arima(order=None, seasonal_order=None)
    model.fit(y)
    expected_pred = np.array(
        [48.75149485301368, 49.23650349452236, 49.721512136031045]
    )

    assert model.best_params_ == {
        'order': (0, 1, 1),
        'seasonal_order': (0, 0, 0),
        'm': 1,
        'fit_intercept': False,
        'include_drift': True,
        'lambda_bc': None
    }
    pred = model.predict(steps=3)
    np.testing.assert_allclose(pred, expected_pred, **tol_pred)

    model._set_params(**model.best_params_)
    np.testing.assert_array_almost_equal(model.predict(steps=3), pred)

    model_frozen = Arima(order=None, seasonal_order=None)
    model_frozen.set_params(**model.best_params_)
    model_frozen.fit(y)
    assert model_frozen.coef_names_ == ['ma1', 'drift']
    np.testing.assert_allclose(model_frozen.predict(steps=3), pred, **tol_pred)


def test_arima_predict_auto_arima_with_drift_and_leading_missing_values():
    """
    Test that, when auto ARIMA selects a drift term and the series starts with
    missing values (dropped before fitting), the drift continues from its last
    training value. Before the fix, the future drift was computed from the
    length of the series including the missing values, shifting the forecasts
    by drift * number of leading missing values.
    """
    rng = np.random.RandomState(123)
    y = np.cumsum(2.0 + rng.normal(0, 0.5, 120))
    y[:10] = np.nan

    model = Arima(order=None, seasonal_order=None)
    model.fit(y)
    pred = model.predict(steps=3)

    assert model.coef_names_ == ['drift']
    np.testing.assert_array_almost_equal(model.coef_, np.array([2.0243631214236544]))
    np.testing.assert_array_almost_equal(
        pred, np.array([242.9929197291593, 245.01728285058294, 247.0416459720066])
    )


def test_arima_predict_box_cox_with_manual_order():
    """
    Test that predictions of a manual order with `lambda_bc=0` are the
    back-transformed predictions of the same model fitted on log(y). Before
    the fix, `lambda_bc` was ignored when the order was set manually.
    """
    model = Arima(order=(0, 1, 1), seasonal_order=(0, 1, 1), m=12, lambda_bc=0.0)
    model.fit(air_passengers.to_numpy())
    pred = model.predict(steps=3)
    pred_interval = model.predict_interval(steps=3, level=[0.95])

    expected_pred = np.array([450.4227889244881, 425.7168169945266, 479.00433358066545])

    np.testing.assert_allclose(pred, expected_pred, **tol_pred)
    np.testing.assert_array_almost_equal(pred_interval['mean'].to_numpy(), pred)
    assert np.all(pred_interval['lower_0.95'].to_numpy() < pred)
    assert np.all(pred_interval['upper_0.95'].to_numpy() > pred)


def test_arima_predict_after_freezing_best_params_with_box_cox():
    """
    Test that freezing the parameters selected by auto ARIMA with `lambda_bc`
    (as done by `backtesting_stats` with `freeze_params=True`) keeps the Box-Cox
    transformation, so refitting with the frozen parameters gives the same
    predictions.
    """
    y = air_passengers.to_numpy()
    model = Arima(order=None, seasonal_order=None, m=12, lambda_bc='auto')
    model.fit(y)
    pred = model.predict(steps=3)

    model_frozen = Arima(order=None, seasonal_order=None, m=12, lambda_bc='auto')
    model_frozen.set_params(**model.best_params_)
    model_frozen.fit(y)

    assert isinstance(model.best_params_['lambda_bc'], float)
    assert model_frozen.lambda_bc == model.best_params_['lambda_bc']
    np.testing.assert_array_almost_equal(model_frozen.predict(steps=3), pred)


def test_arima_predict_auto_arima_multi_seasonal_data():
    """
    Test predict works correctly with auto ARIMA mode when applied to
    multi_seasonal dataset and returns exact values.
    """

    expected_pred = np.array([
        174.22831851, 174.13324908, 174.86422913, 174.85907826,
        174.81533986, 174.81629778, 174.81890523, 174.81880912,
        174.81865425, 174.81866231
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
    np.testing.assert_allclose(pred, expected_pred, **tol_pred)


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
        -1.61696432125585, -1.19545972, -0.89550894,
        -0.68205819, -0.53016252
    ])
    np.testing.assert_array_almost_equal(pred_after, expected_pred, decimal=5)
