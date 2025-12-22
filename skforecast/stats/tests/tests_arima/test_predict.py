# Unit test predict method - Arima
# ==============================================================================
import numpy as np
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
    msg = r"Length of `exog` \(3\) must match `steps` \(5\)\."
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
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    # Test 10 steps prediction
    pred = model.predict(steps=10)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    expected_pred = np.array([-0.13859403, -0.13859403, -0.13859403, -0.13859403, -0.13859403,
                              -0.13859403, -0.13859403, -0.13859403, -0.13859403, -0.13859403])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=6)
    
    # Test 1 step prediction
    pred = model.predict(steps=1)
    assert pred.shape == (1,)
    np.testing.assert_almost_equal(pred[0], -0.13859403, decimal=6)
    
    # Test 50 steps prediction - all should be same value for converged AR(1)
    pred = model.predict(steps=50)
    assert pred.shape == (50,)
    assert np.allclose(pred, -0.13859403, atol=1e-6)


def test_arima_predict_returns_finite_and_exact_values():
    """
    Test that predictions are finite and match exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    pred = model.predict(steps=20)
    assert np.all(np.isfinite(pred))
    # Check first 5 values
    expected_pred = np.array([-0.13215602, -0.13215602, -0.13215602, -0.13215602, -0.13215602])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred, decimal=6)
    # For ARIMA(1,0,1) all predictions converge to same value
    assert np.allclose(pred, pred[0], atol=1e-6)


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
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    # Check exact coefficients
    expected_coef = np.array([0.69886375, -0.10619333, -0.0348415, -0.06402661])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    assert model.n_exog_features_in_ == 2
    
    exog_pred = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]])
    pred = model.predict(steps=3, exog=exog_pred)
    
    # Check exact prediction values
    expected_pred = np.array([-0.09160078, -0.14103484, -0.12078588])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=6)


def test_arima_predict_with_exog_1d_array():
    """
    Test predict with 1D exogenous array.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80)  # 1D
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    np.random.seed(42)  # Reset seed for reproducible exog_pred
    exog_pred = np.random.randn(5)  # 1D
    pred = model.predict(steps=5, exog=exog_pred)
    
    assert pred.shape == (5,)
    assert model.n_exog_features_in_ == 1
    # Check exact prediction values
    expected_pred = np.array([-0.27009679, -0.27135228, -0.26979829, -0.26806756, -0.27154187])
    np.testing.assert_array_almost_equal(pred, expected_pred, decimal=6)


def test_arima_predict_consistency():
    """
    Test that predictions are consistent across multiple calls and match exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    pred1 = model.predict(steps=10)
    pred2 = model.predict(steps=10)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred1, pred2)
    
    # Check exact values
    expected_pred = np.array([-0.13215602, -0.13215602, -0.13215602, -0.13215602, -0.13215602,
                              -0.13215602, -0.13215602, -0.13215602, -0.13215602, -0.13215602])
    np.testing.assert_array_almost_equal(pred1, expected_pred, decimal=6)


def test_arima_predict_seasonal_model():
    """
    Test predict for seasonal ARIMA model with exact values.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    pred = model.predict(steps=24)
    assert pred.shape == (24,)
    assert np.all(np.isfinite(pred))
    # Check first 5 values - seasonal model predictions converge
    expected_pred_start = np.array([2.49797772, 2.49797772, 2.49797772, 2.49797772, 2.49797772])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred_start, decimal=5)


def test_arima_predict_ar_model_stays_bounded():
    """
    Test that predictions from stationary AR model converge and match exact values.
    """
    y = ar1_series(100, phi=0.5, seed=42)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    pred = model.predict(steps=100)
    # For stationary process, long-term predictions should converge to constant
    assert np.all(np.abs(pred) < 1000)
    
    # Check first 5 values
    expected_pred_start = np.array([-0.09060905, -0.09060905, -0.09060905, -0.09060905, -0.09060905])
    np.testing.assert_array_almost_equal(pred[:5], expected_pred_start, decimal=6)
    
    # All predictions should be same (converged)
    assert np.allclose(pred, -0.09060905, atol=1e-6)
    
    # Verify max absolute value
    assert np.max(np.abs(pred)) < 0.1
