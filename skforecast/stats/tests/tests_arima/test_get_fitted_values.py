# Unit test get_fitted_values method - Arima
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
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


def test_get_fitted_values_raises_NotFittedError_when_not_fitted():
    """
    Test that get_fitted_values raises NotFittedError when the model has not
    been fitted.
    """
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))

    msg = re.escape(
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.get_fitted_values()


def test_get_fitted_values_raises_ValueError_after_reduce_memory():
    """
    Test that get_fitted_values raises ValueError when reduce_memory() has
    been called.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    model.reduce_memory()

    msg = re.escape(
        "Cannot call get_fitted_values(): model memory has been reduced via reduce_memory() "
        "to reduce memory usage. Refit the model to restore full functionality."
    )
    with pytest.raises(ValueError, match=msg):
        model.get_fitted_values()


def test_get_fitted_values_shape():
    """
    Test that get_fitted_values returns an array with the same length as the
    training series.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()

    assert fitted.shape == (100,)


def test_get_fitted_values_equals_fitted_values_attribute():
    """
    Test that get_fitted_values returns the same values as the fitted_values_
    attribute.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()

    np.testing.assert_array_equal(fitted, model.fitted_values_)


def test_get_fitted_values_exact_values():
    """
    Test that get_fitted_values returns expected exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()

    expected_first5 = np.array(
        [-0.15533905,  0.17204575, -0.63307972,  0.07744067,  0.71009339]
    )
    np.testing.assert_array_almost_equal(fitted[:5], expected_first5, decimal=5)


def test_get_fitted_values_all_finite():
    """
    Test that get_fitted_values returns only finite values (no NaN or Inf).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(2, 1, 1), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()

    assert np.all(np.isfinite(fitted))


def test_get_fitted_values_returns_ndarray():
    """
    Test that get_fitted_values returns a numpy ndarray.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()

    assert isinstance(fitted, np.ndarray)


def test_get_fitted_values_with_exog():
    """
    Test get_fitted_values when the model was fitted with exogenous variables.
    """
    np.random.seed(111)
    y = ar1_series(100, seed=42)
    exog = np.random.randn(100)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)

    fitted = model.get_fitted_values()

    expected_first5 = np.array(
        [-0.15522718,  0.17195077, -0.6331881 ,  0.07757076,  0.71015509]
    )
    assert fitted.shape == (100,)
    np.testing.assert_array_almost_equal(fitted[:5], expected_first5, decimal=5)


def test_get_fitted_values_with_seasonal_model():
    """
    Test get_fitted_values with a seasonal model.
    """
    from .fixtures_arima import air_passengers

    model = Arima(order=(1, 1, 0), seasonal_order=(0, 1, 1), m=12)
    model.fit(air_passengers)

    fitted = model.get_fitted_values()

    assert fitted.shape == air_passengers.shape
    assert np.all(np.isfinite(fitted))
    assert isinstance(fitted, np.ndarray)


def test_get_fitted_values_residuals_mean_close_to_zero():
    """
    Test that the mean of the residuals (y - fitted_values) is close to zero.
    """
    y = ar1_series(200, seed=99)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)

    fitted = model.get_fitted_values()
    residuals = y - fitted

    np.testing.assert_almost_equal(np.mean(residuals), 0.0, decimal=1)
