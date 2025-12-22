# Unit test get_residuals method - Arima
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


def test_get_residuals_raises_error_for_unfitted_model():
    """
    Test that get_residuals raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.get_residuals()


def test_get_residuals_returns_correct_shape():
    """
    Test that get_residuals returns array of correct shape.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    residuals = model.get_residuals()
    
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (100,)


def test_get_residuals_are_finite():
    """
    Test that all residuals are finite values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    residuals = model.get_residuals()
    
    assert np.all(np.isfinite(residuals))


def test_get_residuals_mean_near_zero():
    """
    Test that residuals have mean close to zero.
    """
    y = ar1_series(200, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    residuals = model.get_residuals()
    
    # Mean of residuals should be close to zero
    assert np.abs(np.mean(residuals)) < 0.2


def test_get_residuals_equals_observed_minus_fitted():
    """
    Test that residuals equal observed - fitted values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    residuals = model.get_residuals()
    fitted = model.get_fitted_values()
    
    expected_residuals = model.y_ - fitted
    
    np.testing.assert_array_almost_equal(residuals, expected_residuals)


def test_get_residuals_after_reduce_memory():
    """
    Test that get_residuals raises error after reduce_memory is called.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Should work before reduce_memory
    residuals = model.get_residuals()
    assert residuals is not None
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise error after reduce_memory
    msg = "Cannot call get_residuals\\(\\): model memory has been reduced via"
    with pytest.raises(ValueError, match=msg):
        model.get_residuals()


def test_get_residuals_with_exog():
    """
    Test get_residuals when model is fitted with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog)
    
    residuals = model.get_residuals()
    
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (80,)
    assert np.all(np.isfinite(residuals))


def test_get_residuals_seasonal_model():
    """
    Test get_residuals for seasonal ARIMA model.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    residuals = model.get_residuals()
    
    assert residuals.shape == (100,)
    assert np.all(np.isfinite(residuals))


def test_get_residuals_consistency():
    """
    Test that get_residuals returns same values across multiple calls.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    residuals1 = model.get_residuals()
    residuals2 = model.get_residuals()
    
    np.testing.assert_array_equal(residuals1, residuals2)
