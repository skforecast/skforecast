# Unit test get_score method - Arima
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


def test_get_score_raises_error_for_unfitted_model():
    """
    Test that get_score raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.get_score()


def test_get_score_returns_float():
    """
    Test that get_score returns a float value.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    score = model.get_score()
    
    assert isinstance(score, (float, np.floating))


def test_get_score_between_zero_and_one():
    """
    Test that R² score is generally between 0 and 1 for good fits.
    """
    y = ar1_series(100, phi=0.7, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    score = model.get_score()
    
    # For a well-fitted AR model, R² should be positive and reasonably high
    assert 0 <= score <= 1


def test_get_score_after_reduce_memory():
    """
    Test that get_score raises error after reduce_memory is called.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Should work before reduce_memory
    score = model.get_score()
    assert isinstance(score, (float, np.floating))
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise error after reduce_memory
    msg = "Cannot call get_score\\(\\): model memory has been reduced via"
    with pytest.raises(ValueError, match=msg):
        model.get_score()


def test_get_score_with_exog():
    """
    Test get_score when model is fitted with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)
    
    score = model.get_score()
    
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_get_score_ignores_y_parameter():
    """
    Test that get_score ignores the y parameter (sklearn compatibility).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    score1 = model.get_score()
    score2 = model.get_score(y=None)
    score3 = model.get_score(y="ignored")
    
    assert score1 == score2 == score3


def test_get_score_consistency():
    """
    Test that get_score returns same value across multiple calls.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    score1 = model.get_score()
    score2 = model.get_score()
    
    assert score1 == score2


def test_get_score_different_models():
    """
    Test that different model configurations give different scores.
    """
    y = ar1_series(100, seed=42)
    
    model1 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model1.fit(y)
    score1 = model1.get_score()
    
    model2 = Arima(order=(0, 0, 1), seasonal_order=(0, 0, 0))
    model2.fit(y)
    score2 = model2.get_score()
    
    # Scores should be different (though both should be reasonable)
    assert score1 != score2


def test_get_score_seasonal_model():
    """
    Test get_score for seasonal ARIMA model.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    score = model.get_score()
    
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)
