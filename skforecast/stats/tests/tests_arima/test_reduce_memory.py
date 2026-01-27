# Unit test reduce_memory method - Arima
# ==============================================================================
import re
import pytest
import numpy as np
import warnings
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


def test_reduce_memory_raises_error_for_unfitted_model():
    """
    Test that reduce_memory raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.reduce_memory()


def test_reduce_memory_returns_self():
    """
    Test that reduce_memory returns self for method chaining.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    result = model.reduce_memory()
    
    assert result is model


def test_reduce_memory_sets_flag():
    """
    Test that reduce_memory sets is_memory_reduced flag to True.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    assert model.is_memory_reduced is False
    
    model.reduce_memory()
    
    assert model.is_memory_reduced is True


def test_reduce_memory_emits_warning():
    """
    Test that reduce_memory emits a UserWarning.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.reduce_memory()
        
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Memory reduced" in str(w[0].message)


def test_reduce_memory_deletes_large_attributes():
    """
    Test that reduce_memory deletes expected attributes.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Check attributes exist before
    assert hasattr(model, 'y_train_')
    assert hasattr(model, 'fitted_values_')
    assert hasattr(model, 'in_sample_residuals_')
    
    model.reduce_memory()
    
    # Check attributes are deleted
    assert not hasattr(model, 'y_train_')
    assert not hasattr(model, 'fitted_values_')
    assert not hasattr(model, 'in_sample_residuals_')


def test_reduce_memory_keeps_model_for_predictions():
    """
    Test that reduce_memory keeps model_ attribute for predictions.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    model.reduce_memory()
    
    # model_ should still exist
    assert hasattr(model, 'model_')
    assert model.model_ is not None


def test_reduce_memory_predictions_still_work():
    """
    Test that predictions still work after reduce_memory.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Get predictions before
    pred_before = model.predict(steps=10)
    
    model.reduce_memory()
    
    # Predictions should still work
    pred_after = model.predict(steps=10)
    
    np.testing.assert_array_almost_equal(pred_before, pred_after)


def test_reduce_memory_predict_interval_still_works():
    """
    Test that predict_interval still works after reduce_memory.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    model.reduce_memory()
    
    # predict_interval should still work
    result = model.predict_interval(steps=10)
    
    assert result.shape[0] == 10
    assert 'mean' in result.columns


def test_reduce_memory_diagnostic_methods_fail():
    """
    Test that diagnostic methods fail after reduce_memory.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    model.reduce_memory()
    
    # get_residuals should fail
    error_msg = re.escape(
        "Cannot call get_residuals(): model memory has been reduced via"
    )
    with pytest.raises(ValueError, match=error_msg):
        model.get_residuals()
    
    # get_fitted_values should fail
    error_msg = re.escape(
        "Cannot call get_fitted_values(): model memory has been reduced via"
    )
    with pytest.raises(ValueError, match=error_msg):
        model.get_fitted_values()
    
    # get_score should fail
    error_msg = re.escape(
        "Cannot call get_score(): model memory has been reduced via"
    )
    with pytest.raises(ValueError, match=error_msg):
        model.get_score()


def test_reduce_memory_with_exog():
    """
    Test reduce_memory when model was fitted with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog_train = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog_train)
    
    model.reduce_memory()
    
    # Predictions with exog should still work
    exog_pred = np.random.randn(10, 2)
    pred = model.predict(steps=10, exog=exog_pred)
    
    assert pred.shape == (10,)


def test_reduce_memory_preserves_coefficients():
    """
    Test that reduce_memory preserves important attributes like coefficients.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    coef_before = model.coef_.copy()
    aic_before = model.aic_
    
    model.reduce_memory()
    
    # These should still be accessible
    np.testing.assert_array_equal(model.coef_, coef_before)
    assert model.aic_ == aic_before
    assert hasattr(model, 'sigma2_')
    assert hasattr(model, 'loglik_')


def test_reduce_memory_method_chaining():
    """
    Test reduce_memory with method chaining.
    """
    y = ar1_series(100, seed=42)
    
    pred = (Arima(order=(1, 0, 1))
            .fit(y)
            .reduce_memory()
            .predict(steps=10))
    
    assert pred.shape == (10,)
