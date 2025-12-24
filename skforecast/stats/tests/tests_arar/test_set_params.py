# Unit test set_params method - Arar
# ==============================================================================
import numpy as np
import pytest
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_set_params_raises_error_for_invalid_parameter():
    """
    Test that set_params raises ValueError for invalid parameter names.
    """
    model = Arar()
    
    with pytest.raises(
        ValueError,
        match="Invalid parameter 'invalid_param' for estimator Arar. Valid parameters are:"
    ):
        model.set_params(invalid_param=10)
    
    with pytest.raises(
        ValueError,
        match="Invalid parameter 'max_depth' for estimator Arar. Valid parameters are:"
    ):
        model.set_params(max_depth=20)


def test_set_params_updates_valid_parameters():
    """
    Test that set_params correctly updates valid parameters.
    """
    model = Arar(max_ar_depth=10, max_lag=20, safe=True)
    
    assert model.max_ar_depth == 10
    assert model.max_lag == 20
    assert model.safe is True
    
    # Update single parameter
    result = model.set_params(max_ar_depth=15)
    assert result is model  # Returns self for method chaining
    assert model.max_ar_depth == 15
    assert model.max_lag == 20  # Unchanged
    assert model.safe is True  # Unchanged
    
    # Update multiple parameters
    model.set_params(max_lag=30, safe=False)
    assert model.max_ar_depth == 15
    assert model.max_lag == 30
    assert model.safe is False


def test_set_params_resets_fitted_state():
    """
    Test that set_params resets all fitted state attributes.
    """
    y = ar1_series(100)
    model = Arar(max_ar_depth=10)
    model.fit(y)
    
    # Verify model is fitted
    assert model.is_fitted is True
    assert model.model_ is not None
    assert model.y_train_ is not None
    assert model.coef_ is not None
    assert model.lags_ is not None
    assert model.sigma2_ is not None
    assert model.psi_ is not None
    assert model.sbar_ is not None
    assert model.fitted_values_ is not None
    assert model.in_sample_residuals_ is not None
    assert model.aic_ is not None
    assert model.bic_ is not None
    assert model.is_memory_reduced is False
    
    # Set params - should reset fitted state
    model.set_params(max_ar_depth=15)
    
    # Verify all fitted attributes are reset
    assert model.is_fitted is False
    assert model.model_ is None
    assert model.y_train_ is None
    assert model.coef_ is None
    assert model.lags_ is None
    assert model.sigma2_ is None
    assert model.psi_ is None
    assert model.sbar_ is None
    assert model.fitted_values_ is None
    assert model.in_sample_residuals_ is None
    assert model.aic_ is None
    assert model.bic_ is None
    assert model.is_memory_reduced is False


def test_set_params_resets_fitted_state_with_exog():
    """
    Test that set_params resets fitted state including exog-related attributes.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    # Verify exog-related attributes are set
    assert model.exog_model_ is not None
    assert model.coef_exog_ is not None
    assert model.n_exog_features_in_ == 2
    
    # Set params - should reset all state
    model.set_params(max_ar_depth=20)
    
    # Verify exog-related attributes are reset
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.n_exog_features_in_ is None


def test_set_params_requires_refit_before_predict():
    """
    Test that predict raises error after set_params until model is refitted.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    
    # Predict works after fit
    pred_before = model.predict(steps=5)
    assert pred_before.shape == (5,)
    
    # Set params resets fitted state
    model.set_params(max_ar_depth=20)
    
    # Predict should raise error
    with pytest.raises(
        TypeError,
        match="This Arar instance is not fitted yet. Call 'fit' with appropriate arguments"
    ):
        model.predict(steps=5)
    
    # Refit and predict should work again
    model.fit(y)
    pred_after = model.predict(steps=5)
    assert pred_after.shape == (5,)


def test_set_params_requires_refit_before_predict_interval():
    """
    Test that predict_interval raises error after set_params until model is refitted.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    
    # predict_interval works after fit
    result_before = model.predict_interval(steps=5, level=(95,))
    assert result_before.shape == (5, 3)  # mean, lower_95, upper_95
    
    # Set params resets fitted state
    model.set_params(safe=False)
    
    # predict_interval should raise error
    with pytest.raises(
        TypeError,
        match="This Arar instance is not fitted yet. Call 'fit' with appropriate arguments"
    ):
        model.predict_interval(steps=5)
    
    # Refit and predict_interval should work again
    model.fit(y)
    result_after = model.predict_interval(steps=5, level=(95,))
    assert result_after.shape == (5, 3)


def test_set_params_method_chaining():
    """
    Test that set_params returns self for method chaining.
    """
    y = ar1_series(100)
    model = Arar()
    
    # Method chaining: set_params -> fit -> predict
    predictions = model.set_params(max_ar_depth=15).fit(y).predict(steps=5)
    
    assert predictions.shape == (5,)
    assert model.max_ar_depth == 15
    assert model.is_fitted is True


def test_set_params_preserves_unchanged_parameters():
    """
    Test that set_params only changes specified parameters.
    """
    model = Arar(max_ar_depth=10, max_lag=20, safe=True)
    
    # Change only max_ar_depth
    model.set_params(max_ar_depth=25)
    assert model.max_ar_depth == 25
    assert model.max_lag == 20
    assert model.safe is True
    
    # Change only safe
    model.set_params(safe=False)
    assert model.max_ar_depth == 25
    assert model.max_lag == 20
    assert model.safe is False


def test_set_params_with_empty_dict():
    """
    Test that set_params with no parameters still resets fitted state.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    
    assert model.is_fitted is True
    
    # Call set_params with no arguments
    result = model.set_params()
    
    # Should still reset fitted state
    assert result is model
    assert model.is_fitted is False
    assert model.model_ is None


def test_set_params_after_reduce_memory():
    """
    Test that set_params works correctly after reduce_memory.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    model.reduce_memory()
    
    assert model.is_memory_reduced is True
    assert model.fitted_values_ is None
    
    # Set params should reset is_memory_reduced flag
    model.set_params(max_ar_depth=20)
    
    assert model.is_memory_reduced is False
    assert model.is_fitted is False
