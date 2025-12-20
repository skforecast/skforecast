# Unit test set_params method - Ets
# ==============================================================================
import numpy as np
import pytest
from ..._ets import Ets


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
    model = Ets()
    
    with pytest.raises(
        ValueError,
        match="Invalid parameter 'invalid_param' for estimator Ets. Valid parameters are:"
    ):
        model.set_params(invalid_param=10)
    
    with pytest.raises(
        ValueError,
        match="Invalid parameter 'max_depth' for estimator Ets. Valid parameters are:"
    ):
        model.set_params(max_depth=20)


def test_set_params_updates_valid_parameters():
    """
    Test that set_params correctly updates valid parameters.
    """
    model = Ets(m=12, model="AAN", alpha=0.1, beta=0.05, damped=False)
    
    assert model.m == 12
    assert model.model == "AAN"
    assert model.alpha == 0.1
    assert model.beta == 0.05
    assert model.damped is False
    
    # Update single parameter
    result = model.set_params(m=4)
    assert result is model  # Returns self for method chaining
    assert model.m == 4
    assert model.model == "AAN"  # Unchanged
    assert model.alpha == 0.1  # Unchanged
    
    # Update multiple parameters
    model.set_params(model="AAA", alpha=0.2, beta=0.1, damped=True)
    assert model.m == 4
    assert model.model == "AAA"
    assert model.alpha == 0.2
    assert model.beta == 0.1
    assert model.damped is True


def test_set_params_resets_fitted_state():
    """
    Test that set_params resets all fitted state attributes.
    """
    y = ar1_series(100)
    model = Ets(m=1, model="AAN")
    model.fit(y)
    
    # Verify model is fitted
    assert hasattr(model, 'model_') and model.model_ is not None
    assert hasattr(model, 'y_') and model.y_ is not None
    assert hasattr(model, 'config_') and model.config_ is not None
    assert hasattr(model, 'params_') and model.params_ is not None
    assert hasattr(model, 'fitted_values_') and model.fitted_values_ is not None
    assert hasattr(model, 'residuals_in_') and model.residuals_in_ is not None
    assert model.memory_reduced_ is False
    
    # Set params - should reset fitted state
    model.set_params(m=4)
    
    # Verify all fitted attributes are reset
    assert model.model_ is None
    assert model.y_ is None
    assert model.config_ is None
    assert model.params_ is None
    assert model.fitted_values_ is None
    assert model.residuals_in_ is None
    assert model.memory_reduced_ is False


def test_set_params_requires_refit_before_predict():
    """
    Test that predict raises error after set_params until model is refitted.
    """
    y = ar1_series(100)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    # Predict works after fit
    pred_before = model.predict(steps=5)
    assert pred_before.shape == (5,)
    
    # Set params resets fitted state
    model.set_params(model="AAN")
    
    # Predict should raise error
    with pytest.raises(Exception):  # Could be NotFittedError or similar
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
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    # predict_interval works after fit
    result_before = model.predict_interval(steps=5, level=(95,))
    assert result_before.shape == (5, 3)  # mean, lower_95, upper_95
    
    # Set params resets fitted state
    model.set_params(alpha=0.2)
    
    # predict_interval should raise error
    with pytest.raises(Exception):
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
    model = Ets()
    
    # Method chaining: set_params -> fit -> predict
    predictions = model.set_params(m=1, model="AAN").fit(y).predict(steps=5)
    
    assert predictions.shape == (5,)
    assert model.m == 1
    assert model.model == "AAN"


def test_set_params_preserves_unchanged_parameters():
    """
    Test that set_params only changes specified parameters.
    """
    model = Ets(m=12, model="AAA", alpha=0.1, beta=0.05, gamma=0.03)
    
    # Change only m
    model.set_params(m=4)
    assert model.m == 4
    assert model.model == "AAA"
    assert model.alpha == 0.1
    assert model.beta == 0.05
    assert model.gamma == 0.03
    
    # Change only alpha
    model.set_params(alpha=0.2)
    assert model.m == 4
    assert model.model == "AAA"
    assert model.alpha == 0.2
    assert model.beta == 0.05
    assert model.gamma == 0.03


def test_set_params_with_empty_dict():
    """
    Test that set_params with no parameters still resets fitted state.
    """
    y = ar1_series(100)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.model_ is not None
    
    # Call set_params with no arguments
    result = model.set_params()
    
    # Should still reset fitted state
    assert result is model
    assert model.model_ is None


def test_set_params_after_reduce_memory():
    """
    Test that set_params works correctly after reduce_memory.
    """
    y = ar1_series(100)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    model.reduce_memory()
    
    assert model.memory_reduced_ is True
    assert model.fitted_values_ is None
    
    # Set params should reset memory_reduced_ flag
    model.set_params(model="AAN")
    
    assert model.memory_reduced_ is False


def test_set_params_with_all_valid_parameters():
    """
    Test that set_params accepts all valid ETS parameters.
    """
    model = Ets()
    
    # Valid parameters from ETS.__init__
    valid_params = {
        'm': 12,
        'model': 'AAA',
        'damped': True,
        'alpha': 0.1,
        'beta': 0.05,
        'gamma': 0.03,
        'phi': 0.9,
        'lambda_param': 0.5,
        'lambda_auto': True,
        'bias_adjust': False,
        'bounds': 'admissible',
        'seasonal': False,
        'trend': True,
        'ic': 'bic',
        'allow_multiplicative': False,
        'allow_multiplicative_trend': True
    }
    
    result = model.set_params(**valid_params)
    
    assert result is model
    assert model.m == 12
    assert model.model == 'AAA'
    assert model.damped is True
    assert model.alpha == 0.1
    assert model.beta == 0.05
    assert model.gamma == 0.03
    assert model.phi == 0.9
    assert model.lambda_param == 0.5
    assert model.lambda_auto is True
    assert model.bias_adjust is False
    assert model.bounds == 'admissible'
    assert model.seasonal is False
    assert model.trend is True
    assert model.ic == 'bic'
    assert model.allow_multiplicative is False
    assert model.allow_multiplicative_trend is True
