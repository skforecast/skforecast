# Unit test set_params method - Arima
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


def test_set_params_raises_error_for_invalid_parameter():
    """
    Test that set_params raises ValueError for invalid parameter names.
    """
    model = Arima()
    
    msg = "Invalid parameter 'invalid_param'. Valid parameters are:"
    with pytest.raises(ValueError, match=msg):
        model.set_params(invalid_param=10)
    
    msg = "Invalid parameter 'max_depth'. Valid parameters are:"
    with pytest.raises(ValueError, match=msg):
        model.set_params(max_depth=20)


def test_set_params_updates_valid_parameters():
    """
    Test that set_params correctly updates valid parameters.
    """
    model = Arima(order=(1, 0, 0), m=1, method="CSS-ML")
    
    assert model.order == (1, 0, 0)
    assert model.m == 1
    assert model.method == "CSS-ML"
    
    # Update single parameter
    result = model.set_params(order=(2, 1, 1))
    assert result is model  # Returns self for method chaining
    assert model.order == (2, 1, 1)
    
    # Update multiple parameters
    model.set_params(m=12, method="ML")
    assert model.m == 12
    assert model.method == "ML"


def test_set_params_all_parameters():
    """
    Test that set_params can update all valid parameters.
    """
    model = Arima()
    
    new_params = {
        'order': (2, 1, 2),
        'seasonal_order': (1, 1, 1),
        'm': 12,
        'include_mean': False,
        'transform_pars': False,
        'method': 'ML',
        'n_cond': 15,
        'SSinit': 'Rossignol2011',
        'optim_method': 'L-BFGS-B',
        'optim_control': {'maxiter': 200},
        'kappa': 1e5
    }
    
    model.set_params(**new_params)
    
    for key, value in new_params.items():
        assert getattr(model, key) == value


def test_set_params_resets_fitted_state():
    """
    Test that set_params resets all fitted attributes.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    # Check that model is fitted
    assert hasattr(model, 'model_')
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'y_train_')
    assert model.is_memory_reduced is False
    
    # Change a parameter
    model.set_params(order=(2, 0, 0))
    
    # Fitted attributes should be removed
    assert not hasattr(model, 'model_')
    assert not hasattr(model, 'coef_')
    assert not hasattr(model, 'y_')
    assert not hasattr(model, 'coef_names_')
    assert not hasattr(model, 'sigma2_')
    assert not hasattr(model, 'loglik_')
    assert not hasattr(model, 'aic_')
    assert not hasattr(model, 'bic_')
    assert not hasattr(model, 'arma_')
    assert not hasattr(model, 'converged_')
    assert not hasattr(model, 'n_features_in_')
    assert not hasattr(model, 'n_exog_features_in_')
    assert not hasattr(model, 'fitted_values_')
    assert not hasattr(model, 'in_sample_residuals_')
    assert not hasattr(model, 'var_coef_')
    assert model.is_memory_reduced is False


def test_set_params_after_fit_requires_refit():
    """
    Test that after set_params, model needs to be refitted before prediction.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    # Predictions work after fitting
    pred1 = model.predict(steps=5)
    assert pred1.shape == (5,)
    
    # Change parameters
    model.set_params(order=(2, 0, 0))
    
    # Predictions should fail without refitting
    from sklearn.exceptions import NotFittedError
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.predict(steps=5)
    
    # Refit and predictions should work again
    model.fit(y)
    pred2 = model.predict(steps=5)
    assert pred2.shape == (5,)


def test_set_params_returns_self():
    """
    Test that set_params returns self for method chaining.
    """
    model = Arima()
    result = model.set_params(order=(1, 1, 1))
    
    assert result is model


def test_set_params_method_chaining():
    """
    Test that set_params supports method chaining.
    """
    y = ar1_series(100, seed=42)
    
    model = (Arima()
             .set_params(order=(1, 0, 1))
             .set_params(m=12)
             .fit(y))
    
    assert model.order == (1, 0, 1)
    assert model.m == 12
    assert hasattr(model, 'model_')


def test_set_params_preserves_unmodified_parameters():
    """
    Test that set_params only changes specified parameters.
    """
    model = Arima(
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1),
        m=12,
        method="ML",
        kappa=1e5
    )
    
    # Only change one parameter
    model.set_params(order=(2, 0, 2))
    
    # Other parameters should remain unchanged
    assert model.order == (2, 0, 2)  # Changed
    assert model.seasonal_order == (1, 0, 1)  # Unchanged
    assert model.m == 12  # Unchanged
    assert model.method == "ML"  # Unchanged
    assert model.kappa == 1e5  # Unchanged


def test_set_params_on_fitted_model_with_exog():
    """
    Test that set_params resets state even when model was fitted with exog.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog)
    
    assert model.n_exog_features_in_ == 2
    
    # Change parameters
    model.set_params(order=(2, 0, 0))
    
    # Exog-related attributes should also be reset
    assert not hasattr(model, 'n_exog_features_in_')


def test_set_params_empty_call():
    """
    Test that calling set_params with no arguments still resets fitted state.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    
    assert hasattr(model, 'model_')
    
    # Call with no parameters
    model.set_params()
    
    # Should still reset fitted state
    assert not hasattr(model, 'model_')


def test_set_params_resets_memory_reduced_flag():
    """
    Test that set_params resets is_memory_reduced flag.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0))
    model.fit(y)
    model.reduce_memory()
    
    assert model.is_memory_reduced is True
    
    model.set_params(order=(2, 0, 0))
    
    assert model.is_memory_reduced is False
