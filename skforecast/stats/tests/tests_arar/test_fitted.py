# Unit test fitted_ method - Arar
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


def test_estimator_fitted_helper():
    """
    Test basic fitted_ functionality.
    """
    y = ar1_series(70)
    est = Arar().fit(y)
    f = est.fitted_()
    
    assert f.shape == y.shape


def test_arar_fitted_values_with_exog():
    """
    Test that fitted values include exogenous component.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    fitted = model.fitted_()
    
    assert fitted.shape == y.shape
    # First few values may be NaN due to ARAR lag structure
    assert np.isnan(fitted).any()
    # But at least half of values should be finite (depends on selected lags)
    assert np.sum(~np.isnan(fitted)) > n * 0.5


def test_fitted_raises_error_after_reduce_memory():
    """
    Test that fitted_() raises error after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Verify fitted works before reduction
    fitted_before = est.fitted_()
    assert fitted_before is not None
    assert fitted_before.shape == y.shape
    
    # Reduce memory
    est.reduce_memory()
    
    # fitted_() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.fitted_()
