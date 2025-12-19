# Unit test get_fitted_values method - Ets
# ==============================================================================
import numpy as np
import pytest
from ..._ets import Ets


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Generate AR(1) series for testing"""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_estimator_fitted_helper():
    """Test get_fitted_values() helper method"""
    y = ar1_series(70)
    est = Ets(m=1, model="AAN").fit(y)

    f = est.get_fitted_values()
    assert f.shape == y.shape


def test_get_fitted_values_raises_error_after_reduce_memory():
    """Test that get_fitted_values() raises error after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Verify fitted works before reduction
    fitted_before = est.get_fitted_values()
    assert fitted_before is not None
    assert fitted_before.shape == y.shape
    
    # Reduce memory
    est.reduce_memory()
    
    # get_fitted_values() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.get_fitted_values()
