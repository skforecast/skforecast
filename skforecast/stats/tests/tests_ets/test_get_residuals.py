# Unit test get_residuals method - Ets
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


def test_estimator_residuals_helper():
    """Test get_residuals() helper method"""
    y = ar1_series(70)
    est = Ets(m=1, model="AAN").fit(y)

    r = est.get_residuals()
    f = est.get_fitted_values()

    assert r.shape == y.shape
    assert f.shape == y.shape

    # Residuals should equal y - fitted (approximately)
    assert np.allclose(r, y - f, atol=1e-10)
    
    # Check exact residual values (first 10)
    expected_residuals = np.array([
        -0.58403773, -0.90187138,  0.58714496,  0.407959,  1.00341891,
         0.99247986, -0.27350356,  0.10948028, -0.52014891, -0.91011355
    ])
    np.testing.assert_array_almost_equal(r[:10], expected_residuals, decimal=6)


def test_get_residuals_raises_error_after_reduce_memory():
    """Test that get_residuals() raises error after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Verify residuals work before reduction
    residuals_before = est.get_residuals()
    assert residuals_before is not None
    assert residuals_before.shape == y.shape
    
    # Reduce memory
    est.reduce_memory()
    
    # get_residuals() should raise error
    with pytest.raises(
        ValueError,
        match="Cannot call residuals_\\(\\): model memory has been reduced"
    ):
        est.get_residuals()
