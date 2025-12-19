# Unit test residuals_ method - Arar
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


def test_estimator_residuals_helper():
    """
    Test basic residuals_ functionality.
    """
    y = ar1_series(70)
    est = Arar().fit(y)
    r = est.residuals_()
    f = est.fitted_()
    
    assert r.shape == y.shape
    assert f.shape == y.shape
    
    mask = ~np.isnan(f)
    assert np.allclose(r[mask], y[mask] - f[mask])


def test_arar_residuals_with_exog():
    """
    Test that residuals are computed correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    residuals = model.residuals_()
    fitted = model.fitted_()
    
    # Residuals should be: original y - fitted values
    assert np.allclose(residuals, y - fitted, equal_nan=True)
    # Check that residuals + fitted = original y (where both are finite)
    mask = ~np.isnan(fitted)
    assert np.allclose(residuals[mask] + fitted[mask], y[mask])


def test_residuals_raises_error_after_reduce_memory():
    """
    Test that residuals_() raises error after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Verify residuals work before reduction
    residuals_before = est.residuals_()
    assert residuals_before is not None
    assert residuals_before.shape == y.shape
    
    # Reduce memory
    est.reduce_memory()
    
    # residuals_() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.residuals_()
