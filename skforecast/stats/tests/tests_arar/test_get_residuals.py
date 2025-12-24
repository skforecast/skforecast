# Unit test get_residuals method - Arar
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


def test_get_estimator_residuals_output():
    """
    Test basic get_residuals functionality.
    """
    y = ar1_series(70)
    model = Arar().fit(y)
    residuals = model.get_residuals()
    fitted_values = model.get_fitted_values()
    
    assert residuals.shape == y.shape
    assert fitted_values.shape == y.shape
    
    mask = ~np.isnan(fitted_values)
    assert np.allclose(residuals[mask], y[mask] - fitted_values[mask])
    
    # Check exact residual values (first 10 non-NaN)
    expected_residuals = np.array([
        -0.32910035,  1.42716468, -0.70544396, -0.34080345,  0.50415606,
        -2.28900364,  1.17407058,  1.4133774 ,  1.44340035,  0.7033159
    ])
    np.testing.assert_array_almost_equal(residuals[mask][:10], expected_residuals, decimal=6)


def test_get_residuals_with_exog():
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
    
    residuals = model.get_residuals()
    fitted_values = model.get_fitted_values()
    
    # Residuals should be: original y - fitted values
    assert np.allclose(residuals, y - fitted_values, equal_nan=True)
    # Check that residuals + fitted = original y (where both are finite)
    mask = ~np.isnan(fitted_values)
    assert np.allclose(residuals[mask] + fitted_values[mask], y[mask])
    
    # Check exact residual values (first 10 non-NaN)
    expected_residuals = np.array([
        -1.13182379,  0.49289273, -0.40344324, -0.38759575, -0.45856898,
         2.25060686, -0.43544925, -1.02018638,  1.11042531, -0.98902931
    ])
    np.testing.assert_array_almost_equal(residuals[mask][:10], expected_residuals, decimal=6)


def test_residuals_raises_error_after_reduce_memory():
    """
    Test that get_residuals() raises error after reduce_memory().
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    
    # Verify residuals work before reduction
    residuals_before = model.get_residuals()
    assert residuals_before is not None
    assert residuals_before.shape == y.shape
    
    model.reduce_memory()
    
    with pytest.raises(
        ValueError, 
        match="Cannot call residuals_\\(\\): model memory has been reduced"
    ):
        model.get_residuals()
