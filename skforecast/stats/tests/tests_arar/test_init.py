# Unit test __init__ method - Arar
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


def test_arar_init_default_params():
    """
    Test Arar initialization with default parameters.
    """
    est = Arar()
    
    assert est.max_ar_depth is None
    assert est.max_lag is None
    assert est.safe is True


def test_arar_init_with_explicit_params():
    """
    Test Arar initialization with explicit parameters.
    """
    est = Arar(max_ar_depth=8, max_lag=15, safe=False)
    
    assert est.max_ar_depth == 8
    assert est.max_lag == 15
    assert est.safe is False


def test_arar_with_explicit_params_propagated_to_estimator():
    """
    Test that initialization parameters are properly propagated after fitting.
    """
    y = ar1_series(60)
    est = Arar(max_ar_depth=8, max_lag=15, safe=True).fit(y)
    
    assert isinstance(est.max_ar_depth, int)
    assert isinstance(est.max_lag, int)
    assert est.max_ar_depth >= 4
    assert est.max_lag >= est.max_ar_depth
