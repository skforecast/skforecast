# Unit test get_score method - Arar
# ==============================================================================
import re
import pytest
import numpy as np
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_estimator_score():
    """
    Test basic get_score functionality.
    """
    y = ar1_series(100)
    est = Arar().fit(y)
    
    score = est.get_score()
    assert np.isfinite(score) or np.isnan(score)
    expected_score = 0.46961209858721265
    np.testing.assert_almost_equal(score, expected_score, decimal=10)


def test_arar_score_with_exog():
    """
    Test that get_score() works correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    score = model.get_score()
    
    expected_score = 0.8982829747101195
    np.testing.assert_almost_equal(score, expected_score, decimal=10)


def test_get_score_raises_error_after_reduce_memory():
    """
    Test that get_score() raises error after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Verify score works before reduction
    score_before = est.get_score()
    assert score_before is not None
    
    # Reduce memory
    est.reduce_memory()
    
    # get_score() should raise error
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot call get_score(): model memory has been reduced")
    ):
        est.get_score()
