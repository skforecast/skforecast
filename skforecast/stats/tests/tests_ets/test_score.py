# Unit test score method - Ets
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


def test_estimator_score():
    """Test Ets estimator score method"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    score = est.score()

    # R^2 should be between -inf and 1
    assert score <= 1.0
    assert np.isfinite(score) or np.isnan(score)


def test_score_raises_error_after_reduce_memory():
    """Test that score() raises error after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Verify score works before reduction
    score_before = est.score()
    assert score_before is not None
    
    # Reduce memory
    est.reduce_memory()
    
    # score() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.score()
