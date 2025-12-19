# Unit test summary method - Ets
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


def test_estimator_summary(capsys):
    """Test Ets estimator summary output"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    est.summary()
    captured = capsys.readouterr().out

    assert "ETS Model Summary" in captured
    assert "Number of observations:" in captured
    assert "Smoothing parameters:" in captured
    assert "alpha (level):" in captured


def test_summary_raises_error_after_reduce_memory(capsys):
    """Test that summary() raises error after reduce_memory()"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN")
    est.fit(y)
    
    # Verify summary works before reduction
    est.summary()
    out_before = capsys.readouterr().out
    assert "ETS Model Summary" in out_before
    
    # Reduce memory
    est.reduce_memory()
    
    # summary() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.summary()
