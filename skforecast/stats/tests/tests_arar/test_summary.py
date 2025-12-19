# Unit test summary method - Arar
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


def test_estimator_summary(capsys):
    """
    Test basic summary functionality.
    """
    y = ar1_series(100)
    est = Arar().fit(y)
    est.summary()
    
    out = capsys.readouterr().out
    assert "ARAR Model Summary" in out


def test_arar_summary_with_exog(capsys):
    """
    Test that summary() includes exogenous model information.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    model.summary()
    
    captured = capsys.readouterr().out
    assert "ARAR Model Summary" in captured
    assert "Exogenous Model (Linear Regression)" in captured
    assert "Intercept:" in captured
    assert "Coefficients:" in captured


def test_summary_raises_error_after_reduce_memory(capsys):
    """
    Test that summary() raises error after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Verify summary works before reduction
    est.summary()
    out_before = capsys.readouterr().out
    assert "ARAR Model Summary" in out_before
    
    # Reduce memory
    est.reduce_memory()
    
    # summary() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.summary()
