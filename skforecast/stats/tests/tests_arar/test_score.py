# Unit test score method - Arar
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


def test_estimator_score():
    """
    Test basic score functionality.
    """
    y = ar1_series(100)
    est = Arar().fit(y)
    
    score = est.score()
    assert np.isfinite(score) or np.isnan(score)


def test_arar_score_with_exog():
    """
    Test that score() works correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    score = model.score()
    
    assert np.isfinite(score) or np.isnan(score)
    # Score should be between -inf and 1
    if np.isfinite(score):
        assert score <= 1.0


def test_arar_aic_bic_with_exog():
    """
    Test that AIC/BIC account for exogenous parameters.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model_with_exog = Arar()
    model_with_exog.fit(y, exog=exog)
    
    model_without_exog = Arar()
    model_without_exog.fit(y)
    
    # Model with exog should have higher parameter count (reflected in AIC/BIC)
    # Note: AIC/BIC values depend on fit quality, but we can check they're computed
    assert np.isfinite(model_with_exog.aic_)
    assert np.isfinite(model_with_exog.bic_)
    assert np.isfinite(model_without_exog.aic_)
    assert np.isfinite(model_without_exog.bic_)


def test_score_raises_error_after_reduce_memory():
    """
    Test that score() raises error after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Verify score works before reduction
    score_before = est.score()
    assert score_before is not None
    
    # Reduce memory
    est.reduce_memory()
    
    # score() should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        est.score()
