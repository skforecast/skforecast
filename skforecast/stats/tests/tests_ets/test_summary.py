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

    # Check exact output
    expected_output = """ETS Model Summary
============================================================
Model: ETS(AAN)
Number of observations: 100
Seasonal period (m): 1

Smoothing parameters:
  alpha (level):       0.1000
  beta (trend):        0.0100

Initial states:
  Level (l0):          0.5697
  Trend (b0):          0.0143

Model fit statistics:
  sigma^2:             1.536046
  Log-likelihood:      -19.42
  AIC:                 48.84
  BIC:                 61.86

Residual statistics:
  Mean:                -0.017768
  Std Dev:             1.220320
  MAE:                 0.966249
  RMSE:                1.214333

Time Series Summary Statistics:
  Mean:                0.2885
  Std Dev:             1.1514
  Min:                 -2.4900
  25%:                 -0.4976
  Median:              0.2289
  75%:                 1.0181
  Max:                 3.2281
"""
    assert captured == expected_output


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
    with pytest.raises(
        ValueError,
        match="Cannot call summary\\(\\): model memory has been reduced"
    ):
        est.summary()
