# Unit test summary method - Ets
# ==============================================================================
import re
import pytest
import numpy as np
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

    # Check the output: same text, numbers equal up to one unit in the last
    # printed digit (the estimates may differ by rounding across platforms)
    expected_output = """ETS Model Summary
============================================================
Model: Ets(AAN)
Seasonal period (m): 1

Smoothing parameters:
  alpha (level):       0.8430
  beta (trend):        0.0001

Initial states:
  Level (l0):          -0.0306
  Trend (b0):          0.0061

Model fit statistics:
  sigma^2:             0.976924
  Log-likelihood:      3.21
  AIC:                 3.58
  BIC:                 16.61

Residual statistics:
  Mean:                -0.000001
  Std Dev:             0.973304
  MAE:                 0.789177
  RMSE:                0.968425

Time Series Summary Statistics:
Number of observations: 100
  Mean:                 0.2885
  Std Dev:              1.1514
  Min:                  -2.4900
  25%:                  -0.4976
  Median:               0.2289
  75%:                  1.0181
  Max:                  3.2281
"""
    number = r"-?\d+\.\d+"
    assert re.sub(number, "#", captured) == re.sub(number, "#", expected_output)
    for value, expected in zip(
        re.findall(number, captured), re.findall(number, expected_output)
    ):
        decimals = len(expected.split(".")[1])
        assert abs(float(value) - float(expected)) <= 1.01 * 10 ** -decimals


def test_summary_is_shorter_after_reduce_memory(capsys):
    """
    Test that summary() output is shorter after reduce_memory().
    """
    y = ar1_series(100, seed=42)
    model = Ets(m=1, model="AAN")
    model.fit(y)

    model.summary()
    captured = capsys.readouterr()
    assert "ETS Model Summary" in captured.out
    assert "Time Series Summary Statistics" in captured.out
    
    model.reduce_memory()
    model.summary()
    captured = capsys.readouterr()
    assert "ETS Model Summary" in captured.out
    assert "Time Series Summary Statistics" not in captured.out
