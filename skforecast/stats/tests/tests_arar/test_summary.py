# Unit test summary method - Arar
# ==============================================================================
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


def test_estimator_summary(capsys):
    """
    Test basic summary functionality.
    """
    y = ar1_series(100)
    est = Arar().fit(y)
    est.summary()
    
    out = capsys.readouterr().out
    
    # Check exact output
    expected_output = """Arar(lags=(1, 3, 16, 17)) Model Summary
------------------
Number of observations: 100
Selected AR lags: (1, 3, 16, 17)
AR coefficients (phi): [ 0.6876 -0.1565  0.102  -0.1787]
Residual variance (sigma^2): 0.7354
Mean of shortened series (sbar): 0.2885
Length of memory-shortening filter (psi): 1

Time Series Summary Statistics
Mean: 0.2885
Std Dev: 1.1514
Min: -2.4900
25%: -0.4976
Median: 0.2289
75%: 1.0181
Max: 3.2281

Model Diagnostics
AIC: 224.5980
BIC: 239.1111
"""
    assert out == expected_output


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
    model.fit(y, exog=exog, suppress_warnings=True)
    model.summary()
    
    captured = capsys.readouterr().out
    
    # Check exact output
    expected_output = """Arar(lags=(1, 5, 8, 25)) Model Summary
------------------
Number of observations: 100
Selected AR lags: (1, 5, 8, 25)
AR coefficients (phi): [ 0.019   0.1824 -0.1076  0.1259]
Residual variance (sigma^2): 0.7978
Mean of shortened series (sbar): -0.1081
Length of memory-shortening filter (psi): 2

Time Series Summary Statistics
Mean: -6.2559
Std Dev: 5.1307
Min: -14.5761
25%: -9.6461
Median: -7.6509
75%: -3.1440
Max: 8.5967

Model Diagnostics
AIC: 213.2099
BIC: 233.9465

Exogenous Model (Linear Regression)
-----------------------------------
Number of features: 2
Intercept: -6.4133
Coefficients: [0.4044 2.1382]
"""
    assert captured == expected_output


def test_summary_is_shorter_after_reduce_memory(capsys):
    """
    Test that summary() output is shorter after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    model = Arar()
    model.fit(y)

    model.summary()
    captured = capsys.readouterr()
    assert "Arar(lags=(1, 4, 8, 20)) Model Summary" in captured.out
    assert "Time Series Summary Statistics" in captured.out
    
    model.reduce_memory()
    model.summary()
    captured = capsys.readouterr()
    assert "Arar(lags=(1, 4, 8, 20)) Model Summary" in captured.out
    assert "Time Series Summary Statistics" not in captured.out
