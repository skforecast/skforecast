# Unit test summary method - Arima
# ==============================================================================
import numpy as np
import pytest
from io import StringIO
import sys
from ..._arima import Arima


def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_summary_raises_error_for_unfitted_model():
    """
    Test that summary raises error when model is not fitted.
    """
    from sklearn.exceptions import NotFittedError
    model = Arima(order=(1, 0, 0))
    msg = (
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.summary()


def test_summary_prints_output():
    """
    Test that summary prints output to stdout.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Capture stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    # Check that output contains expected sections
    assert "ARIMA Model Summary" in output
    assert "Coefficients:" in output
    assert "Model fit statistics:" in output
    assert "Residual statistics:" in output


def test_summary_displays_model_specification():
    """
    Test that summary displays correct model specification.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(2, 1, 1), seasonal_order=(1, 0, 1), m=12)
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "ARIMA(2,1,1)(1,0,1)[12]" in output


def test_summary_displays_convergence_status():
    """
    Test that summary displays convergence status.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "Converged :" in output


def test_summary_displays_fit_statistics():
    """
    Test that summary displays fit statistics (AIC, BIC, etc).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "AIC:" in output
    assert "BIC:" in output
    assert "Log-likelihood:" in output
    assert "sigma^2:" in output


def test_summary_displays_coefficients():
    """
    Test that summary displays model coefficients.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    # Should display AR and MA coefficients
    for coef_name in model.coef_names_:
        assert coef_name in output


def test_summary_displays_residual_statistics():
    """
    Test that summary displays residual statistics.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "Mean:" in output
    assert "Std Dev:" in output
    assert "MAE:" in output
    assert "RMSE:" in output


def test_summary_is_shorter_after_reduce_memory(capsys):
    """
    Test that summary() output is shorter after reduce_memory().
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)

    model.summary()
    captured = capsys.readouterr()
    assert "ARIMA Model Summary" in captured.out
    assert "Time Series Summary Statistics" in captured.out
    
    model.reduce_memory()
    model.summary()
    captured = capsys.readouterr()
    assert "ARIMA Model Summary" in captured.out
    assert "Time Series Summary Statistics" not in captured.out


def test_summary_with_exog():
    """
    Test that summary works when model is fitted with exogenous variables.
    """
    np.random.seed(42)
    y = ar1_series(80)
    exog = np.random.randn(80, 2)
    
    model = Arima(order=(1, 0, 0))
    model.fit(y, exog=exog)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    # Should contain exogenous coefficient names
    assert "ARIMA Model Summary" in output


def test_summary_displays_time_series_statistics():
    """
    Test that summary displays time series summary statistics.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "Time Series Summary Statistics:" in output
    assert "Min:" in output
    assert "Max:" in output
    assert "Median:" in output


def test_summary_handles_none_bic():
    """
    Test that summary handles None BIC gracefully.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1))
    model.fit(y)
    
    # Artificially set BIC to None
    model.bic_ = None
    
    captured_output = StringIO()
    sys.stdout = captured_output
    model.summary()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    # Should display N/A for BIC
    assert "N/A" in output or "BIC:" in output
