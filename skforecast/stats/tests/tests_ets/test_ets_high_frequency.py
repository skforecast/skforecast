"""
Tests for ETS models with high frequency (m > 24)

This module tests the handling of seasonal periods greater than 24,
which are not supported by ETS due to computational constraints.
The tests validate that:
1. auto_ets() correctly excludes seasonal models when m > 24
2. ets() with explicit seasonal models raises clear error messages
3. Ets class with model='ZZZ' works correctly for m > 24
4. Boundary case m=24 still allows seasonal models
"""

import pytest
import numpy as np
import warnings
from skforecast.stats._ets import ets, auto_ets, Ets


def test_auto_ets_excludes_seasonal_with_high_frequency():
    """
    Test that auto_ets() automatically excludes seasonal models when m > 24
    and fits non-seasonal models successfully.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    # Should fit non-seasonal model without errors
    model = auto_ets(y, m=30, seasonal=True, verbose=False)
    
    # Verify model is non-seasonal
    assert model.config.season == "N", "Model should be non-seasonal when m > 24"
    assert model.config.m == 30, "Seasonal period should be preserved"
    

def test_auto_ets_verbose_message_for_high_frequency():
    """
    Test that auto_ets() prints informative message when excluding seasonality
    due to high frequency.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    # Capture stdout to check verbose message
    import io
    import sys
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        model = auto_ets(y, m=30, seasonal=True, verbose=True)
        output = captured_output.getvalue()
    finally:
        sys.stdout = sys.__stdout__
    
    assert "Frequency too high" in output or "m=30" in output, \
        "Should print informative message about high frequency"


def test_ets_explicit_seasonal_model_high_frequency_raises_error():
    """
    Test that ets() with explicit seasonal model (e.g., 'AAA') and m > 24
    raises a clear ValueError with helpful message.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    with pytest.raises(ValueError) as excinfo:
        ets(y, m=30, model='AAA')
    
    error_msg = str(excinfo.value).lower()
    assert "frequency too high" in error_msg or "m=30" in error_msg, \
        "Error message should mention high frequency"
    assert "m>24" in error_msg or "24" in error_msg, \
        "Error message should mention the limit of 24"
    assert "zzz" in error_msg, \
        "Error message should suggest using ZZZ for automatic selection"


def test_ets_non_seasonal_model_high_frequency_works():
    """
    Test that ets() with non-seasonal model works fine even when m > 24.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    # Non-seasonal models should work regardless of m value
    model = ets(y, m=30, model='ANN')
    
    assert model.config.season == "N"
    assert model.config.m == 30
    assert model.aic is not None
    

def test_ets_class_zzz_high_frequency():
    """
    Test that Ets class with model='ZZZ' handles m > 24 correctly
    by automatically selecting non-seasonal models.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    estimator = Ets(m=30, model='ZZZ')
    estimator.fit(y)
    
    # Should fit non-seasonal model
    assert estimator.config_.season == "N"
    assert estimator.config_.m == 30
    
    # Should be able to forecast
    forecasts = estimator.predict(steps=10)
    assert len(forecasts) == 10
    assert np.all(np.isfinite(forecasts))


def test_ets_boundary_m24_allows_seasonal():
    """
    Test that m=24 (boundary case) still allows seasonal models.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    # m=24 should still allow seasonal models
    estimator = Ets(m=24, model='ZZZ')
    estimator.fit(y)
    
    # Could be seasonal or non-seasonal depending on data, 
    # but should not raise error
    assert estimator.config_.m == 24
    assert estimator.model_ is not None


def test_ets_boundary_m25_excludes_seasonal():
    """
    Test that m=25 (just over boundary) excludes seasonal models.
    """
    np.random.seed(42)
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 2 + np.random.normal(0, 0.1, 100)
    
    estimator = Ets(m=25, model='ZZZ')
    estimator.fit(y)
    
    # Should fit non-seasonal model when m=25
    assert estimator.config_.season == "N"
    assert estimator.config_.m == 25


def test_auto_ets_various_high_frequencies():
    """
    Test auto_ets with various high frequency values (m > 24).
    """
    np.random.seed(123)
    y = np.random.randn(200) + 10
    
    for m in [25, 30, 48, 168, 336]:
        model = auto_ets(y, m=m, seasonal=True, verbose=False)
        assert model.config.season == "N", f"Should be non-seasonal for m={m}"
        assert model.config.m == m


def test_ets_seasonal_model_names_high_frequency():
    """
    Test that various seasonal model specifications all raise errors for m > 24.
    """
    np.random.seed(42)
    y = np.random.randn(100) + 10
    
    seasonal_models = ['ANA', 'ANM', 'AAA', 'AAM', 'MAA', 'MAM', 'MNA', 'MNM']
    
    for model_spec in seasonal_models:
        with pytest.raises(ValueError) as excinfo:
            ets(y, m=30, model=model_spec)
        
        assert "24" in str(excinfo.value), \
            f"Error for model {model_spec} should mention the limit"


def test_auto_ets_respects_seasonal_false_high_frequency():
    """
    Test that auto_ets with seasonal=False works correctly for m > 24.
    """
    np.random.seed(42)
    y = np.random.randn(100) + 10
    
    # Should not try any seasonal models
    model = auto_ets(y, m=30, seasonal=False, verbose=False)
    
    assert model.config.season == "N"
    assert model.config.m == 30
