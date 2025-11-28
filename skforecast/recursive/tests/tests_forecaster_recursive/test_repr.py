# Unit test __repr__ and _repr_html_ ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterRecursive" in result_repr
    assert "LinearRegression" in result_repr
    assert "Estimator:" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterRecursive" in result_html
    assert "LinearRegression" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "https://skforecast.org" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags
    forecaster2 = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=[1, 2, 3, 5]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert "[1 2 3 5]" in result_repr2 or "1 2 3 5" in result_repr2
    
    # Test 3: Forecaster with differentiation and forecaster_id
    forecaster3 = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
        differentiation=1,
        forecaster_id="test_forecaster"
    )
    result_repr3 = repr(forecaster3)
    assert "Differentiation order: 1" in result_repr3
    assert "Forecaster id: test_forecaster" in result_repr3
    
    # Test 4: Fitted forecaster without exog
    forecaster4 = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3
    )
    y = pd.Series(np.random.rand(50), index=pd.date_range('2020-01-01', periods=50))
    forecaster4.fit(y)
    
    result_repr4 = repr(forecaster4)
    result_html4 = forecaster4._repr_html_()
    
    assert "Training range:" in result_repr4
    assert "Training range: None" not in result_repr4
    assert "Last fit date:" in result_repr4
    assert "Last fit date: None" not in result_repr4
    assert "2020" in result_html4
    
    # Test 5: Fitted forecaster with exog
    forecaster5 = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )
    y = pd.Series(np.random.rand(50), index=pd.date_range('2020-01-01', periods=50))
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50))
    
    forecaster5.fit(y, exog=exog)
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Exogenous included: True" in result_repr5
    assert "exog_1" in result_repr5 or "Exogenous names:" in result_repr5
    assert "StandardScaler" in result_repr5
    assert "Exogenous included" in result_html5 or "Exogenous Variables" in result_html5
    assert "True" in result_html5
    
    # Test 6: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Exogenous Variables",
        "Data Transformations",
        "Training Information",
        "Estimator Parameters"
    ]
    
    for section in expected_sections:
        assert section in result_html5, f"Section '{section}' not found in HTML output"
