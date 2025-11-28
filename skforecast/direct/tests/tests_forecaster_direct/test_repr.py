# Unit test __repr__ and _repr_html_ ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirect


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterDirect(
        estimator=LinearRegression(),
        steps=5,
        lags=3
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterDirect" in result_repr
    assert "LinearRegression" in result_repr
    assert "Estimator:" in result_repr
    assert "Maximum steps to predict: 5" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterDirect" in result_html
    assert "LinearRegression" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "Maximum steps to predict" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags and steps
    forecaster2 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=10,
        lags=[1, 2, 3, 5, 7]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert "[1 2 3 5 7]" in result_repr2 or "1 2 3 5 7" in result_repr2
    assert "Maximum steps to predict: 10" in result_repr2
    
    # Test 3: Forecaster with differentiation, n_jobs and forecaster_id
    forecaster3 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=3,
        lags=5,
        differentiation=1,
        n_jobs=2,
        forecaster_id="test_direct_forecaster"
    )
    result_repr3 = repr(forecaster3)
    assert "Differentiation order: 1" in result_repr3
    assert "Forecaster id: test_direct_forecaster" in result_repr3
    assert forecaster3.n_jobs == 2
    
    # Test 4: Fitted forecaster without exog
    forecaster4 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=4,
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
    forecaster5 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=3,
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
    
    # Test 7: Test with n_jobs='auto'
    forecaster7 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=2,
        lags=3,
        n_jobs='auto'
    )
    result_repr7 = repr(forecaster7)
    assert isinstance(result_repr7, str)
    assert "ForecasterDirect" in result_repr7
    
    # Test 8: Test with fit_kwargs
    forecaster8 = ForecasterDirect(
        estimator=LinearRegression(),
        steps=3,
        lags=2,
        fit_kwargs={'sample_weight': None}
    )
    result_repr8 = repr(forecaster8)
    assert "fit_kwargs:" in result_repr8
