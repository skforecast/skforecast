# Unit test __repr__ and _repr_html_ ForecasterDirectMultiVariate
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=5,
        lags=3
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterDirectMultiVariate" in result_repr
    assert "LinearRegression" in result_repr
    assert "Estimator:" in result_repr
    assert "Target series (level): serie_1" in result_repr
    assert "Maximum steps to predict: 5" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterDirectMultiVariate" in result_html
    assert "LinearRegression" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "Target series (level)" in result_html
    assert "serie_1" in result_html
    assert "Maximum steps to predict" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags and steps
    forecaster2 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='target',
        steps=10,
        lags=[1, 2, 3, 5, 7]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert "Target series (level): target" in result_repr2
    assert "Maximum steps to predict: 10" in result_repr2
    
    # Test 3: Forecaster with dict lags
    forecaster3 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=3,
        lags={'serie_1': 3, 'serie_2': [1, 2, 5]}
    )
    result_repr3 = repr(forecaster3)
    assert "Lags:" in result_repr3
    assert isinstance(result_repr3, str)
    
    # Test 4: Forecaster with differentiation, n_jobs and forecaster_id
    forecaster4 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='y',
        steps=3,
        lags=5,
        differentiation=1,
        n_jobs=2,
        forecaster_id="test_multivariate_forecaster",
        transformer_series=None  # Override default StandardScaler
    )
    result_repr4 = repr(forecaster4)
    assert "Differentiation order: 1" in result_repr4
    assert "Forecaster id: test_multivariate_forecaster" in result_repr4
    assert forecaster4.n_jobs == 2
    
    # Test 5: Fitted forecaster without exog
    forecaster5 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=4,
        lags=3,
        transformer_series=StandardScaler()
    )
    series = pd.DataFrame({
        'serie_1': np.random.rand(50),
        'serie_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50))
    forecaster5.fit(series=series)
    
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Training range:" in result_repr5
    assert "Training range: None" not in result_repr5
    assert "Last fit date:" in result_repr5
    assert "Last fit date: None" not in result_repr5
    assert "2020" in result_html5
    assert "Multivariate series:" in result_repr5
    
    # Test 6: Fitted forecaster with exog
    forecaster6 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=3,
        lags=3,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler()
    )
    series = pd.DataFrame({
        'serie_1': np.random.rand(50),
        'serie_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50))
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50))
    
    forecaster6.fit(series=series, exog=exog)
    result_repr6 = repr(forecaster6)
    result_html6 = forecaster6._repr_html_()
    
    assert "Exogenous included: True" in result_repr6
    assert "exog_1" in result_repr6 or "Exogenous names:" in result_repr6
    assert "StandardScaler" in result_repr6
    assert "Transformer for series:" in result_repr6
    assert "Exogenous included" in result_html6 or "Exogenous Variables" in result_html6
    assert "True" in result_html6
    
    # Test 7: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Exogenous Variables",
        "Data Transformations",
        "Training Information",
        "Estimator Parameters"
    ]
    
    for section in expected_sections:
        assert section in result_html6, f"Section '{section}' not found in HTML output"
    
    # Test 8: Test with n_jobs='auto'
    forecaster8 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='data',
        steps=2,
        lags=3,
        n_jobs='auto'
    )
    result_repr8 = repr(forecaster8)
    assert isinstance(result_repr8, str)
    assert "ForecasterDirectMultiVariate" in result_repr8
    assert "Target series (level): data" in result_repr8
    
    # Test 9: Test with transformer_series as dict
    forecaster9 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=3,
        lags=2,
        transformer_series={'serie_1': StandardScaler(), 'serie_2': None}
    )
    result_repr9 = repr(forecaster9)
    assert "Transformer for series:" in result_repr9
    
    # Test 10: Test with fit_kwargs
    forecaster10 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='y',
        steps=3,
        lags=2,
        fit_kwargs={'sample_weight': None}
    )
    result_repr10 = repr(forecaster10)
    assert "fit_kwargs:" in result_repr10
    
    # Test 11: Verify level appears in both repr and html
    forecaster11 = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='my_target_series',
        steps=5,
        lags=3
    )
    result_repr11 = repr(forecaster11)
    result_html11 = forecaster11._repr_html_()
    assert "my_target_series" in result_repr11
    assert "my_target_series" in result_html11
