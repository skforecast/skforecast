# Unit test __repr__ and _repr_html_ ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterRecursiveMultiSeries" in result_repr
    assert "LinearRegression" in result_repr
    assert "Estimator:" in result_repr
    assert "Lags:" in result_repr
    assert "Window size:" in result_repr
    assert "Series encoding:" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterRecursiveMultiSeries" in result_html
    assert "LinearRegression" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "Series encoding" in result_html
    assert "https://skforecast.org" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags
    forecaster2 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=[1, 2, 3, 5, 7]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert isinstance(result_repr2, str)
    
    # Test 3: Forecaster with different encoding options
    for encoding in ['ordinal', 'ordinal_category', 'onehot', None]:
        forecaster3 = ForecasterRecursiveMultiSeries(
            estimator=LinearRegression(),
            lags=3,
            encoding=encoding
        )
        result_repr3 = repr(forecaster3)
        assert "Series encoding:" in result_repr3
        if encoding is not None:
            assert encoding in result_repr3
    
    # Test 4: Forecaster with differentiation and forecaster_id
    forecaster4 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=5,
        differentiation=1,
        forecaster_id="test_multiseries_forecaster"
    )
    result_repr4 = repr(forecaster4)
    assert "Differentiation order: 1" in result_repr4
    assert "Forecaster id: test_multiseries_forecaster" in result_repr4
    
    # Test 5: Fitted forecaster without exog
    forecaster5 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        transformer_series=StandardScaler()
    )
    series = pd.DataFrame({
        'serie_1': np.random.rand(50),
        'serie_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    forecaster5.fit(series=series)
    
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Training range:" in result_repr5
    assert "Training range: None" not in result_repr5
    assert "Last fit date:" in result_repr5
    assert "Last fit date: None" not in result_repr5
    assert "2020" in result_html5
    assert "Series names (levels):" in result_repr5
    
    # Test 6: Fitted forecaster with exog
    forecaster6 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler()
    )
    series = pd.DataFrame({
        'serie_1': np.random.rand(50),
        'serie_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    
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
    
    # Test 8: Forecaster with weight_func and series_weights
    def custom_weight_func(index):
        return np.ones(len(index))
    
    forecaster8 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        weight_func=custom_weight_func,
        series_weights={'serie_1': 2.0, 'serie_2': 1.0}
    )
    result_repr8 = repr(forecaster8)
    assert "Weight function included: True" in result_repr8
    assert "Series weights:" in result_repr8
    
    # Test 9: Test with transformer_series as dict
    forecaster9 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=2,
        transformer_series={
            'serie_1': StandardScaler(), 
            'serie_2': None,
            '_unknown_level': StandardScaler()
        }
    )
    result_repr9 = repr(forecaster9)
    assert "Transformer for series:" in result_repr9
    
    # Test 10: Test with fit_kwargs
    forecaster10 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=2,
        fit_kwargs={'sample_weight': None}
    )
    result_repr10 = repr(forecaster10)
    assert "fit_kwargs:" in result_repr10
    
    # Test 11: Test with differentiation as dict
    forecaster11 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        differentiation={'serie_1': 1, 'serie_2': 2, '_unknown_level': 1}
    )
    result_repr11 = repr(forecaster11)
    assert "Differentiation order:" in result_repr11
    
    # Test 12: Verify window_features appear in repr
    from skforecast.preprocessing import RollingFeatures
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=3)
    forecaster12 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        window_features=rolling
    )
    result_repr12 = repr(forecaster12)
    result_html12 = forecaster12._repr_html_()
    assert "Window features:" in result_repr12
    assert "Window features" in result_html12
    
    # Test 13: Test dropna_from_series parameter
    forecaster13 = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        dropna_from_series=True
    )
    assert forecaster13.dropna_from_series is True
    result_repr13 = repr(forecaster13)
    assert isinstance(result_repr13, str)
