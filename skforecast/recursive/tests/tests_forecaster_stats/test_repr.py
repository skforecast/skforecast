# Unit test __repr__ and _repr_html_ ForecasterStats
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterStats
from skforecast.stats import Sarimax, Arima


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterStats" in result_repr
    assert "Estimators:" in result_repr
    assert "Series name:" in result_repr
    assert "Exogenous included:" in result_repr
    assert "Transformer for y:" in result_repr
    assert "Transformer for exog:" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterStats" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with forecaster_id
    forecaster2 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0)),
        forecaster_id="test_stats_forecaster"
    )
    result_repr2 = repr(forecaster2)
    assert "Forecaster id: test_stats_forecaster" in result_repr2
    
    # Test 3: Fitted forecaster without exog
    forecaster3 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster3.fit(y)
    
    result_repr3 = repr(forecaster3)
    result_html3 = forecaster3._repr_html_()
    
    assert "Training range:" in result_repr3
    assert "Training range: None" not in result_repr3
    assert "Last fit date:" in result_repr3
    assert "Last fit date: None" not in result_repr3
    assert "2020" in result_html3
    assert "Series name: y" in result_repr3
    
    # Test 4: Fitted forecaster with exog
    forecaster4 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0)),
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    
    forecaster4.fit(y, exog=exog)
    result_repr4 = repr(forecaster4)
    result_html4 = forecaster4._repr_html_()
    
    assert "Exogenous included: True" in result_repr4
    assert "exog_1" in result_repr4 or "Exogenous names:" in result_repr4
    assert "StandardScaler" in result_repr4
    assert "Transformer for y:" in result_repr4
    assert "Transformer for exog:" in result_repr4
    assert "Exogenous" in result_html4 or "Exogenous Variables" in result_html4
    assert "True" in result_html4
    
    # Test 5: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Exogenous Variables",
        "Data Transformations",
        "Training Information",
        "Estimator Parameters",
        "Fit Kwargs"
    ]
    
    for section in expected_sections:
        assert section in result_html4, f"Section '{section}' not found in HTML output"
    
    # Test 6: Verify creation date and skforecast version appear
    forecaster6 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    result_repr6 = repr(forecaster6)
    result_html6 = forecaster6._repr_html_()
    
    assert "Creation date:" in result_repr6
    assert "Skforecast version:" in result_repr6
    assert "Python version:" in result_repr6
    assert "Creation date" in result_html6
    assert "Skforecast version" in result_html6
    
    # Test 7: Fitted forecaster - verify series name is captured
    forecaster7 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='my_custom_series'
    )
    forecaster7.fit(y)
    
    result_repr7 = repr(forecaster7)
    assert "Series name: my_custom_series" in result_repr7
    
    # Test 8: Verify training index type for DatetimeIndex
    forecaster8 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster8.fit(y)
    
    result_repr8 = repr(forecaster8)
    assert "Training index type: DatetimeIndex" in result_repr8
    assert "Training index frequency:" in result_repr8
    
    # Test 9: Forecaster with different SARIMAX orders
    forecaster9 = ForecasterStats(
        estimator=Sarimax(order=(2, 1, 1), seasonal_order=(1, 0, 1, 12))
    )
    result_repr9 = repr(forecaster9)
    assert "ForecasterStats" in result_repr9
    assert "Estimator parameters:" in result_repr9
    
    # Test 10: Verify extended_index_ appears in repr after fitting
    forecaster10 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster10.fit(y)
    
    result_repr10 = repr(forecaster10)
    assert "Index seen by the forecaster:" in result_repr10
    
    # Test 11: Verify fit_kwargs appears in repr
    forecaster11 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    result_repr11 = repr(forecaster11)
    assert "fit_kwargs:" in result_repr11
    
    # Test 12: Verify window_size appears in HTML
    forecaster12 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    result_html12 = forecaster12._repr_html_()
    assert "Window size" in result_html12
    
    # Test 13: Verify Exogenous included is False when no exog
    forecaster13 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster13.fit(y)
    
    result_repr13 = repr(forecaster13)
    assert "Exogenous included: False" in result_repr13
    assert "Exogenous names: None" in result_repr13
    
    # Test 14: Verify transformer_y and transformer_exog are None when not set
    forecaster14 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    result_repr14 = repr(forecaster14)
    assert "Transformer for y: None" in result_repr14
    assert "Transformer for exog: None" in result_repr14
    
    # Test 15: Fitted forecaster with only transformer_y
    forecaster15 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0)),
        transformer_y=StandardScaler()
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster15.fit(y)
    
    result_repr15 = repr(forecaster15)
    assert "Transformer for y: StandardScaler()" in result_repr15
    assert "Transformer for exog: None" in result_repr15
    
    # Test 16: Verify params display correctly (long params)
    forecaster16 = ForecasterStats(
        estimator=Sarimax(
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 12),
            trend='c'
        )
    )
    result_repr16 = repr(forecaster16)
    assert "Estimator parameters:" in result_repr16
    
    # Test 17: Verify HTML contains link to documentation
    forecaster17 = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 0))
    )
    result_html17 = forecaster17._repr_html_()
    assert "API Reference" in result_html17
    assert "User Guide" in result_html17
    assert "forecasting-sarimax-arima" in result_html17


def test_repr_and_repr_html_multiple_estimators():
    """
    Test for __repr__ and _repr_html_ methods with multiple estimators.
    """
    # Test 1: Multiple estimators - not fitted
    forecaster = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Arima(order=(2, 1, 1))
        ]
    )
    
    result_repr = repr(forecaster)
    result_html = forecaster._repr_html_()
    
    assert "ForecasterStats" in result_repr
    assert "Estimators:" in result_repr
    assert "skforecast.Sarimax" in result_repr
    assert "skforecast.Arima" in result_repr
    assert "ForecasterStats" in result_html
    assert "skforecast.Sarimax" in result_html
    assert "skforecast.Arima" in result_html
    
    # Verify estimator_ids are generated correctly
    assert len(forecaster.estimator_ids) == 2
    assert forecaster.n_estimators == 2
    
    # Test 2: Multiple estimators with same type - check unique IDs
    forecaster2 = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Sarimax(order=(2, 1, 1)),
            Arima(order=(1, 1, 1))
        ]
    )
    
    result_repr2 = repr(forecaster2)
    result_html2 = forecaster2._repr_html_()
    
    assert forecaster2.n_estimators == 3
    assert len(forecaster2.estimator_ids) == 3
    # Check that duplicate estimator types get unique IDs
    assert "skforecast.Sarimax" in result_repr2
    assert "skforecast.Sarimax_2" in result_repr2
    assert "skforecast.Arima" in result_repr2
    
    # Test 3: Multiple estimators - fitted without exog
    forecaster3 = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Arima(order=(1, 0, 1))
        ]
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='test_series'
    )
    forecaster3.fit(y)
    
    result_repr3 = repr(forecaster3)
    result_html3 = forecaster3._repr_html_()
    
    assert "Training range:" in result_repr3
    assert "Training range: None" not in result_repr3
    assert "Series name: test_series" in result_repr3
    assert "skforecast.Sarimax" in result_repr3
    assert "skforecast.Arima" in result_repr3
    assert "2020" in result_html3
    
    # Verify estimator_names_ are populated after fitting
    assert forecaster3.estimator_names_[0] is not None
    assert forecaster3.estimator_names_[1] is not None
    
    # Test 4: Multiple estimators - fitted with exog
    forecaster4 = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Arima(order=(1, 0, 1))
        ],
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    
    forecaster4.fit(y, exog=exog)
    result_repr4 = repr(forecaster4)
    result_html4 = forecaster4._repr_html_()
    
    assert "Exogenous included: True" in result_repr4
    assert "exog_1" in result_repr4 or "Exogenous names:" in result_repr4
    assert "StandardScaler" in result_repr4
    assert "skforecast.Sarimax" in result_repr4
    assert "skforecast.Arima" in result_repr4
    assert "Exogenous Variables" in result_html4
    
    # Test 5: Multiple estimators with forecaster_id
    forecaster5 = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Arima(order=(1, 0, 1))
        ],
        forecaster_id="multi_estimator_forecaster"
    )
    result_repr5 = repr(forecaster5)
    assert "Forecaster id: multi_estimator_forecaster" in result_repr5
    
    # Test 6: Verify HTML sections with multiple estimators
    forecaster6 = ForecasterStats(
        estimator=[
            Sarimax(order=(1, 0, 0)),
            Arima(order=(1, 0, 1))
        ]
    )
    y = pd.Series(
        np.random.rand(50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster6.fit(y)
    
    result_html6 = forecaster6._repr_html_()
    
    expected_sections = [
        "General Information",
        "Exogenous Variables",
        "Data Transformations",
        "Training Information",
        "Estimator Parameters",
        "Fit Kwargs"
    ]
    
    for section in expected_sections:
        assert section in result_html6, f"Section '{section}' not found in HTML output"
    
    # Verify HTML contains list items for each estimator
    assert "<li>" in result_html6
    assert "skforecast.Sarimax" in result_html6
    assert "skforecast.Arima" in result_html6
    
    # Test 7: Estimator parameters for multiple estimators
    forecaster7 = ForecasterStats(
        estimator=[
            Sarimax(order=(2, 1, 1), seasonal_order=(1, 0, 1, 12)),
            Arima(order=(1, 1, 1))
        ]
    )
    result_repr7 = repr(forecaster7)
    
    assert "Estimator parameters:" in result_repr7
    # Verify parameters for each estimator are shown
    assert "skforecast.Sarimax:" in result_repr7
    assert "skforecast.Arima:" in result_repr7
