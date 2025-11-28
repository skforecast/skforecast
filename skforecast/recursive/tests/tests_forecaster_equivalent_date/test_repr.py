# Unit test __repr__ and _repr_html_ ForecasterEquivalentDate
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterEquivalentDate


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster with integer offset - not fitted
    forecaster = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterEquivalentDate" in result_repr
    assert "Offset: 7" in result_repr
    assert "Number of offsets: 1" in result_repr
    assert "Aggregation function:" in result_repr
    assert "Window size:" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterEquivalentDate" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with multiple offsets
    forecaster2 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=3,
        agg_func=np.mean
    )
    result_repr2 = repr(forecaster2)
    assert "Number of offsets: 3" in result_repr2
    assert "Window size: 21" in result_repr2
    assert "mean" in result_repr2
    
    # Test 3: Forecaster with different aggregation function
    forecaster3 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=2,
        agg_func=np.median
    )
    result_repr3 = repr(forecaster3)
    assert "median" in result_repr3
    
    # Test 4: Forecaster with forecaster_id
    forecaster4 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1,
        forecaster_id="test_equivalent_date_forecaster"
    )
    result_repr4 = repr(forecaster4)
    assert "Forecaster id: test_equivalent_date_forecaster" in result_repr4
    
    # Test 5: Fitted forecaster with integer offset
    forecaster5 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    y = pd.Series(
        np.random.rand(100), 
        index=pd.date_range('2020-01-01', periods=100, freq='D'),
        name='y'
    )
    forecaster5.fit(y)
    
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Training range:" in result_repr5
    assert "Training range: None" not in result_repr5
    assert "Last fit date:" in result_repr5
    assert "Last fit date: None" not in result_repr5
    assert "2020" in result_html5
    assert "Series name:" in result_repr5
    
    # Test 6: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Training Information"
    ]
    
    for section in expected_sections:
        assert section in result_html5, f"Section '{section}' not found in HTML output"
    
    # Test 7: Forecaster with DateOffset
    forecaster7 = ForecasterEquivalentDate(
        offset=pd.DateOffset(weeks=1),
        n_offsets=2,
        agg_func=np.mean
    )
    result_repr7 = repr(forecaster7)
    assert "Offset:" in result_repr7
    assert "Number of offsets: 2" in result_repr7
    
    # Test 8: Fitted forecaster with DateOffset
    forecaster8 = ForecasterEquivalentDate(
        offset=pd.DateOffset(weeks=1),
        n_offsets=1
    )
    y = pd.Series(
        np.random.rand(100), 
        index=pd.date_range('2020-01-01', periods=100, freq='D'),
        name='y'
    )
    forecaster8.fit(y)
    
    result_repr8 = repr(forecaster8)
    result_html8 = forecaster8._repr_html_()
    
    assert "Training range:" in result_repr8
    assert "Training index type:" in result_repr8
    assert "Training index frequency:" in result_repr8
    assert "DatetimeIndex" in result_repr8
    assert "2020" in result_html8
    
    # Test 9: Verify window size calculation
    forecaster9 = ForecasterEquivalentDate(
        offset=5,
        n_offsets=4
    )
    result_repr9 = repr(forecaster9)
    assert "Window size: 20" in result_repr9
    
    # Test 10: Forecaster with custom binner_kwargs
    forecaster10 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1,
        binner_kwargs={'n_bins': 5, 'method': 'linear'}
    )
    result_repr10 = repr(forecaster10)
    assert "ForecasterEquivalentDate" in result_repr10
    
    # Test 11: Verify creation date and skforecast version appear
    forecaster11 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    result_repr11 = repr(forecaster11)
    result_html11 = forecaster11._repr_html_()
    
    assert "Creation date:" in result_repr11
    assert "Skforecast version:" in result_repr11
    assert "Python version:" in result_repr11
    assert "Creation date" in result_html11
    assert "Skforecast version" in result_html11
    
    # Test 12: Fitted forecaster - verify series name is captured
    forecaster12 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    y = pd.Series(
        np.random.rand(50), 
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='my_custom_series'
    )
    forecaster12.fit(y)
    
    result_repr12 = repr(forecaster12)
    assert "Series name: my_custom_series" in result_repr12
    
    # Test 13: Forecaster with Business Day offset
    forecaster13 = ForecasterEquivalentDate(
        offset=pd.offsets.BDay(5),
        n_offsets=2
    )
    result_repr13 = repr(forecaster13)
    assert "Offset:" in result_repr13
    assert "Number of offsets: 2" in result_repr13
    
    # Test 14: Verify aggregation functions display correctly
    for agg_func in [np.min, np.max, np.sum]:
        forecaster14 = ForecasterEquivalentDate(
            offset=7,
            n_offsets=2,
            agg_func=agg_func
        )
        result_repr14 = repr(forecaster14)
        assert agg_func.__name__ in result_repr14
    
    # Test 15: Verify Training index type for RangeIndex
    forecaster15 = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    y = pd.Series(
        np.random.rand(50), 
        index=pd.RangeIndex(start=0, stop=50, step=1),
        name='y'
    )
    forecaster15.fit(y)
    
    result_repr15 = repr(forecaster15)
    assert "Training index type: RangeIndex" in result_repr15
