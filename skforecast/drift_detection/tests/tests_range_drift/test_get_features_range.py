# Unit test _get_features_range
# ==============================================================================
import pytest
import pandas as pd
import numpy as np
from skforecast.drift_detection import RangeDriftDetector


def test_get_features_range_TypeError_when_input_not_DataFrame_or_Series():
    """
    Test that _get_features_range raises TypeError when input is not DataFrame or Series.
    """
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame or Series."):
        RangeDriftDetector._get_features_range("invalid_input")

    with pytest.raises(TypeError, match="Input must be a pandas DataFrame or Series."):
        RangeDriftDetector._get_features_range([1, 2, 3])

    with pytest.raises(TypeError, match="Input must be a pandas DataFrame or Series."):
        RangeDriftDetector._get_features_range(123)

def test_get_features_range_Series_numeric():
    """
    Test _get_features_range with numeric Series.
    """
    # Test with integer series
    series = pd.Series([1, 2, 3, 4, 5], name='numeric_series')
    result = RangeDriftDetector._get_features_range(series)
    expected = (1, 5)
    assert result == expected

    # Test with float series
    series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], name='float_series')
    result = RangeDriftDetector._get_features_range(series)
    expected = (1.1, 5.5)
    assert result == expected

    # Test with negative values
    series = pd.Series([-5, -2, 0, 3, 7], name='negative_series')
    result = RangeDriftDetector._get_features_range(series)
    expected = (-5, 7)
    assert result == expected


def test_get_features_range_Series_categorical():
    """
    Test _get_features_range with categorical Series.
    """
    # Test with string series
    series = pd.Series(['a', 'b', 'c', 'a', 'b'], name='string_series')
    result = RangeDriftDetector._get_features_range(series)
    expected = {'a', 'b', 'c'}
    assert result == expected

    # Test with mixed types (treated as categorical)
    series = pd.Series([1, 'a', 2, 'b'], name='mixed_series')
    result = RangeDriftDetector._get_features_range(series)
    expected = {1, 'a', 2, 'b'}
    assert result == expected

def test_get_features_range_Series_with_NaN():
    """
    Test _get_features_range with Series containing NaN values.
    # NaN should be ignored.
    """
    # Numeric series with NaN
    series = pd.Series([1, 2, np.nan, 4, 5], name='numeric_with_nan')
    result = RangeDriftDetector._get_features_range(series)
    expected = (1, 5) 
    assert result == expected

    # Categorical series with NaN
    series = pd.Series(['a', 'b', np.nan, 'c'], name='categorical_with_nan')
    result = RangeDriftDetector._get_features_range(series)
    expected = {'a', 'b', 'c'}
    assert result == expected

def test_get_features_range_DataFrame_mixed_types():
    """
    Test _get_features_range with DataFrame containing mixed column types.
    """
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'categorical_col': ['a', 'b', 'c', 'a', 'b'],
        'mixed_col': [1, 'x', 2, 'y', 3],  # Treated as categorical
        'numeric_col_nan': (1, 2, 3, 5, np.nan),
        'categorical_col_nan': ['a', 'b', 'c', 'a',  np.nan]
    })

    result = RangeDriftDetector._get_features_range(df)
    expected = {
        'numeric_col': (1, 5),
        'float_col': (1.1, 5.5),
        'numeric_col_nan': (1, 5),
        'categorical_col': {'a', 'b', 'c'},
        'mixed_col': {1, 'x', 2, 'y', 3},
        'categorical_col_nan': {'a', 'b', 'c'}
    }
    assert result == expected

def test_get_features_range_empty_Series():
    """
    Test _get_features_range with empty Series.
    """
    # Empty numeric series
    series = pd.Series([], dtype='float64', name='empty_numeric')
    result = RangeDriftDetector._get_features_range(series)
    expected = (np.nan, np.nan)
    assert pd.isna(result[0]) and pd.isna(result[1])

    # Empty categorical series
    series = pd.Series([], dtype='object', name='empty_categorical')
    result = RangeDriftDetector._get_features_range(series)
    expected = set()
    assert result == expected

def test_get_features_range_empty_DataFrame():
    """
    Test _get_features_range with empty DataFrame.
    """
    df = pd.DataFrame({
        'numeric_col': pd.Series([], dtype='float64'),
        'categorical_col': pd.Series([], dtype='object')
    })

    result = RangeDriftDetector._get_features_range(df)
    expected = {
        'numeric_col': (np.nan, np.nan),
        'categorical_col': set()
    }
    assert pd.isna(result['numeric_col'][0]) and pd.isna(result['numeric_col'][1])
    assert result['categorical_col'] == set()