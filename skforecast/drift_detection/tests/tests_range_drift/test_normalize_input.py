# Unit test _normalize_input
# ==============================================================================
import re
import pytest
import pandas as pd
import numpy as np
from skforecast.drift_detection import RangeDriftDetector
from skforecast.exceptions import IgnoredArgumentWarning


def test_normalize_input_Series_without_name():
    """
    Test _normalize_input with a pandas Series without a name raises ValueError.
    """
    series = pd.Series([1, 2, 3])
    msg = "test must have a name when a pandas Series is provided."
    with pytest.raises(ValueError, match=msg):
        RangeDriftDetector._normalize_input(series, name="test")

def test_normalize_input_Series_with_name():
    """
    Test _normalize_input with a pandas Series that has a name.
    """
    series = pd.Series([1, 2, 3], name='y')
    result = RangeDriftDetector._normalize_input(series, name="series")
    expected = {'y': series}
    assert result == expected

def test_normalize_input_DataFrame_no_multiindex():
    """
    Test _normalize_input with a pandas DataFrame without MultiIndex.
    """
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    result = RangeDriftDetector._normalize_input(df, name="series")
    expected = {'col1': pd.Series([1, 2], name='col1'), 'col2': pd.Series([3, 4], name='col2')}
    assert result.keys() == expected.keys()
    for key in result:
        pd.testing.assert_series_equal(result[key], expected[key])


def test_normalize_input_DataFrame_multiindex_series():
    """
    Test _normalize_input with a pandas DataFrame with MultiIndex for 'series'.
    """
    index = pd.MultiIndex.from_tuples(
        [('series_1', 0), ('series_1', 1), ('series_2', 0), ('series_2', 1)],
        names=['series', 'time']
    )
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index)
    result = RangeDriftDetector._normalize_input(df, name="series")
    expected = {
        'series_1': pd.Series([1, 2], name='series_1'),
        'series_2': pd.Series([3, 4], name='series_2')
    }
    assert result.keys() == expected.keys()
    for key in result:
        pd.testing.assert_series_equal(result[key], expected[key], check_names=False)


def test_normalize_input_DataFrame_multiindex_last_window():
    """
    Test _normalize_input with a pandas DataFrame with MultiIndex for 'last_window'.
    """
    index = pd.MultiIndex.from_tuples(
        [('series_1', 0), ('series_1', 1), ('series_2', 0), ('series_2', 1)],
        names=['series', 'datetime']
    )
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index)
    result = RangeDriftDetector._normalize_input(df, name="last_window")
    expected = {
        'series_1': pd.Series([1, 2], name='series_1').rename_axis('datetime', axis='index'),
        'series_2': pd.Series([3, 4], name='series_2').rename_axis('datetime', axis='index')
    }
    assert result.keys() == expected.keys()
    for key in result:
        pd.testing.assert_series_equal(result[key], expected[key])


def test_normalize_input_DataFrame_multiindex_exog():
    """
    Test _normalize_input with a pandas DataFrame with MultiIndex for 'exog'.
    """
    index = pd.MultiIndex.from_tuples(
        [('series_1', 0), ('series_1', 1), ('series_2', 0), ('series_2', 1)],
        names=['series', 'datetime']
    )
    df = pd.DataFrame({'exog1': [1, 2, 3, 4], 'exog2': [5, 6, 7, 8]}, index=index)
    result = RangeDriftDetector._normalize_input(df, name="exog")
    expected = {
        'series_1': pd.DataFrame({'exog1': [1, 2], 'exog2': [5, 6]}, index=pd.Index([0, 1], name='datetime')),
        'series_2': pd.DataFrame({'exog1': [3, 4], 'exog2': [7, 8]}, index=pd.Index([0, 1], name='datetime'))
    }
    assert result.keys() == expected.keys()
    for key in result:
        pd.testing.assert_frame_equal(result[key], expected[key])


def test_normalize_warning_input_DataFrame_multiindex_series_multiple_columns():
    """
    Test _normalize_input with MultiIndex DataFrame for 'series' with multiple
    columns, should warn and use first.
    """
    index = pd.MultiIndex.from_tuples(
        [("series_1", 0), ("series_1", 1), ("series_2", 0), ("series_2", 1)],
        names=['series', 'datetime']
    )
    df = pd.DataFrame({'value1': [1, 2, 3, 4], 'value2': [5, 6, 7, 8]}, index=index)
    msg = re.escape(
        "`series` DataFrame has multiple columns. Only the "
        "first column, 'value1', will be used. Others ignored.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=msg):
        result = RangeDriftDetector._normalize_input(df, name="series")
    expected = {
        'series_1': pd.Series([1, 2], name='series_1').rename_axis('datetime', axis='index'),
        'series_2': pd.Series([3, 4], name='series_2').rename_axis('datetime', axis='index')
    }
    assert result.keys() == expected.keys()
    for key in result:
        pd.testing.assert_series_equal(result[key], expected[key])


def test_normalize_input_dict_valid():
    """
    Test _normalize_input with a valid dict.
    """
    series1 = pd.Series([1, 2], name='s1')
    series2 = pd.Series([3, 4], name='s2')
    data = {'key1': series1, 'key2': series2}
    result = RangeDriftDetector._normalize_input(data, name="test")
    assert result == data


def test_normalize_raise_error_input_dict_invalid_value():
    """
    Test _normalize_input with a dict containing invalid value type.
    """
    data = {'key1': pd.Series([1, 2]), 'key2': [3, 4]}
    msg = re.escape(
        "All values in `test` must be a pandas Series or DataFrame. " 
        "Review the value for key 'key2'."
    )
    with pytest.raises(TypeError, match=msg):
        RangeDriftDetector._normalize_input(data, name="test")
