# Unit test check_preprocess_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_preprocess_series
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series_as_dict


def test_TypeError_check_preprocess_series_when_series_is_not_pandas_DataFrame_or_dict():
    """
    Test TypeError is raised when series is not a pandas DataFrame or dict.
    """
    series = np.array([1, 2, 3])

    err_msg = re.escape(
        (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
         f"Got {type(series)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series)


def test_ValueError_check_preprocess_series_when_all_series_values_are_missing_DataFrame():
    """
    Test ValueError is raised when all series values are missing when series
    is a pandas DataFrame.
    """
    series_nan = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                               '2': pd.Series([np.nan]*7)})
    series_nan.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')

    err_msg = re.escape("All values of series '2' are NaN.")
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series_nan)


def test_check_preprocess_series_when_series_is_pandas_DataFrame():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame.
    """
    series_dict, series_indexes = check_preprocess_series(series=series)

    expected_series_indexes = {
        col: pd.RangeIndex(start=0, stop=len(series), step=1) 
        for col in series.columns
    }

    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k, v in series_dict.items():
        assert isinstance(v, pd.Series)
        assert k == v.name

    assert isinstance(series_indexes, dict)
    assert list(series_indexes.keys()) == ['1', '2']
    for k in series_indexes:
        pd.testing.assert_index_equal(series_indexes[k], expected_series_indexes[k])
        assert series_indexes[k].step == expected_series_indexes[k].step


def test_check_preprocess_series_when_series_is_pandas_DataFrame_with_DatetimeIndex():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(
        start='2000-01-01', periods=len(series_datetime), freq='D'
    )

    series_dict, series_indexes = check_preprocess_series(series=series_datetime)

    expected_series_indexes = {series: pd.date_range(
        start='2000-01-01', periods=len(series_datetime), freq='D'
    ) for series in series_datetime.columns}

    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k, v in series_dict.items():
        assert isinstance(v, pd.Series)
        assert k == v.name
        pd.testing.assert_series_equal(v, series_datetime[k])

    assert isinstance(series_indexes, dict)
    assert list(series_indexes.keys()) == ['1', '2']
    for k in series_indexes:
        pd.testing.assert_index_equal(series_indexes[k], expected_series_indexes[k])
        assert series_indexes[k].freq == expected_series_indexes[k].freq


def test_check_preprocess_series_when_series_is_pandas_DataFrame_with_no_pandas_Index():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with
    no pandas Index.
    """
    series_no_pandas_idx = series.copy()
    series_no_pandas_idx.index = np.arange(6, 6 + len(series))
    series_dict, series_indexes = check_preprocess_series(series=series_no_pandas_idx)

    expected_series_indexes = {
        col: pd.RangeIndex(start=0, stop=len(series_no_pandas_idx), step=1) 
        for col in series_no_pandas_idx.columns
    }

    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k, v in series_dict.items():
        assert isinstance(v, pd.Series)
        assert k == v.name

    assert isinstance(series_indexes, dict)
    assert list(series_indexes.keys()) == ['1', '2']
    for k in series_indexes:
        pd.testing.assert_index_equal(series_indexes[k], expected_series_indexes[k])
        assert series_indexes[k].step == expected_series_indexes[k].step


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when series is dict not containing 
    a pandas Series or DataFrame.
    """
    series_dict = {'l1': np.array([1, 2, 3]),
                   'l2': series}

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must be a named pandas Series "
         "or a pandas DataFrame with a single column. Review series: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_series_is_dict_with_a_DataFrame_with_2_columns():
    """
    Test ValueError is raised when series is dict containing a pandas DataFrame 
    with 2 columns.
    """
    series_dict = {'l1': series['1'],
                   'l2': series}

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must be a named pandas Series "
         "or a pandas DataFrame with a single column. Review series: 'l2'")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_DatetimeIndex():
    """
    Test TypeError is raised when series is dict containing series with no
    DatetimeIndex.
    """
    series_dict = {'l1': series['1'],
                   'l2': series['2']}

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Review series: ['l1', 'l2']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_series_is_dict_with_different_freqs():
    """
    Test ValueError is raised when series is dict containing series with 
    DatetimeIndex but different frequencies.
    """
    series_dict = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='MS'
    )

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Found frequencies: ['<Day>', '<MonthBegin>']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_all_series_values_are_missing_dict():
    """
    Test ValueError is raised when all series values are missing when series
    is a dict.
    """
    series_nan = pd.DataFrame({'l1': pd.Series(np.arange(7)), 
                               'l2': pd.Series([np.nan]*7)})
    series_nan.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')
    series_nan = series_nan.to_dict("series")

    err_msg = re.escape("All values of series 'l2' are NaN.")
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series_nan)


def test_check_preprocess_series_when_series_is_dict():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_dict = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }

    series_dict['l1'].index = pd.date_range(start='2000-01-01', periods=len(series_dict['l1']), freq='D')
    series_dict['l2'] = series_dict['l2'].to_frame()
    series_dict['l2'].index = pd.date_range(start='2000-01-01', periods=len(series_dict['l2']), freq='D')

    series_dict, series_indexes = check_preprocess_series(series=series_dict)

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_dict['l1'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series['1']), freq='D'),
                  name  ='l1'
              ),
        'l2': pd.Series(
                  data  = series_dict['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series['2']), freq='D'),
                  name  ='l2'
              ),
    }

    expected_series_indexes = {
        k: v.index
        for k, v in expected_series_dict.items()
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(series_dict[k].index, pd.DatetimeIndex)

    indexes_freq = [f'{v.index.freq}' 
                    for v in series_dict.values()]
    assert len(set(indexes_freq)) == 1

    assert isinstance(series_indexes, dict)
    assert list(series_indexes.keys()) == ['l1', 'l2']
    for k in series_indexes:
        pd.testing.assert_index_equal(series_indexes[k], expected_series_indexes[k])
        assert series_indexes[k].freq == expected_series_indexes[k].freq