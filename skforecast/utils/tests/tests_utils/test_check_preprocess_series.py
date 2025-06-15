# Unit test check_preprocess_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_preprocess_series
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import (
    series_long_dt,
    series_dict_dt,
    series_dict_range,
    exog_dict_nans,
    series_wide_dt
) 

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


def test_ValueError_check_preprocess_series_when_all_values_in_a_series_are_nan():
    """
    Test ValueError is raised when all values in a series are NaN.
    """
    series = {
        '1': pd.Series(np.arange(7), index = pd.date_range(start='2022-01-01', periods=7, freq='D')),
        '2': pd.Series([np.nan] * 7, index = pd.date_range(start='2022-01-01', periods=7, freq='D'))
    }

    err_msg = re.escape("All values of series '2' are NaN.")
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series)


def test_TypeError_check_preprocess_series_when_series_is_pandas_DataFrame_multiindex_without_datetime():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame with 
    a MultiIndex without datetime in the second level.
    """
    series = pd.DataFrame({
        '1': pd.Series(np.arange(6), index=pd.MultiIndex.from_product([['a', 'b'], range(3)])),
    })

    err_msg = re.escape(
        "The second level of the MultiIndex in `series` must be a pandas "
        "DatetimeIndex with the same frequency for each series. "
        "Found <class 'pandas.core.indexes.base.Index'>."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series=series)


def test_check_preprocess_series_when_series_is_pandas_DataFrame_multiindex_datetime():
    """
    Test check_preprocess_series when `series` is a pandas DataFrame.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_long_dt)

    expected_series_indexes = {
        col: pd.date_range(start='2000-01-01', periods=len(series_long_dt.loc[col]), freq='D')
        for col in series_long_dt.index.levels[0]
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
        assert series_indexes[k].freq == expected_series_indexes[k].freq


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when series is dict not containing 
    a pandas Series or DataFrame.
    """
    series_dict = {'l1': np.array([1, 2, 3]), 'l2': pd.Series([1, 2, 3])}
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
    series_dict = {'l1': series_wide_dt, 'l2': series_wide_dt}
    err_msg = re.escape(
        ("If `series` is a dictionary, all series must be a named pandas Series "
         "or a pandas DataFrame with a single column. Review series: 'l2'")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_TypeError_check_preprocess_series_when_series_is_dict_with_no_DatetimeIndex_RangeIndex():
    """
    Test TypeError is raised when series is dict containing series with no
    DatetimeIndex.
    """
    series_dict = {
        'l1': pd.Series(np.arange(3), index=[0, 1, 2]),
        'l2': pd.Series(np.arange(3), index=pd.RangeIndex(start=0, stop=3, step=1))
        }

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas RangeIndex or DatetimeIndex "
        "with the same step/frequency. Review series: ['l1']"
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_series(series = series_dict)


def test_ValueError_check_preprocess_series_when_series_is_dict_with_different_freqs():
    """
    Test ValueError is raised when series is dict containing series with 
    DatetimeIndex but different frequencies.
    """
    series_dict = series_dict_dt.copy()
    series_dict['l1'] = series_dict['l1'].asfreq('2D')

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas RangeIndex or "
        "DatetimeIndex with the same step/frequency. Found frequencies: ['2D', 'D']"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series = series_dict)

def test_ValueError_check_preprocess_series_when_series_is_dataframe_multiindex_with_different_freqs():
    """
    Test ValueError is raised when series is a pandas DataFrame with a MultiIndex
    containing different frequencies.
    """
    series = pd.DataFrame({
        'series_id': ['1', '1', '1', '2', '2', '2'],
        'datetime': pd.date_range(start='2000-01-01', periods=6, freq='D'),
        'value': np.random.rand(6)
    }
    )
    series = series.groupby("series_id", sort=False).apply(
        lambda x: x.set_index("datetime").asfreq("D"), include_groups=False
    )
    series = series.drop(index=('1', '2000-01-02'))

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas RangeIndex "
        "or DatetimeIndex with the same step/frequency. Found frequencies: ['2D', 'D']"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series)


def test_ValueError_check_preprocess_series_when_series_is_dataframe_multiindex_with_any_series_without_freq():
    """
    Test ValueError is raised when series is a pandas DataFrame with a MultiIndex
    containing at least one series without frequency.
    """
    series = pd.DataFrame({
            'series_id': ['1', '1', '1', '1', '2', '2', '2', '2'],
            'datetime': pd.date_range(start='2000-01-01', periods=8, freq='D'),
            'value': np.random.rand(8)
        }
        )
    series = series.groupby("series_id", sort=False).apply(
        lambda x: x.set_index("datetime").asfreq("D"), include_groups=False
    )
    series = series.drop(index=('1', '2000-01-03'))

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas RangeIndex "
        "or DatetimeIndex with the same step/frequency. Found series with no frequency or step."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series)

def test_ValueError_check_preprocess_series_when_all_series_values_are_missing_dict():
    """
    Test ValueError is raised when all series values are missing when series
    is a dict.
    """
    series_nan = pd.DataFrame({'l1': pd.Series(np.arange(7)), 
                               'l2': pd.Series([np.nan] * 7)})
    series_nan.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')
    series_nan = series_nan.to_dict("series")

    err_msg = re.escape("All values of series 'l2' are NaN.")
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_series(series=series_nan)


def test_check_preprocess_series_when_series_is_dict_with_a_pandas_Series_and_a_DataFrame():
    """
    Test check_preprocess_series when `series` is a dict containing a pandas Series
    and a pandas DataFrame with a single column.
    """
    series_dict = series_dict_dt.copy()
    series_dict['l2'] = series_dict['l2'].to_frame()

    series_dict, series_indexes = check_preprocess_series(series=series_dict)

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_dict['l1'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_dict['l1']), freq='D'),
                  name  ='l1'
              ),
        'l2': pd.Series(
                  data  = series_dict['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_dict['l2']), freq='D'),
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