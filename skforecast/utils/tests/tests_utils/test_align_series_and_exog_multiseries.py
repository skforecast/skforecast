# Unit test align_series_and_exog_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import check_preprocess_series
from skforecast.utils import check_preprocess_exog_multiseries
from skforecast.utils import align_series_and_exog_multiseries
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series_2
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series_as_dict
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_as_dict
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_as_dict_datetime
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_predict


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_and_exog_None():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is None.
    """
    input_series_is_dict = isinstance(series, dict)
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict = {'1': None, '2': None}

    expected_series_dict = {
        '1': pd.Series(
                 data  = series['1'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '1'
             ),
        '2': pd.Series(
                 data  = series['2'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '2'
             ),
    }
    expected_exog_dict = {'1': None, '2': None}
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(exog_dict[k], type(None))


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_exog_None():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and `exog` is None.
    """
    series_diff = series_2.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict = {'l1': None, 'l2': None}

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_2['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_2['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {'l1': None, 'l2': None}
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(exog_dict[k], type(None))


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_nans_in_between():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and NaNs in between with `exog` is None.
    """
    series_diff = series_2.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan
    series_diff.iloc[20:23, 0] = np.nan
    series_diff.iloc[33:49, 1] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict = {'l1': None, 'l2': None}

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {'l1': None, 'l2': None}
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(exog_dict[k], type(None))


def test_output_align_series_and_exog_multiseries_when_series_is_dict_different_lengths_and_nans_in_between():
    """
    Test align_series_and_exog_multiseries when `series` is a dict 
    with series of different lengths and NaNs in between with `exog` is None.
    """
    series_diff = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }
    n_nans = 10
    series_diff['l1'].iloc[:n_nans, ] = np.nan
    series_diff['l1'].iloc[20:23, ] = np.nan
    series_diff['l2'].iloc[33:49, ] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict = {'l1': None, 'l2': None}

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {'l1': None, 'l2': None}
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
        assert isinstance(exog_dict[k], type(None))


def test_output_align_series_and_exog_multiseries_when_input_series_is_DataFrame_and_exog_DataFrame():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is a pandas DataFrame.
    """
    input_series_is_dict = isinstance(series, dict)
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['1', '2'],
                       exog                 = exog,
                       exog_dict            = {'1': None, '2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        '1': pd.Series(
                 data  = series['1'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '1'
             ),
        '2': pd.Series(
                 data  = series['2'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '2'
             ),
    }
    expected_exog_dict = {
        '1': pd.DataFrame(
                 data    = exog.to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
        '2': pd.DataFrame(
                 data    = exog.to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_nans_in_between_and_exog_DataFrame():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and NaNs in between with `exog` is a 
    pandas DataFrame with no datetime index.
    """
    series_diff = series.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan
    series_diff.iloc[20:23, 0] = np.nan
    series_diff.iloc[33:49, 1] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['1', '2'],
                       exog                 = exog,
                       exog_dict            = {'1': None, '2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        '1': pd.Series(
                 data  = series_diff['1'].to_numpy()[-40:],
                 index = pd.RangeIndex(start=10, stop=50),
                 name  = '1'
             ),
        '2': pd.Series(
                 data  = series_diff['2'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '2'
             ),
    }
    expected_exog_dict = {
        '1': pd.DataFrame(
                 data    = exog.to_numpy()[-40:],
                 index   = pd.RangeIndex(start=10, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
        '2': pd.DataFrame(
                 data    = exog.to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_nans_in_between_and_exog_DataFrame_datetime():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and NaNs in between with `exog` is a 
    pandas DataFrame with datetime index.
    """
    series_diff = series_2.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan
    series_diff.iloc[20:23, 0] = np.nan
    series_diff.iloc[33:49, 1] = np.nan

    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_datetime,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_datetime.to_numpy()[-40:],
                  index   = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_datetime.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_input_series_is_DataFrame_and_exog_dict():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is a dict.
    """
    exog_dict = {
        '1': exog.copy(),
        '2': exog['exog_1'].copy()
    }

    input_series_is_dict = isinstance(series, dict)
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['1', '2'],
                       exog                 = exog_dict,
                       exog_dict            = {'1': None, '2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        '1': pd.Series(
                 data  = series['1'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '1'
             ),
        '2': pd.Series(
                 data  = series['2'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '2'
             ),
    }
    expected_exog_dict = {
        '1': pd.DataFrame(
                 data    = exog.to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
        '2': pd.DataFrame(
                 data    = exog['exog_1'].to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1']
             ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_nans_in_between_and_exog_dict():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and NaNs in between with `exog` is a 
    dict with no datetime index.
    """
    series_diff = series.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan
    series_diff.iloc[20:23, 0] = np.nan
    series_diff.iloc[33:49, 1] = np.nan

    exog_dict = {
        '1': exog.copy(),
        '2': exog['exog_1'].copy()
    }

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['1', '2'],
                       exog                 = exog_dict,
                       exog_dict            = {'1': None, '2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        '1': pd.Series(
                 data  = series_diff['1'].to_numpy()[-40:],
                 index = pd.RangeIndex(start=10, stop=50),
                 name  = '1'
             ),
        '2': pd.Series(
                 data  = series_diff['2'].to_numpy(),
                 index = pd.RangeIndex(start=0, stop=50),
                 name  = '2'
             ),
    }
    expected_exog_dict = {
        '1': pd.DataFrame(
                 data    = exog.to_numpy()[-40:],
                 index   = pd.RangeIndex(start=10, stop=50),
                 columns = ['exog_1', 'exog_2']
             ).astype({'exog_1': float}),
        '2': pd.DataFrame(
                 data    = exog['exog_1'].to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1']
             ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['1', '2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_DataFrame_different_lengths_and_nans_in_between_and_exog_dict_datetime():
    """
    Test align_series_and_exog_multiseries when `series` is a pandas DataFrame 
    with series of different lengths and NaNs in between with `exog` is a 
    dict with datetime index.
    """
    series_diff = series_2.copy()
    n_nans = 10
    series_diff.iloc[:n_nans, 0] = np.nan
    series_diff.iloc[20:23, 0] = np.nan
    series_diff.iloc[33:49, 1] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_as_dict_datetime,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_as_dict_datetime['l1'].to_numpy()[-40:],
                  index   = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_dict_and_exog_dict_datetime():
    """
    Test align_series_and_exog_multiseries when `series` is a dict and `exog` is 
    a dict. Datetime index is a must.
    """
    input_series_is_dict = isinstance(series_as_dict, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_as_dict)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_as_dict_datetime,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_as_dict['l1'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_as_dict['l1']), freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_as_dict['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_as_dict['l2']), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_as_dict_datetime['l1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog_as_dict_datetime['l1']), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog_as_dict_datetime['l2']), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_dict_different_lengths_and_nans_in_between_and_exog_dict_datetime():
    """
    Test align_series_and_exog_multiseries when `series` is a dict 
    with series of different lengths and NaNs in between with `exog` is a 
    dict with datetime index.
    """
    series_diff = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }
    n_nans = 10
    series_diff['l1'].iloc[:n_nans, ] = np.nan
    series_diff['l1'].iloc[20:23, ] = np.nan
    series_diff['l2'].iloc[33:49, ] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_as_dict_datetime,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    series_dict, exog_dict = align_series_and_exog_multiseries(
                                 series_dict          = series_dict,
                                 input_series_is_dict = input_series_is_dict,
                                 exog_dict            = exog_dict
                             )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_as_dict_datetime['l1'].to_numpy()[-40:],
                  index   = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_dict_and_length_intersection_with_exog_is_0():
    """
    Test align_series_and_exog_multiseries when `series` is a dict and the
    intersection with `exog` is 0.
    """
    series_diff = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }
    n_nans = 10
    series_diff['l1'].iloc[:n_nans, ] = np.nan
    series_diff['l1'].iloc[20:23, ] = np.nan
    series_diff['l2'].iloc[33:49, ] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_no_intersection = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': exog_as_dict_datetime['l2'].copy()
    }
    exog_no_intersection['l1'].index = pd.date_range(
        start='2001-01-01', periods=len(exog_no_intersection['l1']), freq='D'
    )

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_no_intersection,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    warn_msg = re.escape(
        ("Series 'l1' and its `exog` do not have the same index. "
         "All exog values will be NaN for the period of the series.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict          = series_dict,
                                     input_series_is_dict = input_series_is_dict,
                                     exog_dict            = exog_dict
                                 )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = np.nan,
                  index   = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_2': object}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  columns = ['exog_1']
              ),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)


def test_output_align_series_and_exog_multiseries_when_series_is_dict_and_different_length_intersection_with_exog():
    """
    Test align_series_and_exog_multiseries when `series` is a dict and the
    intersection with `exog` do not have the same length as series.
    """
    series_diff = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy()
    }
    n_nans = 10
    series_diff['l1'].iloc[:n_nans, ] = np.nan
    series_diff['l1'].iloc[20:23, ] = np.nan
    series_diff['l2'].iloc[33:49, ] = np.nan

    input_series_is_dict = isinstance(series_diff, dict)
    series_dict, series_indexes = check_preprocess_series(series=series_diff)

    exog_half_intersection = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': exog_as_dict_datetime['l2'].copy()
    }
    exog_half_intersection['l1'].iloc[20:23, :] = np.nan
    exog_half_intersection['l1'].index = pd.date_range(
        start='2000-01-15', periods=len(exog_half_intersection['l1']), freq='D'
    )
    exog_half_intersection['l2'].iloc[:10] = np.nan
    exog_half_intersection['l2'].iloc[33:49] = np.nan

    exog_dict, _ = check_preprocess_exog_multiseries(
                       input_series_is_dict = input_series_is_dict,
                       series_indexes       = series_indexes,
                       series_names_in_     = ['l1', 'l2'],
                       exog                 = exog_half_intersection,
                       exog_dict            = {'l1': None, 'l2': None}
                   )

    warn_msg = re.escape(
        ("Series 'l1' and its `exog` do not have the same length. "
         "Exog values will be NaN for the not matched period of the series.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict          = series_dict,
                                     input_series_is_dict = input_series_is_dict,
                                     exog_dict            = exog_dict
                                 )

    expected_series_dict = {
        'l1': pd.Series(
                  data  = series_diff['l1'].to_numpy()[-40:],
                  index = pd.date_range(start='2000-01-11', periods=len(series_2)-n_nans, freq='D'),
                  name  = 'l1'
              ),
        'l2': pd.Series(
                  data  = series_diff['l2'].to_numpy(),
                  index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D'),
                  name  = 'l2'
              ),
    }
    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = np.vstack((
                                np.full((4, len(['exog_1', 'exog_2'])), np.nan), 
                                exog_half_intersection['l1'].to_numpy()[:40-4]
                            )),
                  index   = pd.date_range(start='2000-01-11', periods=len(expected_series_dict['l1']), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float, 'exog_2': object}),
        'l2': pd.DataFrame(
                  data    = exog_half_intersection['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog_half_intersection['l2']), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    
    assert isinstance(series_dict, dict)
    assert list(series_dict.keys()) == ['l1', 'l2']
    for k in series_dict:
        assert isinstance(series_dict[k], pd.Series)
        pd.testing.assert_series_equal(series_dict[k], expected_series_dict[k])
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)