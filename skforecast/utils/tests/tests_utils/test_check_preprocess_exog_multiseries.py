# Unit test check_preprocess_exog_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingExogWarning
from skforecast.utils import check_preprocess_series
from skforecast.utils import check_preprocess_exog_multiseries
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series_2
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series_as_dict
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_as_dict
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_as_dict_datetime
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import exog_predict


def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_valid_type():
    """
    Test TypeError is raised when exog is not a pandas Series, DataFrame or dict.
    """
    _, series_indexes = check_preprocess_series(series=series)

    not_valid_exog = 'not_valid_exog'

    err_msg = re.escape(
        ("`exog` must be a pandas Series, DataFrame, dictionary of pandas "
         "Series/DataFrames or None. Got <class 'str'>.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


@pytest.mark.parametrize("not_valid_exog", 
                         [pd.Series([1, 2, 3], name='exog1'),
                          pd.DataFrame([1, 2, 3], columns=['exog1'])], 
                         ids=lambda exog: f'exog: {type(exog)}')
def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_dict_and_series_is_dict(not_valid_exog):
    """
    Test TypeError is raised when exog is a pandas Series or DataFrame and
    input series is a dict (input_series_is_dict = True).
    """
    _, series_indexes = check_preprocess_series(series=series_as_dict)

    err_msg = re.escape(
        (f"`exog` must be a dict of DataFrames or Series if "
         f"`series` is a dict. Got {type(not_valid_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = True,
            series_indexes       = series_indexes,
            series_names_in_     = ['l1', 'l2'],
            exog                 = not_valid_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


@pytest.mark.parametrize("not_valid_exog", 
                         [exog['exog_1'].iloc[:30].copy(),
                          exog.iloc[:30].copy()], 
                         ids=lambda exog: f'exog: {type(exog)}')
def test_ValueError_check_preprocess_exog_multiseries_when_exog_pandas_with_different_length_from_series(not_valid_exog):
    """
    Test ValueError is raised when exog is a pandas Series or DataFrame with a
    different length from input series (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series)

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(not_valid_exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_pandas_with_different_index_from_series():
    """
    Test ValueError is raised when exog is a pandas Series or DataFrame with a
    different index from input series (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series)

    err_msg = re.escape(
        ("Different index for `series` and `exog`. They must be equal "
         "to ensure the correct alignment of values.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = exog_predict,
            exog_dict            = {'1': None, '2': None}
        )
    

def test_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame.
    """
    _, series_indexes = check_preprocess_series(series=series)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['1', '2'],
                                    exog                 = exog,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog,
        '2': exog
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0
    

def test_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame_with_RangeIndex_not_starting_0():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame with 
    a RangeIndex that doesn't start at 0.
    """
    series_range = series.copy()
    series_range.index = pd.RangeIndex(start=3, stop=3+len(series), step=1)
    _, series_indexes = check_preprocess_series(series=series_range)
    
    exog_range = exog.copy()
    exog_range.index = pd.RangeIndex(start=3, stop=3+len(exog_range), step=1)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['1', '2'],
                                    exog                 = exog_range,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog_range,
        '2': exog_range
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0
    

def test_output_check_preprocess_exog_multiseries_when_series_is_pandas_DataFrame_with_DatetimeIndex():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame with 
    a DatetimeIndex.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    _, series_indexes = check_preprocess_series(series=series_datetime)

    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(
        start='2000-01-01', periods=len(exog_datetime), freq='D'
    )

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['1', '2'],
                                    exog                 = exog_datetime,
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': exog_datetime,
        '2': exog_datetime
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']

    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_indexes['1'])
    
    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_dict_with_no_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when exog is dict not containing 
    a pandas Series or DataFrame.
    """
    _, series_indexes = check_preprocess_series(series=series_2)

    not_valid_exog = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': np.array([1, 2, 3]),
        'l3': None
    }

    err_msg = re.escape(
        ("If `exog` is a dictionary, all exog must be a named pandas "
         "Series, a pandas DataFrame or None. Review exog: ['l2']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['l1', 'l2'],
            exog                 = not_valid_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


def test_MissingValuesWarning_check_preprocess_exog_multiseries_when_exog_is_dict_without_all_series():
    """
    Test MissingValuesWarning is issues when exog is a dict without all the 
    series as keys.
    """
    _, series_indexes = check_preprocess_series(series=series_2)

    incomplete_exog = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': exog_as_dict_datetime['l2'].copy()
    }
    incomplete_exog.pop('l1')

    warn_msg = re.escape(
        ("{'l1'} not present in `exog`. All values "
         "of the exogenous variables for these series will be NaN.")
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['l1', 'l2'],
            exog                 = incomplete_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_dict_with_different_length_from_series():
    """
    Test ValueError is raised when exog is a dict with different length from 
    input series as pandas DataFrame (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series_2)

    not_valid_exog = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': exog_as_dict_datetime['l2'].copy()
    }
    not_valid_exog['l2'] = not_valid_exog['l2'].iloc[:30]

    err_msg = re.escape(
        ("`exog` for series 'l2' must have same number of samples as `series`. "
         "length `exog`: (30), length `series`: (50)")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['l1', 'l2'],
            exog                 = not_valid_exog,
            exog_dict            = {'l1': None, 'l2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_dict_with_different_index_from_series():
    """
    Test ValueError is raised when exog is a dict with different index from 
    input series as pandas DataFrame (series_indexes).
    """
    _, series_indexes = check_preprocess_series(series=series)

    not_valid_exog = {
        '1': exog.copy(),
        '2': exog_predict['exog_1'].copy()
    }

    err_msg = re.escape(
        ("Different index for series '2' and its exog. "
         "When `series` is a pandas DataFrame, they must "
         "be equal to ensure the correct alignment of values.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_dict_with_different_index_from_series():
    """
    Test TypeError is raised when exog is a dict with not DatetimeIndex index 
    when input series is a dict (input_series_is_dict=True).
    """
    _, series_indexes = check_preprocess_series(series=series_as_dict)

    err_msg = re.escape(
        ("All exog must have a Pandas DatetimeIndex as index with the "
         "same frequency. Check exog for series: ['l1', 'l2']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = True,
            series_indexes       = series_indexes,
            series_names_in_     = ['l1', 'l2'],
            exog                 = exog_as_dict,
            exog_dict            = {'l1': None, 'l2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_dict_with_different_dtypes_same_column():
    """
    Test TypeError is raised when exog is a dict with different dtypes for the 
    same column.
    """
    _, series_indexes = check_preprocess_series(series=series)

    not_valid_exog = {
        '1': exog.copy(),
        '2': exog['exog_1'].astype(str).copy()
    }

    err_msg = re.escape("Exog/s: ['exog_1'] have different dtypes in different series.")
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_has_columns_named_as_series():
    """
    Test ValueError is raised when exog has columns named as the series.
    """
    _, series_indexes = check_preprocess_series(series=series)

    duplicate_exog = exog['exog_1'].copy()
    duplicate_exog.name = '1'

    not_valid_exog = {
        '1': duplicate_exog
    }

    err_msg = re.escape(
        ("`exog` cannot contain a column named the same as one of the series.\n"
         "    `series` columns : ['1', '2'].\n"
         "    `exog`   columns : ['1'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            input_series_is_dict = False,
            series_indexes       = series_indexes,
            series_names_in_     = ['1', '2'],
            exog                 = not_valid_exog,
            exog_dict            = {'1': None, '2': None}
        )


def test_output_check_preprocess_exog_multiseries_when_series_is_DataFrame_and_exog_pandas_Series():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is a pandas Series.
    """
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['1', '2'],
                                    exog                 = exog['exog_1'],
                                    exog_dict            = {'1': None, '2': None}
                                )

    expected_exog_dict = {
        '1': pd.DataFrame(
                 data    = exog['exog_1'].to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1']
             ),
        '2': pd.DataFrame(
                 data    = exog['exog_1'].to_numpy(),
                 index   = pd.RangeIndex(start=0, stop=50),
                 columns = ['exog_1']
             ),
    }
    expected_exog_names_in_ = ['exog_1']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert exog_names_in_ == expected_exog_names_in_


def test_output_check_preprocess_exog_multiseries_when_series_is_DataFrame_and_exog_pandas_DataFrame():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is a pandas DataFrame.
    """
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['1', '2'],
                                    exog                 = exog,
                                    exog_dict            = {'1': None, '2': None}
                                )

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
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['1', '2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_output_check_preprocess_exog_multiseries_when_series_is_DataFrame_and_exog_dict():
    """
    Test check_preprocess_exog_multiseries when `series` is a pandas DataFrame 
    and `exog` is a dict.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_2)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = False,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['l1', 'l2'],
                                    exog                 = exog_as_dict_datetime,
                                    exog_dict            = {'l1': None, 'l2': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_as_dict_datetime['l1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_output_check_preprocess_exog_multiseries_when_series_is_dict_and_exog_dict():
    """
    Test check_preprocess_exog_multiseries when `series` is a dict (input_series_is_dict = True)
    and `exog` is a dict.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_as_dict)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = True,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['l1', 'l2'],
                                    exog                 = exog_as_dict_datetime,
                                    exog_dict            = {'l1': None, 'l2': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_as_dict_datetime['l1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_as_dict_datetime['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_output_check_preprocess_exog_multiseries_when_series_is_dict_and_exog_dict_with_different_lengths():
    """
    Test check_preprocess_exog_multiseries when `series` is a dict (input_series_is_dict = True)
    and `exog` is a dict with series of different lengths and exog None.
    """
    series_dict_3 = {
        'l1': series_as_dict['l1'].copy(),
        'l2': series_as_dict['l2'].copy(),
        'l3': series_as_dict['l1'].copy()
    }
    series_dict, series_indexes = check_preprocess_series(series=series_dict_3)

    exog_test = {
        'l1': exog_as_dict_datetime['l1'].copy(),
        'l2': exog_as_dict_datetime['l2'].copy(),
        'l3': None
    }
    exog_test['l1'] = exog_test['l1'].iloc[10:30]
    exog_test['l2'] = exog_test['l2'].iloc[:40]

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    input_series_is_dict = True,
                                    series_indexes       = series_indexes,
                                    series_names_in_     = ['l1', 'l2', 'l3'],
                                    exog                 = exog_test,
                                    exog_dict            = {'l1': None, 'l2': None, 'l3': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_test['l1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-11', periods=len(exog_test['l1']), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_test['l2'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog_test['l2']), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
        'l3': None
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2', 'l3']
    for k in exog_dict:
        if k == 'l3':
            assert exog_dict[k] is None
        else:
            assert isinstance(exog_dict[k], pd.DataFrame)
            pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
            index_intersection = (
                series_dict[k].index.intersection(exog_dict[k].index)
            )
            assert len(index_intersection) == len(exog_dict[k])

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0