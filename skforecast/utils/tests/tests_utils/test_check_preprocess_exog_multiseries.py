# Unit test check_preprocess_exog_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import InputTypeWarning, MissingExogWarning
from skforecast.utils import check_preprocess_series
from skforecast.utils import check_preprocess_exog_multiseries
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import (
    series_wide_range,
    series_dict_range,
    series_dict_dt,
    exog_wide_range,
    exog_wide_dt,
    exog_long_dt,
    exog_dict_range,
    exog_dict_dt,
)


def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_valid_type():
    """
    Test TypeError is raised when exog is not a pandas Series, DataFrame or dict.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)

    not_valid_exog = 'not_valid_exog'

    err_msg = re.escape(
        "`exog` must be a pandas Series, DataFrame, dictionary of pandas "
        "Series/DataFrames or None. Got <class 'str'>."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = not_valid_exog,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_dict_and_series_is_dict():
    """
    Test TypeError is raised when exog is a pandas Series or DataFrame and
    input series is a dict (input_series_is_dict = True).
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)
    not_valid_exog = pd.DataFrame({
        '1': pd.Series(np.arange(6), index=pd.MultiIndex.from_product([['a', 'b'], range(3)])),
    })

    err_msg = re.escape(
        "When input data are pandas MultiIndex DataFrame, "
        "`series` and `exog` second level index must be a "
        "pandas DatetimeIndex. Found `exog` index type: "
        "<class 'pandas.core.indexes.base.Index'>."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = not_valid_exog,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_pandas_with_different_index_type_from_series():
    """
    Test TypeError is raised when exog is a pandas Series or DataFrame with a
    different index from input series.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)

    err_msg = re.escape(
        "`exog` must have the same index type as `series`, pandas "
        "RangeIndex or pandas DatetimeIndex.\n"
        "    `series` index type : <class 'pandas.core.indexes.datetimes.DatetimeIndex'>.\n"
        "    `exog`   index type : <class 'pandas.core.indexes.range.RangeIndex'>."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = exog_wide_range,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_is_dict_with_no_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when exog is dict not containing 
    a pandas Series or DataFrame.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)
    not_valid_exog = {
        'l1': exog_dict_dt['l1'].copy(),
        'l2': np.array([1, 2, 3]),
        'l3': None
    }

    err_msg = re.escape(
        "If `exog` is a dictionary, all exog must be a named pandas "
        "Series, a pandas DataFrame or None. Review exog: ['l2']"
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = not_valid_exog,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_MissingValuesWarning_check_preprocess_exog_multiseries_when_exog_is_dict_without_all_series():
    """
    Test MissingValuesWarning is issues when exog is a dict without all the 
    series as keys.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)

    incomplete_exog = {
        'l1': exog_dict_dt['l1'].copy(),
        'l2': exog_dict_dt['l2'].copy()
    }
    incomplete_exog.pop('l1')

    warn_msg = re.escape(
        "No `exog` for series {'l1'}. All values "
        "of the exogenous variables for these series will be NaN."
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                        series_names_in_  = ['l1', 'l2'],
                                        series_index_type = type(series_indexes['l1']),
                                        exog              = incomplete_exog,
                                        exog_dict         = {'l1': None, 'l2': None}
                                    )

    expected_exog_dict = {
        'l1': None,
        'l2': exog_dict_dt['l2'].to_frame()
    }
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        if k == 'l1':
            assert exog_dict[k] is None
        else:
            assert isinstance(exog_dict[k], pd.DataFrame)
            pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
            index_intersection = (
                series_dict_dt[k].index.intersection(exog_dict[k].index)
            )
            assert len(index_intersection) == len(exog_dict[k])

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_TypeError_check_preprocess_exog_multiseries_when_exog_dict_with_different_index_from_series():
    """
    Test TypeError is raised when exog is a dict with not DatetimeIndex index 
    when input series is a dict (input_series_is_dict=True).
    """
    _, series_indexes = check_preprocess_series(series=series_dict_dt)

    err_msg = re.escape(
        "All exog must have the same index type as `series`, which can be "
        "either a pandas RangeIndex or a pandas DatetimeIndex. If either "
        "`series` or `exog` is a pandas DataFrame with a MultiIndex, then "
        "both must be pandas DatetimeIndex. Review exog for series: ['l1', 'l2']."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = exog_dict_range,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_TypeError_check_preprocess_exog_multiseries_when_exog_dict_with_different_dtypes_same_column():
    """
    Test TypeError is raised when exog is a dict with different dtypes for the 
    same column.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_range)

    not_valid_exog = {
        'l1': exog_wide_range.copy(),
        'l2': exog_wide_range['exog_1'].astype(str).copy()
    }

    err_msg = re.escape(
        "Exog/s: ['exog_1'] have different dtypes in different "
        "series. If any of these variables are categorical, note that this "
        "error can also occur when their internal categories "
        "(`series.cat.categories`) differ between series. Please ensure "
        "that all series have the same categories (and category order) "
        "for each categorical variable."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = not_valid_exog,
            exog_dict         = {'l1': None, 'l2': None}
        )


def test_ValueError_check_preprocess_exog_multiseries_when_exog_has_columns_named_as_series():
    """
    Test ValueError is raised when exog has columns named as the series.
    """
    _, series_indexes = check_preprocess_series(series=series_dict_range)

    duplicate_exog = exog_wide_range['exog_1'].copy()
    duplicate_exog.name = 'l1'

    not_valid_exog = {
        'l1': duplicate_exog
    }

    err_msg = re.escape(
        "`exog` cannot contain a column named the same as one of the series.\n"
        "    `series` columns : ['l1', 'l2'].\n"
        "    `exog`   columns : ['l1']."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_preprocess_exog_multiseries(
            series_names_in_  = ['l1', 'l2'],
            series_index_type = type(series_indexes['l1']),
            exog              = not_valid_exog,
            exog_dict         = {'l1': None, 'l2': None}
        )


@pytest.mark.parametrize("series", 
                         [series_wide_range, series_dict_range],
                         ids = lambda series: f'series type: {type(series)}')
def test_output_check_preprocess_exog_multiseries_when_exog_pandas_Series(series):
    """
    Test check_preprocess_exog_multiseries when `exog` is a pandas Series.
    """
    if isinstance(series, pd.DataFrame):
        series = series.rename(columns={'1': 'l1', '2': 'l2'})
    
    series_dict, series_indexes = check_preprocess_series(series=series)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    series_names_in_  = ['l1', 'l2'],
                                    series_index_type = type(series_indexes['l1']),
                                    exog              = exog_wide_range['exog_1'],
                                    exog_dict         = {'l1': None, 'l2': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_wide_range['exog_1'].to_numpy(),
                  index   = pd.RangeIndex(start=0, stop=50),
                  columns = ['exog_1']
              ),
        'l2': pd.DataFrame(
                  data    = exog_wide_range['exog_1'].to_numpy(),
                  index   = pd.RangeIndex(start=0, stop=50),
                  columns = ['exog_1']
              ),
    }
    expected_exog_names_in_ = ['exog_1']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert exog_names_in_ == expected_exog_names_in_


def test_output_check_preprocess_exog_multiseries_when_exog_pandas_DataFrame():
    """
    Test check_preprocess_exog_multiseries when `exog` is a pandas DataFrame.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_dict_dt)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    series_names_in_  = ['l1', 'l2'],
                                    series_index_type = type(series_indexes['l1']),
                                    exog              = exog_wide_dt,
                                    exog_dict         = {'l1': None, 'l2': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_wide_dt.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_wide_dt.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
                  columns = ['exog_1', 'exog_2']
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


def test_output_check_preprocess_exog_multiseries_when_exog_DataFrame_MultiIndex():
    """
    Test check_preprocess_exog_multiseries when `exog` is a pandas DataFrame with MultiIndex.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_dict_dt)

    warn_msg = re.escape(
        "Using a long-format DataFrame as `exog` requires additional transformations, "
        "which can increase computational time. It is recommended to use a dictionary of "
        "Series or DataFrames instead. For more information, see: "
        "https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#input-data"
    )
    with pytest.warns(InputTypeWarning, match=warn_msg):
        exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                        series_names_in_  = ['l1', 'l2'],
                                        series_index_type = type(series_indexes['l1']),
                                        exog              = exog_long_dt,
                                        exog_dict         = {'l1': None, 'l2': None}
                                    )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_wide_dt.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_wide_dt.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
    }
    expected_exog_dict['l1'].index.freq = None
    expected_exog_dict['l2'].index.freq = None
    expected_exog_names_in_ = ['exog_1', 'exog_2']
    
    assert isinstance(exog_dict, dict)
    assert list(exog_dict.keys()) == ['l1', 'l2']
    for k in exog_dict:
        assert isinstance(exog_dict[k], pd.DataFrame)
        pd.testing.assert_frame_equal(exog_dict[k], expected_exog_dict[k])
        pd.testing.assert_index_equal(exog_dict[k].index, series_dict[k].index)

    assert len(set(exog_names_in_) - set(expected_exog_names_in_)) == 0


def test_output_check_preprocess_exog_multiseries_when_exog_MultiIndex_with_different_lengths():
    """
    Test check_preprocess_exog_multiseries when `series` is a dict 
    and `exog` is a pandas DataFrame with MultiIndex and different lengths.
    """
    series_dict_3 = {
        'l1': series_dict_dt['l1'].copy(),
        'l2': series_dict_dt['l2'].copy(),
        'l3': series_dict_dt['l1'].copy()
    }
    series_dict, series_indexes = check_preprocess_series(series=series_dict_3)

    exog_test_l1 = exog_dict_dt['l1'].iloc[10:30, :].assign(series_id="l1")
    exog_test_l2 = exog_dict_dt['l2'].iloc[:40].to_frame().assign(series_id="l2")
    exog_long_test = pd.concat([exog_test_l1, exog_test_l2])
    exog_long_test.index.name = "datetime"
    exog_long_test = exog_long_test.set_index(["series_id", exog_long_test.index])

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    series_names_in_  = ['l1', 'l2', 'l3'],
                                    series_index_type = type(series_indexes['l1']),
                                    exog              = exog_long_test,
                                    exog_dict         = {'l1': None, 'l2': None, 'l3': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_test_l1[['exog_1', 'exog_2']].to_numpy(),
                  index   = pd.date_range(start='2000-01-11', periods=len(exog_test_l1), freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_test_l2['exog_1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=len(exog_test_l2), freq='D'),
                  columns = ['exog_1']
              ).astype({'exog_1': float}),
        'l3': None
    }
    expected_exog_dict['l1'].index.freq = None
    expected_exog_dict['l2'].index.freq = None
    expected_exog_dict['l2']['exog_2'] = np.nan
    expected_exog_dict['l2']['exog_2'] = expected_exog_dict['l2']['exog_2'].astype(object)
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


def test_output_check_preprocess_exog_multiseries_when_series_is_dict_and_exog_dict():
    """
    Test check_preprocess_exog_multiseries when `series` is a dict and `exog` is a dict.
    """
    series_dict, series_indexes = check_preprocess_series(series=series_dict_dt)

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    series_names_in_  = ['l1', 'l2'],
                                    series_index_type = type(series_indexes['l1']),
                                    exog              = exog_dict_dt,
                                    exog_dict         = {'l1': None, 'l2': None}
                                )

    expected_exog_dict = {
        'l1': pd.DataFrame(
                  data    = exog_wide_dt.to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
                  columns = ['exog_1', 'exog_2']
              ).astype({'exog_1': float}),
        'l2': pd.DataFrame(
                  data    = exog_wide_dt['exog_1'].to_numpy(),
                  index   = pd.date_range(start='2000-01-01', periods=50, freq='D'),
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
    Test check_preprocess_exog_multiseries when `series` is a dict 
    and `exog` is a dict with series of different lengths and exog None.
    """
    series_dict_3 = {
        'l1': series_dict_dt['l1'].copy(),
        'l2': series_dict_dt['l2'].copy(),
        'l3': series_dict_dt['l1'].copy()
    }
    series_dict, series_indexes = check_preprocess_series(series=series_dict_3)

    exog_test = {
        'l1': exog_dict_dt['l1'].copy(),
        'l2': exog_dict_dt['l2'].copy(),
        'l3': None
    }
    exog_test['l1'] = exog_test['l1'].iloc[10:30]
    exog_test['l2'] = exog_test['l2'].iloc[:40]

    exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                    series_names_in_  = ['l1', 'l2', 'l3'],
                                    series_index_type = type(series_indexes['l1']),
                                    exog              = exog_test,
                                    exog_dict         = {'l1': None, 'l2': None, 'l3': None}
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
