# Unit test _create_data_to_return_dict ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate

rolling = RollingFeatures(
    stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
)


def test_create_data_to_return_dict_ValueError_when_lags_keys_not_in_series():
    """
    Test ValueError is raised when lags keys are not in series column names.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))})
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags={'l1': 2, '2': 3}
    )
    
    series_names_in_ = list(series.columns)

    err_msg = re.escape(
        f"When `lags` parameter is a `dict`, its keys must be the "
        f"same as `series` column names. If don't want to include lags, "
         "add '{column: None}' to the lags dict." 
        f"  Lags keys        : {list(forecaster.lags.keys())}.\n"
        f"  `series` columns : {series_names_in_}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_data_to_return_dict(series_names_in_=series_names_in_)

  
def test_create_data_to_return_dict_when_same_lags_all_series():
    """
    Test _create_data_to_return_dict output when lags is the same for all series.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=1, lags=3
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'both', 'l2': 'X'},
        ['l1', 'l2']
    )

    for k in forecaster.lags_.keys():
        np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array([1, 2, 3]))
    for k in forecaster.lags_names.keys():
        assert forecaster.lags_names[k] == [f'{k}_lag_{i}' for i in range(1, 4)]
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]

  
def test_create_data_to_return_dict_when_interspersed_lags_all_series():
    """
    Test _create_data_to_return_dict output when interspersed lags for all series.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=1, lags=[1, 5]
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'both', 'l2': 'X'},
        ['l1', 'l2']
    )

    for k in forecaster.lags_.keys():
        np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array([1, 5]))
    for k in forecaster.lags_names.keys():
        assert forecaster.lags_names[k] == [f'{k}_lag_1', f'{k}_lag_5']
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]

  
def test_create_data_to_return_dict_when_lags_dict_with_all_series():
    """
    Test _create_data_to_return_dict output when lags is a dict with lags for 
    all series.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l2', steps=1, lags={'l1': 3, 'l2': 4}
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'X', 'l2': 'both'},
        ['l1', 'l2']
    )

    for k in forecaster.lags_.keys():
        expected_lags = [1, 2, 3] if k == 'l1' else [1, 2, 3, 4]
        np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array(expected_lags))
    for k in forecaster.lags_names.keys():
        range_ = range(1, 4) if k == 'l1' else range(1, 5)
        expected_names = [f'{k}_lag_{i}' for i in range_]
        assert forecaster.lags_names[k] == expected_names
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]


def test_create_data_to_return_dict_when_lags_dict_with_no_lags_for_level():
    """
    Test _create_data_to_return_dict output when lags is a dict with no lags for
    the level.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l2', steps=1, lags={'l1': 3, 'l2': None}
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'X', 'l2': 'y'},
        ['l1']
    )

    for k in forecaster.lags_.keys():
        if forecaster.lags_[k] is not None:
            np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array([1, 2, 3]))
        else:
            assert forecaster.lags_[k] is None
    for k in forecaster.lags_names.keys():
        if forecaster.lags_names[k] is not None:
            assert forecaster.lags_names[k] == [f'{k}_lag_{i}' for i in range(1, 4)]
        else:
            assert forecaster.lags_names[k] is None
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]


def test_create_data_to_return_dict_when_lags_dict_with_lags_only_for_level():
    """
    Test _create_data_to_return_dict output when lags is a dict with lags only for
    level.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=1, lags={'l1': 3, 'l2': None}
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'both'},
        ['l1']
    )

    for k in forecaster.lags_.keys():
        if forecaster.lags_[k] is not None:
            np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array([1, 2, 3]))
        else:
            assert forecaster.lags_[k] is None
    for k in forecaster.lags_names.keys():
        if forecaster.lags_names[k] is not None:
            assert forecaster.lags_names[k] == [f'{k}_lag_{i}' for i in range(1, 4)]
        else:
            assert forecaster.lags_names[k] is None
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]


def test_create_data_to_return_dict_when_lags_dict_with_lags_only_for_level_and_window_features():
    """
    Test _create_data_to_return_dict output when lags is a dict with lags only for
    level and window_features.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=1, lags={'l1': 3, 'l2': None}, window_features=rolling
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'both'},
        ['l1', 'l2']
    )

    for k in forecaster.lags_.keys():
        if forecaster.lags_[k] is not None:
            np.testing.assert_array_almost_equal(forecaster.lags_[k], np.array([1, 2, 3]))
        else:
            assert forecaster.lags_[k] is None
    for k in forecaster.lags_names.keys():
        if forecaster.lags_names[k] is not None:
            assert forecaster.lags_names[k] == [f'{k}_lag_{i}' for i in range(1, 4)]
        else:
            assert forecaster.lags_names[k] is None
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]


def test_create_data_to_return_dict_when_lags_is_None():
    """
    Test _create_data_to_return_dict output when lags is None.
    """    
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=1, lags=None, window_features=rolling
    )
    
    results = forecaster._create_data_to_return_dict(series_names_in_=['l1', 'l2'])
    expected = (
        {'l1': 'y'},
        ['l1', 'l2']
    )

    for k in forecaster.lags_.keys():
        assert forecaster.lags_[k] is None
    for k in forecaster.lags_names.keys():
        assert forecaster.lags_names[k] is None
    for k in results[0].keys():
        assert results[0][k] == expected[0][k]
    assert results[1] == expected[1]
