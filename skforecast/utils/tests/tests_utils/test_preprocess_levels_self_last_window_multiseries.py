# Unit test preprocess_levels_self_last_window_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.utils import preprocess_levels_self_last_window_multiseries


def test_preprocess_levels_self_last_window_multiseries_IgnoredArgumentWarning_when_not_available_self_last_window_for_some_levels():
    """
    Test IgnoredArgumentWarning is raised when last window is not available for 
    levels because it was not stored during fit.
    """

    levels = ['1', '2', '3']
    input_levels_is_list = False
    last_window_ = {
        '1': pd.Series(np.arange(10), name='1'),
        '3': pd.Series(np.arange(10), name='3'),
    }

    warn_msg = re.escape(
        ("Levels {'2'} are excluded from prediction "
         "since they were not stored in `last_window_` attribute "
         "during training. If you don't want to retrain the "
         "Forecaster, provide `last_window` as argument.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        levels, last_window = preprocess_levels_self_last_window_multiseries(
                                  levels               = levels,
                                  input_levels_is_list = input_levels_is_list, 
                                  last_window_         = last_window_
                              )

    expected = (
        ['1', '3'],
        pd.DataFrame({'1': np.arange(10), '3': np.arange(10)})
    )
    
    assert levels == expected[0]
    pd.testing.assert_frame_equal(last_window, expected[1])


@pytest.mark.parametrize("last_window_",
                         [{'1': pd.Series(np.arange(10), name='1')}, None],
                         ids=lambda lw: f"last_window: {type(lw)}")
def test_preprocess_levels_self_last_window_multiseries_ValueError_when_not_available_self_last_window_for_levels(last_window_):
    """
    Test ValueError is raised when last window is not available for all 
    levels because it was not stored during fit.
    """
    
    levels = ['2']
    input_levels_is_list = False

    err_msg = re.escape(
        ("No series to predict. None of the series {'2'} are present in "
         "`last_window_` attribute. Provide `last_window` as argument "
         "in predict method.")
    )
    with pytest.raises(ValueError, match = err_msg):
        levels, last_window = preprocess_levels_self_last_window_multiseries(
                                  levels               = levels,
                                  input_levels_is_list = input_levels_is_list, 
                                  last_window_         = last_window_
                              )


def test_preprocess_levels_self_last_window_multiseries_IgnoredArgumentWarning_when_levels_is_list_and_different_last_index_in_self_last_window_DatetimeIndex():
    """
    Test IgnoredArgumentWarning is raised when levels is a list and have 
    different last index in last_window_ attribute using a DatetimeIndex.
    """

    levels = ['1', '2']
    input_levels_is_list = True
    last_window_ = {
        '1': pd.Series(np.arange(10), name='1'),
        '2': pd.Series(np.arange(5), name='2'),
    }
    last_window_['1'].index = pd.date_range(start='2020-01-01', periods=10)
    last_window_['2'].index = pd.date_range(start='2020-01-01', periods=5)

    expected = (
        ['1'],
        pd.DataFrame({'1': np.arange(10)},
                     index=pd.date_range(start='2020-01-01', periods=10))
    )
    
    warn_msg = re.escape(
        ("Only series whose last window ends at the same index "
         "can be predicted together. Series that do not reach the "
         "maximum index, '2020-01-10 00:00:00', are excluded "
         "from prediction: {'2'}.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        levels, last_window = preprocess_levels_self_last_window_multiseries(
                                  levels               = levels,
                                  input_levels_is_list = input_levels_is_list, 
                                  last_window_         = last_window_
                              )

    assert levels == expected[0]
    pd.testing.assert_frame_equal(last_window, expected[1])


@pytest.mark.parametrize("input_levels_is_list", 
                         [True, False], 
                         ids=lambda levels: f'input_levels_is_list: {levels}')
def test_output_preprocess_levels_self_last_window_multiseries(input_levels_is_list):
    """
    Test output preprocess_levels_self_last_window_multiseries for different inputs.
    """

    levels = ['1', '2', '3']
    last_window_ = {
        '1': pd.Series(np.arange(10), name='1'),
        '2': pd.Series(np.arange(10, 20), name='2'),
        '3': pd.Series(np.arange(20, 30), name='3'),
    }
    last_window_['1'].index = pd.date_range(start='2020-01-01', periods=10)
    last_window_['2'].index = pd.date_range(start='2020-01-01', periods=10)
    last_window_['3'].index = pd.date_range(start='2020-01-01', periods=10)

    expected = (
        ['1', '2', '3'],
        pd.DataFrame(
            {'1': np.arange(10),
             '2': np.arange(10, 20),
             '3': np.arange(20, 30)},
             index=pd.date_range(start='2020-01-01', periods=10))
    )

    levels, last_window = preprocess_levels_self_last_window_multiseries(
                              levels               = levels,
                              input_levels_is_list = input_levels_is_list, 
                              last_window_         = last_window_
                          )

    assert levels == expected[0]
    pd.testing.assert_frame_equal(last_window, expected[1])
