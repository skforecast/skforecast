# Unit test initialize_lags
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import initialize_lags


def test_ValueError_initialize_lags_when_lags_is_int_lower_than_1():
    """
    Test ValueError is raised when lags is initialized with int lower than 1.
    """
    err_msg = re.escape("Minimum value of lags allowed is 1.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = -10
        )


def test_ValueError_initialize_lags_when_lags_numpy_ndarray_with_more_than_1_dimension():
    """
    Test ValueError is raised when lags is numpy ndarray with more than 1 dimension.
    """
    lags = np.ones((2, 2))

    err_msg = re.escape("`lags` must be a 1-dimensional array.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [[], (), range(0), np.array([])], 
                         ids = lambda lags: f'lags type: {type(lags)}')
def test_ValueError_initialize_lags_when_lags_list_tuple_range_or_numpy_ndarray_with_no_values(lags):
    """
    Test ValueError is raised when lags is list, tuple, range or numpy ndarray 
    with no values.
    """
    err_msg = re.escape("Argument `lags` must contain at least one value.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [[1, 1.5], 
                          (1, 1.5), 
                          np.array([1.2, 1.5])], 
                         ids = lambda lags: f'lags: {lags}')
def test_TypeError_initialize_lags_when_lags_list_tuple_or_numpy_array_with_values_not_int(lags):
    """
    Test TypeError is raised when lags is list, tuple or numpy ndarray with 
    values not int.
    """
    err_msg = re.escape("All values in `lags` must be integers.")
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [[0, 1], (0, 1), range(0, 2), np.arange(0, 2)], 
                         ids = lambda lags: f'lags: {lags}')
def test_ValueError_initialize_lags_when_lags_has_values_lower_than_1(lags):
    """
    Test ValueError is raised when lags is initialized with any value lower than 1.
    """
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [1.5, 'not_valid_type'], 
                         ids = lambda lags: f'lags: {lags}')
def test_TypeError_initialize_lags_when_lags_is_not_valid_type(lags):
    """
    Test TypeError is raised when lags is not a valid type.
    """
    err_msg = re.escape(
        (f"`lags` argument must be an int, 1d numpy ndarray, range, tuple or list. "
         f"Got {type(lags)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterRecursive',
            lags            = lags
        )


@pytest.mark.parametrize("lags", 
                         [1.5, 'not_valid_type'], 
                         ids = lambda lags: f'lags: {lags}')
def test_TypeError_initialize_lags_when_lags_is_not_valid_type_ForecasterAutoregMultiVariate(lags):
    """
    Test TypeError is raised when lags is not a valid type in ForecasterDirectMultiVariate.
    """    
    err_msg = re.escape(
        (f"`lags` argument must be a dict, int, 1d numpy ndarray, range, tuple or list. "
         f"Got {type(lags)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags(
            forecaster_name = 'ForecasterDirectMultiVariate',
            lags            = lags
        )


@pytest.mark.parametrize("lags             , expected", 
                         [(None            , (None, None, None)),
                          (10              , (np.arange(10) + 1, [f'lag_{i}' for i in range(1, 11)], 10)), 
                          ([1, 2, 3]       , (np.array([1, 2, 3]), ['lag_1', 'lag_2', 'lag_3'], 3)),
                          ((4, 5, 6)       , (np.array((4, 5, 6)), ['lag_4', 'lag_5', 'lag_6'], 6)),  
                          (range(1, 4)     , (np.array([1, 2, 3]), ['lag_1', 'lag_2', 'lag_3'], 3)), 
                          (np.arange(1, 10), (np.arange(1, 10), [f'lag_{i}' for i in range(1, 10)], 9))], 
                         ids = lambda values: f'values: {values}')
def test_initialize_lags_input_lags_parameter(lags, expected):
    """
    Test creation of attribute lags with different arguments.
    """
    lags, lags_names, max_lag = initialize_lags(
                                    forecaster_name = 'ForecasterRecursive',
                                    lags            = lags
                                )

    if lags is None:
        assert lags is expected[0]
        assert lags_names is expected[1]
        assert max_lag is expected[2]
    else:
        np.testing.assert_array_almost_equal(lags, expected[0])
        assert lags_names == expected[1]
        assert max_lag == expected[2]
