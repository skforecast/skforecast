# Unit test initialize_steps
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import initialize_steps


def test_TypeError_initialize_steps_ForecasterAutoregDirect_when_steps_is_not_int_list_or_equivalent():
    """
    Test TypeError is raised when steps is not an int, 1d array, range, tuple or list.
    """
    steps = 'not_valid_type'
    err_msg = re.escape(
                 "`steps` argument must be an int, 1d numpy ndarray, range, tuple or list. "
                 f"Got {type(steps)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        initialize_steps(forecaster_name = 'ForecasterAutoregDirect', steps = steps)


def test_TypeError_initialize_steps_ForecasterAutoregMultiVariate_when_steps_is_not_int_list_or_equivalent():
    """
    Test TypeError is raised when steps is not a dict, int, 1d array, range, tuple or list.
    """
    steps = 'not_valid_type'
    err_msg = re.escape(
                 "`steps` argument must be a dict, int, 1d numpy ndarray, range, tuple or list. "
                 f"Got {type(steps)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        initialize_steps(forecaster_name = 'ForecasterAutoregMultiVariate', steps = steps)


def test_ValueError_initialize_steps_when_steps_is_int_less_than_1():
    """
    Test ValueError is raised when steps is less than 1.
    """
    steps = 0
    err_msg = re.escape(f"`steps` argument must be greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_steps(forecaster_name = 'ForecasterAutoregDirect', steps=steps)


def test_ValueError_initialize_steps_when_steps_numpy_ndarray_with_more_than_1_dimension():
    """
    Test ValueError is raised when steps is numpy ndarray with more than 1 dimension.
    """
    steps = np.ones((2, 2))
    err_msg = re.escape("`steps` must be a 1-dimensional array.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_steps(
            forecaster_name = 'ForecasterAutoregDirect',
            steps           = steps
        )


@pytest.mark.parametrize("steps",
                         [[], (), range(0), np.array([])],
                         ids = lambda steps : f'steps type: {type(steps)}')
def test_ValueError_initialize_steps_when_steps_list_tuple_range_or_numpy_ndarray_with_no_values(steps):
    """
    Test ValueError is raised when steps is list, tuple, range or numpy ndarray
    with no values.
    """
    err_msg = re.escape("Argument `steps` must contain at least one value.")
    with pytest.raises(ValueError, match = err_msg):
        initialize_steps(
            forecaster_name = 'ForecasterAutoregDirect',
            steps            = steps
        )


@pytest.mark.parametrize("steps",
                         [[1, 1.5],
                          (1, 1.5),
                          np.array([1.2, 1.5])],
                         ids = lambda steps : f'steps: {steps}')
def test_TypeError_initialize_steps_when_steps_list_tuple_or_numpy_array_with_values_not_int(steps):
    """
    Test TypeError is raised when steps is list, tuple or numpy ndarray with
    values not int.
    """
    err_msg = re.escape("All values in `steps` must be integers.")
    with pytest.raises(TypeError, match = err_msg):
        initialize_steps(
            forecaster_name = 'ForecasterAutoregDirect',
            steps           = steps
        )


@pytest.mark.parametrize("steps",
                         [[0, 1], (0, 1), range(0, 2), np.arange(0, 2)],
                         ids = lambda steps : f'steps: {steps}')
def test_ValueError_initialize_steps_when_steps_has_values_lower_than_1(steps):
    """
    Test ValueError is raised when steps is initialized with any value lower than 1.
    """
    err_msg = re.escape('Minimum value of steps allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        initialize_steps(
            forecaster_name = 'ForecasterAutoreg',
            steps           = steps
        )


@pytest.mark.parametrize("steps             , expected",
                         [(10              , np.arange(10) + 1),
                          ([1, 2, 3]       , np.array([1, 2, 3])),
                          ((1, 2, 3)       , np.array((1, 2, 3))),
                          (range(1, 4)     , np.array(range(1, 4))),
                          (np.arange(1, 10), np.arange(1, 10))],
                         ids = lambda values : f'values: {values}' )


def test_initialize_steps_input_steps_parameter(steps, expected):
    """
    Test creation of attribute steps with different arguments.
    """
    steps = initialize_steps(
               forecaster_name = 'ForecasterAutoregDirect',
               steps           = steps
           )
    assert (steps == expected).all()