# Unit test prepare_steps_direct
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import prepare_steps_direct


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_TypeError_prepare_steps_direct_when_steps_list_contain_floats(steps):
    """
    Test TypeError when steps list contain floats.
    """

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        prepare_steps_direct(max_step=5, steps=steps)


@pytest.mark.parametrize("steps", [(1, 2), range(1, 3), np.array([1, 2]), 2.0],
                         ids=lambda steps: f'steps: {type(steps).__name__}')
def test_TypeError_prepare_steps_direct_when_steps_is_not_int_list_or_None(steps):
    """
    Test TypeError when steps is not an int, a list or None.
    """

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        prepare_steps_direct(max_step=5, steps=steps)


@pytest.mark.parametrize("steps", [0, -1, np.int64(0)],
                         ids=['zero', 'negative', 'numpy_zero'])
def test_ValueError_prepare_steps_direct_when_steps_is_int_less_than_1(steps):
    """
    Test ValueError when steps is an integer less than 1.
    """

    err_msg = re.escape(
        f"`steps` must be an integer greater than or equal to 1. Got {steps}."
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_steps_direct(max_step=5, steps=steps)


def test_ValueError_prepare_steps_direct_when_steps_is_empty_list():
    """
    Test ValueError when steps is an empty list.
    """

    err_msg = re.escape("`steps` cannot be an empty list.")
    with pytest.raises(ValueError, match = err_msg):
        prepare_steps_direct(max_step=5, steps=[])


@pytest.mark.parametrize("steps, max_step, expected_steps", 
                         [(4, 5, [1, 2, 3, 4]),
                          (np.int64(4), 5, [1, 2, 3, 4]),
                          (None, 5, [1, 2, 3, 4, 5]),
                          (None, [1, 3], [1, 3]),
                          ([1, 3], 5, [1, 3]),
                          ([np.int64(1), 3], 5, [1, 3])],
                         ids=lambda param: f'steps: {param}')
def test_output_prepare_steps_direct(steps, max_step, expected_steps):
    """
    Test output prepare_steps_direct for different inputs.
    """
    steps = prepare_steps_direct(
                max_step = max_step, 
                steps    = steps
            )
    
    assert steps == expected_steps