# Unit test prepare_steps_direct
# ==============================================================================
import re
import pytest
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


@pytest.mark.parametrize("steps, expected_steps", 
                         [(4, [1, 2, 3, 4]),
                          (None, [1, 2, 3, 4, 5]),
                          ([1, 3], [1, 3])], 
                         ids=lambda param: f'steps: {param}')
def test_output_prepare_steps_direct(steps, expected_steps):
    """
    Test output prepare_steps_direct for different inputs.
    """
    steps = prepare_steps_direct(
                max_step = 5, 
                steps    = steps
            )
    
    assert steps == expected_steps