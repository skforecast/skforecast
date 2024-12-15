# Unit test crps_from_predictions
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import coverage

def test_coverage_raise_error_when_no_valid_inputs():

    y = np.array([1, 2])
    lower_bound = np.array([1, 2])
    upper_bound = np.array([2, 3])

    msg = "`y`, must be a 1D numpy array."
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y='invalid value', lower_bound=lower_bound, upper_bound=upper_bound)

    msg = "`lower_bound`, must be a 1D numpy array."
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y=y, lower_bound='invalid value', upper_bound=upper_bound)

    msg = "`upper_bound`, must be a 1D numpy array."
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y=y, lower_bound=lower_bound, upper_bound='invalid value')

    msg = "`y`, `lower_bound`, and `upper_bound` must have the same shape."
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y, lower_bound=np.array([1, 2, 3]), upper_bound=upper_bound)
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y, lower_bound=lower_bound, upper_bound=np.array([1, 2, 3]))
    with pytest.raises(TypeError, match=re.escape(msg)):
        coverage(y=np.array([1, 2, 3]), lower_bound=lower_bound, upper_bound=upper_bound)

    
def test_coverage_output():
    """
    Test the output of the coverage function when 10 out of 100 values are outside
    the upper and lower bounds.
    """
    lower_bound = np.random.normal(10, 2, 100)
    upper_bound = lower_bound * 2
    y = (lower_bound + upper_bound) / 2
    y[[10, 12, 15, 20, 25, 30, 35, 40, 45, 50]] = (
        y[[10, 12, 15, 20, 25, 30, 35, 40, 45, 50]] * 10
    )
    results = coverage(y, lower_bound, upper_bound)
    expected = 0.9

    assert results == expected
