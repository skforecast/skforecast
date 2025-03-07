# Unit test crps_from_predictions
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import calculate_coverage


def test_calculate_coverage_raise_error_when_no_valid_inputs():
    """
    Test that coverage function raises an error when no valid inputs are provided.
    """
    y_true = pd.Series(np.array([1, 2]))
    lower_bound = np.array([1, 2])
    upper_bound = np.array([2, 3])

    msg = "`y_true` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        calculate_coverage(y_true='invalid value', lower_bound=lower_bound, upper_bound=upper_bound)

    msg = "`lower_bound` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        calculate_coverage(y_true=y_true, lower_bound='invalid value', upper_bound=upper_bound)

    msg = "`upper_bound` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        calculate_coverage(y_true=y_true, lower_bound=lower_bound, upper_bound='invalid value')

    msg = "`y_true`, `lower_bound` and `upper_bound` must have the same shape."
    with pytest.raises(ValueError, match=re.escape(msg)):
        calculate_coverage(y_true, lower_bound=np.array([1, 2, 3]), upper_bound=upper_bound)
    with pytest.raises(ValueError, match=re.escape(msg)):
        calculate_coverage(y_true, lower_bound=lower_bound, upper_bound=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match=re.escape(msg)):
        calculate_coverage(y_true=np.array([1, 2, 3]), lower_bound=lower_bound, upper_bound=upper_bound)

    
def test_calculate_coverage_output_10_out_of_100_values_outside_bounds():
    """
    Test the output of the coverage function when 10 out of 100 values are outside
    the upper and lower bounds.
    """
    lower_bound = np.random.normal(10, 2, 100)
    upper_bound = lower_bound * 2
    y_true = (lower_bound + upper_bound) / 2
    y_true[[10, 12, 15, 20, 25, 30, 35, 40, 45, 50]] = (
        y_true[[10, 12, 15, 20, 25, 30, 35, 40, 45, 50]] * 10
    )
    results = calculate_coverage(y_true, lower_bound, upper_bound)
    expected = 0.9

    assert results == expected


def test_calculate_coverage_output_100_out_of_100_values_outside_bounds():
    """
    Test the output of the coverage function when 10 out of 100 values are outside
    the upper and lower bounds.
    """
    lower_bound = np.random.normal(10, 2, 100)
    upper_bound = lower_bound * 2
    y_true = upper_bound * 10
    results = calculate_coverage(y_true, lower_bound, upper_bound)
    expected = 0

    assert results == expected


def test_calculate_coverage_output_100_out_of_100_values_inside_bounds():
    """
    Test the output of the coverage function when all values are inside the upper and
    lower bounds.
    """
    lower_bound = np.random.normal(10, 2, 100)
    upper_bound = lower_bound * 2
    y_true = (lower_bound + upper_bound) / 2
    results = calculate_coverage(y_true, lower_bound, upper_bound)
    expected = 1.0

    assert results == expected
