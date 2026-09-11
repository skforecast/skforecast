# Unit test check_level
# ==============================================================================
import re
import pytest
import numpy as np

from skforecast.stats._utils import check_level


@pytest.mark.parametrize("level, expected",
                         [(0.8, [0.8]),
                          (1, [1.0]),
                          (np.float64(0.95), [0.95]),
                          ([0.8, 0.95], [0.8, 0.95]),
                          ((0.025, 0.975), [0.025, 0.975]),
                          (np.array([0.5, 0.9]), [0.5, 0.9])],
                         ids = lambda value: f'level: {value}')
def test_check_level_output_when_valid_coverage_proportions(level, expected):
    """
    Test check_level returns a list of floats when `level` is within (0, 1].
    """
    results = check_level(level)

    assert results == expected
    assert all(isinstance(v, float) for v in results)


@pytest.mark.parametrize("level",
                         [90, [80, 95], (95,), 0, -0.1, [0.9, 95]],
                         ids = lambda value: f'level: {value}')
def test_check_level_ValueError_when_level_is_out_of_range(level):
    """
    Test ValueError is raised when `level` is outside the (0, 1] range, for
    example when given as percentiles. Support for percentiles was removed in
    skforecast 0.25.0.
    """
    err_msg = re.escape(
        f"All values in `level` must be coverage proportions in the (0, 1] "
        f"range, e.g. `level=[0.8, 0.95]`. Got {level}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_level(level)
