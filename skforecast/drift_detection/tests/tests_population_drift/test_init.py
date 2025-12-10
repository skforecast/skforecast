# Unit test init
# ==============================================================================
import re
import pytest
from ....drift_detection import PopulationDriftDetector


def test_init_exception_when_chunk_size_not_valid():
    """
    Test that an exception is raised when an invalid chunk_size is provided.
    """

    error_msg = re.escape(
        "`chunk_size` must be a positive integer, a string compatible with "
        "pandas frequencies (e.g., 'D', 'W', 'MS'), or None."
    )

    chunk_size = ['a']
    with pytest.raises(TypeError, match=f"{error_msg} Got {type(chunk_size)}."):
        PopulationDriftDetector(chunk_size=chunk_size, threshold=0.95)

    chunk_size = 0
    with pytest.raises(ValueError, match=f"{error_msg} Got {chunk_size}."):
        PopulationDriftDetector(chunk_size=chunk_size, threshold=0.95)

    chunk_size = "non-valid"
    with pytest.raises(ValueError, match=f"{error_msg} Got {type(chunk_size)}."):
        PopulationDriftDetector(chunk_size=chunk_size, threshold=0.95)


def test_init_ValueError_when_threshold_method_not_valid():
    """
    Test that an ValueError is raised when threshold is not between 0 and 1.
    """

    threshold_method = 'not_valid'
    valid_threshold_methods = ['quantile', 'std']

    error_msg = re.escape(
        f"`threshold_method` must be one of {valid_threshold_methods}. "
        f"Got '{threshold_method}'."
    )
    with pytest.raises(ValueError, match=error_msg):
        PopulationDriftDetector(
            chunk_size='MS', threshold=0.5, threshold_method=threshold_method
        )


@pytest.mark.parametrize("threshold", 
                         [-0.1, 1.1], 
                         ids = lambda threshold: f'threshold: {threshold}')
def test_init_ValueError_when_threshold_not_between_0_and_1_method_quantile(threshold):
    """
    Test that an ValueError is raised when threshold is not between 0 and 1.
    """

    error_msg = re.escape(
        f"When `threshold_method='quantile'`, `threshold` must be between "
        f"0 and 1. Got {threshold}."
    )
    with pytest.raises(ValueError, match=error_msg):
        PopulationDriftDetector(
            chunk_size='MS', threshold=threshold, threshold_method='quantile'
        )


def test_init_ValueError_when_threshold_less_than_0_method_std():
    """
    Test that an ValueError is raised when threshold is not between 0 and 1.
    """

    threshold = -0.1

    error_msg = re.escape(
        f"When `threshold_method='std'`, `threshold` must be >= 0. "
        f"Got {threshold}."
    )
    with pytest.raises(ValueError, match=error_msg):
        PopulationDriftDetector(
            chunk_size='MS', threshold=threshold, threshold_method='std'
        )
