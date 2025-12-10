# Unit test init
# ==============================================================================
import pytest
import re
from ....drift_detection import PopulationDriftDetector


def test_init_exception_when_chunk_size_not_valid():
    """
    Test that an exception is raised when an invalid chunk_size is provided.
    """

    error_msg = re.escape(
        "`chunk_size` must be a positive integer, a string compatible with "
        "pandas frequencies (e.g., 'D', 'W', 'M'), or None."
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


def test_init_exception_when_threshold_not_between_0_and_1_method_quantile():
    """
    Test that an exception is raised when threshold is not between 0 and 1.
    """

    threshold = -0.1
    error_msg = re.escape(f"`threshold` must be between 0 and 1. Got {threshold}.")
    with pytest.raises(ValueError, match=error_msg):
        PopulationDriftDetector(chunk_size='MS', threshold=threshold, threshold_method='quantile')

    threshold = 1.1
    error_msg = re.escape(f"`threshold` must be between 0 and 1. Got {threshold}.")
    with pytest.raises(ValueError, match=error_msg):
        PopulationDriftDetector(chunk_size='MS', threshold=threshold, threshold_method='quantile')





