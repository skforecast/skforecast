# Unit test _format_lags_warning
# ==============================================================================
import pytest
import numpy as np
from skforecast.model_selection._utils import _format_lags_warning


@pytest.mark.parametrize(
    "lags, expected",
    [
        (np.array([1, 2, 3]), "[1, 2, 3]"),
        (np.arange(1, 7), "[1, 2, 3, 4, 5, 6]"),
        (np.arange(1, 25), "[1, 2, ..., 24]"),
        ([1, 7, 14], "[1, 7, 14]"),
        (None, "None"),
        ("lags_24", "lags_24"),
        ({'l1': np.arange(1, 9), 'l2': None}, "{'l1': [1, 2, ..., 8], 'l2': None}"),
    ],
    ids=["array", "6 lags", "24 lags", "list", "None", "label", "dict"],
)
def test_format_lags_warning_output(lags, expected):
    """
    Test the representation of the lags of a candidate: arrays with more than
    6 lags show the first two and the last, also inside a dict
    (ForecasterDirectMultiVariate), and labels are kept.
    """
    assert _format_lags_warning(lags) == expected
