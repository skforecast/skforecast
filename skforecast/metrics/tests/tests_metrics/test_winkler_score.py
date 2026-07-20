# Unit test winkler_score
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import winkler_score


def test_winkler_score_raise_error_when_invalid_inputs():
    """
    Test that winkler_score raises the expected TypeError and ValueError when
    the inputs are not valid.
    """
    y_true = np.array([100.0, 200.0])
    lower = np.array([90.0, 180.0])
    upper = np.array([110.0, 220.0])

    msg = "`y_true` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        winkler_score(y_true="invalid", lower_bound=lower, upper_bound=upper, alpha=0.05)

    msg = "`lower_bound` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        winkler_score(y_true=y_true, lower_bound="invalid", upper_bound=upper, alpha=0.05)

    msg = "`upper_bound` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        winkler_score(y_true=y_true, lower_bound=lower, upper_bound="invalid", alpha=0.05)

    msg = "`y_true`, `lower_bound`, and `upper_bound` must have the same shape."
    with pytest.raises(ValueError, match=re.escape(msg)):
        winkler_score(
            y_true=y_true,
            lower_bound=np.array([90.0, 180.0, 140.0]),
            upper_bound=upper,
            alpha=0.05,
        )

    msg = "`y_true` must have at least one element."
    with pytest.raises(ValueError, match=re.escape(msg)):
        winkler_score(
            y_true=np.array([]),
            lower_bound=np.array([]),
            upper_bound=np.array([]),
            alpha=0.05,
        )

    msg = "All values in `upper_bound` must be >= corresponding `lower_bound`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        winkler_score(
            y_true=np.array([100.0]),
            lower_bound=np.array([110.0]),
            upper_bound=np.array([90.0]),
            alpha=0.05,
        )


@pytest.mark.parametrize(
    "alpha",
    [0.0, 1.0, 1.5, "not_float"],
    ids=lambda alpha: f"alpha: {alpha}",
)
def test_winkler_score_ValueError_when_alpha_not_valid(alpha):
    """
    Test that winkler_score raises a ValueError when alpha is not a float
    strictly between 0 and 1.
    """
    y_true = np.array([100.0, 200.0])
    lower = np.array([90.0, 180.0])
    upper = np.array([110.0, 220.0])

    msg = "`alpha` must be a float strictly between 0 and 1."
    with pytest.raises(ValueError, match=re.escape(msg)):
        winkler_score(y_true=y_true, lower_bound=lower, upper_bound=upper, alpha=alpha)


@pytest.mark.parametrize(
    "y_true, lower_bound, upper_bound, alpha, expected",
    [
        (
            np.array([100.0, 200.0, 150.0]),
            np.array([90.0, 190.0, 140.0]),
            np.array([110.0, 210.0, 160.0]),
            0.05,
            20.0,
        ),
        (
            np.array([80.0]),
            np.array([100.0]),
            np.array([120.0]),
            0.05,
            820.0,
        ),
        (
            np.array([130.0]),
            np.array([100.0]),
            np.array([120.0]),
            0.05,
            420.0,
        ),
        (
            np.array([100.0, 200.0, 150.0, 80.0]),
            np.array([90.0, 180.0, 140.0, 100.0]),
            np.array([110.0, 220.0, 160.0, 120.0]),
            0.05,
            225.0,
        ),
        (
            np.array([80.0]),
            np.array([100.0]),
            np.array([120.0]),
            0.20,
            220.0,
        ),
    ],
    ids=["all_covered", "below_lower", "above_upper", "mixed", "alpha_80"],
)
def test_winkler_score_output(y_true, lower_bound, upper_bound, alpha, expected):
    """
    Test the numerical output of winkler_score for observations inside and
    outside the interval and for different significance levels.
    """
    result = winkler_score(y_true, lower_bound, upper_bound, alpha=alpha)

    assert isinstance(result, float)
    assert np.isclose(result, expected)


def test_winkler_score_output_with_pandas_series_input():
    """
    Test that winkler_score accepts pandas Series inputs and returns the same
    value as with numpy array inputs.
    """
    y_true = pd.Series([100.0, 200.0, 150.0])
    lower_bound = pd.Series([90.0, 190.0, 140.0])
    upper_bound = pd.Series([110.0, 210.0, 160.0])

    result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)

    assert isinstance(result, float)
    assert np.isclose(result, 20.0)
