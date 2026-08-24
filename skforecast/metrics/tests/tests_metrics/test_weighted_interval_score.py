# Unit test weighted_interval_score
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import weighted_interval_score


def test_weighted_interval_score_raise_error_when_invalid_inputs():
    """
    Test that weighted_interval_score raises the expected TypeError and
    ValueError when the inputs are not valid.
    """
    y_true = np.array([100.0, 200.0])
    pf = np.array([98.0, 195.0])
    lb = np.array([[88.0, 80.0], [175.0, 165.0]])
    ub = np.array([[108.0, 118.0], [215.0, 225.0]])
    alphas = np.array([0.20, 0.05])

    msg = "`y_true` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        weighted_interval_score("invalid", pf, lb, ub, alphas)

    msg = "`y_pred` must be a 1D numpy array or pandas Series."
    with pytest.raises(TypeError, match=re.escape(msg)):
        weighted_interval_score(y_true, "invalid", lb, ub, alphas)

    msg = "`lower_bounds` must be a 2D array with shape (n_observations, n_intervals)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, np.array([88.0, 175.0]), ub, alphas)

    msg = "`upper_bounds` must be a 2D array with shape (n_observations, n_intervals)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, lb, np.array([108.0, 215.0]), alphas)

    msg = "`alphas` must be a 1D array of significance levels."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, lb, ub, np.array([[0.20, 0.05]]))

    msg = "`y_true` must have at least one element."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(
            np.array([]), np.array([]), np.zeros((0, 2)), np.zeros((0, 2)), alphas
        )

    msg = "`y_true` and `y_pred` must have the same length."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, np.array([98.0]), lb, ub, alphas)

    msg = "`lower_bounds` must have shape (2, 2). Got (2, 1)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, np.array([[88.0], [175.0]]), ub, alphas)

    msg = "`upper_bounds` must have shape (2, 2). Got (2, 1)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, lb, np.array([[108.0], [215.0]]), alphas)

    msg = "All values in `alphas` must be strictly between 0 and 1."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(y_true, pf, lb, ub, np.array([0.0, 0.05]))

    msg = "All values in `upper_bounds` must be >= corresponding `lower_bounds`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        weighted_interval_score(
            y_true,
            pf,
            np.array([[88.0, 80.0], [175.0, 165.0]]),
            np.array([[108.0, 118.0], [160.0, 225.0]]),
            alphas,
        )


@pytest.mark.parametrize(
    "y_true, y_pred, lower_bounds, upper_bounds, alphas, expected",
    [
        (
            np.array([100.0]),
            np.array([100.0]),
            np.array([[90.0]]),
            np.array([[110.0]]),
            np.array([0.05]),
            0.3333333333333333,
        ),
        (
            np.array([100.0]),
            np.array([98.0]),
            np.array([[88.0, 80.0]]),
            np.array([[108.0, 118.0]]),
            np.array([0.20, 0.05]),
            1.58,
        ),
        (
            np.array([100.0, 200.0, 150.0]),
            np.array([98.0, 195.0, 155.0]),
            np.array([[88.0, 80.0], [175.0, 165.0], [138.0, 128.0]]),
            np.array([[108.0, 118.0], [215.0, 225.0], [168.0, 178.0]]),
            np.array([0.20, 0.05]),
            2.4933333333333336,
        ),
        (
            np.array([100.0, 200.0]),
            np.array([100.0, 200.0]),
            np.array([[100.0, 100.0], [200.0, 200.0]]),
            np.array([[100.0, 100.0], [200.0, 200.0]]),
            np.array([0.20, 0.05]),
            0.0,
        ),
    ],
    ids=[
        "single_interval_covered",
        "two_intervals_covered",
        "three_observations",
        "perfect_and_degenerate",
    ],
)
def test_weighted_interval_score_output(
    y_true, y_pred, lower_bounds, upper_bounds, alphas, expected
):
    """
    Test the numerical output of weighted_interval_score for single and
    multiple intervals, following the definition of Bracher et al. (2021).
    """
    result = weighted_interval_score(
        y_true, y_pred, lower_bounds, upper_bounds, alphas
    )

    assert isinstance(result, float)
    assert np.isclose(result, expected)


def test_weighted_interval_score_increases_when_observation_outside_interval():
    """
    Test that weighted_interval_score increases when the observation moves
    outside the prediction intervals.
    """
    y_pred = np.array([100.0])
    lower_bounds = np.array([[88.0, 80.0]])
    upper_bounds = np.array([[108.0, 118.0]])
    alphas = np.array([0.20, 0.05])

    result_inside = weighted_interval_score(
        np.array([100.0]), y_pred, lower_bounds, upper_bounds, alphas
    )
    result_outside = weighted_interval_score(
        np.array([200.0]), y_pred, lower_bounds, upper_bounds, alphas
    )

    assert result_outside > result_inside


def test_weighted_interval_score_output_with_pandas_series_and_list_alphas():
    """
    Test that weighted_interval_score accepts pandas Series inputs and a list
    of alphas, returning the same value as with numpy array inputs.
    """
    y_true = pd.Series([100.0])
    y_pred = pd.Series([98.0])
    lower_bounds = np.array([[88.0, 80.0]])
    upper_bounds = np.array([[108.0, 118.0]])

    result = weighted_interval_score(
        y_true, y_pred, lower_bounds, upper_bounds, [0.20, 0.05]
    )

    assert isinstance(result, float)
    assert np.isclose(result, 1.58)


def test_weighted_interval_score_output_with_pandas_dataframe_bounds():
    """
    Test that weighted_interval_score accepts pandas DataFrame bounds and
    returns the same value as with numpy array bounds.
    """
    y_true = np.array([100.0, 200.0, 150.0])
    y_pred = np.array([98.0, 195.0, 155.0])
    lower_bounds = pd.DataFrame(
        [[88.0, 80.0], [175.0, 165.0], [138.0, 128.0]], columns=["0.8", "0.95"]
    )
    upper_bounds = pd.DataFrame(
        [[108.0, 118.0], [215.0, 225.0], [168.0, 178.0]], columns=["0.8", "0.95"]
    )
    alphas = np.array([0.20, 0.05])

    result = weighted_interval_score(
        y_true, y_pred, lower_bounds, upper_bounds, alphas
    )

    assert isinstance(result, float)
    assert np.isclose(result, 2.4933333333333336)
