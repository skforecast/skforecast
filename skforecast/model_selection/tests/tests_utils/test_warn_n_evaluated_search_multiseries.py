# Unit test _warn_n_evaluated_search_multiseries
# ==============================================================================
import re
import warnings
import pytest
from skforecast.exceptions import MissingValuesWarning
from skforecast.model_selection._utils import _warn_n_evaluated_search_multiseries


@pytest.mark.parametrize(
    "n_evaluated",
    [[100, 100, 100], [100, 99, 100], [0, 0], []],
    ids=lambda n: f"n_evaluated: {n}",
)
def test_warn_n_evaluated_search_multiseries_no_warning(n_evaluated):
    """
    Test that no warning is raised when every candidate evaluates at least 99%
    of the points of the most complete one (99% included), or there are no
    candidates.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_n_evaluated_search_multiseries(
            n_evaluated = n_evaluated,
            candidates  = [f"candidate {i}" for i in range(len(n_evaluated))]
        )


def test_warn_n_evaluated_search_multiseries_MissingValuesWarning_when_best_is_affected():
    """
    Test the warning when the best candidate (the first) evaluates less than
    99% of the points of the most complete one: candidates listed in the order
    of the ranking, up to `max_candidates`, with the rest counted.
    """
    warn_msg = re.escape(
        "The candidates are not evaluated on the same points, so their metrics "
        "are not comparable. The most complete candidate evaluates 1,000 points; "
        "4 of 5 candidates evaluate less than 99% of them:\n"
        "  - c0: 989 points (98.9%)\n"
        "  - c2: 500 points (50.0%)\n"
        "  ... and 2 more\n"
        "The best candidate of the ranking (first row of the results) is one of "
        "them, so the ranking may favor it. Common causes: series shorter than "
        "the window of the candidate (`window_size`), missing values in the last "
        "window or NaN predictions. To compare the candidates on the same points, "
        "pass in `levels` only the series that every candidate predicts in every "
        "fold (for short series, those with more observations than the largest "
        "`window_size` before the first fold), or remove the candidates with "
        "larger windows. See the column `n_evaluated` of the results."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_n_evaluated_search_multiseries(
            n_evaluated    = [989, 1000, 500, 10, 0],
            candidates     = ["c0", "c1", "c2", "c3", "c4"],
            max_candidates = 2
        )


def test_warn_n_evaluated_search_multiseries_MissingValuesWarning_when_best_is_not_affected():
    """
    Test that the warning says when the best candidate is not one of the
    candidates affected.
    """
    warn_msg = re.escape(
        "1 of 2 candidates evaluate less than 99% of them:\n"
        "  - c1: 50 points (50.0%)\n"
        "The best candidate of the ranking (first row of the results) is not one "
        "of them."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_n_evaluated_search_multiseries(
            n_evaluated = [100, 50],
            candidates  = ["c0", "c1"]
        )
