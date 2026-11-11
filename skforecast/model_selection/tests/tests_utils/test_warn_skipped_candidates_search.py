# Unit test _warn_skipped_candidates_search
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
from skforecast.model_selection._utils import _warn_skipped_candidates_search


def test_warn_skipped_candidates_search_no_warning_when_no_candidate_skipped():
    """
    Test that no warning is raised when no candidate is skipped.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_skipped_candidates_search(skipped=[], n_candidates=3)


def test_warn_skipped_candidates_search_grouped_by_error_and_lags():
    """
    Test that the candidates are grouped by error, and that a value of `lags`
    whose every combination of parameters raises the same error is summarized
    in one line.
    """
    error_80 = ValueError("Window 80.")
    error_90 = ValueError("Window 90.")
    skipped = [
        (np.arange(1, 81), {'alpha': 0.1}, error_80, None),
        (np.arange(1, 81), {'alpha': 1.0}, error_80, None),
        (np.arange(1, 91), {'alpha': 0.1}, error_90, None),
        (np.arange(1, 91), {'alpha': 1.0}, error_90, None),
        ('lags_3', {'alpha': 1.0}, TypeError("Other."), None),
    ]
    warn_msg = re.escape(
        "Parameters skipped: 5 of 6 candidates raise an error, so they are not "
        "in the results.\n"
        "\n"
        "ValueError: Window 80.\n"
        "  - lags: [1, 2, ..., 80] (2 candidates)\n"
        "\n"
        "ValueError: Window 90.\n"
        "  - lags: [1, 2, ..., 90] (2 candidates)\n"
        "\n"
        "TypeError: Other.\n"
        "  - lags: lags_3, params: {'alpha': 1.0}"
    )
    with pytest.warns(RuntimeWarning, match=f"^{warn_msg}$"):
        _warn_skipped_candidates_search(
            skipped=skipped, n_candidates=6, n_params=2
        )


def test_warn_skipped_candidates_search_max_errors_and_candidates():
    """
    Test that at most `max_errors` errors and `max_candidates` candidates per
    error are listed, and the rest are counted. When every candidate is
    skipped, the warning says that the results are empty.
    """
    skipped = (
        [(None, {'p': i}, ValueError("Same."), None) for i in range(4)]
        + [(None, {'p': i}, ValueError(f"Error {i}."), None) for i in range(4, 7)]
    )
    warn_msg = re.escape(
        "Parameters skipped: 7 of 7 candidates raise an error, so the results "
        "are empty.\n"
        "\n"
        "ValueError: Same.\n"
        "  - params: {'p': 0}\n"
        "  - params: {'p': 1}\n"
        "  ... and 2 more\n"
        "\n"
        "ValueError: Error 4.\n"
        "  - params: {'p': 4}\n"
        "\n"
        "... and 2 more errors"
    )
    with pytest.warns(RuntimeWarning, match=f"^{warn_msg}$"):
        _warn_skipped_candidates_search(
            skipped        = skipped,
            n_candidates   = 7,
            max_errors     = 2,
            max_candidates = 2
        )


def test_warn_skipped_candidates_search_trials_identified_by_number():
    """
    Test that the trials of a Bayesian search are identified by their number
    and listed in order, whatever the order in which they are skipped (the
    trials of Optuna can run in threads).
    """
    error = ValueError("Window 80.")
    skipped = [
        (None, {'lags': 80, 'alpha': 0.5}, error, 6),
        (None, {'lags': 80, 'alpha': 0.1}, error, 1),
    ]
    warn_msg = re.escape(
        "Parameters skipped: 2 of 8 trials raise an error, so they are not in "
        "the results.\n"
        "\n"
        "ValueError: Window 80.\n"
        "  - trial 1, params: {'lags': 80, 'alpha': 0.1}\n"
        "  - trial 6, params: {'lags': 80, 'alpha': 0.5}"
    )
    with pytest.warns(RuntimeWarning, match=f"^{warn_msg}$"):
        _warn_skipped_candidates_search(
            skipped=skipped, n_candidates=8, candidate_name='trial'
        )
