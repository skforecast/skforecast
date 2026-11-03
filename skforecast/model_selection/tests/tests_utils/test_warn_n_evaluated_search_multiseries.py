# Unit test _warn_n_evaluated_search_multiseries
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning
from skforecast.model_selection._utils import _warn_n_evaluated_search_multiseries


@pytest.mark.parametrize(
    "n_evaluated",
    [[100, 100, 100], [100, 95, 100], [21, 20], [0, 0], []],
    ids=lambda n: f"n_evaluated: {n}",
)
def test_warn_n_evaluated_search_multiseries_no_warning(n_evaluated):
    """
    Test that no warning is raised when every candidate evaluates at least 95%
    of the points of the most complete one (95% included, 20 of 21 points is
    95.2%), or there are no candidates.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated,
            n_evaluated_levels = [[n] for n in n_evaluated],
            candidates         = [f"c{i}" for i in range(len(n_evaluated))],
            levels             = ['l1']
        )


def test_warn_n_evaluated_search_multiseries_MissingValuesWarning_when_best_is_affected():
    """
    Test the warning when the best candidate (the first) evaluates less than
    95% of the points of the most complete one: candidates listed in the order
    of the ranking, up to `max_candidates`, with the rest counted. The levels
    that every candidate evaluates on the same points are listed when they are
    the shortest list, and the levels without points evaluated are left out.
    """
    n_evaluated_levels = np.array([
        [100, 300, 500, 0],
        [100, 400, 500, 0],
        [100, 400, 0,   0],
        [100, 0,   0,   0],
        [100, 0,   0,   0],
    ])
    warn_msg = re.escape(
        "The candidates are not evaluated on the same points, so their metrics "
        "are not comparable. The most complete candidate evaluates 1,000 points "
        "(all series and folds); 4 of 5 candidates evaluate fewer than 950 (95%):\n"
        "  - c0: 900 points (90.0%)\n"
        "  - c2: 500 points (50.0%)\n"
        "  ... and 2 more\n"
        "The best candidate of the ranking (first row of the results) is one of "
        "them, so the ranking may favor it.\n"
        "To compare the candidates on the same points, pass in `levels` the 1 of 3 "
        "series that every candidate evaluates on the same points:\n"
        "  ['l1']\n"
        "or remove the candidates with larger windows.\n"
        "Common causes: series with 24 observations or fewer up to 2021-06-01 "
        "(the largest `window_size`), missing values in the last window or NaN "
        "predictions. See the column `n_evaluated` of the results."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1", "c2", "c3", "c4"],
            levels             = ['l1', 'l2', 'l3', 'l4'],
            window_size        = 24,
            last_train_date    = pd.Timestamp('2021-06-01'),
            max_candidates     = 2
        )


@pytest.mark.parametrize(
    "levels_other, expected",
    [
        (
            ['l07'],
            "pass in `levels` the 11 of 12 series that every candidate evaluates "
            "on the same points, all except:\n  ['l07']\n"
            "or remove the candidates with larger windows."
        ),
        (
            ['l00', 'l01', 'l02', 'l03', 'l04', 'l05'],
            "pass in `levels` the 6 of 12 series that every candidate evaluates "
            "on the same points:\n  ['l06', 'l07', 'l08', 'l09', 'l10', 'l11']\n"
            "or remove the candidates with larger windows."
        ),
        (
            [f'l{i:02d}' for i in range(12)],
            "No series is evaluated on the same points by every candidate, so "
            "restricting `levels` does not help: remove the candidates with "
            "larger windows."
        ),
    ],
    ids=["all except", "same points", "none"],
)
def test_warn_n_evaluated_search_multiseries_levels_message(levels_other, expected):
    """
    Test the levels given in the warning: the shortest list, the levels that
    every candidate evaluates on the same points or all except the rest, and
    a message when there are none. The best candidate is not affected.
    """
    levels = [f'l{i:02d}' for i in range(12)]
    n_evaluated_levels = np.array([
        [10] * 12,
        [0 if level in levels_other else 10 for level in levels],
    ])
    n_evaluated = n_evaluated_levels.sum(axis=1)

    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated,
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1"],
            levels             = levels
        )

    message = str(record[0].message)
    assert expected in message
    assert (
        "The best candidate of the ranking (first row of the results) is not one "
        "of them."
    ) in message
    assert "Common causes: series shorter than the window of the candidate," in message


def test_warn_n_evaluated_search_multiseries_levels_message_when_both_lists_are_long():
    """
    Test that, when both lists of levels are longer than `max_levels`, only
    their size and some of the levels not evaluated on the same points are
    shown, and the position of the training end when the index is not a
    DatetimeIndex.
    """
    levels = [f'l{i:02d}' for i in range(30)]
    n_evaluated_levels = np.array([
        [10] * 30,
        [10] * 15 + [5] * 15,
    ])

    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1"],
            levels             = levels,
            window_size        = 7,
            last_train_date    = 99
        )

    message = str(record[0].message)
    assert (
        "pass in `levels` the 15 of 30 series that every candidate evaluates on "
        "the same points (the other 15 include 'l15', 'l16', ..., 'l29'), or "
        "remove the candidates with larger windows."
    ) in message
    assert "series with 7 observations or fewer up to position 99" in message


def test_warn_n_evaluated_search_multiseries_one_level():
    """
    Test that, with one level (for example, ForecasterDirectMultiVariate), the
    warning does not suggest `levels`.
    """
    warn_msg = re.escape(
        "The most complete candidate evaluates 100 points (all folds); 1 of 2 "
        "candidates evaluate fewer than 95 (95%):\n"
        "  - c1: 50 points (50.0%)\n"
        "The best candidate of the ranking (first row of the results) is not one "
        "of them.\n"
        "To compare the candidates on the same points, remove the candidates with "
        "larger windows.\n"
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = [100, 50],
            n_evaluated_levels = [[100], [50]],
            candidates         = ["c0", "c1"],
            levels             = ['l1']
        )
