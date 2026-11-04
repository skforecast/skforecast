# Unit test _warn_n_evaluated_search_multiseries
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
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
    of the ranking, up to `max_candidates`, with the rest counted, and the
    levels that every candidate evaluates on the same points. The levels
    without points evaluated are left out.
    """
    n_evaluated_levels = np.array([
        [100, 300, 500, 0],
        [100, 400, 500, 0],
        [100, 400, 0,   0],
        [100, 0,   0,   0],
        [100, 0,   0,   0],
    ])
    warn_msg = re.escape(
        "4 of 5 candidates are evaluated on fewer points than the most complete "
        "one, so their metrics are not comparable. The best candidate of the "
        "ranking is one of them, so the ranking may favor it. To compare the "
        "candidates on the same points, pass the series listed below in `levels` "
        "or remove the candidates with larger windows. See the column "
        "`n_evaluated` of the results.\n"
        "\n"
        "Candidates with less than 95% of the 1,000 points of the most complete "
        "one:\n"
        "  - c0: 900 (90.0%)\n"
        "  - c2: 500 (50.0%)\n"
        "  ... and 2 more\n"
        "\n"
        "Series that every candidate evaluates on the same points:\n"
        "  levels=['l1']"
    )
    with pytest.warns(MissingValuesWarning, match=f"^{warn_msg}"):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1", "c2", "c3", "c4"],
            levels             = ['l1', 'l2', 'l3', 'l4'],
            max_candidates     = 2
        )


@pytest.mark.parametrize(
    "levels_other, expected",
    [
        (
            ['l07'],
            "Series that every candidate evaluates on the same points:\n"
            "  levels: every series except ['l07']"
        ),
        (
            [f'l{i:02d}' for i in range(12) if i not in (3, 5)],
            "Series that every candidate evaluates on the same points:\n"
            "  levels=['l03', 'l05']"
        ),
        (
            [f'l{i:02d}' for i in range(12)],
            "No series is evaluated on the same points by every candidate, so "
            "restricting `levels` does not help: remove the candidates with "
            "larger windows."
        ),
    ],
    ids=["every series except", "levels", "none"],
)
def test_warn_n_evaluated_search_multiseries_levels_and_one_candidate_not_the_best(
    levels_other, expected
):
    """
    Test the levels given in the warning: the levels that every candidate
    evaluates on the same points, every level except the rest, or a message
    when there are none. With one candidate affected, which is not the best,
    the warning is written in singular.
    """
    levels = [f'l{i:02d}' for i in range(12)]
    n_evaluated_levels = np.array([
        [10] * 12,
        [0 if level in levels_other else 10 for level in levels],
    ])

    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1"],
            levels             = levels
        )

    message = str(record[0].message)
    assert message.startswith(
        "1 of 2 candidates is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. It is not the best candidate of "
        "the ranking."
    )
    assert expected in message
    if levels_other == levels:
        assert "Series that every candidate" not in message


def test_warn_n_evaluated_search_multiseries_levels_when_both_lists_are_long():
    """
    Test that, when both lists of levels are longer than `max_levels`, only
    their size and some of the levels left out are shown. With one candidate
    affected, which is the best, the warning is written in singular.
    """
    levels = [f'l{i:02d}' for i in range(30)]
    n_evaluated_levels = np.array([
        [10] * 15 + [5] * 15,
        [10] * 30,
    ])

    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            candidates         = ["c0", "c1"],
            levels             = levels
        )

    message = str(record[0].message)
    assert (
        "It is the best candidate of the ranking, so the ranking may favor it."
    ) in message
    assert (
        "To compare the candidates on the same points, pass in `levels` only the "
        "series that every candidate evaluates on the same points (counted below) "
        "or remove the candidates with larger windows."
    ) in message
    assert (
        "Series that every candidate evaluates on the same points:\n"
        "  levels: 15 of 30 series (the series left out include 'l15', 'l16', "
        "..., 'l29')"
    ) in message


def test_warn_n_evaluated_search_multiseries_one_level():
    """
    Test that, with one level (for example, ForecasterDirectMultiVariate), the
    warning does not suggest `levels`.
    """
    warn_msg = re.escape(
        "1 of 2 candidates is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. It is not the best candidate of "
        "the ranking. To compare the candidates on the same points, remove the "
        "candidates with larger windows. See the column `n_evaluated` of the "
        "results.\n"
        "\n"
        "Candidates with less than 95% of the 100 points of the most complete "
        "one:\n"
        "  - c1: 50 (50.0%)"
    )
    with pytest.warns(MissingValuesWarning, match=f"^{warn_msg}"):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = [100, 50],
            n_evaluated_levels = [[100], [50]],
            candidates         = ["c0", "c1"],
            levels             = ['l1']
        )


def test_warn_n_evaluated_search_multiseries_when_best_is_not_first():
    """
    Test that the warning does not say whether the best candidate is affected
    when the best candidate is not among the candidates (`best_is_first=False`).
    """
    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = [50, 100],
            n_evaluated_levels = [[50], [100]],
            candidates         = ["c0", "c1"],
            levels             = ['l1'],
            best_is_first      = False
        )

    message = str(record[0].message)
    assert message.startswith(
        "1 of 2 candidates is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. To compare the candidates"
    )
    assert "best candidate" not in message

