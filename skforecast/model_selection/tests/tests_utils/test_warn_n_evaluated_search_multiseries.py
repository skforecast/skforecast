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
            lags               = [np.arange(1, 4)] * len(n_evaluated),
            candidate_ids      = list(range(len(n_evaluated))),
            levels             = ['l1']
        )


def test_warn_n_evaluated_search_multiseries_MissingValuesWarning_when_best_is_affected():
    """
    Test the warning when the best candidate (the first) evaluates less than
    95% of the points of the most complete one: candidates listed in the order
    of the ranking with their row, up to `max_lines`, with the rest counted,
    and the levels that every candidate evaluates on the same points. The
    levels without points evaluated are left out.
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
        "  - lags: [1, 2, 3], row 0: 900 (90.0%)\n"
        "  - lags: [1, 2, 3, 4, 5], row 2: 500 (50.0%)\n"
        "  ... and 2 more candidates\n"
        "\n"
        "Series that every candidate evaluates on the same points:\n"
        "  levels=['l1']"
    )
    with pytest.warns(MissingValuesWarning, match=f"^{warn_msg}"):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated_levels.sum(axis=1),
            n_evaluated_levels = n_evaluated_levels,
            lags               = [np.arange(1, k) for k in range(4, 9)],
            candidate_ids      = [0, 1, 2, 3, 4],
            levels             = ['l1', 'l2', 'l3', 'l4'],
            max_lines          = 2
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
            lags               = [np.arange(1, 4)] * 2,
            candidate_ids      = [0, 1],
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
            lags               = [np.arange(1, 4)] * 2,
            candidate_ids      = [0, 1],
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
        "  - lags: [1, 2, 3], row 1: 50 (50.0%)"
    )
    with pytest.warns(MissingValuesWarning, match=f"^{warn_msg}"):
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = [100, 50],
            n_evaluated_levels = [[100], [50]],
            lags               = [np.arange(1, 4)] * 2,
            candidate_ids      = [0, 1],
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
            lags               = [np.arange(1, 4)] * 2,
            candidate_ids      = [0, 1],
            levels             = ['l1'],
            best_is_first      = False
        )

    message = str(record[0].message)
    assert message.startswith(
        "1 of 2 candidates is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. To compare the candidates"
    )
    assert "best candidate" not in message



@pytest.mark.parametrize(
    "candidate_name, candidate_ids, expected",
    [
        (
            "candidate",
            [0, 1, 2, 3, 4, 5],
            "3 of 6 candidates are evaluated on fewer points",
        ),
        (
            "trial",
            [5, 7, 8, 1, 3, 0],
            "3 of 6 trials are evaluated on fewer points",
        ),
    ],
    ids=["rows", "trials"],
)
def test_warn_n_evaluated_search_multiseries_grouped_by_lags(
    candidate_name, candidate_ids, expected
):
    """
    Test that the candidates with the same lags and number of points are listed
    in one line, with their rows in the results (grid and random search) or
    their trial numbers (Bayesian search).
    """
    lags = [np.arange(1, 36)] * 3 + [np.arange(1, 8)] * 3
    n_evaluated = [184, 184, 184, 276, 276, 276]
    expected_line = {
        "candidate": "  - lags: [1, 2, ..., 35], 3 candidates (rows 0, 1, 2): 184 (66.7%)",
        "trial": "  - lags: [1, 2, ..., 35], 3 trials (5, 7, 8): 184 (66.7%)",
    }[candidate_name]

    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated,
            n_evaluated_levels = [[n] for n in n_evaluated],
            lags               = lags,
            candidate_ids      = candidate_ids,
            levels             = ['l1'],
            candidate_name     = candidate_name
        )

    message = record[0].message.message
    assert message.startswith(expected)
    assert message.endswith(
        f"{candidate_name.capitalize()}s with less than 95% of the 276 points of "
        f"the most complete one:\n{expected_line}"
    )
    assert record[0].filename == __file__


def test_warn_n_evaluated_search_multiseries_max_ids_and_max_lines():
    """
    Test that at most `max_ids` identifiers are listed in a line, at most
    `max_lines` lines are listed, and the candidates of the rest are counted.
    A group of one candidate shows its identifier.
    """
    lags = (
        [np.arange(1, 36)] * 3 + [np.arange(1, 61)]
        + [np.arange(1, 25)] + [np.arange(1, 8)]
    )
    n_evaluated = [184, 184, 184, 184, 200, 276]
    expected = (
        "Trials with less than 95% of the 276 points of the most complete one:\n"
        "  - lags: [1, 2, ..., 35], 3 trials (5, 7, ...): 184 (66.7%)\n"
        "  - lags: [1, 2, ..., 60], trial 1: 184 (66.7%)\n"
        "  ... and 1 more trial"
    )
    with pytest.warns(MissingValuesWarning) as record:
        _warn_n_evaluated_search_multiseries(
            n_evaluated        = n_evaluated,
            n_evaluated_levels = [[n] for n in n_evaluated],
            lags               = lags,
            candidate_ids      = [5, 7, 8, 1, 3, 0],
            levels             = ['l1'],
            candidate_name     = 'trial',
            max_lines          = 2,
            max_ids            = 2
        )

    assert record[0].message.message.endswith(expected)
