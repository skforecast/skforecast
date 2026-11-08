# Unit test _SearchWarnings
# ==============================================================================
import threading
import warnings
import pytest
import skforecast.utils.utils as utils_module
from skforecast.exceptions import (
    DataTransformationWarning,
    LongTrainingWarning,
    MissingValuesWarning
)
from skforecast.model_selection._utils import _SearchWarnings
from skforecast.utils import _reemit_worker_warnings

UTILS_FILE = utils_module.__file__


def _worker_warning(text, category=MissingValuesWarning, lineno=10):
    """
    Warning recorded in a fold, as returned by `_run_recording_worker_warnings`,
    raised in `skforecast/utils/utils.py`.
    """
    return (category(text), category, UTILS_FILE, lineno)


def _messages(record, category=None):
    """
    Texts of the warnings recorded, without the suffix of the skforecast
    warnings.
    """
    return [
        getattr(w.message, "message", str(w.message))
        for w in record
        if category is None or issubclass(w.category, category)
    ]


def test_SearchWarnings_same_text_in_every_candidate_shown_once_as_it_is():
    """
    Test that a warning with the same text in every candidate is shown once,
    as it is, after exiting the context.
    """
    search_warnings = _SearchWarnings(n_candidates=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            for candidate in range(3):
                search_warnings.set_candidate(candidate)
                warnings.warn("Use a transformer. Details.", DataTransformationWarning)
        assert record == []
        search_warnings.emit()

    assert len(record) == 1
    assert record[0].category is DataTransformationWarning
    assert _messages(record) == ["Use a transformer. Details."]


def test_SearchWarnings_same_text_in_some_candidates_shown_with_count():
    """
    Test that a warning with the same text in some candidates is shown once,
    with the number of candidates after its first sentence. A warning raised
    outside the candidates is shown as it is.
    """
    search_warnings = _SearchWarnings(n_candidates=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            for candidate in range(3):
                search_warnings.set_candidate(candidate)
                if candidate > 0:
                    warnings.warn(
                        "`exog` has missing values. Prediction may fail.",
                        MissingValuesWarning
                    )
            search_warnings.set_candidate(None)
            warnings.warn("Refit warning", UserWarning)
        search_warnings.emit()

    assert _messages(record) == [
        "`exog` has missing values (in 2 of 3 candidates). Prediction may fail.",
        "Refit warning",
    ]
    assert record[0].category is MissingValuesWarning
    assert record[1].category is UserWarning


def test_SearchWarnings_different_texts_with_same_first_sentence_grouped():
    """
    Test that warnings with the same first sentence and different texts are
    shown once, with the text of the first one and the number of candidates
    (trials) after its first sentence.
    """
    search_warnings = _SearchWarnings(n_candidates=4, candidate_name="trial")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            for candidate in range(4):
                search_warnings.set_candidate(candidate)
                if candidate < 3:
                    warnings.warn(
                        f"Some levels are not predicted. Window {candidate}.",
                        MissingValuesWarning
                    )
        search_warnings.emit()

    assert _messages(record) == [
        "Some levels are not predicted (in 3 of 4 trials; details of the first "
        "one). Window 0."
    ]


def test_SearchWarnings_same_text_outside_candidates_shown_once_as_it_is():
    """
    Test that a warning raised only outside the candidates, with the same text
    every time, is shown once, as it is.
    """
    search_warnings = _SearchWarnings(n_candidates=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(None)
            warnings.warn("Refit warning. Text 1.", UserWarning)
            warnings.warn("Refit warning. Text 1.", UserWarning)
        search_warnings.emit()

    assert _messages(record) == ["Refit warning. Text 1."]


def test_SearchWarnings_different_texts_outside_candidates_grouped():
    """
    Test that warnings with the same first sentence and different texts,
    raised only outside the candidates (for example, when the training
    matrices of a `OneStepAheadFold` are created for each value of `lags`),
    are shown once: the text of the first one, with the number of different
    texts after its first sentence. A warning with another first sentence is
    shown apart.
    """
    search_warnings = _SearchWarnings(n_candidates=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(None)
            for n_rows in (8, 16, 30, 16):
                warnings.warn(
                    f"Some rows have missing values. {n_rows} rows are dropped.",
                    MissingValuesWarning
                )
            warnings.warn("Refit warning. Text 1.", UserWarning)
        search_warnings.emit()

    assert _messages(record) == [
        "Some rows have missing values (3 times in the search, details of the "
        "first one). 8 rows are dropped.",
        "Refit warning. Text 1.",
    ]
    assert record[0].category is MissingValuesWarning
    assert record[1].category is UserWarning


def test_SearchWarnings_different_texts_outside_and_inside_candidates_grouped_by_candidates():
    """
    Test that, when warnings with the same first sentence and different texts
    are raised outside and inside the candidates, they are shown once with the
    number of candidates in which they are raised, as when they are raised
    only inside the candidates.
    """
    search_warnings = _SearchWarnings(n_candidates=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(None)
            warnings.warn(
                "Some rows have missing values. 8 rows are dropped.",
                MissingValuesWarning
            )
            for candidate in range(3):
                search_warnings.set_candidate(candidate)
                if candidate < 2:
                    warnings.warn(
                        f"Some rows have missing values. {candidate} rows are dropped.",
                        MissingValuesWarning
                    )
        search_warnings.emit()

    assert _messages(record) == [
        "Some rows have missing values (in 2 of 3 candidates; details of the "
        "first one). 8 rows are dropped."
    ]


def test_SearchWarnings_warnings_point_to_the_code_of_the_user():
    """
    Test that the warnings shown by `emit` and LongTrainingWarning point to
    the code that uses `_SearchWarnings` (the tests count as code of the
    user), not to the internal lines of skforecast.
    """
    search_warnings = _SearchWarnings(n_candidates=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn("Text.", UserWarning)
            warnings.warn("The forecaster will be fit 10 times.", LongTrainingWarning)
            search_warnings.add_skipped(
                lags=None, params={'alpha': 1}, error=ValueError("Wrong.")
            )
        search_warnings.emit()

    assert [w.category for w in record] == [
        LongTrainingWarning, UserWarning, RuntimeWarning
    ]
    assert all(w.filename == __file__ for w in record)


def test_SearchWarnings_emit_with_n_candidates():
    """
    Test that `n_candidates` of `emit` replaces the one of the initialization,
    for example, the trials run when the study stops before `n_trials`.
    """
    search_warnings = _SearchWarnings(n_candidates=10, candidate_name="trial")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn("Text.", UserWarning)
            search_warnings.set_candidate(1)
        search_warnings.emit(n_candidates=2)

    assert _messages(record) == ["Text (in 1 of 2 trials)."]


def test_SearchWarnings_respects_filters_and_restores_them():
    """
    Test that the warnings ignored by the filters active when entering the
    context (for example, `suppress_warnings=True`) are not recorded, that the
    filters and `warnings.showwarning` are restored on exit and that the
    warnings are shown through the filters active when calling `emit`.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=MissingValuesWarning)
        filters_before = list(warnings.filters)
        showwarning_before = warnings.showwarning

        search_warnings = _SearchWarnings(n_candidates=1)
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn("Ignored.", MissingValuesWarning)
            warnings.warn("Recorded.", DataTransformationWarning)

        assert list(warnings.filters) == filters_before
        assert warnings.showwarning is showwarning_before
        assert [r[2] for r in search_warnings.records] == ["Recorded."]

        warnings.filterwarnings("ignore", category=DataTransformationWarning)
        search_warnings.emit()

    assert record == []


def test_SearchWarnings_LongTrainingWarning_shown_once_with_fits_of_the_search():
    """
    Test that the first LongTrainingWarning is shown when raised, inside the
    context, with the number of fits of the whole search, and the rest are
    discarded.
    """
    search_warnings = _SearchWarnings(n_candidates=6)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            for candidate in range(6):
                search_warnings.set_candidate(candidate)
                warnings.warn(
                    "The forecaster will be fit 60 times. This can take "
                    "substantial amounts of time. If not feasible, try with "
                    "`refit = False`.\n",
                    LongTrainingWarning
                )
                if candidate == 0:
                    assert len(record) == 1
        search_warnings.emit()

    assert len(record) == 1
    assert record[0].category is LongTrainingWarning
    assert _messages(record) == [
        "Each candidate fits the forecaster about 60 times, so the search (6 "
        "candidates) fits it about 360 times. This can take substantial amounts "
        "of time. If not feasible, try with `refit = False`."
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "The forecaster will be fit 600 times (60 folds * 10 estimators). "
            "This can take substantial amounts of time.",
            "Each trial fits the forecaster about 600 times (60 folds * 10 "
            "estimators), so the search (1 trial) fits it about 600 times. This "
            "can take substantial amounts of time. If not feasible, try with "
            "`refit = False`."
        ),
        (
            "Another text.",
            "Another text."
        ),
    ],
    ids=["estimators", "unknown text"],
)
def test_SearchWarnings_LongTrainingWarning_text(text, expected):
    """
    Test the text of LongTrainingWarning with the number of estimators of
    ForecasterDirect, and that a text without the number of fits is shown
    unchanged.
    """
    search_warnings = _SearchWarnings(n_candidates=1, candidate_name="trial")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn(text, LongTrainingWarning)

    assert _messages(record) == [expected]


def test_SearchWarnings_candidate_per_thread():
    """
    Test that the candidate is stored per thread, so the warnings raised by
    trials run in threads are tagged with their own trial.
    """
    search_warnings = _SearchWarnings(n_candidates=4, candidate_name="trial")
    barrier = threading.Barrier(2)

    def trial(candidate):
        search_warnings.set_candidate(candidate)
        barrier.wait()
        warnings.warn(f"Trial {candidate}.", UserWarning)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with search_warnings:
            for pair in [(0, 1), (2, 3)]:
                threads = [threading.Thread(target=trial, args=(c,)) for c in pair]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

    assert sorted((r[0], r[2]) for r in search_warnings.records) == [
        (0, "Trial 0."), (1, "Trial 1."), (2, "Trial 2."), (3, "Trial 3.")
    ]


def test_SearchWarnings_skipped_candidates():
    """
    Test that the candidates skipped are reported in a single RuntimeWarning
    after the other warnings.
    """
    search_warnings = _SearchWarnings(n_candidates=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn("Text.", UserWarning)
            search_warnings.add_skipped(
                lags=None, params={'alpha': 1}, error=ValueError("Wrong.")
            )
        search_warnings.emit()

    assert [w.category for w in record] == [UserWarning, RuntimeWarning]
    assert _messages(record)[1] == (
        "Parameters skipped: 1 of 2 candidates raises an error, so it is not in "
        "the results.\n"
        "\n"
        "ValueError: Wrong.\n"
        "  - params: {'alpha': 1}"
    )


@pytest.mark.parametrize("action", ["default", "module", "once"])
def test_SearchWarnings_counts_with_filters_that_show_a_warning_once(action):
    """
    Test that the repetitions are counted when the filters show a warning
    once ('default', 'module', 'once'), and that the filters with a message
    or a module are kept.
    """
    search_warnings = _SearchWarnings(n_candidates=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter(action)
        warnings.filterwarnings("ignore", message="Ignored", module="skforecast")
        warnings.filterwarnings("ignore", message="Ignored", category=UserWarning)
        filters_before = list(warnings.filters)
        with search_warnings:
            for candidate in range(3):
                search_warnings.set_candidate(candidate)
                if candidate > 0:
                    warnings.warn("Repeated.", MissingValuesWarning)
                warnings.warn("Ignored text.", UserWarning)
        assert list(warnings.filters) == filters_before
        search_warnings.emit()

    assert _messages(record) == ["Repeated (in 2 of 3 candidates)."]


def test_SearchWarnings_discard():
    """
    Test that `discard` removes the warnings recorded of a category whose text
    starts with the text given, and keeps the rest.
    """
    search_warnings = _SearchWarnings(n_candidates=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            search_warnings.set_candidate(0)
            warnings.warn("Covered warning. Details.", MissingValuesWarning)
            warnings.warn("Other warning.", MissingValuesWarning)
            warnings.warn("Covered warning, other category.", UserWarning)
        search_warnings.discard(MissingValuesWarning, "Covered warning")
        search_warnings.emit()

    assert _messages(record) == [
        "Other warning (in 1 of 2 candidates).",
        "Covered warning, other category (in 1 of 2 candidates).",
    ]


def test_reemit_worker_warnings_collected_by_SearchWarnings_with_candidate():
    """
    Test that, inside a search, the warnings of the folds are collected by
    `_SearchWarnings` with the candidate being evaluated, so they are grouped
    with the rest of the warnings of the search.
    """
    search_warnings = _SearchWarnings(n_candidates=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with search_warnings:
            for candidate in range(3):
                search_warnings.set_candidate(candidate)
                worker_warnings = [_worker_warning("Missing values. Details.")] if candidate else None
                _reemit_worker_warnings([("output", worker_warnings), ("output", worker_warnings)])
        assert record == []
        assert [record[0] for record in search_warnings.records] == [1, 1, 2, 2]
        search_warnings.emit()

    assert len(record) == 1
    assert record[0].message.message == "Missing values (in 2 of 3 candidates). Details."
