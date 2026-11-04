# Unit test _SearchWarnings
# ==============================================================================
import threading
import warnings
import pytest
from skforecast.exceptions import (
    DataTransformationWarning,
    LongTrainingWarning,
    MissingValuesWarning
)
from skforecast.model_selection._utils import _SearchWarnings


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
    shown once, with the first sentence, the number of candidates (trials) and
    how to see the details.
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
        "Some levels are not predicted (in 3 of 4 trials). Run the backtesting "
        "of a candidate to see the details."
    ]


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
