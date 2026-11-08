# Unit test _reemit_fold_warnings
# ==============================================================================
import warnings
import skforecast.utils.utils as utils_module
from skforecast.exceptions import MissingValuesWarning
from skforecast.model_selection._utils import (
    _reemit_fold_warnings,
    _SearchWarnings
)

UTILS_FILE = utils_module.__file__


def _fold_warning(text, category=MissingValuesWarning, lineno=10):
    """
    Warning recorded in a fold, as returned by `_run_fold_recording_warnings`,
    raised in `skforecast/utils/utils.py`.
    """
    return (category(text), category, UTILS_FILE, lineno)


def test_reemit_fold_warnings_shown_in_order_with_location_and_outputs_returned():
    """
    Test that the warnings of the folds are shown in the order of the folds,
    with their category and original location, and that the outputs of the
    folds are returned. Folds run in the main process have no warnings (`None`).
    """
    results = [
        ("output_0", [_fold_warning("First fold.", lineno=11)]),
        ("output_1", None),
        ("output_2", [_fold_warning("Third fold.", category=UserWarning, lineno=12)]),
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        outputs = _reemit_fold_warnings(results)

    assert outputs == ["output_0", "output_1", "output_2"]
    assert [w.category for w in record] == [MissingValuesWarning, UserWarning]
    assert [str(w.message).split("\n")[0] for w in record] == ["First fold.", "Third fold."]
    assert [(w.filename, w.lineno) for w in record] == [(UTILS_FILE, 11), (UTILS_FILE, 12)]


def test_reemit_fold_warnings_same_warning_shown_in_every_fold_with_default_action():
    """
    Test that the same warning raised in several folds is shown once per fold
    with the action 'default' (no registry of the warnings already shown), as
    with `n_jobs=1`.
    """
    results = [("output", [_fold_warning("Same text.")]) for _ in range(3)]
    with warnings.catch_warnings(record=True) as record:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        _reemit_fold_warnings(results)

    assert len(record) == 3


def test_reemit_fold_warnings_filters_of_the_user_by_category_and_module():
    """
    Test that the warnings go through the filters of the main process: by
    category, and by module, since the module of their file is passed.
    """
    results = [
        ("output", [_fold_warning("Missing values."), _fold_warning("Other.", UserWarning)])
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", category=MissingValuesWarning)
        _reemit_fold_warnings(results)

    assert [w.category for w in record] == [UserWarning]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", module=r"skforecast\.utils\.utils")
        _reemit_fold_warnings(results)

    assert record == []


def test_reemit_fold_warnings_collected_by_SearchWarnings_with_candidate():
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
                fold_warnings = [_fold_warning("Missing values. Details.")] if candidate else None
                _reemit_fold_warnings([("output", fold_warnings), ("output", fold_warnings)])
        assert record == []
        assert [record[0] for record in search_warnings.records] == [1, 1, 2, 2]
        search_warnings.emit()

    assert len(record) == 1
    assert record[0].message.message == "Missing values (in 2 of 3 candidates). Details."
