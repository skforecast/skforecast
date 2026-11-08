# Unit test _reemit_worker_warnings
# ==============================================================================
import sys
import types
import warnings
from importlib.machinery import BuiltinImporter
import skforecast.utils.utils as utils_module
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import _reemit_worker_warnings

UTILS_FILE = utils_module.__file__


def _worker_warning(text, category=MissingValuesWarning, lineno=10):
    """
    Warning recorded in a fold, as returned by `_run_recording_worker_warnings`,
    raised in `skforecast/utils/utils.py`.
    """
    return (category(text), category, UTILS_FILE, lineno)


def test_reemit_worker_warnings_shown_in_order_with_location_and_outputs_returned():
    """
    Test that the warnings of the folds are shown in the order of the folds,
    with their category and original location, and that the outputs of the
    folds are returned. Folds run in the main process have no warnings (`None`).
    """
    results = [
        ("output_0", [_worker_warning("First fold.", lineno=11)]),
        ("output_1", None),
        ("output_2", [_worker_warning("Third fold.", category=UserWarning, lineno=12)]),
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        outputs = _reemit_worker_warnings(results)

    assert outputs == ["output_0", "output_1", "output_2"]
    assert [w.category for w in record] == [MissingValuesWarning, UserWarning]
    assert [str(w.message).split("\n")[0] for w in record] == ["First fold.", "Third fold."]
    assert [(w.filename, w.lineno) for w in record] == [(UTILS_FILE, 11), (UTILS_FILE, 12)]


def test_reemit_worker_warnings_same_warning_shown_in_every_fold_with_default_action():
    """
    Test that the same warning raised in several folds is shown once per fold
    with the action 'default' (no registry of the warnings already shown), as
    with `n_jobs=1`.
    """
    results = [("output", [_worker_warning("Same text.")]) for _ in range(3)]
    with warnings.catch_warnings(record=True) as record:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        _reemit_worker_warnings(results)

    assert len(record) == 3


def test_reemit_worker_warnings_filters_of_the_user_by_category_and_module():
    """
    Test that the warnings go through the filters of the main process: by
    category, and by module, since the module of their file is passed.
    """
    results = [
        ("output", [_worker_warning("Missing values."), _worker_warning("Other.", UserWarning)])
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", category=MissingValuesWarning)
        _reemit_worker_warnings(results)

    assert [w.category for w in record] == [UserWarning]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", module=r"skforecast\.utils\.utils")
        _reemit_worker_warnings(results)

    assert record == []


def test_reemit_worker_warnings_module_without_spec_shown_without_extra_warnings():
    """
    Test that a warning raised in a module without `__spec__` (as `__main__`
    when the code is read from the standard input) is shown as it is, without
    a `DeprecationWarning` or an `ImportError` from its loader.
    """
    module = types.ModuleType("module_without_spec")
    module.__file__ = "/path/to/module_without_spec.py"
    module.__spec__ = None
    module.__loader__ = BuiltinImporter
    sys.modules["module_without_spec"] = module
    try:
        results = [
            ("output", [(UserWarning("Warning of the user."), UserWarning, module.__file__, 3)])
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _reemit_worker_warnings(results)
    finally:
        del sys.modules["module_without_spec"]

    assert [(w.category, str(w.message)) for w in record] == [
        (UserWarning, "Warning of the user.")
    ]
