# Unit test _run_fold_recording_warnings
# ==============================================================================
import os
import warnings
import pytest
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import manage_warnings
from skforecast.model_selection._utils import _run_fold_recording_warnings

# NOTE: A `main_pid` other than the identifier of this process simulates a fold
# that runs in a worker process of joblib.
OTHER_PID = -1


def fold_warns(value, n_warnings=1):
    """
    Function of a fold that raises `n_warnings` MissingValuesWarning.
    """
    for _ in range(n_warnings):
        warnings.warn("`exog` has missing values. Details.", MissingValuesWarning)

    return value * 2


@manage_warnings
def fold_warns_suppressed(value, suppress_warnings=False):
    """
    Function of a fold whose skforecast warnings are silenced by
    `suppress_warnings`, as the `fit` and `predict` of the forecasters.
    """
    warnings.warn("`exog` has missing values. Details.", MissingValuesWarning)
    warnings.warn("A warning of another library.", UserWarning)

    return value


def test_run_fold_recording_warnings_in_main_process_warnings_shown_as_usual():
    """
    Test that, in the main process (`n_jobs=1` or a backend of threads), the
    function of the fold is called and its warnings are shown as usual, so no
    warnings are returned.
    """
    with pytest.warns(MissingValuesWarning, match="`exog` has missing values"):
        result, fold_warnings = _run_fold_recording_warnings(
            fold_function=fold_warns, main_pid=os.getpid(), value=3
        )

    assert result == 6
    assert fold_warnings is None


def test_run_fold_recording_warnings_in_worker_process_warnings_returned():
    """
    Test that, in a worker process, the warnings of the fold are returned with
    their category, message, filename and line, every repetition included
    (action 'always', whatever the filters are), and none is shown.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        result, fold_warnings = _run_fold_recording_warnings(
            fold_function=fold_warns, main_pid=OTHER_PID, value=3, n_warnings=2
        )

    assert record == []
    assert result == 6
    assert len(fold_warnings) == 2
    for message, category, filename, lineno in fold_warnings:
        assert category is MissingValuesWarning
        assert isinstance(message, MissingValuesWarning)
        assert message.message == "`exog` has missing values. Details."
        assert filename == __file__
        assert isinstance(lineno, int)


def test_run_fold_recording_warnings_in_worker_process_suppressed_not_recorded():
    """
    Test that the warnings silenced inside the function of the fold
    (`suppress_warnings=True`) are not recorded, and the rest are.
    """
    _, fold_warnings = _run_fold_recording_warnings(
        fold_function=fold_warns_suppressed, main_pid=OTHER_PID, value=1,
        suppress_warnings=True
    )

    assert [category for _, category, _, _ in fold_warnings] == [UserWarning]


def test_run_fold_recording_warnings_warning_not_serializable_sent_as_RuntimeWarning():
    """
    Test that a warning whose category cannot be serialized (a class defined
    inside a function) is returned as a `RuntimeWarning` with its text, after
    the name of its category, so the result of the fold can be sent to the
    main process.
    """

    class LocalWarning(UserWarning):
        pass

    def fold_local_warning():
        warnings.warn("Warning of a local class.", LocalWarning)
        return "result"

    _, fold_warnings = _run_fold_recording_warnings(
        fold_function=fold_local_warning, main_pid=OTHER_PID
    )

    message, category, filename, _ = fold_warnings[0]
    assert category is RuntimeWarning
    assert isinstance(message, RuntimeWarning)
    assert str(message) == "LocalWarning: Warning of a local class."
    assert filename == __file__


class WarningWithTwoArguments(UserWarning):
    """
    Warning whose `__init__` takes other arguments than its text: it is
    serialized, but it cannot be rebuilt from its `args`.
    """

    def __init__(self, n_rows, total):
        super().__init__(f"{n_rows} of {total} rows are dropped.")


def test_run_fold_recording_warnings_warning_not_rebuilt_sent_as_RuntimeWarning():
    """
    Test that a warning that can be serialized but not rebuilt (its `__init__`
    takes other arguments than its text) is returned as a `RuntimeWarning`
    with its text.
    """

    def fold_two_arguments_warning():
        warnings.warn(WarningWithTwoArguments(2, 10))
        return "result"

    _, fold_warnings = _run_fold_recording_warnings(
        fold_function=fold_two_arguments_warning, main_pid=OTHER_PID
    )

    message, category, _, _ = fold_warnings[0]
    assert category is RuntimeWarning
    assert str(message) == "WarningWithTwoArguments: 2 of 10 rows are dropped."


def test_run_fold_recording_warnings_skforecast_warning_text_not_duplicated():
    """
    Test that a skforecast warning, which adds a text in `__str__`, is returned
    as it is, so its text is not duplicated when it is shown.
    """
    _, fold_warnings = _run_fold_recording_warnings(
        fold_function=fold_warns, main_pid=OTHER_PID, value=1
    )

    message, category, _, _ = fold_warnings[0]
    assert category is MissingValuesWarning
    assert message.message == "`exog` has missing values. Details."
    assert str(message).count("`exog` has missing values. Details.") == 1
