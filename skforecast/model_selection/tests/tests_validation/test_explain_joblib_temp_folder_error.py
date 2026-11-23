# Unit test _explain_joblib_temp_folder_error
# ==============================================================================
import pickle
import pytest
from skforecast.model_selection._validation import _explain_joblib_temp_folder_error


# NOTE: Text of the cause (`_RemoteTraceback`) of the error that loky raises
# when joblib cannot write an array of more than 1 MB to its temporary folder.
remote_traceback_temp_folder = (
    '\n"""\n'
    'Traceback (most recent call last):\n'
    '  File "site-packages/joblib/externals/loky/backend/queues.py", line 159, '
    'in _feed\n'
    '    obj_ = dumps(obj, reducers=reducers)\n'
    '  File "site-packages/joblib/_memmapping_reducer.py", line 442, in __call__\n'
    '    os.makedirs(self._temp_folder)\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '/not/writable'\n"
    '"""'
)
# NOTE: The same with Python 3.14, which adds to the errors raised while
# pickling a note for each object that contains the one that failed, so the
# line of the error is no longer the last one.
remote_traceback_temp_folder_notes = (
    '\n"""\n'
    'Traceback (most recent call last):\n'
    '  File "site-packages/joblib/externals/loky/backend/queues.py", line 159, '
    'in _feed\n'
    '    obj_ = dumps(obj, reducers=reducers)\n'
    '  File "site-packages/joblib/_memmapping_reducer.py", line 442, in __call__\n'
    '    os.makedirs(self._temp_folder)\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '/not/writable'\n"
    'when serializing tuple item 0\n'
    'when serializing joblib.parallel.BatchedCalls object\n'
    "when serializing dict item 'fn'\n"
    'when serializing joblib.externals.loky.process_executor._CallItem object\n'
    '"""'
)
# NOTE: The same, when an argument of the task cannot be pickled.
remote_traceback_unpicklable = (
    '\n"""\n'
    'Traceback (most recent call last):\n'
    '  File "site-packages/joblib/externals/loky/backend/queues.py", line 159, '
    'in _feed\n'
    '    obj_ = dumps(obj, reducers=reducers)\n'
    "TypeError: cannot pickle '_thread.lock' object\n"
    '"""'
)
# NOTE: The same, with an array of dtype object that cannot be pickled: every
# array goes through `_memmapping_reducer`, also those pickled with the task.
remote_traceback_unpicklable_array = (
    '\n"""\n'
    'Traceback (most recent call last):\n'
    '  File "site-packages/joblib/externals/loky/backend/queues.py", line 159, '
    'in _feed\n'
    '    obj_ = dumps(obj, reducers=reducers)\n'
    '  File "site-packages/joblib/_memmapping_reducer.py", line 529, in __call__\n'
    '    return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))\n'
    "TypeError: cannot pickle '_thread.lock' object\n"
    'when serializing numpy.ndarray object\n'
    'when serializing tuple item 0\n'
    '"""'
)
explanation = (
    "joblib could not write to its temporary folder the large arrays that it "
    "sends to the worker processes (more than 1 MB by default). Set the "
    "environment variable `JOBLIB_TEMP_FOLDER` to a writable folder with enough "
    "free space, run the function inside `with "
    "joblib.parallel_config(max_nbytes=None):` to send them without that "
    "folder, or do not use parallel processes (`n_jobs=1`)."
)


@pytest.mark.parametrize(
    "remote_traceback",
    [remote_traceback_temp_folder, remote_traceback_temp_folder_notes],
    ids=["error in the last line", "notes after the error (Python 3.14)"]
)
def test_explain_joblib_temp_folder_error_message_extended(remote_traceback):
    """
    Test that the message of the `PicklingError` that joblib raises when it
    cannot write to its temporary folder is extended with the cause and how to
    solve it, in the same error (same type and same cause).
    """
    cause = RuntimeError(remote_traceback)
    error = pickle.PicklingError(
        "Could not pickle the task to send it to the workers."
    )
    error.__cause__ = cause
    expected_message = (
        "Could not pickle the task to send it to the workers. " + explanation
    )

    _explain_joblib_temp_folder_error(error)

    assert type(error) is pickle.PicklingError
    assert str(error) == expected_message
    assert error.args == (expected_message,)
    assert error.__cause__ is cause


@pytest.mark.parametrize(
    "error, cause",
    [
        (pickle.PicklingError("Could not pickle the task."), None),
        (
            pickle.PicklingError("Could not pickle the task."),
            RuntimeError(remote_traceback_unpicklable)
        ),
        (
            pickle.PicklingError("Could not pickle the task."),
            RuntimeError(remote_traceback_unpicklable_array)
        ),
        (ValueError("Error in a fold."), RuntimeError(remote_traceback_temp_folder)),
        (OSError("No space left on device"), None),
    ],
    ids=[
        "PicklingError without cause",
        "PicklingError with another cause",
        "PicklingError of an array that cannot be pickled",
        "other error with the cause",
        "other error without cause",
    ]
)
def test_explain_joblib_temp_folder_error_other_errors_not_changed(error, cause):
    """
    Test that the message is not changed when the error is not a
    `PicklingError`, or when its cause is not the failure of joblib to write
    to its temporary folder (no cause, or an argument of the task that cannot
    be pickled, an array included).
    """
    error.__cause__ = cause
    expected_args = error.args
    expected_message = str(error)

    _explain_joblib_temp_folder_error(error)

    assert error.args == expected_args
    assert str(error) == expected_message
    assert error.__cause__ is cause
