################################################################################
#                          skforecast.utils._warnings                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from functools import wraps
import os
import pickle
import sys
from typing import Any, Callable, ParamSpec, TypeVar
import warnings
from ..exceptions import warn_skforecast_categories

# Type variables for the manage_warnings decorator. ParamSpec preserves the
# decorated function's parameter signature, and TypeVar preserves its return
# type, so that type checkers see the original signatures through the wrapper.
P = ParamSpec('P')
R = TypeVar('R')


def manage_warnings(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that safely manages skforecast warning suppression using
    `warnings.catch_warnings()` context manager. If the decorated function
    receives a `suppress_warnings=True` keyword argument, all skforecast
    warnings are suppressed within its execution scope. Warning filter state
    is automatically saved and restored, making this safe for nested calls
    and exception scenarios.

    By using `warnings.catch_warnings()`, the filter state is saved on entry and 
    restored on exit — even if an exception is raised — so nested decorated 
    functions never interfere with each other's suppression settings.

    The decorator's type signature uses module-level type variables:

    - `P` (`ParamSpec`): Captures the full parameter specification
      (positional and keyword arguments) of the decorated function, so that
      type checkers preserve the original call signature through the wrapper.
    - `R` (`TypeVar`): Captures the return type of the decorated function,
      ensuring the wrapper advertises the same return type.

    Parameters
    ----------
    func : Callable[P, R]
        The function to decorate. Expected to accept a `suppress_warnings`
        keyword argument.

    Returns
    -------
    Callable[P, R]
        The wrapped function with safe warning management.

    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        suppress = kwargs.get('suppress_warnings', False)
        with warnings.catch_warnings():
            if suppress:
                for category in warn_skforecast_categories:
                    warnings.filterwarnings('ignore', category=category)
            return func(*args, **kwargs)
    return wrapper


def _first_frame_outside_skforecast(frame: Any) -> tuple[Any, int]:
    """
    First frame, starting from `frame` and going up the stack, whose code is
    not in the package skforecast. The tests of skforecast count as code
    outside the package.

    Parameters
    ----------
    frame : frame
        Frame where the search starts.

    Returns
    -------
    frame : frame, None
        First frame outside skforecast, `None` if every frame is in it.
    n_frames : int
        Number of frames of skforecast skipped.

    """

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_prefix = package_dir + os.sep
    tests_dir = os.sep + "tests" + os.sep

    n_frames = 0
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(package_prefix) or tests_dir in filename[len(package_dir):]:
            break
        frame = frame.f_back
        n_frames += 1

    return frame, n_frames


def _find_stack_level() -> int:
    """
    `stacklevel` of `warnings.warn`, called in the function that calls this
    one, so the warning points to the code of the user (the first frame
    outside skforecast) instead of the internal line that raises it. If every
    frame is in skforecast, the warning points to the line that raises it.

    Returns
    -------
    stacklevel : int
        `stacklevel` of `warnings.warn`.

    """

    frame, n_frames = _first_frame_outside_skforecast(sys._getframe(1))

    return 1 + n_frames if frame is not None else 1


_WORKER_WARNINGS_ATTRIBUTE = "_skforecast_worker_warnings"


def _run_recording_worker_warnings(
    function: Callable,
    main_pid: int,
    **kwargs
) -> tuple[Any, list[tuple] | None]:
    """
    Run the function of a fold of the backtesting. If it runs in a worker
    process of joblib (a process other than `main_pid`), its warnings are
    recorded and returned, so the main process shows them with
    `_reemit_worker_warnings`. Otherwise, the warnings of a worker process would
    be printed by the worker, without the filters of the user and outside the
    registry of the warnings of a search (`_SearchWarnings`).

    - In the main process (`n_jobs=1`, or a backend of threads), the function
    is called and its warnings are shown as usual.
    - The warnings are recorded with the action `"always"`, so the filters of
    the main process decide which ones are shown. Those ignored inside the
    function (for example, by `suppress_warnings`) are not recorded.
    - Each warning is returned as it is if it can be serialized and rebuilt
    with `pickle`. Otherwise, its text is returned as a `RuntimeWarning`,
    starting with the name of its category.
    - If the function raises an error, the warnings raised before it are
    stored in the attribute `_WORKER_WARNINGS_ATTRIBUTE` of the error, which
    reaches the main process with it, and the error is raised again.

    Parameters
    ----------
    function : Callable
        Function of the fold, defined at module level.
    main_pid : int
        Identifier of the main process (`os.getpid()` before `joblib.Parallel`).
    **kwargs
        Arguments of `function`.

    Returns
    -------
    result : Any
        Output of `function`.
    worker_warnings : list, None
        Warnings raised in the fold, as tuples `(message, category, filename,
        lineno)`. `None` if the fold runs in the main process.

    """

    if os.getpid() == main_pid:
        return function(**kwargs), None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(**kwargs)
        except Exception as error:
            try:
                setattr(
                    error, _WORKER_WARNINGS_ATTRIBUTE, _serializable_warnings(caught)
                )
            except Exception:
                pass
            raise

    return result, _serializable_warnings(caught)


def _serializable_warnings(caught: list) -> list[tuple]:
    """
    Warnings recorded in a worker process, in a form that can be sent to the
    main process. Each warning is kept as it is if it can be serialized and
    rebuilt with `pickle`. Otherwise, its text is kept as a `RuntimeWarning`,
    starting with the name of its category.

    Parameters
    ----------
    caught : list
        Warnings recorded by `warnings.catch_warnings(record=True)`.

    Returns
    -------
    worker_warnings : list
        Warnings, as tuples `(message, category, filename, lineno)`.

    """

    worker_warnings = []
    for warning in caught:
        message, category = warning.message, warning.category
        try:
            # NOTE: The warning is also rebuilt: a category whose `__init__` takes
            # other arguments than its text is serialized, but fails to load in
            # the main process.
            pickle.loads(pickle.dumps((message, category)))
        except Exception:
            message = RuntimeWarning(f"{category.__name__}: {message}")
            category = RuntimeWarning
        worker_warnings.append((message, category, warning.filename, warning.lineno))

    return worker_warnings


def _module_name_from_filename(filename: str) -> str | None:
    """
    Name of the module loaded from `filename`, if any.

    Parameters
    ----------
    filename : str
        Path of the file of the module.

    Returns
    -------
    module_name : str, None
        Name of the module in `sys.modules`, `None` if no module is loaded from
        `filename`.

    """

    for module_name, module in list(sys.modules.items()):
        if getattr(module, "__file__", None) == filename:
            return module_name

    return None


def _reemit_worker_warnings(results: list[tuple]) -> list:
    """
    Show, in the main process and in the order of the folds, the warnings
    recorded by `_run_recording_worker_warnings` in the worker processes of
    joblib, and return the outputs of the folds.

    Each warning is shown with `warnings.warn_explicit`, with its category,
    location, module and the globals of its module (if it has `__spec__`), as
    with `n_jobs=1`. It goes through the filters of the main process: those of
    the user, those of `suppress_warnings` and, in a search, the registry of
    its warnings, which groups them. No registry of the warnings already shown
    is used: with `n_jobs=1`, the `catch_warnings` of each `fit` and `predict`
    resets them, so the same warning is shown in every fold.

    Parameters
    ----------
    results : list
        Outputs of `_run_recording_worker_warnings`, one per fold, as tuples
        `(result, worker_warnings)`.

    Returns
    -------
    results : list
        Outputs of the function of each fold.

    """

    module_names = {}
    outputs = []
    for result, worker_warnings in results:
        outputs.append(result)
        for message, category, filename, lineno in worker_warnings or ():
            if filename not in module_names:
                module_names[filename] = _module_name_from_filename(filename)
            module_name = module_names[filename]
            module = sys.modules.get(module_name) if module_name is not None else None
            # NOTE: The globals of a module without `__spec__` (`__main__` of a
            # script or of the standard input) make `warn_explicit` raise a
            # `DeprecationWarning`, or an `ImportError` from its loader, while
            # looking for the source line. `warnings.warn` does not use them.
            module_globals = (
                vars(module)
                if module is not None and getattr(module, "__spec__", None) is not None
                else None
            )
            warnings.warn_explicit(
                message        = message,
                category       = category,
                filename       = filename,
                lineno         = lineno,
                module         = module_name,
                registry       = None,
                module_globals = module_globals
            )

    return outputs
