# Unit test _optimize_study
# ==============================================================================
import logging
import pytest
import optuna
from optuna.trial import TrialState
from skforecast.model_selection._utils import (
    _optimize_study,
    _TrialEvaluationError,
    _FailedTrialsLogFilter
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
LOGGER_NAME = "optuna.study._optimize"


class _ListHandler(logging.Handler):
    """
    Handler that stores the messages of a logger.
    """

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _objective(trial):
    """
    Objective whose trials 1 and 3 fail with `_TrialEvaluationError` and the
    trial 2 with `ZeroDivisionError`.
    """
    if trial.number in (1, 3):
        raise _TrialEvaluationError("Evaluation failed.")
    if trial.number == 2:
        raise ZeroDivisionError("division by zero")
    return trial.suggest_float("x", 0, 1)


@pytest.mark.parametrize(
    "filter_logs", [True, False], ids=lambda x: f"filter_logs: {x}"
)
def test_optimize_study_trials_failed_and_catch_of_the_user(filter_logs):
    """
    Test that the trials that fail with `_TrialEvaluationError` and with the
    errors of the `catch` of the user (a single class) are marked as failed
    and the study continues. The messages of Optuna about the former are not
    logged if `filter_logs=True`, the messages about the latter are always
    logged. The filter is removed from the logger at the end.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handler = _ListHandler()
    logger.addHandler(handler)
    filters_before = list(logger.filters)
    try:
        study = optuna.create_study()
        _optimize_study(
            study                 = study,
            objective             = _objective,
            n_trials              = 5,
            kwargs_study_optimize = {'catch': ZeroDivisionError},
            filter_logs           = filter_logs
        )
    finally:
        logger.removeHandler(handler)

    assert [trial.state for trial in study.trials] == [
        TrialState.COMPLETE, TrialState.FAIL, TrialState.FAIL,
        TrialState.FAIL, TrialState.COMPLETE
    ]
    assert logger.filters == filters_before
    failed_trials = sorted({
        message.split()[1] for message in handler.messages
        if message.startswith("Trial") and "failed" in message
    })
    assert failed_trials == (['2'] if filter_logs else ['1', '2', '3'])


@pytest.mark.parametrize(
    "filter_logs", [True, False], ids=lambda x: f"filter_logs: {x}"
)
def test_optimize_study_other_errors_stop_the_study(filter_logs):
    """
    Test that an error that is not `_TrialEvaluationError` nor in the `catch`
    of the user (for example, of the `search_space`) stops the study, and the
    filter is removed from the logger. The messages of Optuna about that
    trial are not logged if `filter_logs=True`, since the error is raised.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handler = _ListHandler()
    logger.addHandler(handler)
    filters_before = list(logger.filters)
    study = optuna.create_study()
    try:
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            _optimize_study(
                study                 = study,
                objective             = _objective,
                n_trials              = 5,
                kwargs_study_optimize = {'catch': (KeyError,)},
                filter_logs           = filter_logs
            )
    finally:
        logger.removeHandler(handler)

    assert len(study.trials) == 3
    assert logger.filters == filters_before
    logged_trial_2 = any(m.startswith("Trial 2 failed") for m in handler.messages)
    assert logged_trial_2 is not filter_logs


def _record(message, exc=None):
    """
    Record of the logger of Optuna, with the error `exc` as `exc_info`.
    """
    return logging.LogRecord(
        LOGGER_NAME, logging.WARNING, __file__, 0, message, None,
        (type(exc), exc, None) if exc is not None else None
    )


def test_FailedTrialsLogFilter_drops_trials_not_in_catch_of_the_user():
    """
    Test that the filter drops the two messages of the trials that fail with
    `_TrialEvaluationError` (reported in the warning of the candidates
    skipped) or with an error that is not in the `catch` of the user (raised
    by `study.optimize`), and keeps those of the errors in the `catch` of
    the user and the other messages.
    """
    log_filter = _FailedTrialsLogFilter(catch=(ZeroDivisionError,))
    error_trial = _TrialEvaluationError("Evaluation failed.")
    error_raised = KeyError("alpha")
    error_catch = ZeroDivisionError("division by zero")
    failed = "Trial {} failed with parameters: {{}} because of ..."
    value = "Trial {} failed with value None."

    assert log_filter.filter(_record("Trial 1 finished with value: 1.0")) is True
    assert log_filter.filter(_record(failed.format(2), exc=error_trial)) is False
    assert log_filter.filter(_record(value.format(2))) is False
    assert log_filter.filter(_record(failed.format(3), exc=error_raised)) is False
    assert log_filter.filter(_record(value.format(3))) is False
    assert log_filter.filter(_record(failed.format(4), exc=error_catch)) is True
    assert log_filter.filter(_record(value.format(4))) is True


def test_FailedTrialsLogFilter_keeps_trials_failed_without_error():
    """
    Test that the filter keeps the messages of the trials that fail without
    an error (for example, the objective returns NaN), which do not stop the
    search and are not reported elsewhere.
    """
    log_filter = _FailedTrialsLogFilter()
    message = "Trial 5 failed with parameters: {} because of the value nan."

    assert log_filter.filter(_record(message)) is True
    assert log_filter.filter(_record("Trial 5 failed with value nan.")) is True
