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
    logged if `filter_logs=True`, the messages about the latter are. The
    filter is removed from the logger at the end.
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


def test_optimize_study_other_errors_stop_the_study():
    """
    Test that an error that is not `_TrialEvaluationError` nor in the `catch`
    of the user (for example, of the `search_space`) stops the study, and the
    filter is removed from the logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    filters_before = list(logger.filters)
    study = optuna.create_study()
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        _optimize_study(
            study                 = study,
            objective             = _objective,
            n_trials              = 5,
            kwargs_study_optimize = {'catch': (KeyError,)}
        )

    assert len(study.trials) == 3
    assert logger.filters == filters_before


def test_FailedTrialsLogFilter_keeps_other_messages():
    """
    Test that the filter keeps the messages that are not about a trial that
    fails with `_TrialEvaluationError`.
    """
    log_filter = _FailedTrialsLogFilter()
    error = _TrialEvaluationError("Evaluation failed.")

    def record(message, exc=None):
        return logging.LogRecord(
            LOGGER_NAME, logging.WARNING, __file__, 0, message, None,
            (type(exc), exc, None) if exc is not None else None
        )

    assert log_filter.filter(record("Trial 4 finished with value: 1.0")) is True
    assert log_filter.filter(record("Trial 5 failed with value None.")) is True
    assert log_filter.filter(
        record("Trial 4 failed with parameters: {} because of ...", exc=error)
    ) is False
    assert log_filter.filter(record("Trial 4 failed with value None.")) is False
