# Unit test _get_search_direction and _check_study_direction
# ==============================================================================
import re
import pytest
import warnings
import optuna
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, zero_one_loss
from skforecast.metrics import add_y_train_argument, _get_metric
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.model_selection._utils import (
    _get_search_direction,
    _check_study_direction
)


forecaster_reg = ForecasterRecursive(estimator=LinearRegression(), lags=2)
forecaster_clf = ForecasterRecursiveClassifier(
    estimator=LogisticRegression(), lags=2
)


def custom_metric(y_true, y_pred):  # pragma: no cover
    """
    Custom metric without a known direction.
    """
    return 0.0


@pytest.mark.parametrize(
    'metric, forecaster, expected',
    [
        (r2_score, forecaster_reg, 'maximize'),
        (add_y_train_argument(r2_score), forecaster_clf, 'maximize'),
        (mean_absolute_error, forecaster_reg, 'minimize'),
        (zero_one_loss, forecaster_clf, 'minimize'),
        (_get_metric('mean_absolute_error'), forecaster_clf, 'minimize'),
        (_get_metric('accuracy_score'), forecaster_reg, 'maximize'),
        (custom_metric, forecaster_reg, 'minimize'),
        (custom_metric, forecaster_clf, 'maximize'),
        (_get_metric('mean_absolute_scaled_error'), forecaster_reg, 'minimize'),
    ],
    ids=lambda dt: f'{getattr(dt, "__name__", dt)}'
)
def test_get_search_direction_output(metric, forecaster, expected):
    """
    Test that the direction follows the metric when it is known, whatever the
    task, and the task of the forecaster otherwise (custom functions and the
    metrics of skforecast).
    """
    assert _get_search_direction(metric=metric, forecaster=forecaster) == expected


@pytest.mark.parametrize(
    'study_direction, metric',
    [
        ('maximize', r2_score),
        ('minimize', mean_absolute_error),
        ('maximize', custom_metric),
        ('minimize', custom_metric),
    ],
    ids=lambda dt: f'{getattr(dt, "__name__", dt)}'
)
def test_check_study_direction_output_and_no_warning_when_no_conflict(
    study_direction, metric
):
    """
    Test that the direction of the study is returned without warnings when it
    matches the direction of the metric or the direction of the metric is
    unknown.
    """
    study = optuna.create_study(direction=study_direction)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        direction = _check_study_direction(
            study=study, metric=add_y_train_argument(metric)
        )

    assert direction == study_direction


@pytest.mark.parametrize(
    'study_direction, metric, metric_direction',
    [
        ('minimize', r2_score, 'maximize'),
        (optuna.study.StudyDirection.MAXIMIZE, mean_absolute_error, 'minimize'),
    ],
    ids=lambda dt: f'{getattr(dt, "__name__", dt)}'
)
def test_check_study_direction_UserWarning_when_conflict_with_metric(
    study_direction, metric, metric_direction
):
    """
    Test UserWarning is raised when the direction of the study contradicts the
    known direction of the metric, and that the direction of the study is
    returned.
    """
    study = optuna.create_study(direction=study_direction)
    expected_direction = study.direction.name.lower()

    warn_msg = re.escape(
        f"The optuna study direction is '{expected_direction}', but greater "
        f"values of the metric `{metric.__name__}` are "
    )
    with pytest.warns(UserWarning, match=warn_msg):
        direction = _check_study_direction(
            study=study, metric=add_y_train_argument(metric)
        )

    assert direction == expected_direction
    assert direction != metric_direction
