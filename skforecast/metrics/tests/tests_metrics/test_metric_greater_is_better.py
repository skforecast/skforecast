# Unit test _metric_greater_is_better
# ==============================================================================
import re
import pytest
import numpy as np
from functools import partial
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    d2_absolute_error_score,
    mean_absolute_error,
    max_error,
    mean_pinball_loss,
    mean_tweedie_deviance,
    accuracy_score,
    matthews_corrcoef,
    zero_one_loss,
    hamming_loss,
    brier_score_loss,
)
from skforecast.metrics import (
    add_y_train_argument,
    create_mean_pinball_loss,
    mean_absolute_scaled_error,
    winkler_score,
    _get_metric,
)
from skforecast.metrics.metrics import _metric_greater_is_better


def custom_metric(y_true, y_pred):  # pragma: no cover
    """
    Custom metric without a known direction.
    """
    return np.mean(np.abs(y_true - y_pred))


def custom_score(y_true, y_pred):  # pragma: no cover
    """
    Custom metric whose name follows the scikit-learn convention.
    """
    return np.mean(np.abs(y_true - y_pred))


@pytest.mark.parametrize(
    'metric, expected',
    [
        (r2_score, True),
        (explained_variance_score, True),
        (d2_absolute_error_score, True),
        (accuracy_score, True),
        (matthews_corrcoef, True),
        (mean_absolute_error, False),
        (max_error, False),
        (mean_pinball_loss, False),
        (mean_tweedie_deviance, False),
        (zero_one_loss, False),
        (hamming_loss, False),
        (brier_score_loss, False),
    ],
    ids=lambda dt: f'{getattr(dt, "__name__", dt)}'
)
def test_metric_greater_is_better_output_when_scikit_learn_metric(metric, expected):
    """
    Test that the direction of the scikit-learn metrics is inferred from their
    names, also when they are wrapped by `add_y_train_argument` or
    `functools.partial`.
    """
    assert _metric_greater_is_better(metric) is expected
    assert _metric_greater_is_better(add_y_train_argument(metric)) is expected
    assert _metric_greater_is_better(partial(metric)) is expected
    assert _metric_greater_is_better(add_y_train_argument(partial(metric))) is expected


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('mean_squared_error', False),
        ('root_mean_squared_error', False),
        ('mean_absolute_error', False),
        ('mean_absolute_percentage_error', False),
        ('mean_squared_log_error', False),
        ('mean_absolute_scaled_error', None),
        ('root_mean_squared_scaled_error', None),
        ('median_absolute_error', False),
        ('symmetric_mean_absolute_percentage_error', None),
        ('accuracy_score', True),
        ('balanced_accuracy_score', True),
        ('f1_score', True),
        ('precision_score', True),
        ('recall_score', True),
    ],
    ids=lambda dt: f'{dt}'
)
def test_metric_greater_is_better_output_when_metric_from_get_metric(metric, expected):
    """
    Test the direction of every metric accepted by name. The metrics of
    skforecast have no known direction, so the search uses the direction of
    the task (they are regression metrics, which are minimized).
    """
    assert _metric_greater_is_better(_get_metric(metric=metric)) is expected


@pytest.mark.parametrize(
    'metric',
    [
        custom_metric,
        custom_score,
        lambda y_true, y_pred: 0.0,
        partial(custom_metric),
        mean_absolute_scaled_error,
        create_mean_pinball_loss(alpha=0.9),
        winkler_score,
    ],
    ids=[
        'custom_metric',
        'custom_score',
        'lambda',
        'partial',
        'mean_absolute_scaled_error',
        'pinball',
        'winkler_score',
    ]
)
def test_metric_greater_is_better_None_when_direction_unknown(metric):
    """
    Test that `None` is returned for functions that are not scikit-learn
    metrics and have no attribute `greater_is_better`, so that the search
    uses the direction of the task. The scikit-learn naming convention is not
    applied to them (`custom_score`, `winkler_score`).
    """
    assert _metric_greater_is_better(metric) is None
    assert _metric_greater_is_better(add_y_train_argument(metric)) is None


@pytest.mark.parametrize(
    'greater_is_better',
    [True, False, np.bool_(True), np.bool_(False)],
    ids=lambda dt: f'greater_is_better: {dt}'
)
def test_metric_greater_is_better_output_when_attribute_is_defined(greater_is_better):
    """
    Test that the attribute `greater_is_better` defines the direction and takes
    precedence over the name, also through wrappers.
    """
    def my_metric(y_true, y_pred, y_train=None):  # pragma: no cover
        return 0.0

    def my_score(y_true, y_pred):  # pragma: no cover
        return 0.0

    my_metric.greater_is_better = greater_is_better
    my_score.greater_is_better = greater_is_better
    my_partial = partial(r2_score)
    my_partial.greater_is_better = greater_is_better

    for metric in [my_metric, my_score, my_partial]:
        assert _metric_greater_is_better(metric) is bool(greater_is_better)
        assert (
            _metric_greater_is_better(add_y_train_argument(metric))
            is bool(greater_is_better)
        )


def test_metric_greater_is_better_TypeError_when_attribute_is_not_bool():
    """
    Test TypeError is raised when the attribute `greater_is_better` is not a
    boolean.
    """
    def my_metric(y_true, y_pred):  # pragma: no cover
        return 0.0

    for greater_is_better in ['maximize', 1, 0.0]:
        my_metric.greater_is_better = greater_is_better
        err_msg = re.escape(
            f"The attribute `greater_is_better` of the metric must be a boolean: "
            f"`True` or `False`. Got {type(greater_is_better)}."
        )
        with pytest.raises(TypeError, match=err_msg):
            _metric_greater_is_better(my_metric)
