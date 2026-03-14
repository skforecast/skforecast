# Unit test any_metric_needs_y_train
# ==============================================================================
import pytest
from skforecast.metrics import (
    add_y_train_argument,
    any_metric_needs_y_train,
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


@pytest.mark.parametrize(
    'metrics, expected',
    [
        ([add_y_train_argument(mean_absolute_scaled_error),
          add_y_train_argument(root_mean_squared_scaled_error)], True),
        ([add_y_train_argument(mean_absolute_error),
          add_y_train_argument(mean_squared_error)], False),
        ([add_y_train_argument(mean_absolute_error),
          add_y_train_argument(mean_absolute_scaled_error)], True),
        ([add_y_train_argument(mean_absolute_scaled_error)], True),
        ([add_y_train_argument(mean_absolute_error)], False),
        ([], False),
    ],
    ids=[
        'all_need_y_train',
        'none_need_y_train',
        'mixed',
        'single_needs_y_train',
        'single_no_y_train',
        'empty_list',
    ],
)
def test_any_metric_needs_y_train_with_wrapped_metrics(metrics, expected):
    """
    Test any_metric_needs_y_train with metrics processed by
    add_y_train_argument.
    """
    assert any_metric_needs_y_train(metrics) is expected


@pytest.mark.parametrize(
    'needs_attr, expected',
    [
        (None, True),
        (True, True),
        (False, False),
    ],
    ids=[
        'attribute_missing_defaults_True',
        '_needs_y_train=True',
        '_needs_y_train=False',
    ],
)
def test_any_metric_needs_y_train_with_custom_metric(needs_attr, expected):
    """
    Test any_metric_needs_y_train with custom metrics that may or may not
    have the `_needs_y_train` attribute.
    """
    def custom_metric(y_true, y_pred):
        return 0.0

    if needs_attr is not None:
        custom_metric._needs_y_train = needs_attr

    assert any_metric_needs_y_train([custom_metric]) is expected
