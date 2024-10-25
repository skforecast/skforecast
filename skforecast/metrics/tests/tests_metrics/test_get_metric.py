# Unit test _get_metric
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import _get_metric
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error


def test_get_metric_ValueError_when_metric_not_in_metrics_allowed():
    """
    Test ValueError is raised when metric is not in metrics allowed.
    """
    metric = 'not_a_metric'
    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
        "median_absolute_error"
    ]
    
    err_msg = re.escape(f"Allowed metrics are: {allowed_metrics}. Got {metric}.")
    with pytest.raises(ValueError, match = err_msg):
        _get_metric(metric)


@pytest.mark.parametrize("metric_str, metric_callable", 
                         [('mean_squared_error', mean_squared_error),
                          ('mean_absolute_error', mean_absolute_error),
                          ('mean_absolute_percentage_error', mean_absolute_percentage_error),
                          ('mean_squared_log_error', mean_squared_log_error),
                          ('median_absolute_error', median_absolute_error)], 
                         ids = lambda dt: f'mertic_str, metric_callable: {dt}')
def test_get_metric_output_for_all_metrics(metric_str, metric_callable):
    """
    Test output for all metrics allowed.
    """
    y_true = np.array([3, 1, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    metric = _get_metric(metric_str)
    expected = metric_callable(y_true=y_true, y_pred=y_pred)
    
    assert metric(y_true=y_true, y_pred=y_pred) == expected


@pytest.mark.parametrize("metric_str, metric_callable", 
                         [('mean_absolute_scaled_error', mean_absolute_scaled_error),
                          ('root_mean_squared_scaled_error', root_mean_squared_scaled_error)], 
                         ids = lambda dt : f'mertic_str, metric_callable: {dt}')
def test_get_metric_output_for_all_metrics_y_train(metric_str, metric_callable):
    """
    Test output for all metrics allowed with y_train argument.
    """
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    y_train = np.array([2, 3, 4, 5])

    metric = _get_metric(metric_str)
    expected = metric_callable(y_true=y_true, y_pred=y_pred, y_train=y_train)
    
    assert metric(y_true=y_true, y_pred=y_pred, y_train=y_train) == expected