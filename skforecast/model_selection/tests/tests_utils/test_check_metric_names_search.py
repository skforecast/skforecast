# Unit test _check_metric_names_search
# ==============================================================================
import re
import pytest
from skforecast.model_selection._utils import _check_metric_names_search


def test_check_metric_names_search_ValueError_when_duplicated_names():
    """
    Test ValueError is raised when the names of the metrics are not unique.
    """
    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match=err_msg):
        _check_metric_names_search(
            metric_names=['mean_absolute_error', 'mean_absolute_error']
        )


def test_check_metric_names_search_ValueError_when_metric_named_n_evaluated():
    """
    Test ValueError is raised when a metric is named `n_evaluated`, the column
    of the backtesting metrics with the number of points evaluated.
    """
    err_msg = re.escape(
        "The backtesting metrics include the column 'n_evaluated' with the "
        "number of points evaluated. Rename the metric 'n_evaluated' passed in "
        "`metric`."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_metric_names_search(metric_names=['mean_absolute_error', 'n_evaluated'])


@pytest.mark.parametrize(
    "metric_names",
    [['mean_absolute_error'], ['mean_absolute_error', 'coverage', 'interval_width']],
    ids=lambda names: f"metric_names: {names}",
)
def test_check_metric_names_search_valid_names(metric_names):
    """
    Test that unique names other than `n_evaluated` are valid. The searches do
    not compute interval metrics, so `coverage` and `interval_width` are allowed.
    """
    assert _check_metric_names_search(metric_names=metric_names) is None
