# Unit test _calculate_interval_metrics
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _calculate_interval_metrics


@pytest.mark.parametrize(
    "y_true, expected",
    [([1.0, 2.0, 3.0, 4.0], [0.75, 1.25]),
     ([0.5, 2.5, 3.5, 5.0], [1.0, 1.25]),
     ([0.0, 3.0, 5.0, 6.0], [0.0, 1.25])],
    ids = lambda dt: f'y_true, expected: {dt}'
)
def test_calculate_interval_metrics_output(y_true, expected):
    """
    Test output of _calculate_interval_metrics: coverage (bounds included) and
    mean width of the interval, with numpy arrays and pandas Series.
    """
    lower_bound = np.array([0.5, 1.5, 3.5, 3.0])
    upper_bound = np.array([1.5, 2.5, 4.5, 5.0])

    results = _calculate_interval_metrics(
        y_true      = np.array(y_true),
        lower_bound = lower_bound,
        upper_bound = upper_bound
    )
    results_series = _calculate_interval_metrics(
        y_true      = pd.Series(y_true),
        lower_bound = pd.Series(lower_bound),
        upper_bound = pd.Series(upper_bound)
    )

    np.testing.assert_array_almost_equal(results, expected)
    np.testing.assert_array_almost_equal(results_series, expected)


def test_calculate_interval_metrics_output_when_missing_values():
    """
    Test output of _calculate_interval_metrics when `y_true`, `lower_bound` or
    `upper_bound` have missing values: those rows are ignored.
    """
    y_true = np.array([1.0, np.nan, 3.0, 4.0, 10.0, 2.0])
    lower_bound = np.array([0.5, 1.5, np.nan, 3.0, 9.0, 0.0])
    upper_bound = np.array([1.5, 2.5, 4.5, 5.0, 9.5, np.nan])

    results = _calculate_interval_metrics(
        y_true      = y_true,
        lower_bound = lower_bound,
        upper_bound = upper_bound
    )
    expected = [0.6666666666666666, 1.1666666666666667]

    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_interval_metrics_output_when_all_values_missing():
    """
    Test output of _calculate_interval_metrics when no row is left after
    removing the missing values: both metrics are NaN.
    """
    results = _calculate_interval_metrics(
        y_true      = np.array([np.nan, 2.0]),
        lower_bound = np.array([0.5, np.nan]),
        upper_bound = np.array([1.5, 2.5])
    )

    assert np.isnan(results).all()
    assert len(results) == 2
