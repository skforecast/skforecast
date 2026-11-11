# Unit test _calculate_metrics_backtesting
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from skforecast.metrics import (
    add_y_train_argument,
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
)
from skforecast.model_selection._utils import _calculate_metrics_backtesting

# Fixtures
index = pd.date_range(start='2020-01-01', periods=6, freq='D')
predictions = pd.DataFrame(
    {
        'fold': [0, 0, 0, 1, 1, 1],
        'step': [1, 2, 3, 1, 2, 3],
        'y_true': [1., np.nan, 3., 4., np.nan, 6.],
        'pred': [1.5, 2., 2., 4.5, 5., 7.],
        'lower_bound': [1., 1., 2.5, 4.2, 4., 5.],
        'upper_bound': [2., 3., 3.5, 4.8, 6., 8.],
    },
    index=index
)
y_train = pd.Series([1., 2., np.nan, 4., 7.])


@pytest.mark.parametrize(
    "pred_column",
    ['pred', 'q_0.5'],
    ids=lambda pred_column: f"pred_column: {pred_column}"
)
def test_calculate_metrics_backtesting_output_when_y_true_has_NaN(pred_column):
    """
    Test that the dates without an observed value are left out of the
    metrics, the interval metrics and `n_evaluated`, that the metrics never
    receive a missing value, and that the predictions are not modified. The
    expected values are calculated by hand with the 4 dates with a value.
    """
    predictions_input = predictions.rename(columns={'pred': pred_column})
    predictions_copy = predictions_input.copy()

    received = []

    def spy_metric(y_true, y_pred, y_train=None):
        received.append((np.asarray(y_true), np.asarray(y_pred)))
        return 0.

    metrics = [
        add_y_train_argument(mean_absolute_error),
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
        spy_metric,
    ]
    metric_values = _calculate_metrics_backtesting(
                        predictions      = predictions_input,
                        metrics          = metrics,
                        y_train          = y_train,
                        interval_columns = ['lower_bound', 'upper_bound'],
                        pred_column      = pred_column
                    )

    # Dates with a value: errors [0.5, 1, 0.5, 1]. Differences of y_train
    # [1, NaN, NaN, 3]. Coverage: 4 is outside [4.2, 4.8]. Widths [1, 1, 0.6, 3].
    expected = [
        0.75,
        0.75 / 2.,
        np.sqrt(0.625) / np.sqrt(5.),
        0.,
        0.75,
        1.4,
        4,
    ]
    np.testing.assert_allclose(metric_values, expected)
    assert len(received) == 1
    np.testing.assert_array_equal(received[0][0], [1., 3., 4., 6.])
    np.testing.assert_array_equal(received[0][1], [1.5, 2., 4.5, 7.])
    pd.testing.assert_frame_equal(predictions_input, predictions_copy)


def test_calculate_metrics_backtesting_output_without_interval_columns():
    """
    Test that, without `interval_columns`, only the metrics and `n_evaluated`
    are returned.
    """
    metric_values = _calculate_metrics_backtesting(
                        predictions = predictions,
                        metrics     = [add_y_train_argument(mean_absolute_error)]
                    )

    assert metric_values == [0.75, 4]


def test_calculate_metrics_backtesting_ValueError_when_no_observed_values():
    """
    Test ValueError is raised when no date of the test set has an observed
    value, so no metric can be calculated.
    """
    predictions_nan = predictions.copy()
    predictions_nan['y_true'] = np.nan

    err_msg = re.escape(
        "The series has no observed values in the test set of the folds, so the "
        "metrics cannot be calculated. Review `initial_train_size` of `cv`, or "
        "the span of the series."
    )
    with pytest.raises(ValueError, match=err_msg):
        _calculate_metrics_backtesting(
            predictions = predictions_nan,
            metrics     = [add_y_train_argument(mean_absolute_error)]
        )


def test_calculate_metrics_backtesting_ValueError_NaN_predictions_before_dropping():
    """
    Test that a NaN prediction at a date with an observed value raises the
    error of `_check_nan_predictions_backtesting`, with the estimator, before
    the dates without an observed value are left out.
    """
    predictions_nan = predictions.copy()
    predictions_nan['y_true'] = [np.nan, np.nan, 3., np.nan, np.nan, np.nan]
    predictions_nan['pred'] = [1., 2., np.nan, 4., 5., 6.]

    err_msg = re.escape(
        "1 backtesting predictions of estimator 'arima' are NaN at dates with "
        "observed values (first: 2020-01-03 00:00:00)"
    )
    with pytest.raises(ValueError, match=err_msg):
        _calculate_metrics_backtesting(
            predictions  = predictions_nan,
            metrics      = [add_y_train_argument(mean_absolute_error)],
            estimator_id = 'arima'
        )
