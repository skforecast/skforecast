# Unit test _scaled_metric_per_origin
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.metrics import (
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error
)
from skforecast.metrics.metrics import _scaled_metric_per_origin


def test_scaled_metric_per_origin_output_mean_of_scaled_errors():
    """
    Test that MASE is the mean of the absolute errors divided by the scale of
    each point, and RMSSE the root of the mean of the squared errors divided by
    the squared scale of each point. With the same scale for every point, they
    are equal to `mean_absolute_scaled_error` and
    `root_mean_squared_scaled_error` with that `y_train`.
    """
    y_true = pd.Series([3.0, 5.0, 2.5, 7.0])
    y_pred = pd.Series([2.5, 5.5, 2.0, 8.0])
    scale_abs = np.array([1.0, 1.0, 2.0, 2.0])
    scale_sq = np.array([1.0, 1.0, 4.0, 4.0])
    errors = (y_true - y_pred).to_numpy()

    mase, n_mase = _scaled_metric_per_origin(
        mean_absolute_scaled_error, y_true, y_pred, scale_abs, scale_sq
    )
    rmsse, n_rmsse = _scaled_metric_per_origin(
        root_mean_squared_scaled_error, y_true, y_pred, scale_abs, scale_sq
    )

    assert mase == np.mean(np.abs(errors) / scale_abs)
    np.testing.assert_almost_equal(rmsse, np.sqrt(np.mean(errors ** 2 / scale_sq)))
    assert n_mase == 0
    assert n_rmsse == 0

    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mase_fixed, _ = _scaled_metric_per_origin(
        mean_absolute_scaled_error, y_true, y_pred, np.full(4, 1.0), np.full(4, 1.0)
    )
    np.testing.assert_almost_equal(
        mase_fixed, mean_absolute_scaled_error(y_true, y_pred, y_train)
    )
    rmsse_fixed, _ = _scaled_metric_per_origin(
        root_mean_squared_scaled_error, y_true, y_pred, np.full(4, 1.0), np.full(4, 1.0)
    )
    np.testing.assert_almost_equal(
        rmsse_fixed, root_mean_squared_scaled_error(y_true, y_pred, y_train)
    )


def test_scaled_metric_per_origin_leaves_out_points_without_scale_or_error():
    """
    Test that the points whose scale is missing or zero are left out and
    counted, the points without an error (missing true value) are left out
    without being counted, and the metric is NaN when no point is left.
    """
    y_true = np.array([3.0, 5.0, np.nan, 7.0, 1.0])
    y_pred = np.array([2.5, 5.5, 2.0, 8.0, 1.0])
    scale_abs = np.array([1.0, np.nan, 2.0, 0.0, 2.0])
    scale_sq = scale_abs ** 2

    mase, n_undefined = _scaled_metric_per_origin(
        mean_absolute_scaled_error, y_true, y_pred, scale_abs, scale_sq
    )
    assert mase == np.mean([0.5 / 1.0, 0.0 / 2.0])
    assert n_undefined == 2

    mase, n_undefined = _scaled_metric_per_origin(
        mean_absolute_scaled_error, y_true, y_pred,
        np.full(5, np.nan), np.full(5, np.nan)
    )
    assert np.isnan(mase)
    assert n_undefined == 4
