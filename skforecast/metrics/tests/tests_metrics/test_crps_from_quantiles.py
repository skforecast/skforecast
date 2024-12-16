# Unit test crps_from_predictions
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import crps_from_quantiles


def test_crps_from_quantiles_raises_error_when_pred_quantiles_is_not_1d_array():
    """
    This test verifies that the function `crps_from_quantiles` raises a
    `TypeError` when the input `pred_quantiles` is not a 1D array.
    """
    y_true = 1
    pred_quantiles = "non valid input"
    quantile_levels = np.array([0.1, 0.5, 0.9])

    err_msg = re.escape("`pred_quantiles` must be a 1D numpy array.")
    with pytest.raises(TypeError, match = err_msg):
        crps_from_quantiles(y_true, pred_quantiles, quantile_levels)


def test_crps_from_quantiles_raises_error_when_quantile_levels_is_not_1d_array():
    """
    This test verifies that the function `crps_from_quantiles` raises a
    `TypeError` when the input `quantile_levels` is not a 1D array.
    """
    y_true = 1
    pred_quantiles = np.array([1, 2, 3])
    quantile_levels = "no valid input"

    err_msg = re.escape("`quantile_levels` must be a 1D numpy array.")
    with pytest.raises(TypeError, match = err_msg):
        crps_from_quantiles(y_true, pred_quantiles, quantile_levels)


def test_crps_from_quantiles_raises_error_when_y_true_is_not_int_or_float():
    """
    This test verifies that the function `crps_from_quantiles` raises a
    `TypeError` when the input `y_true` is not an integer or float.
    """

    y_true = "no valid input"
    pred_quantiles = np.array([1, 2, 3])
    quantile_levels = np.array([0.1, 0.5, 0.9])

    err_msg = re.escape("`y_true` must be a float or integer.")
    with pytest.raises(TypeError, match = err_msg):
        crps_from_quantiles(y_true, pred_quantiles, quantile_levels)


def test_crps_from_quantiles_output():
    """
    This test verifies that the function `crps_from_quantiles` output.
    """
    y_true = 3.0
    quantile_levels = np.array([
        0.00, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 1.00
    ])
    pred_quantiles = np.array([
        0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5
    ])
    expected = 1.7339183102042313
    result = crps_from_quantiles (y_true, pred_quantiles, quantile_levels)

    np.testing.assert_almost_equal(result, expected)