# Unit test crps_from_predictions
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import crps_from_predictions


def test_crps_from_predictions_raises_error_when_y_true_is_not_int_or_float():
    """
    This test verifies that the function `crps_from_predictions` raises a
    `TypeError` when `y_true` is not an integer or float.
    """

    y_true = "invalid-value"
    y_pred = np.random.normal(500, 10, 100)
    
    err_msg = re.escape("`y_true` must be a float or integer.")
    with pytest.raises(TypeError, match = err_msg):
        crps_from_predictions(y_true, y_pred)


def test_crps_from_predictions_raises_error_when_y_pred_is_not_1d_array():
    """
    This test verifies that the function `crps_from_predictions` raises a
    `TypeError` when the input `y_pred` is not a 1D array.
    """

    y_true = 1
    y_pred = np.array([[1, 2], [3, 4]])
    
    err_msg = re.escape("`y_pred` must be a 1D numpy array.")
    with pytest.raises(TypeError, match = err_msg):
        crps_from_predictions(y_true, y_pred)


def test_crps_from_predictions_output():
    """
    Test output crps_from_predictions
    """
    y_true = 20
    y_pred = np.array([
                8.98516149, 25.84367949, 16.40895419, 19.72014324, 14.3529477,
                5.38841901,  7.27408446, 11.32718453, 25.35152174, 27.30077998,
                18.3414763, 16.43394644, 17.2256319, 23.21058096, 18.88273853,
                1.73470249, 22.20651515, 25.3752937,  7.25976185, 34.54323398,
                26.38369077, 16.9818026, 13.81680905, 31.882123, 13.51698506,
                12.96658991, 21.12310745,  0.67550665, 27.14469549,  1.27185001,
                29.36689157, 19.23097391, 32.77156121, 20.30667739, 31.09958335,
                11.1869588,  8.69795603, 33.23353555, 32.41018021, 28.44995924,
                32.89498895, 28.16942487, 33.00331489, 16.93914131, 34.39509022,
                27.11909626, 13.83730754, 13.65824833, 38.60984957, 22.6927353
            ])
    result = crps_from_predictions(y_true, y_pred)
    expected = 2.637792405408

    np.testing.assert_almost_equal(result, expected)