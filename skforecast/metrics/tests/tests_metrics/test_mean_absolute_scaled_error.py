# Unit test mean_absolute_scaled_error
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import mean_absolute_scaled_error


def test_mean_absolute_scaled_error_input_types():
    """
    Test input types of mean_absolute_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        mean_absolute_scaled_error([1, 2, 3], y_pred, y_train)
    
    err_msg = re.escape("`y_pred` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        mean_absolute_scaled_error(y_true, [1, 2, 3], y_train)
    
    err_msg = re.escape("`y_train` must be a list, pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        mean_absolute_scaled_error(y_true, y_pred, 'not_valid_input')
    
    err_msg = re.escape(
        ("When `y_train` is a list, each element must be a pandas Series "
         "or numpy ndarray")
    )
    with pytest.raises(TypeError, match = err_msg):
        mean_absolute_scaled_error(y_true, y_pred, [1, 2, 3])


def test_mean_absolute_scaled_error_input_length():
    """
    Test input lengths of mean_absolute_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        mean_absolute_scaled_error(y_true, y_pred, y_train)

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        mean_absolute_scaled_error(y_true, y_pred, y_train)


def test_mean_absolute_scaled_error_empty_input():
    """
    Test empty input of mean_absolute_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` and `y_pred` must have at least one element")
    with pytest.raises(ValueError, match = err_msg):
        mean_absolute_scaled_error(np.array([]), np.array([]), y_train)


def test_mean_absolute_scaled_error_output():
    """
    Check that the output of mean_absolute_scaled_error is correct.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    expected_mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))

    assert np.isclose(mean_absolute_scaled_error(y_true, y_pred, y_train), expected_mase)


def test_mean_absolute_scaled_error_pandas_series_input():
    """
    Test pandas Series input of mean_absolute_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_train_series = pd.Series(y_train)

    expected_mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))

    assert np.isclose(mean_absolute_scaled_error(y_true_series, y_pred_series, y_train_series), expected_mase)
    
