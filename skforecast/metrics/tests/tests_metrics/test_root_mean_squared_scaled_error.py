# Unit test root_mean_squared_scaled_error
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import root_mean_squared_scaled_error


def test_root_mean_squared_scaled_error_input_types():
    """
    Test input types of root_mean_squared_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        root_mean_squared_scaled_error([1, 2, 3], y_pred, y_train)
    
    err_msg = re.escape("`y_pred` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        root_mean_squared_scaled_error(y_true, [1, 2, 3], y_train)
    
    err_msg = re.escape("`y_train` must be a list, pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        root_mean_squared_scaled_error(y_true, y_pred, 'not_valid_input')
    
    err_msg = re.escape(
        ("When `y_train` is a list, each element must be a pandas Series "
         "or numpy ndarray")
    )
    with pytest.raises(TypeError, match = err_msg):
        root_mean_squared_scaled_error(y_true, y_pred, [1, 2, 3])


def test_root_mean_squared_scaled_error_input_length():
    """
    Test input lengths of root_mean_squared_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        root_mean_squared_scaled_error(y_true, y_pred, y_train)

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        root_mean_squared_scaled_error(y_true, y_pred, y_train)


def test_root_mean_squared_scaled_error_empty_input():
    """
    Test empty input of root_mean_squared_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    err_msg = re.escape("`y_true` and `y_pred` must have at least one element")
    with pytest.raises(ValueError, match = err_msg):
        root_mean_squared_scaled_error(np.array([]), np.array([]), y_train)


def test_root_mean_squared_scaled_error_output():
    """
    Test input types of root_mean_squared_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(np.diff(y_train) ** 2))
    
    assert np.isclose(root_mean_squared_scaled_error(y_true, y_pred, y_train), expected_rmsse)
    

def test_pandas_series_input():
    """
    Test pandas Series input of root_mean_squared_scaled_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_train_series = pd.Series(y_train)
    
    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(np.diff(y_train) ** 2))
    
    assert np.isclose(root_mean_squared_scaled_error(y_true_series, y_pred_series, y_train_series), expected_rmsse)