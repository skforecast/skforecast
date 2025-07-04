# Unit test symmetric_mean_absolute_percentage_error
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import symmetric_mean_absolute_percentage_error


def test_symmetric_mean_absolute_percentage_error_input_types():
    """
    Test input types of symmetric_mean_absolute_percentage_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    err_msg = re.escape("`y_true` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        symmetric_mean_absolute_percentage_error([1, 2, 3], y_pred)
    
    err_msg = re.escape("`y_pred` must be a pandas Series or numpy ndarray")
    with pytest.raises(TypeError, match = err_msg):
        symmetric_mean_absolute_percentage_error(y_true, [1, 2, 3])


def test_symmetric_mean_absolute_percentage_error_input_length():
    """
    Test input lengths of symmetric_mean_absolute_percentage_error.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        symmetric_mean_absolute_percentage_error(y_true, y_pred)

    err_msg = re.escape("`y_true` and `y_pred` must have the same length")
    with pytest.raises(ValueError, match = err_msg):
        symmetric_mean_absolute_percentage_error(y_true, y_pred)


def test_symmetric_mean_absolute_percentage_error_empty_input():
    """
    Test empty input of symmetric_mean_absolute_percentage_error.
    """
    y_true = np.array([])
    y_pred = np.array([])

    err_msg = re.escape("`y_true` and `y_pred` must have at least one element")
    with pytest.raises(ValueError, match = err_msg):
        symmetric_mean_absolute_percentage_error(y_true, y_pred)


def test_symmetric_mean_absolute_percentage_error_output():
    """
    Check that the output of symmetric_mean_absolute_percentage_error is correct.
    """
    y_true = np.array([30, 33.3, 38, 42, 31, 29.5, 43.5, 35.9, 37, 40])
    y_pred = np.array([34, 31, 43, 41.4, 35.6, 33, 40, 38, 34, 44.5])

    expected_smape = 9.162048875098746

    assert np.isclose(
        symmetric_mean_absolute_percentage_error(y_true, y_pred), 
        expected_smape
    )


def test_symmetric_mean_absolute_percentage_error_pandas_series_input():
    """
    Test pandas Series input of symmetric_mean_absolute_percentage_error.
    """
    y_true = pd.Series([30, 33.3, 38, 42, 31, 29.5, 43.5, 35.9, 37, 40])
    y_pred = pd.Series([34, 31, 43, 41.4, 35.6, 33, 40, 38, 34, 44.5])

    expected_smape = 9.162048875098746

    assert np.isclose(
        symmetric_mean_absolute_percentage_error(y_true, y_pred), 
        expected_smape
    )    
