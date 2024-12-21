# Unit test calculate_lag_autocorrelation
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from ... import calculate_lag_autocorrelation


def test_calculate_lag_autocorrelation_raise_error_invalid_arguments():
    """
    Test that calculate_lag_autocorrelation raises an error when invalid 
    arguments are passed.
    """
    
    wrong_data = np.arange(10)
    err_msg = re.escape(
        f"`data` must be a pandas Series or DataFrame. Got {type(wrong_data)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        calculate_lag_autocorrelation(data=wrong_data)
    
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    wrong_n_lags = -1
    err_msg = re.escape(f"`n_lags` must be a positive integer. Got {wrong_n_lags}.")
    with pytest.raises(TypeError, match=err_msg):
        calculate_lag_autocorrelation(data=data, n_lags=-1)

    wrong_last_n_samples = -1
    err_msg = re.escape(f"`last_n_samples` must be a positive integer. Got {wrong_last_n_samples}.")
    with pytest.raises(TypeError, match=err_msg):
        calculate_lag_autocorrelation(data=data, n_lags=3, last_n_samples=wrong_last_n_samples)
    
    err_msg = re.escape(
        "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
        "'autocorrelation_abs' or 'autocorrelation'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(data=data, n_lags=4, sort_by="invalid_sort")


def test_calculate_lag_autocorrelation_output():
    """
    Check that calculate_lag_autocorrelation returns the expected output.
    """
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected = pd.DataFrame(
        {
            "lag": [1, 4, 3, 2],
            "partial_autocorrelation_abs": [
                0.7777777777777778,
                0.36070686070686075,
                0.2745098039215686,
                0.22727272727272751,
            ],
            "partial_autocorrelation": [
                0.7777777777777778,
                -0.36070686070686075,
                -0.2745098039215686,
                -0.22727272727272751,
            ],
            "autocorrelation_abs": [
                0.7000000000000001,
                0.0787878787878788,
                0.14848484848484844,
                0.41212121212121194,
            ],
            "autocorrelation": [
                0.7000000000000001,
                -0.0787878787878788,
                0.14848484848484844,
                0.41212121212121194,
            ],
        }
    )

    results = calculate_lag_autocorrelation(data=data, n_lags=4)
    pd.testing.assert_frame_equal(results, expected)


def test_calculate_lag_autocorrelation_output_sort_by_lag():
    """
    Check that calculate_lag_autocorrelation returns the expected output when
    sort_by='lag'.
    """
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected = pd.DataFrame(
        {
            "lag": [1, 2, 3, 4],
            "partial_autocorrelation_abs": [
                0.7777777777777778,
                0.22727272727272751,
                0.2745098039215686,
                0.36070686070686075,
            ],
            "partial_autocorrelation": [
                0.7777777777777778,
                -0.22727272727272751,
                -0.2745098039215686,
                -0.36070686070686075,
            ],
            "autocorrelation_abs": [
                0.7000000000000001,
                0.41212121212121194,
                0.14848484848484844,
                0.0787878787878788,
            ],
            "autocorrelation": [
                0.7000000000000001,
                0.41212121212121194,
                0.14848484848484844,
                -0.0787878787878788,
            ],
        }
    )

    results = calculate_lag_autocorrelation(data=data, n_lags=4, sort_by="lag")
    pd.testing.assert_frame_equal(results, expected)


def test_calculate_lag_autocorrelation_output_last_n_samples():
    """
    Check that calculate_lag_autocorrelation returns the expected output when
    using last_n_samples.
    """
    data = pd.Series(np.arange(20)[::-1])
    expected = pd.DataFrame(
        {
            "lag": [1, 2, 3, 4],
            "partial_autocorrelation_abs": [
                0.7777777777777778,
                0.22727272727272751,
                0.2745098039215686,
                0.36070686070686075,
            ],
            "partial_autocorrelation": [
                0.7777777777777778,
                -0.22727272727272751,
                -0.2745098039215686,
                -0.36070686070686075,
            ],
            "autocorrelation_abs": [
                0.7000000000000001,
                0.41212121212121194,
                0.14848484848484844,
                0.0787878787878788,
            ],
            "autocorrelation": [
                0.7000000000000001,
                0.41212121212121194,
                0.14848484848484844,
                -0.0787878787878788,
            ],
        }
    )

    results = calculate_lag_autocorrelation(
        data=data, n_lags=4, last_n_samples=10, sort_by="lag"
    )
    pd.testing.assert_frame_equal(results, expected)
