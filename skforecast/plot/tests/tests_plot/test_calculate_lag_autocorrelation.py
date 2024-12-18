# Unit test calculate_lag_autocorrelation
# ==============================================================================
import pytest
import pandas as pd
from ... import calculate_lag_autocorrelation


def test_ccalculate_lag_autocorrelation_raise_error_invalid_sort_by():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    msg = (
        "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
        "'autocorrelation_abs' or 'autocorrelation'."
    )
    with pytest.raises(ValueError, match=msg):
        calculate_lag_autocorrelation(data=data, n_lags=4, sort_by="invalid_sort")


def test_calculate_lag_autocorrelation_raise_error_invalid_data():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    msg = "`data` must be a pandas Series."
    with pytest.raises(ValueError, match=msg):
        calculate_lag_autocorrelation(data=data, n_lags=4)


def test_calculate_lag_autocorrelation_output():
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
