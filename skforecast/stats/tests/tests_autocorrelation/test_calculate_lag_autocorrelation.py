# Unit test calculate_lag_autocorrelation
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats import calculate_lag_autocorrelation


def test_calculate_lag_autocorrelation_raise_error_invalid_arguments():
    """
    Test that calculate_lag_autocorrelation raises an error when invalid
    arguments are passed.
    """
    wrong_data = np.arange(10)
    err_msg = re.escape(
        f"`data` must be a pandas Series or a DataFrame with a single column. "
        f"Got {type(wrong_data)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        calculate_lag_autocorrelation(data=wrong_data)

    wrong_data = pd.DataFrame(np.arange(10).reshape(-1, 2))
    err_msg = re.escape(
        f"If `data` is a DataFrame, it must have exactly one column. "
        f"Got {wrong_data.shape[1]} columns."
    )
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(data=wrong_data)

    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    wrong_n_lags = -1
    err_msg = re.escape(f"`n_lags` must be a positive integer. Got {wrong_n_lags}.")
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(data=data, n_lags=-1)

    wrong_last_n_samples = -1
    err_msg = re.escape(
        f"`last_n_samples` must be a positive integer. Got {wrong_last_n_samples}."
    )
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(
            data=data, n_lags=3, last_n_samples=wrong_last_n_samples
        )

    err_msg = re.escape(
        "`sort_by` must be 'lag', 'partial_autocorrelation_abs', "
        "'partial_autocorrelation', 'autocorrelation_abs' or 'autocorrelation'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(data=data, n_lags=4, sort_by="invalid_sort")

    data_short = pd.Series(range(10), dtype=float)  # n // 2 = 5
    err_msg = re.escape(
        "`n_lags` (5) must be less than len(data) // 2 (5). "
        "Partial autocorrelation cannot be computed for more than half the "
        "sample size."
    )
    with pytest.raises(ValueError, match=err_msg):
        calculate_lag_autocorrelation(data=data_short, n_lags=5)


def test_calculate_lag_autocorrelation_output():
    """
    Check that calculate_lag_autocorrelation returns the expected output.
    Values are computed with the internal acf/pacf (biased FFT estimator).
    """
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    expected = pd.DataFrame(
        {
            "lag": [1, 3, 4, 2],
            "partial_autocorrelation_abs": [
                0.7000000000000001,
                0.1549066670522403,
                0.15474911854395293,
                0.15270350564468238,
            ],
            "partial_autocorrelation": [
                0.7000000000000001,
                -0.1549066670522403,
                -0.15474911854395293,
                -0.15270350564468238,
            ],
            "autocorrelation_abs": [
                0.7000000000000001,
                0.1484848484848485,
                0.07878787878787877,
                0.4121212121212121,
            ],
            "autocorrelation": [
                0.7000000000000001,
                0.1484848484848485,
                -0.07878787878787877,
                0.4121212121212121,
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
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    expected = pd.DataFrame(
        {
            "lag": [1, 2, 3, 4],
            "partial_autocorrelation_abs": [
                0.7000000000000001,
                0.15270350564468238,
                0.1549066670522403,
                0.15474911854395293,
            ],
            "partial_autocorrelation": [
                0.7000000000000001,
                -0.15270350564468238,
                -0.1549066670522403,
                -0.15474911854395293,
            ],
            "autocorrelation_abs": [
                0.7000000000000001,
                0.4121212121212121,
                0.1484848484848485,
                0.07878787878787877,
            ],
            "autocorrelation": [
                0.7000000000000001,
                0.4121212121212121,
                0.1484848484848485,
                -0.07878787878787877,
            ],
        }
    )

    results = calculate_lag_autocorrelation(data=data, n_lags=4, sort_by="lag")
    pd.testing.assert_frame_equal(results, expected)


def test_calculate_lag_autocorrelation_last_n_samples():
    """
    Check that last_n_samples trims the series before computing.
    """
    full = pd.Series(range(1, 21), dtype=float)
    trimmed = full.iloc[-10:]

    result_trimmed = calculate_lag_autocorrelation(data=trimmed, n_lags=4)
    result_last = calculate_lag_autocorrelation(data=full, n_lags=4, last_n_samples=10)

    pd.testing.assert_frame_equal(result_trimmed, result_last)


def test_calculate_lag_autocorrelation_accepts_single_column_dataframe():
    """
    Check that a DataFrame with one column is accepted and gives the same
    result as the equivalent Series.
    """
    data_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    data_df = data_series.to_frame(name="value")

    result_series = calculate_lag_autocorrelation(data=data_series, n_lags=4)
    result_df = calculate_lag_autocorrelation(data=data_df, n_lags=4)

    pd.testing.assert_frame_equal(result_series, result_df)


def test_calculate_lag_autocorrelation_output_excludes_lag0():
    """
    Check that lag 0 is excluded from the results.
    """
    data = pd.Series(range(1, 21), dtype=float)
    results = calculate_lag_autocorrelation(data=data, n_lags=5)
    assert 0 not in results["lag"].values
    assert len(results) == 5
