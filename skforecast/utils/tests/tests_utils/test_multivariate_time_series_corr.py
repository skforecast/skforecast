# Unit test multivariate_time_series_corr
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils.utils import multivariate_time_series_corr


def test_output_multivariate_time_series_corr():
    """
    Test the output of the correlation matrix
    """

    time_series = pd.Series(range(10), name='series_1')
    other = pd.DataFrame({
                'series_2': range(10),
                'series_3': [9, 2, 3, 8, 2, 6, 0, 8, 6, 1]
            })

    expected = pd.DataFrame(
                data = {
                    'series_2': [1., 1., 1., 1., 1.],
                    'series_3': [-0.21854656, -0.02836, -0.11964501, -0.45360921, -0.17251639],
                },
                index = pd.RangeIndex(start=0, stop=5, step=1)
            )
    expected.index.name = 'lag'

    results = multivariate_time_series_corr(
                  time_series = time_series,
                  other       = other,
                  lags        = 5
              )

    pd.testing.assert_frame_equal(results, expected)


def test_exception_multivariate_when_inputs_have_different_index():
    """
    Test that an exception is raised when `time_Series` and `other` have different index.
    """

    time_series = pd.Series(range(5), name='series_1')
    other = pd.DataFrame(
                {'series_2': range(5),
                'series_3': [9, 2, 3, 8, 2]},
                index = pd.RangeIndex(start=10, stop=15, step=1)
            )

    err_msg = re.escape("`time_series` and `other` must have the same index.")
    with pytest.raises(ValueError, match = err_msg):
        multivariate_time_series_corr(
            time_series = time_series,
            other       = other,
            lags        = 5
        )


def test_exception_multivariate_when_inputs_have_different_length():
    """
    Test that an exception is raised when `time_Series` and `other` have different lenth.
    """

    time_series = pd.Series(range(10), name='series_1')
    other = pd.DataFrame({
                'series_2': range(5),
                'series_3': [9, 2, 3, 8, 2]
            })

    err_msg = re.escape("`time_series` and `other` must have the same length.")
    with pytest.raises(ValueError, match = err_msg):
        multivariate_time_series_corr(
            time_series = time_series,
            other       = other,
            lags        = 5
        )


@pytest.mark.parametrize("method", 
                         ['pearson', 'spearman', 'kendall'], 
                         ids = lambda method: f'method: {method}')
@pytest.mark.parametrize("lags", 
                         [4, np.int64(4), [1, 3], np.array([0, 2])], 
                         ids = lambda lags: f'lags: {lags!r}')
def test_output_multivariate_time_series_corr_equals_correlation_matrix(method, lags):
    """
    Test the output is the column of `time_series` in the correlation matrix 
    of `time_series` and the lagged values of each column of `other`, for 
    every method, with missing values, and with `lags` as a numpy integer, a 
    list or an array.
    """
    rng = np.random.default_rng(123)
    time_series = pd.Series(rng.normal(size=50).cumsum(), name='series_1')
    time_series.iloc[[3, 20]] = np.nan
    other = pd.DataFrame({
                'series_2': rng.normal(size=50).cumsum(),
                'series_3': rng.normal(size=50)
            })
    other.iloc[[5, 30], 1] = np.nan

    results = multivariate_time_series_corr(
                  time_series = time_series,
                  other       = other,
                  lags        = lags,
                  method      = method
              )

    lags_list = range(lags) if isinstance(lags, (int, np.integer)) else lags
    expected = {}
    for col in other.columns:
        lag_values = pd.DataFrame({lag: other[col].shift(lag) for lag in lags_list})
        lag_values.insert(0, 'y', time_series)
        expected[col] = lag_values.corr(method=method).iloc[1:, 0]
    expected = pd.DataFrame(expected)
    expected.index = expected.index.astype('int64')
    expected.index.name = 'lag'

    pd.testing.assert_frame_equal(results, expected)
