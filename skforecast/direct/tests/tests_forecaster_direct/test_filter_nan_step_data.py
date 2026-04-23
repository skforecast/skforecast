# Unit test _filter_nan_X_y_step ForecasterDirect
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirect


@pytest.mark.parametrize(
    'dropna_from_series',
    [True, False],
    ids=lambda x: f'dropna_from_series: {x}'
)
def test_filter_nan_X_y_step_no_nans(dropna_from_series):
    """
    Test that _filter_nan_X_y_step returns inputs unchanged when there are
    no NaN values in X_train_step or y_train_step.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=dropna_from_series
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])
    idx = pd.date_range('2020-01-01', periods=3, freq='D')

    X_out, y_out, idx_out = forecaster._filter_nan_X_y_step(X, y, idx)

    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)
    pd.testing.assert_index_equal(idx_out, idx)


def test_filter_nan_X_y_step_nan_in_both_X_and_y_dropna_True():
    """
    Test that NaN rows in both y and X are removed when dropna_from_series
    is True. Rows with NaN in y are removed first, then remaining rows with
    NaN in X. Also covers NaN-in-y-only and NaN-in-X-only branches.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=True
    )
    X = np.array([
        [1.0, 2.0],     # row 0: clean
        [np.nan, 4.0],  # row 1: NaN in X
        [5.0, 6.0],     # row 2: clean
        [7.0, 8.0],     # row 3: clean but NaN in y
    ])
    y = np.array([10.0, 20.0, 30.0, np.nan])
    idx = pd.date_range('2020-01-01', periods=4, freq='D')

    X_out, y_out, idx_out = forecaster._filter_nan_X_y_step(X, y, idx)

    # Row 3 removed (y NaN), then row 1 removed (X NaN)
    expected_X = np.array([[1.0, 2.0], [5.0, 6.0]])
    expected_y = np.array([10.0, 30.0])
    expected_idx = pd.DatetimeIndex(['2020-01-01', '2020-01-03'], freq=None)

    np.testing.assert_array_equal(X_out, expected_X)
    np.testing.assert_array_equal(y_out, expected_y)
    pd.testing.assert_index_equal(idx_out, expected_idx)


def test_filter_nan_X_y_step_nan_in_both_X_and_y_dropna_False():
    """
    Test that when dropna_from_series is False, only y NaN rows are removed
    even if X also has NaN. Covers y-always-removed and X-kept branches.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=False
    )
    X = np.array([
        [np.nan, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y = np.array([10.0, np.nan, 30.0])
    idx = pd.date_range('2020-01-01', periods=3, freq='D')

    X_out, y_out, idx_out = forecaster._filter_nan_X_y_step(X, y, idx)

    # Only row 1 removed (y NaN), row 0 with X NaN kept
    expected_X = np.array([[np.nan, 2.0], [5.0, 6.0]])
    expected_y = np.array([10.0, 30.0])
    expected_idx = pd.DatetimeIndex(['2020-01-01', '2020-01-03'], freq=None)

    np.testing.assert_array_equal(X_out, expected_X)
    np.testing.assert_array_equal(y_out, expected_y)
    pd.testing.assert_index_equal(idx_out, expected_idx)


@pytest.mark.parametrize(
    'dropna_from_series, X, y',
    [
        (True,
         np.array([[np.nan, 2.0], [3.0, np.nan]]),
         np.array([10.0, 20.0])),
        (False,
         np.array([[1.0, 2.0], [3.0, 4.0]]),
         np.array([np.nan, np.nan])),
    ],
    ids=['all_X_NaN_dropna_True', 'all_y_NaN_dropna_False']
)
def test_filter_nan_X_y_step_ValueError_when_all_samples_removed(
    dropna_from_series, X, y
):
    """
    Test that ValueError is raised when all samples are removed due to NaN,
    either because all X rows have NaN (dropna=True) or all y values are NaN.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=dropna_from_series
    )
    idx = pd.date_range('2020-01-01', periods=len(y), freq='D')

    err_msg = "All samples have been removed due to NaNs"
    with pytest.raises(ValueError, match=err_msg):
        forecaster._filter_nan_X_y_step(X, y, idx)


def test_filter_nan_X_y_step_train_index_None():
    """
    Test that _filter_nan_X_y_step works correctly when train_index_step
    is None. NaN filtering applies to X and y but no index operations are
    performed and None is returned for the index.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=True
    )
    X = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, np.nan, 30.0])

    X_out, y_out, idx_out = forecaster._filter_nan_X_y_step(X, y)

    # Row 1 removed (y NaN), row 0 removed (X NaN with dropna=True)
    expected_X = np.array([[5.0, 6.0]])
    expected_y = np.array([30.0])

    np.testing.assert_array_equal(X_out, expected_X)
    np.testing.assert_array_equal(y_out, expected_y)
    assert idx_out is None
