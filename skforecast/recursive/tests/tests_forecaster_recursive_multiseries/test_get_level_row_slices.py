# Unit test _get_level_row_slices ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_dict_unordered,
    exog_dict_unordered
)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ('ordinal',
         pd.DataFrame({'lag_1': [1., 2., 3., 4.],
                       '_level_skforecast': [0., 0., 1., 0.]})),
        ('ordinal_category',
         pd.DataFrame({'lag_1': [1., 2., 3., 4.],
                       '_level_skforecast': pd.Categorical([0, 0, 1, 0])})),
        ('onehot',
         pd.DataFrame({'lag_1': [1., 2., 3., 4.],
                       'l1': [1, 1, 0, 1],
                       'l2': [0, 0, 1, 0]})),
    ],
    ids=['encoding: ordinal', 'encoding: ordinal_category', 'encoding: onehot']
)
def test_get_level_row_slices_ValueError_when_rows_of_a_level_are_not_contiguous(
    encoding, X_train
):
    """
    Test ValueError is raised when the rows of a level are not contiguous in
    `X_train`, for example when the matrix has been shuffled.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=1, encoding=encoding
    )
    forecaster.encoding_mapping_ = {'l1': 0, 'l2': 1}

    err_msg = re.escape(
        "The rows of each series in `X_train` must be contiguous, as "
        "returned by `create_train_X_y`. Rows of series 'l1' are not."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._get_level_row_slices(X_train=X_train)


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'ordinal_category', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
def test_get_level_row_slices_output_when_series_unordered_different_lengths_and_dropped(
    encoding
):
    """
    Test the slice of rows of each level when the series are not in alphabetical
    order ('c', 'a', 'd', 'b'), have different lengths and an interspersed NaN
    (4 rows of 'b' are kept), and all the rows of 'd' are removed because it has
    no exog and `dropna_from_series=True`. The slices follow the order of the
    rows, not the alphabetical order of `encoding_mapping_`, and 'd' is not
    included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding=encoding, dropna_from_series=True
    )
    X_train = forecaster._create_train_X_y(
        series=series_dict_unordered, exog=exog_dict_unordered
    )[0]
    results = forecaster._get_level_row_slices(X_train=X_train)

    expected = {'c': slice(0, 6), 'a': slice(6, 13), 'b': slice(13, 17)}

    assert list(results) == list(expected)
    assert results == expected


@pytest.mark.parametrize(
    "dtype",
    [int, float],
    ids=lambda dtype: f'onehot dtype: {dtype.__name__}'
)
def test_get_level_row_slices_output_when_onehot_columns_int_or_float(dtype):
    """
    Test the slice of rows of each level with `encoding='onehot'` when the
    one-hot columns are int or float (as created by `_create_train_X_y`).
    """
    X_train = pd.DataFrame({
        'lag_1': [1., 2., 3., 4., 5., 6.],
        'l1': np.array([0, 0, 0, 1, 1, 0], dtype=dtype),
        'l2': np.array([0, 0, 0, 0, 0, 1], dtype=dtype),
        'l3': np.array([1, 1, 1, 0, 0, 0], dtype=dtype)
    })
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=1, encoding='onehot'
    )
    forecaster.encoding_mapping_ = {'l1': 0, 'l2': 1, 'l3': 2}
    results = forecaster._get_level_row_slices(X_train=X_train)

    expected = {'l3': slice(0, 3), 'l1': slice(3, 5), 'l2': slice(5, 6)}

    assert list(results) == list(expected)
    assert results == expected


@pytest.mark.parametrize(
    "columns",
    [['l1', 'lag_1', 'l2', 'l3'],
     ['lag_1', 'l3', 'l1', 'l2']],
    ids=['onehot columns not contiguous', 'onehot columns not in order']
)
def test_get_level_row_slices_output_when_onehot_columns_not_contiguous_or_unordered(
    columns
):
    """
    Test the slice of rows of each level with `encoding='onehot'` when the
    one-hot columns are not contiguous or not in the order of
    `encoding_mapping_` (for example, a matrix reordered by the user). The
    columns are then selected by name.
    """
    X_train = pd.DataFrame({
        'lag_1': [1., 2., 3., 4., 5., 6.],
        'l1': [0., 0., 0., 1., 1., 0.],
        'l2': [0., 0., 0., 0., 0., 1.],
        'l3': [1., 1., 1., 0., 0., 0.]
    })[columns]
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=1, encoding='onehot'
    )
    forecaster.encoding_mapping_ = {'l1': 0, 'l2': 1, 'l3': 2}
    results = forecaster._get_level_row_slices(X_train=X_train)

    expected = {'l3': slice(0, 3), 'l1': slice(3, 5), 'l2': slice(5, 6)}

    assert list(results) == list(expected)
    assert results == expected


def test_get_level_row_slices_output_when_X_train_is_empty():
    """
    Test that an empty dict is returned when `X_train` has no rows.
    """
    X_train = pd.DataFrame({'lag_1': [], '_level_skforecast': []})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=1, encoding='ordinal'
    )
    forecaster.encoding_mapping_ = {'l1': 0, 'l2': 1}
    results = forecaster._get_level_row_slices(X_train=X_train)

    assert results == {}
