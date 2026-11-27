# Unit test _train_matrix_to_numpy and _train_matrix_to_estimator_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.utils import _train_matrix_to_numpy
from skforecast.utils import _train_matrix_to_estimator_input


def make_block(n_rows=4):
    """
    Training matrix stored in a single float block, as created by the
    forecaster (numpy array with contiguous columns wrapped without copy).
    """
    block = np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3, order='F')
    X = pd.DataFrame(block, columns=['lag_1', 'lag_2', 'exog_1'], copy=False)

    return X, block


def test_train_matrix_to_numpy_output_when_all_columns_are_float64():
    """
    Test that a matrix with only float64 columns is returned as a view of its
    values (no copy) that is writeable, also with pandas 3, where `to_numpy`
    returns a read-only view.
    """
    X, block = make_block()
    results = _train_matrix_to_numpy(X)

    assert isinstance(results, np.ndarray)
    assert results.dtype == np.float64
    assert np.shares_memory(results, block)
    assert results.flags.writeable
    np.testing.assert_array_equal(results, block)


def test_train_matrix_to_numpy_output_when_rows_were_filtered():
    """
    Test that the matrix is also writeable when it comes from a DataFrame
    whose rows were filtered with a boolean mask (rows with NaN dropped).
    """
    X, block = make_block()
    X = X[np.array([True, False, True, True])]
    results = _train_matrix_to_numpy(X)

    assert results.flags.writeable
    np.testing.assert_array_equal(results, block[[0, 2, 3]])


def test_train_matrix_to_numpy_output_when_values_belong_to_read_only_array():
    """
    Test that the matrix is returned as a read-only view, without error, when
    its values belong to an array that is read-only, since it cannot be made
    writeable.
    """
    block = np.arange(12, dtype=float).reshape(4, 3).copy(order='F')
    block.flags.writeable = False
    X = pd.DataFrame(block, columns=['lag_1', 'lag_2', 'exog_1'], copy=False)
    results = _train_matrix_to_numpy(X)

    assert np.shares_memory(results, block)
    assert not results.flags.writeable
    np.testing.assert_array_equal(results, block)


def test_train_matrix_to_numpy_output_when_single_block_is_not_float64():
    """
    Test that the array is float64 and writeable when all the columns have the
    same dtype other than float64 (float32), which pandas 3 converts into a
    read-only array.
    """
    X = pd.DataFrame(
        np.arange(8, dtype='float32').reshape(4, 2), columns=['lag_1', 'exog_1']
    )
    results = _train_matrix_to_numpy(X)

    assert results.dtype == np.float64
    assert results.flags.writeable
    np.testing.assert_array_equal(
        results, np.arange(8, dtype=float).reshape(4, 2)
    )


def test_train_matrix_to_numpy_output_when_mixed_dtypes():
    """
    Test that a matrix with columns of several dtypes is converted to a
    float64 array: the values (not the codes) of a category with integer
    categories, NaN for the missing values of nullable columns and 0/1 for
    booleans.
    """
    X, _ = make_block()
    X.insert(
        2, '_level_skforecast', pd.Categorical([0, 1, 1, 0], categories=range(3))
    )
    X['exog_cat'] = pd.Categorical([10, 30, 20, 10], categories=[30, 10, 20])
    X['exog_int'] = np.array([1, 2, 3, 4], dtype='int64')
    X['exog_bool'] = [True, False, True, True]
    X['exog_Int64'] = pd.array([1, None, 3, 4], dtype='Int64')
    results = _train_matrix_to_numpy(X)

    expected = np.array(
        [[0., 4., 0., 8., 10., 1., 1., 1.],
         [1., 5., 1., 9., 30., 2., 0., np.nan],
         [2., 6., 1., 10., 20., 3., 1., 3.],
         [3., 7., 0., 11., 10., 4., 1., 4.]]
    )

    assert results.dtype == np.float64
    assert results.flags.writeable
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize(
    "column, expected_repr",
    [(pd.Categorical(['a', 'b', 'c', 'a']), "'exog_bad' (category)"),
     (pd.Categorical([0.5, 1.5, 2.5, 0.5]), "'exog_bad' (category)"),
     (pd.Categorical(['1', '2', '3', '1']), "'exog_bad' (category)"),
     (pd.Series(['a', 'b', 'c', 'a'], dtype=object), "'exog_bad' (object)"),
     (pd.date_range('2020-01-01', periods=4, freq='D').astype('datetime64[s]'),
      "'exog_bad' (datetime64[s])"),
     (pd.to_timedelta([1, 2, 3, 4], unit='D').astype('timedelta64[s]'),
      "'exog_bad' (timedelta64[s])")],
    ids=['category of text', 'category of decimals',
         'category of numeric strings', 'text', 'datetime', 'timedelta']
)
def test_train_matrix_to_numpy_TypeError_when_column_not_convertible(
    column, expected_repr
):
    """
    Test TypeError is raised, with the name and the dtype of the column, when
    a column cannot be passed to the estimator as numeric values: text, dates
    and categories whose categories are not integers.
    """
    X, _ = make_block()
    X['exog_bad'] = column

    err_msg = re.escape(
        f"The following exogenous variables cannot be passed to the estimator "
        f"as numeric values: {expected_repr}. Include them in "
        f"`categorical_features`, or use `categorical_features='auto'`, so "
        f"that the forecaster encodes them. A column of dtype `category` that "
        f"is not included in `categorical_features` must have integer "
        f"categories."
    )
    with pytest.raises(TypeError, match=err_msg):
        _train_matrix_to_numpy(X)


@pytest.mark.parametrize(
    "estimator, cat_indices",
    [(LinearRegression(), []),
     (LGBMRegressor(verbose=-1), [2]),
     (CatBoostRegressor(verbose=0, allow_writing_files=False), []),
     (make_pipeline(
         StandardScaler(), CatBoostRegressor(verbose=0, allow_writing_files=False)
      ), [2])],
    ids=['LinearRegression', 'LGBMRegressor with categorical',
         'CatBoostRegressor without categorical', 'Pipeline ending in CatBoost']
)
def test_train_matrix_to_estimator_input_output_is_numpy(estimator, cat_indices):
    """
    Test that the estimator receives a numpy array, the one of
    `_train_matrix_to_numpy`, unless it is a CatBoost model with categorical
    features (never inside a Pipeline).
    """
    X, block = make_block()
    results = _train_matrix_to_estimator_input(
        X_train=X, estimator=estimator, cat_indices=cat_indices
    )

    assert isinstance(results, np.ndarray)
    assert np.shares_memory(results, block)
    np.testing.assert_array_equal(results, block)


def test_train_matrix_to_estimator_input_output_when_CatBoost_with_categorical():
    """
    Test that a CatBoost model with categorical features receives a DataFrame
    with the names of the columns, where the categorical columns are integers
    with the values (not the codes) of the column and -1 for the missing
    values, and the rest are float. The input matrix is not modified.
    """
    X, _ = make_block()
    X['exog_cat'] = pd.Categorical([10, 30, np.nan, 10], categories=[30, 10])
    X.insert(
        2, '_level_skforecast', pd.Categorical([0, 1, 1, 0], categories=range(3))
    )
    X_copy = X.copy()
    results = _train_matrix_to_estimator_input(
        X_train     = X,
        estimator   = CatBoostRegressor(verbose=0, allow_writing_files=False),
        cat_indices = [2, 4]
    )

    expected = pd.DataFrame(
        {'lag_1': [0., 1., 2., 3.],
         'lag_2': [4., 5., 6., 7.],
         '_level_skforecast': [0, 1, 1, 0],
         'exog_1': [8., 9., 10., 11.],
         'exog_cat': [10, 30, -1, 10]}
    ).astype({'_level_skforecast': int, 'exog_cat': int})

    pd.testing.assert_frame_equal(results, expected)
    pd.testing.assert_frame_equal(X, X_copy)
