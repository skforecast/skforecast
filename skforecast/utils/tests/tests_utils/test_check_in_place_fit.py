# Unit test _copy_rows_to_check and _check_in_place_fit
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.utils import _copy_rows_to_check, _check_in_place_fit


class LinearRegressionAddingColumn(LinearRegression):
    """
    LinearRegression that adds a column to the training matrix in place.
    """

    def fit(self, X, y, sample_weight=None):
        X['extra'] = 1.0
        return super().fit(X, y, sample_weight=sample_weight)


def make_X_y(n_rows):
    """
    Training matrix stored in a single float block (its conversion to numpy
    is a view) and its target.
    """
    values = np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3) ** 1.5
    X = pd.DataFrame(values, columns=['lag_1', 'lag_2', 'exog'])
    y = pd.Series(np.arange(n_rows, dtype=float) * 2.0 + 1.0, name='y')

    return X, y


@pytest.mark.parametrize(
    "n_rows, expected_n_rows, expected_positions",
    [(10, 10, [0, 1, 2, 9]),
     (150, 100, [0, 1, 3, 149])],
    ids=lambda value: f'{value}'
)
def test_copy_rows_to_check_output(n_rows, expected_n_rows, expected_positions):
    """
    Test that `_copy_rows_to_check` returns a copy of up to 100 rows of `X`,
    evenly spaced and including the first and the last ones (first three and
    last positions checked).
    """
    X, _ = make_X_y(n_rows)
    results = _copy_rows_to_check(X)

    assert len(results) == expected_n_rows
    assert results.index[[0, 1, 2, -1]].to_list() == expected_positions
    pd.testing.assert_frame_equal(results, X.loc[results.index])
    assert not np.shares_memory(results.to_numpy(), X.to_numpy())


@pytest.mark.parametrize(
    "n_rows",
    [10, 150],
    ids=lambda n_rows: f'n_rows: {n_rows}'
)
@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(copy_X=False),
     make_pipeline(StandardScaler(copy=False), LinearRegression()),
     LinearRegressionAddingColumn()],
    ids=['LinearRegression(copy_X=False)', 'pipeline StandardScaler(copy=False)',
         'estimator adds a column']
)
def test_check_in_place_fit_ValueError_when_estimator_modifies_X(estimator, n_rows):
    """
    Test ValueError is raised when the estimator modifies the training matrix
    in place, changing its values or its shape, with fewer and with more rows
    than the sample that is compared (100).
    """
    X, y = make_X_y(n_rows)
    X_rows = _copy_rows_to_check(X)
    estimator.fit(X, y)

    err_msg = re.escape(
        "The estimator has modified the training matrix in place during "
        "`fit`. The matrix is used again after training, to calculate the "
        "in-sample residuals or to fit the next candidates of a search "
        "with `OneStepAheadFold`, so the results would be wrong. This "
        "happens with estimators that do not copy their input, such as "
        "`LinearRegression(copy_X=False)` or a pipeline with "
        "`StandardScaler(copy=False)`. Use the default copy behavior of "
        "the estimator (`copy_X=True`, `copy=True`)."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_in_place_fit(X=X, X_rows=X_rows)


def test_check_in_place_fit_no_error_when_imputer_fills_NaN_in_place():
    """
    Test that no error is raised when a step fills in place the cells that
    were NaN before training (`SimpleImputer(copy=False)`), because those
    cells are not compared. The rest of the matrix is not modified.
    """
    X, y = make_X_y(10)
    X.iloc[[2, 5], 0] = np.nan
    X.iloc[7, 2] = np.nan
    X_copy = X.copy()
    estimator = make_pipeline(SimpleImputer(copy=False), LinearRegression())

    X_rows = _copy_rows_to_check(X)
    estimator.fit(X, y)
    _check_in_place_fit(X=X, X_rows=X_rows)

    assert not X.isna().to_numpy().any()
    pd.testing.assert_frame_equal(X.mask(X_copy.isna()), X_copy)


@pytest.mark.parametrize(
    "estimator, X",
    [
        (LinearRegression(),
         pd.DataFrame({'lag_1': [1., 3., 2., 5., 4., 6.],
                       'exog_int': [1, 0, 1, 1, 0, 0],
                       'exog_bool': [True, False, True, True, False, True]})),
        (HistGradientBoostingRegressor(max_iter=2, min_samples_leaf=1),
         pd.DataFrame({'lag_1': [1., 3., 2., 5., 4., 6.],
                       'exog_cat': pd.Categorical([0, 1, 2, 0, 1, 2]),
                       'exog_nan': [1., np.nan, 3., np.nan, 5., 6.]})),
    ],
    ids=['LinearRegression, int and bool columns',
         'HistGradientBoostingRegressor, category and NaN columns']
)
def test_check_in_place_fit_no_error_when_estimator_does_not_modify_X(estimator, X):
    """
    Test that no error is raised and the training matrix is not modified when
    the estimator does not modify its input, with non-float columns and NaN
    values in the matrix.
    """
    y = pd.Series([1., 2., 3., 4., 5., 6.], name='y')
    X_copy = X.copy()

    X_rows = _copy_rows_to_check(X)
    estimator.fit(X, y)
    _check_in_place_fit(X=X, X_rows=X_rows)

    pd.testing.assert_frame_equal(X, X_copy)
    assert hasattr(estimator, 'n_features_in_')
