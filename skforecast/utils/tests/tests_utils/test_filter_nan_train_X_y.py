# Unit test _filter_nan_train_X_y
# ==============================================================================
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import _filter_nan_train_X_y
from skforecast.recursive import ForecasterRecursive

PREDICTORS = "with NaN predictors (lags, window features or exog)."


def make_data(nan_y: list, nan_X: list, kind: str) -> tuple:
    """
    Create a training matrix of 6 rows and 2 columns, its target and its
    index, with NaN in the rows `nan_y` of the target and in the rows `nan_X`
    of the first column of the matrix.
    """
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float) + 100
    X[nan_X, 0] = np.nan
    y[nan_y] = np.nan
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    if kind == "pandas":
        X = pd.DataFrame(X, index=index, columns=["lag_1", "lag_2"])
        y = pd.Series(y, index=index, name="y")

    return X, y, index


def call(**kwargs) -> tuple:
    """
    Call `_filter_nan_train_X_y` and return its output and the warnings
    raised.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        output = _filter_nan_train_X_y(**kwargs)

    return output, record


@pytest.mark.parametrize("kind", ["numpy", "pandas"])
@pytest.mark.parametrize(
    "nan_y, nan_X, dropna_from_series, rows_kept, counts",
    [
        ([1], [], True, [0, 2, 3, 4, 5],
         "1 of 6 rows is dropped, with a NaN target."),
        ([1], [], False, [0, 2, 3, 4, 5],
         "1 of 6 rows is dropped, with a NaN target."),
        ([], [3], True, [0, 1, 2, 4, 5],
         f"1 of 6 rows is dropped, {PREDICTORS}"),
        ([], [3], False, [0, 1, 2, 3, 4, 5],
         f"1 of 6 rows is kept {PREDICTORS}"),
        ([1, 5], [3, 5], True, [0, 2, 4],
         f"3 of 6 rows are dropped: 2 with a NaN target and 1 {PREDICTORS}"),
        ([1, 5], [3, 5], False, [0, 2, 3, 4],
         f"2 of 6 rows are dropped (NaN target) and 1 is kept {PREDICTORS}"),
        ([5], [5], True, [0, 1, 2, 3, 4],
         "1 of 6 rows is dropped, with a NaN target."),
    ],
    ids=[
        "y-True", "y-False", "X-True", "X-False", "both-True", "both-False",
        "same_row-True",
    ]
)
def test_filter_nan_train_X_y_rows_dropped_and_warning(
    nan_y, nan_X, dropna_from_series, rows_kept, counts, kind
):
    """
    Test the rows kept (in `X_train`, `y_train` and `train_index`) and the
    counts of the warning with NaN in the target, in the predictors and in
    both, also in the same row (counted only as a NaN target row), with
    numpy arrays and pandas objects.
    """
    X, y, index = make_data(nan_y=nan_y, nan_X=nan_X, kind=kind)
    X_expected = np.asarray(X)[rows_kept]
    y_expected = np.asarray(y)[rows_kept]
    index_expected = index[rows_kept]

    (X_train, y_train, train_index), record = call(
        X_train            = X,
        y_train            = y,
        dropna_from_series = dropna_from_series,
        series_name        = "`y`",
        train_index        = index
    )

    assert type(X_train) is type(X)
    assert type(y_train) is type(y)
    np.testing.assert_array_equal(np.asarray(X_train), X_expected)
    np.testing.assert_array_equal(np.asarray(y_train), y_expected)
    pd.testing.assert_index_equal(train_index, index_expected)
    if kind == "pandas":
        pd.testing.assert_index_equal(X_train.index, index_expected)
        pd.testing.assert_index_equal(y_train.index, index_expected)
        assert X_train.columns.to_list() == ["lag_1", "lag_2"]
        assert y_train.name == "y"

    assert len(record) == 1
    assert issubclass(record[0].category, MissingValuesWarning)
    assert counts in str(record[0].message)
    assert "interspersed NaNs in `y` or `exog`" in str(record[0].message)


@pytest.mark.parametrize("kind", ["numpy", "pandas"])
def test_filter_nan_train_X_y_without_train_index(kind):
    """
    Test that `None` is returned as `train_index` when it is not passed (the
    way `ForecasterRecursiveMultiSeries` calls it), and that the rows are
    still dropped.
    """
    X, y, _ = make_data(nan_y=[0], nan_X=[2], kind=kind)

    (X_train, y_train, train_index), record = call(
        X_train            = X,
        y_train            = y,
        dropna_from_series = True,
        series_name        = "`series`"
    )

    assert train_index is None
    np.testing.assert_array_equal(np.asarray(X_train), np.asarray(X)[[1, 3, 4, 5]])
    np.testing.assert_array_equal(np.asarray(y_train), np.asarray(y)[[1, 3, 4, 5]])
    assert len(record) == 1
    assert "interspersed NaNs in `series` or `exog`" in str(record[0].message)


@pytest.mark.parametrize("kind", ["numpy", "pandas"])
@pytest.mark.parametrize("dropna_from_series", [True, False])
def test_filter_nan_train_X_y_no_NaN_returns_same_objects_without_warning(
    kind, dropna_from_series
):
    """
    Test that, without NaN, the same objects are returned (not copies) and no
    warning is raised.
    """
    X, y, index = make_data(nan_y=[], nan_X=[], kind=kind)

    (X_train, y_train, train_index), record = call(
        X_train            = X,
        y_train            = y,
        dropna_from_series = dropna_from_series,
        series_name        = "`y`",
        train_index        = index
    )

    assert X_train is X
    assert y_train is y
    assert train_index is index
    assert len(record) == 0


def test_filter_nan_train_X_y_warning_points_to_the_caller():
    """
    Test that the warning points to the file that calls the function (with
    `stacklevel=3`), here this test file and, through `fit`, the file of the
    forecaster.
    """
    X, y, index = make_data(nan_y=[1], nan_X=[3], kind="numpy")
    _, record = call(
        X_train            = X,
        y_train            = y,
        dropna_from_series = True,
        series_name        = "`y`",
        train_index        = index
    )
    assert len(record) == 1
    assert record[0].filename == __file__

    y = pd.Series(
        np.arange(30, dtype=float),
        index=pd.date_range("2020-01-01", periods=30, freq="D")
    )
    y.iloc[10] = np.nan
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, dropna_from_series=True
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        forecaster.fit(y=y)
    record = [w for w in record if issubclass(w.category, MissingValuesWarning)]
    assert len(record) == 1
    assert record[0].filename.endswith("_forecaster_recursive.py")

