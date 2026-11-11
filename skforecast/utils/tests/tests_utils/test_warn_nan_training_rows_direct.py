# Unit test _warn_nan_training_rows_direct
# ==============================================================================
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import _warn_nan_training_rows_direct
from skforecast.direct import ForecasterDirect

PREDICTORS = "with NaN predictors (lags, window features or exog)."
LAGS = 3
MAX_STEP = 3
N_VALUES = 16
N_ROWS = N_VALUES - LAGS - MAX_STEP + 1  # 11 rows per step


def make_matrices(nan_y: list, nan_exog: list | None) -> tuple:
    """
    Create the matrices of a direct forecaster with `lags = 3` and
    `steps = 3` from a series of 16 values with NaN in the positions `nan_y`:
    the target (one column per step, 11 rows), the lags (11 rows) and, if
    `nan_exog` is not `None`, an exogenous variable of 13 rows (11 plus the
    rows of the later steps) with NaN in the rows `nan_exog`.
    """
    y = np.arange(N_VALUES, dtype=float)
    y[nan_y] = np.nan
    X_train_autoreg = np.column_stack(
        [y[LAGS - lag : LAGS - lag + N_ROWS] for lag in range(1, LAGS + 1)]
    )
    y_train = np.column_stack(
        [y[LAGS + step - 1 : LAGS + step - 1 + N_ROWS]
         for step in range(1, MAX_STEP + 1)]
    )
    X_train_exog = None
    if nan_exog is not None:
        X_train_exog = np.arange(N_ROWS + MAX_STEP - 1, dtype=float).reshape(-1, 1)
        X_train_exog[nan_exog, 0] = np.nan

    return y_train, X_train_autoreg, X_train_exog


def call(**kwargs) -> list:
    """
    Call `_warn_nan_training_rows_direct` and return the warnings raised.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_nan_training_rows_direct(**kwargs)

    return record


@pytest.mark.parametrize(
    "nan_y, nan_exog, dropna_from_series, counts",
    [
        ([8], None, True,
         f"4 of 11 rows are dropped per step: 1 with a NaN target and 3 {PREDICTORS}"),
        ([8, 9], None, True,
         f"5 to 6 of 11 rows are dropped per step: 2 with a NaN target and 3 to 4 "
         f"{PREDICTORS}"),
        ([8, 9], None, False,
         f"2 of 11 rows are dropped per step (NaN target) and 3 to 4 are kept "
         f"{PREDICTORS}"),
        ([], [5], True, f"1 of 11 rows is dropped per step, {PREDICTORS}"),
        ([], [0], True, f"0 to 1 of 11 rows are dropped per step, {PREDICTORS}"),
        ([], [12], False, f"0 to 1 of 11 rows are kept per step {PREDICTORS}"),
        ([8], [5], True,
         f"4 of 11 rows are dropped per step: 1 with a NaN target and 3 {PREDICTORS}"),
    ],
    ids=[
        "isolated_y", "block_y-True", "block_y-False", "exog_middle",
        "exog_first_row", "exog_last_row", "exog_same_row_as_target",
    ]
)
def test_warn_nan_training_rows_direct_counts_per_step(
    nan_y, nan_exog, dropna_from_series, counts
):
    """
    Test the counts of the warning: equal in every step with an isolated NaN,
    a range with a block of NaN, with and without `exog`. The exogenous
    variable of step `s` is shifted `s - 1` rows: a NaN in its first row is
    only in step 1, in its last row only in step 3, and a NaN in the row of a
    NaN target (row 5 of `exog` and position 8 of the series) counts only as
    a NaN target row.
    """
    y_train, X_train_autoreg, X_train_exog = make_matrices(
        nan_y=nan_y, nan_exog=nan_exog
    )

    record = call(
        y_train            = y_train,
        X_train_autoreg    = X_train_autoreg,
        X_train_exog       = X_train_exog,
        steps              = [1, 2, 3],
        dropna_from_series = dropna_from_series,
        series_name        = "`y`"
    )

    assert len(record) == 1
    assert issubclass(record[0].category, MissingValuesWarning)
    assert counts in str(record[0].message)
    assert "interspersed NaNs in `y` or `exog`" in str(record[0].message)


@pytest.mark.parametrize("nan_exog", [None, []], ids=["no_exog", "exog"])
@pytest.mark.parametrize("dropna_from_series", [True, False])
def test_warn_nan_training_rows_direct_no_warning_when_no_NaN(
    nan_exog, dropna_from_series
):
    """
    Test that no warning is raised without NaN, with and without `exog`.
    """
    y_train, X_train_autoreg, X_train_exog = make_matrices(
        nan_y=[], nan_exog=nan_exog
    )

    record = call(
        y_train            = y_train,
        X_train_autoreg    = X_train_autoreg,
        X_train_exog       = X_train_exog,
        steps              = [1, 2, 3],
        dropna_from_series = dropna_from_series,
        series_name        = "`y`"
    )

    assert len(record) == 0


def test_warn_nan_training_rows_direct_warning_points_to_the_caller():
    """
    Test that the warning points to the file that calls the function (with
    `stacklevel=3`), here this test file and, through `fit`, the file of the
    forecaster.
    """
    y_train, X_train_autoreg, X_train_exog = make_matrices(
        nan_y=[8], nan_exog=None
    )
    record = call(
        y_train            = y_train,
        X_train_autoreg    = X_train_autoreg,
        X_train_exog       = X_train_exog,
        steps              = [1, 2, 3],
        dropna_from_series = True,
        series_name        = "`y`"
    )
    assert len(record) == 1
    assert record[0].filename == __file__

    y = pd.Series(
        np.arange(30, dtype=float),
        index=pd.date_range("2020-01-01", periods=30, freq="D")
    )
    y.iloc[10] = np.nan
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3, dropna_from_series=True
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        forecaster.fit(y=y)
    record = [w for w in record if issubclass(w.category, MissingValuesWarning)]
    assert len(record) == 1
    assert record[0].filename.endswith("_forecaster_direct.py")
