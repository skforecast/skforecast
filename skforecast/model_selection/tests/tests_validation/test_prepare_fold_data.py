# Unit test _prepare_fold_data
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._validation import _prepare_fold_data
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import exog


def _make_folds(y, steps, initial_train_size, refit, gap=0,
                fixed_train_size=True, window_size=3):
    """
    Helper to create folds using TimeSeriesFold.split.
    """
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=initial_train_size,
        window_size=window_size,
        differentiation=None,
        refit=refit,
        fixed_train_size=fixed_train_size,
        gap=gap,
        skip_folds=None,
        allow_incomplete_fold=True,
        return_all_indexes=False,
        verbose=False,
    )
    return cv.split(X=y, as_pandas=False)


def test_prepare_fold_data_refit_false_no_exog():
    """
    refit=False, no exog. Every fold must have y_train=None, exog_train=None,
    exog_test=None, and last_window_y matching the expected iloc slice.
    """
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 12, refit=False)
    for fold in folds:
        fold[5] = False

    result = _prepare_fold_data(folds, y, exog=None)

    assert len(result) == len(folds)
    for fd, fold in zip(result, folds):
        assert fd['y_train'] is None
        assert fd['exog_train'] is None
        assert fd['exog_test'] is None
        pd.testing.assert_series_equal(
            fd['last_window_y'], y.iloc[fold[2][0]:fold[2][1]]
        )


def test_prepare_fold_data_refit_false_with_exog():
    """
    refit=False with exog. last_window_y and exog_test must be correctly
    sliced; y_train and exog_train must be None.
    """
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 12, refit=False)
    for fold in folds:
        fold[5] = False

    result = _prepare_fold_data(folds, y, exog=exog)

    for fd, fold in zip(result, folds):
        assert fd['y_train'] is None
        assert fd['exog_train'] is None
        pd.testing.assert_series_equal(
            fd['last_window_y'], y.iloc[fold[2][0]:fold[2][1]]
        )
        pd.testing.assert_series_equal(
            fd['exog_test'], exog.iloc[fold[3][0]:fold[3][1]]
        )


def test_prepare_fold_data_refit_true_with_exog():
    """
    refit=True with exog. Refit folds must have y_train, exog_train and
    exog_test correctly sliced; last_window_y must be None.
    """
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 12, refit=True)

    result = _prepare_fold_data(folds, y, exog=exog)

    for fd, fold in zip(result, folds):
        if fold[5] is True:
            pd.testing.assert_series_equal(
                fd['y_train'], y.iloc[fold[1][0]:fold[1][1]]
            )
            pd.testing.assert_series_equal(
                fd['exog_train'], exog.iloc[fold[1][0]:fold[1][1]]
            )
            pd.testing.assert_series_equal(
                fd['exog_test'], exog.iloc[fold[3][0]:fold[3][1]]
            )
            assert fd['last_window_y'] is None


def test_prepare_fold_data_mixed_first_fold_false():
    """
    Simulates the real _backtesting_forecaster pattern: refit=True but
    folds[0][5] set to False after the initial fit. Fold 0 follows the
    no-refit path; remaining folds follow the refit path.
    """
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 12, refit=True)
    folds[0][5] = False

    result = _prepare_fold_data(folds, y, exog=exog)

    # Fold 0: no-refit path
    fd0 = result[0]
    assert fd0['y_train'] is None
    assert fd0['exog_train'] is None
    pd.testing.assert_series_equal(
        fd0['last_window_y'], y.iloc[folds[0][2][0]:folds[0][2][1]]
    )
    pd.testing.assert_series_equal(
        fd0['exog_test'], exog.iloc[folds[0][3][0]:folds[0][3][1]]
    )

    # Remaining folds: refit path
    for fd, fold in zip(result[1:], folds[1:]):
        assert fd['last_window_y'] is None
        assert fd['y_train'] is not None
        assert fd['exog_train'] is not None
        assert fd['exog_test'] is not None


def test_prepare_fold_data_gap_greater_than_zero():
    """
    With gap>0 the test slice starts further ahead. _prepare_fold_data
    must still slice correctly using the fold indices.
    """
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 12,
                        refit=False, gap=2)
    for fold in folds:
        fold[5] = False

    result = _prepare_fold_data(folds, y, exog=exog)

    for fd, fold in zip(result, folds):
        pd.testing.assert_series_equal(
            fd['last_window_y'], y.iloc[fold[2][0]:fold[2][1]]
        )
        pd.testing.assert_series_equal(
            fd['exog_test'], exog.iloc[fold[3][0]:fold[3][1]]
        )
        assert fd['y_train'] is None
        assert fd['exog_train'] is None


def test_prepare_fold_data_dataframe_exog():
    """
    When exog is a DataFrame, slices must preserve columns and content.
    Checks both no-refit (fold 0) and refit (remaining) paths.
    """
    exog_df = pd.DataFrame({'exog_1': exog.values, 'exog_2': exog.values * 2})
    folds = _make_folds(y, steps=4, initial_train_size=len(y) - 8, refit=True)
    folds[0][5] = False

    result = _prepare_fold_data(folds, y, exog=exog_df)

    # Fold 0: no-refit — exog_test is a DataFrame
    fd0 = result[0]
    pd.testing.assert_frame_equal(
        fd0['exog_test'], exog_df.iloc[folds[0][3][0]:folds[0][3][1]]
    )
    assert fd0['exog_train'] is None

    # Remaining refit folds — exog_train and exog_test are DataFrames
    for fd, fold in zip(result[1:], folds[1:]):
        if fold[5] is True:
            pd.testing.assert_frame_equal(
                fd['exog_train'], exog_df.iloc[fold[1][0]:fold[1][1]]
            )
            pd.testing.assert_frame_equal(
                fd['exog_test'], exog_df.iloc[fold[3][0]:fold[3][1]]
            )
