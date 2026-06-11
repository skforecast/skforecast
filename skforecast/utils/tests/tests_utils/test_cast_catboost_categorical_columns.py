# Unit test cast_catboost_categorical_columns
# ==============================================================================
import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from skforecast.utils import cast_catboost_categorical_columns


# ==============================================================================
# Tests: no-op paths (cat_features missing or non-CatBoost estimator)
# ==============================================================================
def test_returns_X_unchanged_when_cat_features_not_in_fit_kwargs():
    """
    Test that the function returns the same array (identity) when
    `fit_kwargs` does not contain the `cat_features` key. CatBoost is the
    only family that uses this key; without it the cast must be skipped
    for every estimator.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs={}, estimator=estimator
    )

    assert result is X


@pytest.mark.parametrize(
    'estimator',
    [
        LinearRegression(),
        LGBMRegressor(verbose=-1),
        XGBRegressor(),
        HistGradientBoostingRegressor(),
    ],
    ids=lambda e: f'estimator: {type(e).__name__}'
)
def test_returns_X_unchanged_for_non_catboost_estimators(estimator):
    """
    Test that non-CatBoost estimators trigger the no-op path even when
    `cat_features` happens to be in `fit_kwargs` (e.g. user-supplied).
    The cast is CatBoost-specific.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    assert result is X


# ==============================================================================
# Tests: CatBoost cast — happy path
# ==============================================================================
def test_catboost_regressor_casts_categorical_columns_to_int_when_no_nan():
    """
    Test that the categorical columns are cast to int and the non-cat
    columns are preserved when no NaN is present.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = np.array([[1.5, 0.0, 7.5], [2.5, 1.0, 8.5]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    assert result.dtype == object
    np.testing.assert_array_equal(result[:, 1], np.array([0, 1]))
    assert all(isinstance(v, (int, np.integer)) for v in result[:, 1])
    np.testing.assert_array_equal(result[:, 0], np.array([1.5, 2.5]))
    np.testing.assert_array_equal(result[:, 2], np.array([7.5, 8.5]))


def test_catboost_regressor_fills_nan_with_minus_one_then_casts_to_int():
    """
    Test that NaN values in the categorical columns (produced by the
    OrdinalEncoder when an unseen level or a missing value is encountered)
    are replaced with `-1` before casting to int. This is the regression
    test for the bug where `np.nan.astype(int)` raised `ValueError`.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = np.array([[1.0, 0.0], [2.0, np.nan], [3.0, 1.0]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    np.testing.assert_array_equal(result[:, 1], np.array([0, -1, 1]))


def test_catboost_classifier_also_triggers_cast():
    """
    Test that `CatBoostClassifier` triggers the same cast. The module-level
    check covers both regressor and classifier without listing them
    individually.
    """
    estimator = CatBoostClassifier(verbose=0, allow_writing_files=False)
    X = np.array([[1.0, 0.0], [2.0, np.nan]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    np.testing.assert_array_equal(result[:, 1], np.array([0, -1]))


def test_multiple_categorical_columns():
    """
    Test that multiple categorical columns are cast independently and
    that NaN handling applies per-column.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = np.array([
        [1.5, 0.0, 9.5, 2.0],
        [2.5, np.nan, 8.5, 1.0],
        [3.5, 1.0, 7.5, np.nan],
    ])
    fit_kwargs = {'cat_features': [1, 3]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    np.testing.assert_array_equal(result[:, 1], np.array([0, -1, 1]))
    np.testing.assert_array_equal(result[:, 3], np.array([2, 1, -1]))
    np.testing.assert_array_equal(result[:, 0], np.array([1.5, 2.5, 3.5]))
    np.testing.assert_array_equal(result[:, 2], np.array([9.5, 8.5, 7.5]))


def test_input_array_is_not_mutated():
    """
    Test that the original X is not modified by the function. The cast
    must return a new array so callers (e.g. the hyperparameter search
    which reuses `X_train`/`X_test` across trials) can keep using the
    original.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = np.array([[1.0, 0.0], [2.0, np.nan]])
    X_copy = X.copy()
    fit_kwargs = {'cat_features': [1]}

    cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator
    )

    np.testing.assert_array_equal(X, X_copy, strict=False)
    assert np.isnan(X[1, 1])


# ==============================================================================
# Tests: Pipeline support
# ==============================================================================
def test_pipeline_wrapping_catboost_triggers_cast():
    """
    Test that the cast runs when the estimator is a Pipeline whose last
    step is CatBoost — matching the unwrapping done by
    `configure_estimator_categorical_features`.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CatBoostRegressor(verbose=0, allow_writing_files=False)),
    ])
    X = np.array([[1.0, 0.0], [2.0, np.nan]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=pipe
    )

    np.testing.assert_array_equal(result[:, 1], np.array([0, -1]))


def test_pipeline_wrapping_non_catboost_returns_X_unchanged():
    """
    Test that a Pipeline whose last step is not CatBoost is treated as a
    non-CatBoost estimator and the cast is skipped.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMRegressor(verbose=-1)),
    ])
    X = np.array([[1.0, 0.0], [2.0, np.nan]])
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns(
        X=X, fit_kwargs=fit_kwargs, estimator=pipe
    )

    assert result is X
