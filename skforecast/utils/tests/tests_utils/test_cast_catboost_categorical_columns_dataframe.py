# Unit test cast_catboost_categorical_columns_dataframe
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from skforecast.utils import cast_catboost_categorical_columns_dataframe


# ==============================================================================
# Tests: no-op paths (cat_features missing or non-CatBoost estimator)
# ==============================================================================
def test_returns_X_unchanged_when_cat_features_not_in_fit_kwargs():
    """
    Test that the function returns the same DataFrame (identity) when
    `fit_kwargs` does not contain the `cat_features` key.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, 1.0]})

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs={}, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
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
    `cat_features` happens to be in `fit_kwargs`.
    """
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, 1.0]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    assert result is X


# ==============================================================================
# Tests: CatBoost cast — float-with-NaN columns
# ==============================================================================
def test_catboost_regressor_casts_float_categorical_columns_to_int_no_nan():
    """
    Test that float categorical columns with no NaN are cast to int and
    the non-cat columns are preserved.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({'lag_1': [1.5, 2.5], 'cat_a': [0.0, 1.0]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    assert result['cat_a'].dtype == np.int64 or result['cat_a'].dtype == np.int32
    np.testing.assert_array_equal(result['cat_a'].to_numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(result['lag_1'].to_numpy(), np.array([1.5, 2.5]))


def test_catboost_regressor_fills_nan_with_minus_one_in_float_columns():
    """
    Test that NaN values in float categorical columns (from the
    OrdinalEncoder) are replaced with `-1` before casting to int.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({
        'lag_1': [1.0, 2.0, 3.0],
        'cat_a': [0.0, np.nan, 1.0],
    })
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    np.testing.assert_array_equal(result['cat_a'].to_numpy(), np.array([0, -1, 1]))


# ==============================================================================
# Tests: CatBoost cast — pandas Categorical dtype columns
# ==============================================================================
def test_catboost_categorical_dtype_uses_cat_codes():
    """
    Test that columns with pandas Categorical dtype are converted via
    `.cat.codes`. This is the path used for the synthetic
    `_level_skforecast` column when `encoding='ordinal_category'` in
    `ForecasterRecursiveMultiSeries`.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({
        'lag_1': [1.0, 2.0, 3.0],
        '_level_skforecast': pd.Categorical(['a', 'b', 'a']),
    })
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', '_level_skforecast'],
    )

    np.testing.assert_array_equal(
        result['_level_skforecast'].to_numpy(), np.array([0, 1, 0])
    )


def test_catboost_categorical_dtype_with_nan_uses_minus_one():
    """
    Test that NaN in a pandas Categorical column becomes `-1` via
    `.cat.codes` (the default encoding for missing categories).
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({
        'lag_1': [1.0, 2.0, 3.0],
        'cat_a': pd.Categorical(['a', None, 'b']),
    })
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    np.testing.assert_array_equal(result['cat_a'].to_numpy(), np.array([0, -1, 1]))


# ==============================================================================
# Tests: mixed dtypes + multiple categorical columns
# ==============================================================================
def test_mixed_categorical_and_float_columns():
    """
    Test that the function handles mixed types in a single call: one
    `pandas.Categorical` column and one float-with-NaN column.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({
        'lag_1': [1.5, 2.5, 3.5],
        '_level_skforecast': pd.Categorical(['a', 'b', 'a']),
        'cat_b': [0.0, np.nan, 1.0],
    })
    fit_kwargs = {'cat_features': [1, 2]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', '_level_skforecast', 'cat_b'],
    )

    np.testing.assert_array_equal(
        result['_level_skforecast'].to_numpy(), np.array([0, 1, 0])
    )
    np.testing.assert_array_equal(result['cat_b'].to_numpy(), np.array([0, -1, 1]))
    np.testing.assert_array_equal(result['lag_1'].to_numpy(), np.array([1.5, 2.5, 3.5]))


# ==============================================================================
# Tests: CatBoostClassifier
# ==============================================================================
def test_catboost_classifier_also_triggers_cast():
    """
    Test that `CatBoostClassifier` triggers the same cast as the
    regressor — the module-level check covers both.
    """
    estimator = CatBoostClassifier(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, np.nan]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    np.testing.assert_array_equal(result['cat_a'].to_numpy(), np.array([0, -1]))


# ==============================================================================
# Tests: input not mutated (copy contract)
# ==============================================================================
def test_input_dataframe_is_not_mutated():
    """
    Test that the input DataFrame is not modified by the function. The
    DataFrame variant copies internally so callers (in particular the
    hyperparameter search loop, which reuses `X_train`/`X_test` across
    trials) can keep using the original.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, np.nan]})
    X_copy = X.copy()
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    pd.testing.assert_frame_equal(X, X_copy)
    assert result is not X


# ==============================================================================
# Tests: Pipeline support
# ==============================================================================
def test_pipeline_wrapping_catboost_triggers_cast():
    """
    Test that the cast runs when the estimator is a Pipeline whose last
    step is CatBoost.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CatBoostRegressor(verbose=0, allow_writing_files=False)),
    ])
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, np.nan]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=pipe,
        feature_names=['lag_1', 'cat_a'],
    )

    np.testing.assert_array_equal(result['cat_a'].to_numpy(), np.array([0, -1]))


def test_pipeline_wrapping_non_catboost_returns_X_unchanged():
    """
    Test that a Pipeline whose last step is not CatBoost triggers the
    no-op path.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMRegressor(verbose=-1)),
    ])
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, np.nan]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=pipe,
        feature_names=['lag_1', 'cat_a'],
    )

    assert result is X
