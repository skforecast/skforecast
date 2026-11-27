# Unit test cast_catboost_categorical_columns_dataframe
# ==============================================================================
import re
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

    assert result['cat_a'].dtype == np.int64
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
# Tests: cat_features given by name or in the constructor of the estimator
# ==============================================================================
@pytest.mark.parametrize(
    'cat_features',
    [['cat_a', 'cat_b'], ['cat_a', 3], ('cat_b', 'cat_a')],
    ids=lambda cf: f'cat_features: {cf}'
)
def test_catboost_cat_features_by_name_in_fit_kwargs(cat_features):
    """
    Test that the categorical columns are cast to int when `cat_features` of
    `fit_kwargs` contains names of columns, alone or mixed with positions.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    X = pd.DataFrame({
        'lag_1': [1.5, 2.5, 3.5],
        'cat_a': [10.0, np.nan, 30.0],
        'exog_1': [0.1, 0.2, 0.3],
        'cat_b': [2.0, 0.0, 1.0],
    })

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs={'cat_features': cat_features}, estimator=estimator,
        feature_names=['lag_1', 'cat_a', 'exog_1', 'cat_b'],
    )

    expected = pd.DataFrame({
        'lag_1': [1.5, 2.5, 3.5],
        'cat_a': [10, -1, 30],
        'exog_1': [0.1, 0.2, 0.3],
        'cat_b': [2, 0, 1],
    })

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'cat_features',
    [(1,), [1], ['cat_a']],
    ids=lambda cf: f'cat_features: {cf}'
)
def test_catboost_cat_features_in_constructor_when_not_in_fit_kwargs(cat_features):
    """
    Test that, when `fit_kwargs` has no `cat_features`, the columns that the
    estimator has in the `cat_features` of its constructor (positions or
    names) are cast to int.
    """
    estimator = CatBoostRegressor(
        verbose=0, allow_writing_files=False, cat_features=cat_features
    )
    X = pd.DataFrame({'lag_1': [1.5, 2.5, 3.5], 'cat_a': [10.0, np.nan, 30.0]})

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs={}, estimator=estimator,
        feature_names=['lag_1', 'cat_a'],
    )

    expected = pd.DataFrame({'lag_1': [1.5, 2.5, 3.5], 'cat_a': [10, -1, 30]})

    pd.testing.assert_frame_equal(result, expected)


def test_catboost_cat_features_of_fit_kwargs_have_priority_over_constructor():
    """
    Test that `cat_features` of `fit_kwargs` is used, and not the one of the
    constructor of the estimator, when both are given.
    """
    estimator = CatBoostRegressor(
        verbose=0, allow_writing_files=False, cat_features=[0]
    )
    X = pd.DataFrame({'cat_a': [1.0, 2.0], 'cat_b': [0.0, np.nan]})

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs={'cat_features': [1]}, estimator=estimator,
        feature_names=['cat_a', 'cat_b'],
    )

    expected = pd.DataFrame({'cat_a': [1.0, 2.0], 'cat_b': [0, -1]})

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'cat_features_constructor, fit_kwargs',
    [(None, {'cat_features': [1]}),
     (None, {'cat_features': ['cat_a']}),
     ((1,), {}),
     (('cat_a',), {})],
    ids=['fit_kwargs by position', 'fit_kwargs by name',
         'constructor by position', 'constructor by name']
)
def test_catboost_TypeError_when_cat_features_column_has_non_integer_values(
    cat_features_constructor, fit_kwargs
):
    """
    Test TypeError is raised, with the name of the column, when a float column
    of `cat_features` (of `fit_kwargs` or of the constructor) has values that
    are not integers, instead of truncating them. NaN is allowed.
    """
    estimator = CatBoostRegressor(
        verbose=0, allow_writing_files=False,
        cat_features=cat_features_constructor
    )
    X = pd.DataFrame({'lag_1': [1.5, 2.5, 3.5], 'cat_a': [0.2, np.nan, 3.0]})

    err_msg = re.escape(
        "The column 'cat_a' is included in `cat_features` of the CatBoost "
        "estimator, but its values are not integers. CatBoost only accepts "
        "integers or strings as categorical features. Remove it from "
        "`cat_features`, or include it in `categorical_features` so that the "
        "forecaster encodes it."
    )
    with pytest.raises(TypeError, match=err_msg):
        cast_catboost_categorical_columns_dataframe(
            X=X, fit_kwargs=fit_kwargs, estimator=estimator,
            feature_names=['lag_1', 'cat_a'],
        )


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
# Tests: Pipeline (never cast)
# ==============================================================================
@pytest.mark.parametrize(
    'model',
    [
        CatBoostRegressor(verbose=0, allow_writing_files=False),
        LGBMRegressor(verbose=-1),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor']
)
def test_pipeline_returns_X_unchanged(model):
    """
    Test that the cast is skipped when the estimator is a Pipeline, also when
    its last step is CatBoost: native categorical features are not configured
    inside a Pipeline, as in `configure_estimator_categorical_features`.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    X = pd.DataFrame({'lag_1': [1.0, 2.0], 'cat_a': [0.0, np.nan]})
    fit_kwargs = {'cat_features': [1]}

    result = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=fit_kwargs, estimator=pipe,
        feature_names=['lag_1', 'cat_a'],
    )

    assert result is X
