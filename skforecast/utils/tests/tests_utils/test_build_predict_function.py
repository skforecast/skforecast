# Unit test _build_predict_function
# ==============================================================================
import numpy as np
import pytest
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.utils.utils import _build_predict_function


# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def regression_data():
    """
    Create a regression dataset used across all tests in this module.
    """
    X, y = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=123
    )
    return X.astype(np.float64), y


# Tests
# ==============================================================================
@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(LinearRegression(), id="LinearRegression"),
        pytest.param(Ridge(alpha=1.0), id="Ridge"),
        pytest.param(Lasso(alpha=0.1), id="Lasso"),
        pytest.param(LGBMRegressor(n_estimators=10, verbose=-1), id="LGBMRegressor"),
        pytest.param(
            XGBRegressor(n_estimators=10, verbosity=0), id="XGBRegressor"
        ),
        pytest.param(
            RandomForestRegressor(n_estimators=10, random_state=123),
            id="RandomForestRegressor",
        ),
        pytest.param(DecisionTreeRegressor(random_state=123), id="DecisionTreeRegressor"),
        pytest.param(
            GradientBoostingRegressor(n_estimators=10, random_state=123),
            id="GradientBoostingRegressor_fallback",
        ),
        pytest.param(
            HistGradientBoostingRegressor(max_iter=10, random_state=123),
            id="HistGradientBoostingRegressor_fallback",
        ),
        pytest.param(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            id="Pipeline_fallback",
        ),
    ],
)
def test_build_predict_function(estimator, regression_data):
    """
    Test that `_build_predict_function` returns a callable that produces the
    same predictions as `estimator.predict()` for both multiple samples and a
    single sample, and that the output is always a 1D numpy array.
    """
    X, y = regression_data
    estimator.fit(X, y)

    predict_fn = _build_predict_function(estimator)
    assert callable(predict_fn)

    # Multiple samples (e.g. ForecasterRecursiveMultiSeries)
    result = predict_fn(X)
    expected = estimator.predict(X).ravel()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape[0] == X.shape[0]
    np.testing.assert_allclose(result, expected)

    # Single sample (e.g. ForecasterRecursive hot loop)
    result_single = predict_fn(X[:1])
    expected_single = estimator.predict(X[:1]).ravel()
    np.testing.assert_allclose(result_single, expected_single)


def test_build_predict_function_catboost_with_cat_features():
    """
    Test that _build_predict_function returns a callable that converts categorical
    columns to int (via object dtype) before passing to CatBoostRegressor.predict.
    cat_indices is resolved once at build time, not per prediction step.
    """
    # Build X with one float column and one categorical column (ordinal-encoded
    # as float, as skforecast produces after OrdinalEncoder).
    rng = np.random.default_rng(42)
    X_num = rng.standard_normal((100, 2))
    X_cat = rng.integers(0, 3, size=(100, 1)).astype(float)  # float-encoded ints
    X = np.concatenate([X_num, X_cat], axis=1)
    y = rng.standard_normal(100)

    estimator = CatBoostRegressor(n_estimators=10, random_state=42, verbose=0, allow_writing_files=False)
    X_train = X.astype(object)
    X_train[:, 2] = X_train[:, 2].astype(int)
    estimator.fit(X_train, y, cat_features=[2])

    predict_fn = _build_predict_function(estimator)
    assert callable(predict_fn)

    # Multiple samples: float X must work without error and match direct predict
    X_obj_ref = X.astype(object)
    X_obj_ref[:, 2] = X_obj_ref[:, 2].astype(int)
    expected = estimator.predict(X_obj_ref).ravel()
    result = predict_fn(X)
    np.testing.assert_allclose(result, expected)

    # Single sample (recursive loop scenario)
    result_single = predict_fn(X[:1])
    expected_single = estimator.predict(X_obj_ref[:1]).ravel()
    np.testing.assert_allclose(result_single, expected_single)


def test_build_predict_function_catboost_without_cat_features():
    """
    Test that _build_predict_function falls back to the generic predict path
    for CatBoostRegressor fitted without any categorical features, i.e. it
    returns the same predictions as estimator.predict.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = rng.standard_normal(100)

    estimator = CatBoostRegressor(n_estimators=10, random_state=42, verbose=0, allow_writing_files=False)
    estimator.fit(X, y)

    predict_fn = _build_predict_function(estimator)
    assert callable(predict_fn)

    expected = estimator.predict(X).ravel()
    result = predict_fn(X)
    np.testing.assert_allclose(result, expected)
