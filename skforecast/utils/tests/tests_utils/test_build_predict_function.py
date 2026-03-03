# Unit test _build_predict_function
# ==============================================================================
import numpy as np
import pytest
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
