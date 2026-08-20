# Unit test estimator_has_native_nan_support
# ==============================================================================
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skforecast.utils import estimator_has_native_nan_support


def _fake_estimator(module: str, name: str):
    """Build a minimal object whose type reports the given module/name."""
    cls = type(name, (), {})
    cls.__module__ = module
    return cls()


@pytest.mark.parametrize(
    "estimator",
    [
        _fake_estimator('lightgbm.sklearn', 'LGBMRegressor'),
        _fake_estimator('xgboost.sklearn', 'XGBRegressor'),
        _fake_estimator('catboost.core', 'CatBoostRegressor'),
        HistGradientBoostingRegressor(),
        HistGradientBoostingClassifier(),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_estimators=2, random_state=123),
        RandomForestClassifier(n_estimators=2, random_state=123),
        ExtraTreesRegressor(n_estimators=2, random_state=123),
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", _fake_estimator('lightgbm.sklearn', 'LGBMRegressor')),
        ]),
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor()),
        ]),
    ],
    ids=[
        'lightgbm',
        'xgboost',
        'catboost',
        'HistGradientBoostingRegressor',
        'HistGradientBoostingClassifier',
        'DecisionTreeRegressor',
        'RandomForestRegressor',
        'RandomForestClassifier',
        'ExtraTreesRegressor',
        'Pipeline-lightgbm',
        'Pipeline-HistGradientBoostingRegressor',
    ],
)
def test_estimator_has_native_nan_support_true(estimator):
    """
    NaN-tolerant estimator families return True, including Pipeline last steps.
    """
    assert estimator_has_native_nan_support(estimator) is True


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        SVR(),
        GradientBoostingRegressor(n_estimators=2, random_state=123),
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        _fake_estimator('unknown_library.module', 'UnknownRegressor'),
    ],
    ids=[
        'LinearRegression',
        'SVR',
        'GradientBoostingRegressor',
        'Pipeline-LinearRegression',
        'unknown-library',
    ],
)
def test_estimator_has_native_nan_support_false(estimator):
    """
    Estimators without native NaN support return False.
    """
    assert estimator_has_native_nan_support(estimator) is False
