# Unit test estimator_has_native_nan_support
# ==============================================================================
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        RandomForestRegressor(n_estimators=2, random_state=123),
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        _fake_estimator('sklearn.ensemble._forest', 'RandomForestRegressor'),
    ],
    ids=[
        'LinearRegression',
        'RandomForestRegressor',
        'Pipeline-LinearRegression',
        'fake-RandomForestRegressor',
    ],
)
def test_estimator_has_native_nan_support_false(estimator):
    """
    Estimators without native NaN support return False.
    """
    assert estimator_has_native_nan_support(estimator) is False
