# Unit test _is_fitted_catboost
# ==============================================================================
import copy
import pickle
import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.utils import _is_fitted_catboost

# Fixtures
X = np.array([
    [0.5, 0., 1.2, 2.],
    [1.5, 1., 0.3, 0.],
    [2.5, 2., 2.2, 1.],
    [3.5, 0., 1.1, 2.],
    [4.5, 1., 0.7, 0.],
    [5.5, 2., 1.9, 1.],
])
y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y_clf = np.array([0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize(
    'estimator, y',
    [
        (CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False),
         y_reg),
        (CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False),
         y_clf),
        (make_pipeline(
            StandardScaler(),
            CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False)
         ), y_reg),
    ],
    ids=['CatBoostRegressor', 'CatBoostClassifier', 'Pipeline']
)
def test_is_fitted_catboost_output_when_CatBoost(estimator, y):
    """
    Test that `True` is returned only when a CatBoost model, or the last step
    of a Pipeline, is already fitted: also for a copy made with pickle or
    deepcopy, but not for a new estimator or an unfitted clone.
    """
    is_fitted_new = _is_fitted_catboost(estimator)
    estimator.fit(X, y)

    assert is_fitted_new is False
    assert _is_fitted_catboost(estimator) is True
    assert _is_fitted_catboost(pickle.loads(pickle.dumps(estimator))) is True
    assert _is_fitted_catboost(copy.deepcopy(estimator)) is True
    assert _is_fitted_catboost(clone(estimator)) is False


@pytest.mark.parametrize(
    'estimator',
    [
        LinearRegression(),
        LGBMRegressor(n_estimators=2, verbose=-1),
        make_pipeline(StandardScaler(), LinearRegression()),
    ],
    ids=['LinearRegression', 'LGBMRegressor', 'Pipeline']
)
def test_is_fitted_catboost_output_when_not_CatBoost(estimator):
    """
    Test that `False` is returned when the estimator, or the last step of a
    Pipeline, is not a CatBoost model, fitted or not.
    """
    is_fitted_new = _is_fitted_catboost(estimator)
    estimator.fit(X, y_reg)

    assert is_fitted_new is False
    assert _is_fitted_catboost(estimator) is False
