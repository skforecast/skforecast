# Unit test _get_catboost_cat_feature_indices
# ==============================================================================
import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from skforecast.utils import _get_catboost_cat_feature_indices

# Fixtures
X = np.array([
    [0.5, 0, 1.2, 2],
    [1.5, 1, 0.3, 0],
    [2.5, 2, 2.2, 1],
    [3.5, 0, 1.1, 2],
    [4.5, 1, 0.7, 0],
    [5.5, 2, 1.9, 1],
], dtype=object)
y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y_clf = np.array([0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize(
    'estimator, y, cat_features, expected',
    [
        (CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False),
         y_reg, [1, 3], np.array([1, 3])),
        (CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False),
         y_clf, [1, 3], np.array([1, 3])),
        (CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False),
         y_reg, None, np.array([], dtype=int)),
    ],
    ids=['CatBoostRegressor', 'CatBoostClassifier', 'no_cat_features']
)
def test_get_catboost_cat_feature_indices_output_when_CatBoost(
    estimator, y, cat_features, expected
):
    """
    Test that the indices of the categorical features are returned for a
    fitted CatBoost regressor or classifier, and an empty array when it was
    fitted without categorical features.
    """
    X_fit = X if cat_features is not None else X.astype(float)
    estimator.fit(X_fit, y, cat_features=cat_features)
    results = _get_catboost_cat_feature_indices(estimator)

    np.testing.assert_array_equal(results, expected)
    assert results.dtype == int


@pytest.mark.parametrize(
    'estimator',
    [LinearRegression(), LGBMRegressor(n_estimators=2, verbose=-1)],
    ids=lambda est: type(est).__name__
)
def test_get_catboost_cat_feature_indices_output_when_not_CatBoost(estimator):
    """
    Test that an empty array is returned when the estimator is not a CatBoost
    model.
    """
    estimator.fit(X.astype(float), y_reg)
    results = _get_catboost_cat_feature_indices(estimator)

    np.testing.assert_array_equal(results, np.array([], dtype=int))
    assert results.dtype == int
