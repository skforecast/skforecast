# Unit test _get_estimator_categorical_set_params
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.utils.utils import _get_estimator_categorical_set_params
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect


# ==============================================================================
# Tests: unsupported estimators return empty dict
# ==============================================================================
@pytest.mark.parametrize(
    'estimator',
    [LinearRegression(), LGBMRegressor(verbose=-1)],
    ids=['linear_regression', 'lgbm']
)
def test_returns_empty_dict_for_unsupported_estimators(estimator):
    """
    Test that estimators that do not use set_params for categoricals
    (LinearRegression, LightGBM — which uses fit_kwargs instead) return an
    empty dict.
    """
    forecaster = ForecasterRecursive(estimator=estimator, lags=2)
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {}


# ==============================================================================
# Tests: XGBoost
# ==============================================================================
def test_xgboost_returns_default_values_before_configure():
    """
    Test that a fresh XGBRegressor (feature_types=None, enable_categorical=False)
    returns those defaults in the dict.
    """
    forecaster = ForecasterRecursive(estimator=XGBRegressor(), lags=2)
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'feature_types': None, 'enable_categorical': False}


def test_xgboost_returns_feature_types_after_set_params():
    """
    Test that after XGBRegressor.set_params has been called (as done by
    configure_estimator_categorical_features), the dict reflects the current
    feature_types and enable_categorical values.
    """
    estimator = XGBRegressor()
    estimator.set_params(feature_types=['q', 'c', 'q'], enable_categorical=True)
    forecaster = ForecasterRecursive(estimator=estimator, lags=2)
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'feature_types': ['q', 'c', 'q'], 'enable_categorical': True}


# ==============================================================================
# Tests: HistGradientBoosting
# ==============================================================================
def test_histgbr_returns_default_categorical_features_before_configure():
    """
    Test that a fresh HistGradientBoostingRegressor returns the sklearn
    default value for categorical_features in the dict.
    """
    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(), lags=2
    )
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'categorical_features': 'from_dtype'}


def test_histgbr_returns_categorical_features_after_set_params():
    """
    Test that after HistGradientBoostingRegressor.set_params has been called,
    the dict reflects the current categorical_features value.
    """
    estimator = HistGradientBoostingRegressor()
    estimator.set_params(categorical_features=[0, 2])
    forecaster = ForecasterRecursive(estimator=estimator, lags=2)
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'categorical_features': [0, 2]}


# ==============================================================================
# Tests: Pipeline support
# ==============================================================================
def test_pipeline_with_xgboost_unwraps_last_step():
    """
    Test that when the estimator is a Pipeline, the last step is unwrapped
    and its XGBoost params are returned.
    """
    xgb = XGBRegressor()
    xgb.set_params(feature_types=['c', 'q'], enable_categorical=True)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', xgb)])
    forecaster = ForecasterRecursive(estimator=pipe, lags=2)
    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'feature_types': ['c', 'q'], 'enable_categorical': True}


# ==============================================================================
# Tests: ForecasterDirect reads from estimators_[1]
# ==============================================================================
def test_forecaster_direct_reads_from_estimators_1_not_template():
    """
    Test that ForecasterDirect reads params from estimators_[1] (the fitted
    per-step clone) and not from the template estimator attribute.
    """
    rng = np.random.default_rng(42)
    y = pd.Series(
        rng.normal(size=50),
        index=pd.date_range('2020', periods=50, freq='D')
    )
    forecaster = ForecasterDirect(estimator=XGBRegressor(), lags=2, steps=3)
    forecaster.fit(y=y)

    # Set a distinct value on the step-1 clone and a different value on the
    # template so we can tell which one the helper reads.
    forecaster.estimator.set_params(feature_types=['q', 'q'], enable_categorical=False)
    forecaster.estimators_[1].set_params(
        feature_types=['c', 'q'], enable_categorical=True
    )

    result = _get_estimator_categorical_set_params(forecaster)
    assert result == {'feature_types': ['c', 'q'], 'enable_categorical': True}
