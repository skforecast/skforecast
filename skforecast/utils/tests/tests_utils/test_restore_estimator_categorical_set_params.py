# Unit test _restore_estimator_categorical_set_params
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from skforecast.utils.utils import _get_estimator_categorical_set_params
from skforecast.utils.utils import _restore_estimator_categorical_set_params
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.direct import ForecasterDirectMultiVariate


# ==============================================================================
# Tests: empty params is a no-op
# ==============================================================================
def test_empty_params_does_not_modify_estimator():
    """
    Test that passing an empty params dict leaves the estimator's params
    unchanged.
    """
    estimator = XGBRegressor()
    estimator.set_params(feature_types=['q', 'c'], enable_categorical=True)
    forecaster = ForecasterRecursive(estimator=estimator, lags=2)

    _restore_estimator_categorical_set_params(forecaster, params={})

    assert estimator.get_params()['feature_types'] == ['q', 'c']
    assert estimator.get_params()['enable_categorical'] is True


# ==============================================================================
# Tests: XGBoost
# ==============================================================================
def test_xgboost_restores_feature_types_and_enable_categorical():
    """
    Test that feature_types and enable_categorical are restored on the
    XGBRegressor to the values captured in the snapshot dict, overwriting
    params that were set by a later configure call.
    """
    forecaster = ForecasterRecursive(estimator=XGBRegressor(), lags=2)

    # Set the initial state and take a snapshot (as on a cache miss).
    forecaster.estimator.set_params(
        feature_types=['q', 'c', 'q'], enable_categorical=True
    )
    snapshot = {'feature_types': ['q', 'c', 'q'], 'enable_categorical': True}

    # Simulate a cache-miss for a different (larger) lag combo overwriting params.
    forecaster.estimator.set_params(
        feature_types=['c', 'q', 'c', 'q', 'c'], enable_categorical=True
    )

    _restore_estimator_categorical_set_params(forecaster, params=snapshot)

    assert forecaster.estimator.get_params()['feature_types'] == ['q', 'c', 'q']
    assert forecaster.estimator.get_params()['enable_categorical'] is True


# ==============================================================================
# Tests: HistGradientBoosting
# ==============================================================================
def test_histgbr_restores_categorical_features():
    """
    Test that categorical_features is restored on HistGradientBoostingRegressor
    to the value captured in the snapshot dict.
    """
    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(), lags=2
    )

    # Set the initial state and take a snapshot (as on a cache miss).
    forecaster.estimator.set_params(categorical_features=[0, 2])
    snapshot = {'categorical_features': [0, 2]}

    # Simulate a cache-miss for a different lag combo overwriting params.
    forecaster.estimator.set_params(categorical_features=[0, 1, 2, 3, 4])

    _restore_estimator_categorical_set_params(forecaster, params=snapshot)

    assert forecaster.estimator.get_params()['categorical_features'] == [0, 2]


# ==============================================================================
# Tests: Pipeline (nothing to restore)
# ==============================================================================
def test_pipeline_is_not_modified():
    """
    Test that, when the estimator is a Pipeline, the params of its last step
    are not modified: native categorical features are not configured inside a
    Pipeline, so `_get_estimator_categorical_set_params` captures nothing and
    there is nothing to restore.
    """
    pipe = Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor())])
    forecaster = ForecasterRecursive(estimator=pipe, lags=2)
    forecaster.estimator[-1].set_params(
        feature_types=['q', 'c', 'q'], enable_categorical=True
    )
    snapshot = _get_estimator_categorical_set_params(forecaster)

    # The user changes the params of the last step after the snapshot.
    forecaster.estimator[-1].set_params(
        feature_types=['c', 'q', 'c', 'q', 'c', 'q', 'q']
    )

    _restore_estimator_categorical_set_params(forecaster, params=snapshot)

    assert snapshot == {}
    assert forecaster.estimator[-1].get_params()['feature_types'] == [
        'c', 'q', 'c', 'q', 'c', 'q', 'q'
    ]
    assert forecaster.estimator[-1].get_params()['enable_categorical'] is True


# ==============================================================================
# Tests: ForecasterDirect / ForecasterDirectMultiVariate restore estimators_[1],
# not the template
# ==============================================================================
@pytest.mark.parametrize(
    'forecaster_cls',
    [ForecasterDirect, ForecasterDirectMultiVariate],
    ids=['ForecasterDirect', 'ForecasterDirectMultiVariate']
)
def test_forecaster_direct_restores_estimators_1_not_template(forecaster_cls):
    """
    Test that ForecasterDirect and ForecasterDirectMultiVariate restore params
    only on estimators_[1] (the fitted per-step clone) and do not modify the
    template estimator attribute.
    """
    rng = np.random.default_rng(42)
    y = pd.Series(
        rng.normal(size=50),
        index=pd.date_range('2020', periods=50, freq='D')
    )

    if forecaster_cls is ForecasterDirect:
        forecaster = ForecasterDirect(estimator=XGBRegressor(), lags=2, steps=3)
        forecaster.fit(y=y)
    else:
        series = pd.DataFrame(
            {'s1': y.values, 's2': rng.normal(size=50)},
            index=y.index
        )
        forecaster = ForecasterDirectMultiVariate(
            estimator=XGBRegressor(), level='s1', lags=2, steps=3
        )
        forecaster.fit(series=series)

    snapshot = {'feature_types': ['q', 'c', 'q'], 'enable_categorical': True}

    # Set a different value on the step-1 clone before restoring.
    forecaster.estimators_[1].set_params(
        feature_types=['c', 'q', 'c', 'q'], enable_categorical=True
    )

    _restore_estimator_categorical_set_params(forecaster, params=snapshot)

    # estimators_[1] should be restored to the snapshot values.
    assert forecaster.estimators_[1].get_params()['feature_types'] == ['q', 'c', 'q']
    assert forecaster.estimators_[1].get_params()['enable_categorical'] is True
    # Template estimator should not have been touched.
    assert forecaster.estimator.get_params()['feature_types'] is None
