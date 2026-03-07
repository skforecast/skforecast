# Unit test set_params ForecasterStats
# ==============================================================================
import re
import numpy as np
import pandas as pd
import pytest
from skforecast.stats import Sarimax, Arar, Ets
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning


def test_ForecasterStats_set_params_params_not_dict():
    """
    Test set_params() raises TypeError when params is not a dictionary.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arar()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    new_params = 'not_a_dict'
    
    err_msg = re.escape(
        f"`params` must be a dictionary. Got {type(new_params).__name__}."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.set_params(new_params)


def test_ForecasterStats_set_params_multiple_estimators_no_match():
    """
    Test set_params() method with multiple estimators when no names match.
    A ValueError should be raised.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arar()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    new_params = {
        'NonExistent1': {'param': 'value'}
    }
    
    err_msg = re.escape(
        f"None of the provided estimator ids ['NonExistent1'] "
        f"match the available estimator ids: {forecaster.estimator_ids}."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.set_params(new_params)


def test_ForecasterStats_set_params_multiple_estimators_partial_match():
    """
    Test set_params() method with multiple estimators when only some names match.
    A warning should be raised for non-matching names.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arar()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    new_params = {
        'skforecast.Sarimax': {'order': (2, 1, 2)},
        'NonExistentEstimator': {'param': 'value'}
    }
    
    warn_msg = re.escape(
        "The following estimator ids do not match any estimator "
        "in the forecaster and will be ignored: ['NonExistentEstimator']. "
        f"Available estimator ids are: {forecaster.estimator_ids}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.set_params(new_params)
    
    # Check that the matching estimator was updated
    assert forecaster.estimators[0].get_params()['order'] == (2, 1, 2)


def test_ForecasterStats_set_params_single_estimator():
    """
    Test set_params() method with a single estimator.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    new_params = {'order': (2, 2, 2), 'seasonal_order': (1, 1, 1, 2)}
    forecaster.set_params(new_params)
    results = forecaster.estimators[0].get_params()

    expected = {
        'order': (2, 2, 2),
        'seasonal_order': (1, 1, 1, 2),
        'trend': None,
        'measurement_error': False,
        'time_varying_regression': False,
        'mle_regression': True,
        'simple_differencing': False,
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'hamilton_representation': False,
        'concentrate_scale': False,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'dates': None,
        'freq': None,
        'missing': 'none',
        'validate_specification': True,
        'method': 'lbfgs',
        'maxiter': 50,
        'start_params': None,
        'disp': False,
        'sm_init_kwargs': {},
        'sm_fit_kwargs': {},
        'sm_predict_kwargs': {}
    }

    assert results == expected


def test_ForecasterStats_set_params_multiple_estimators_all_match():
    """
    Test set_params() method with multiple estimators when all names match.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arar(),
        Ets(trend='add', seasonal=None)
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    new_params = {
        'skforecast.Sarimax': {'order': (2, 1, 2), 'maxiter': 100},
        'skforecast.Arar': {'max_lag': 50},
        'skforecast.Ets': {'damped': True}
    }
    forecaster.set_params(new_params)
    
    # Check Sarimax params
    assert forecaster.estimators[0].get_params()['order'] == (2, 1, 2)
    assert forecaster.estimators[0].get_params()['maxiter'] == 100
    
    # Check Arar params
    assert forecaster.estimators[1].get_params()['max_lag'] == 50
    
    # Check Ets params
    assert forecaster.estimators[2].get_params()['damped'] is True


def test_ForecasterStats_set_params_multiple_estimators_single_update():
    """
    Test set_params() method with multiple estimators updating only one.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arar(),
        Ets(trend='add', seasonal=None)
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    # Store original params
    original_arar_max_lag = forecaster.estimators[1].get_params()['max_lag']
    original_ets_damped = forecaster.estimators[2].get_params()['damped']
    
    new_params = {
        'skforecast.Sarimax': {'maxiter': 200}
    }
    forecaster.set_params(new_params)
    
    # Check Sarimax was updated
    assert forecaster.estimators[0].get_params()['maxiter'] == 200
    
    # Check other estimators remain unchanged
    assert forecaster.estimators[1].get_params()['max_lag'] == original_arar_max_lag
    assert forecaster.estimators[2].get_params()['damped'] == original_ets_damped


def test_ForecasterStats_set_params_sets_is_fitted_to_false():
    """
    Test that set_params sets is_fitted to False after a forecaster has been fitted.
    """
    y = pd.Series(np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    assert forecaster.is_fitted is True
    
    new_params = {'order': (2, 1, 2)}
    forecaster.set_params(new_params)
    
    assert forecaster.is_fitted is False