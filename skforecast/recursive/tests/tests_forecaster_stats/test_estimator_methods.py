# Unit test get_estimator, get_estimator_ids, remove_estimators ForecasterStats
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats import Sarimax, Arima, Ets
from skforecast.recursive import ForecasterStats

# Test get_estimator
# ==============================================================================
def test_get_estimator_raises_KeyError_when_id_not_found():
    """
    Raise KeyError when estimator id is not found.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    err_msg = re.escape(
        "No estimator with id 'invalid_id'. "
        "Available estimators: ['skforecast.Sarimax']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.get_estimator('invalid_id')


def test_get_estimator_returns_correct_estimator():
    """
    Check that get_estimator returns the correct estimator by id.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    estimator = forecaster.get_estimator('skforecast.Arima')
    
    assert isinstance(estimator, Arima)
    assert estimator is forecaster.estimators_[1]


def test_get_estimator_returns_correct_estimator_with_suffix():
    """
    Check that get_estimator returns the correct estimator when id has suffix.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Sarimax(order=(2, 0, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    estimator_1 = forecaster.get_estimator('skforecast.Sarimax')
    estimator_2 = forecaster.get_estimator('skforecast.Sarimax_2')
    
    assert estimator_1.order == (1, 0, 1)
    assert estimator_1 is forecaster.estimators_[0]
    assert estimator_2.order == (2, 0, 1)
    assert estimator_2 is forecaster.estimators_[1]


def test_get_estimator_returns_fitted_estimator():
    """
    Check that the estimator returned by get_estimator is the fitted version,
    not the original unfitted one.
    """
    forecaster = ForecasterStats(estimator=Arima(order=(1, 1, 1)))
    forecaster.fit(y=pd.Series(data=np.arange(50), name='y'))
    
    estimator = forecaster.get_estimator(id='skforecast.Arima')
    
    # estimators_ contains fitted estimators, estimators contains unfitted
    assert estimator is forecaster.estimators_[0]
    assert estimator is not forecaster.estimators[0]


# Test get_estimator_ids
# ==============================================================================
def test_get_estimator_ids_returns_list_of_ids():
    """
    Check that get_estimator_ids returns a list of all estimator ids.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1)), Ets()]
    forecaster = ForecasterStats(estimator=estimators)
    
    ids = forecaster.get_estimator_ids()
    
    assert ids == ['skforecast.Sarimax', 'skforecast.Arima', 'skforecast.Ets']


def test_get_estimator_ids_returns_single_id():
    """
    Check that get_estimator_ids returns a list with single id.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    ids = forecaster.get_estimator_ids()
    
    assert ids == ['skforecast.Sarimax']


def test_get_estimator_ids_with_duplicates():
    """
    Check that get_estimator_ids returns unique ids for duplicate estimators.
    """
    estimators = [Arima(order=(1, 0, 1)), Arima(order=(2, 0, 1)), Arima(order=(1, 1, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    ids = forecaster.get_estimator_ids()
    
    assert ids == ['skforecast.Arima', 'skforecast.Arima_2', 'skforecast.Arima_3']


# Test remove_estimators
# ==============================================================================
def test_remove_estimators_raises_KeyError_when_id_not_found():
    """
    Raise KeyError when estimator id to remove is not found.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    err_msg = re.escape(
        "No estimator(s) with id '['invalid_id']'. "
        "Available estimators: ['skforecast.Sarimax']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.remove_estimators('invalid_id')


def test_remove_estimators_raises_KeyError_when_multiple_ids_not_found():
    """
    Raise KeyError when multiple estimator ids to remove are not found.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    err_msg = re.escape(
        "No estimator(s) with id '['invalid_1', 'invalid_2']'. "
        "Available estimators: ['skforecast.Sarimax', 'skforecast.Arima']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.remove_estimators(['invalid_1', 'invalid_2'])


def test_remove_estimators_single_id():
    """
    Check that remove_estimators removes a single estimator by id.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1)), Ets()]
    forecaster = ForecasterStats(estimator=estimators)
    
    forecaster.remove_estimators('skforecast.Arima')
    
    assert forecaster.n_estimators == 2
    assert forecaster.estimator_ids == ['skforecast.Sarimax', 'skforecast.Ets']
    assert len(forecaster.estimators) == 2
    assert len(forecaster.estimators_) == 2
    assert forecaster.estimator_types == [
        'skforecast.stats._sarimax.Sarimax', 'skforecast.stats._ets.Ets'
    ]
    assert forecaster.estimator_names_ == [None, None]


def test_remove_estimators_multiple_ids_and_fitted():
    """
    Check that remove_estimators removes multiple estimators by ids.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arima(order=(1, 1, 1), seasonal_order=(0, 0, 0)),
        Ets()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))
    
    forecaster.remove_estimators(['skforecast.Sarimax', 'skforecast.Ets'])
    
    assert forecaster.n_estimators == 1
    assert forecaster.estimator_ids == ['skforecast.Arima']
    assert len(forecaster.estimators) == 1
    assert len(forecaster.estimators_) == 1
    assert forecaster.estimator_types == ['skforecast.stats._arima.Arima']
    assert forecaster.estimator_names_ == ['Arima(1,1,1)']


def test_remove_estimators_with_suffix_and_fitted():
    """
    Check that remove_estimators correctly removes estimator with suffix id.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Sarimax(order=(2, 0, 1)), Sarimax(order=(3, 0, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))
    
    forecaster.remove_estimators('skforecast.Sarimax_2')
    
    assert forecaster.n_estimators == 2
    assert forecaster.estimator_ids == ['skforecast.Sarimax', 'skforecast.Sarimax_3']
    assert len(forecaster.estimators) == 2
    assert len(forecaster.estimators_) == 2
    assert forecaster.estimator_types == [
        'skforecast.stats._sarimax.Sarimax', 'skforecast.stats._sarimax.Sarimax'
    ]
    assert forecaster.estimator_names_ == [
        'Sarimax(1,0,1)(0,0,0)[0]', 'Sarimax(3,0,1)(0,0,0)[0]'
    ]


# Test get_estimators_info
# ==============================================================================
def test_get_estimators_info_not_fitted():
    """
    Check that get_estimators_info returns correct DataFrame when not fitted.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arima(order=(1, 1, 1), seasonal_order=(0, 0, 0)),
        Ets()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    results = forecaster.get_estimators_info()

    expected = pd.DataFrame(
        {
            "id": ["skforecast.Sarimax", "skforecast.Arima", "skforecast.Ets"],
            "name": [None, None, None],
            "type": [
                "skforecast.stats._sarimax.Sarimax",
                "skforecast.stats._arima.Arima",
                "skforecast.stats._ets.Ets",
            ],
            "supports_exog": [True, True, False],
            "supports_interval": [True, True, True],
            "params": [
                "{'order': (1, 0, 1), 'seasonal_order': (0, 0, 0, 0), 'trend': None, 'measurement_error': False, 'time_varying_regression': False, 'mle_regression': True, 'simple_differencing': False, 'enforce_stationarity': True, 'enforce_invertibility': True, 'hamilton_representation': False, 'concentrate_scale': False, 'trend_offset': 1, 'use_exact_diffuse': False, 'dates': None, 'freq': None, 'missing': 'none', 'validate_specification': True, 'method': 'lbfgs', 'maxiter': 50, 'start_params': None, 'disp': False, 'sm_init_kwargs': {}, 'sm_fit_kwargs': {}, 'sm_predict_kwargs': {}}",
                "{'order': (1, 1, 1), 'seasonal_order': (0, 0, 0), 'm': 1, 'include_mean': True, 'transform_pars': True, 'method': 'CSS-ML', 'n_cond': None, 'SSinit': 'Gardner1980', 'optim_method': 'BFGS', 'optim_kwargs': {'maxiter': 1000}, 'kappa': 1000000.0}",
                "{'m': 1, 'model': 'ZZZ', 'damped': None, 'alpha': None, 'beta': None, 'gamma': None, 'phi': None, 'seasonal': True, 'trend': None, 'allow_multiplicative': True, 'allow_multiplicative_trend': False}",
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_get_estimators_info_fitted():
    """
    Check that get_estimators_info returns correct DataFrame when fitted.
    """
    estimators = [
        Sarimax(order=(1, 0, 1)),
        Arima(order=(1, 1, 1), seasonal_order=(0, 0, 0)),
        Ets()
    ]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))
    
    results = forecaster.get_estimators_info()

    expected = pd.DataFrame({
        'id': ['skforecast.Sarimax', 'skforecast.Arima', 'skforecast.Ets',],
        'name': ['Sarimax(1,0,1)(0,0,0)[0]', 'Arima(1,1,1)', 'Ets(AAN)',],
        'type': [
            'skforecast.stats._sarimax.Sarimax',
            'skforecast.stats._arima.Arima',
            'skforecast.stats._ets.Ets',
        ],
        'supports_exog': [True, True, False],
        'supports_interval': [True, True, True],
        'params': [
            "{'order': (1, 0, 1), 'seasonal_order': (0, 0, 0, 0), 'trend': None, 'measurement_error': False, 'time_varying_regression': False, 'mle_regression': True, 'simple_differencing': False, 'enforce_stationarity': True, 'enforce_invertibility': True, 'hamilton_representation': False, 'concentrate_scale': False, 'trend_offset': 1, 'use_exact_diffuse': False, 'dates': None, 'freq': None, 'missing': 'none', 'validate_specification': True, 'method': 'lbfgs', 'maxiter': 50, 'start_params': None, 'disp': False, 'sm_init_kwargs': {}, 'sm_fit_kwargs': {}, 'sm_predict_kwargs': {}}",
            "{'order': (1, 1, 1), 'seasonal_order': (0, 0, 0), 'm': 1, 'include_mean': True, 'transform_pars': True, 'method': 'CSS-ML', 'n_cond': None, 'SSinit': 'Gardner1980', 'optim_method': 'BFGS', 'optim_kwargs': {'maxiter': 1000}, 'kappa': 1000000.0}",
            "{'m': 1, 'model': 'ZZZ', 'damped': None, 'alpha': None, 'beta': None, 'gamma': None, 'phi': None, 'seasonal': True, 'trend': None, 'allow_multiplicative': True, 'allow_multiplicative_trend': False}"
            ]
    })

    pd.testing.assert_frame_equal(results, expected)
