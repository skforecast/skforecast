# Unit test set_params ForecasterSarimax
# ==============================================================================
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax


def test_ForecasterSarimax_set_params_skforecast():
    """
    Test set_params() method skforecast.
    """
    forecaster = ForecasterSarimax(regressor = Sarimax(order=(1, 1, 1)))
    new_params = {'order': (2, 2, 2), 'seasonal_order': (1, 1, 1, 2)}
    forecaster.set_params(new_params)
    results = forecaster.regressor.get_params()

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
