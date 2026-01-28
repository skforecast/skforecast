# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
from skforecast.stats import Sarimax, Arima, Ets
from skforecast.recursive import ForecasterStats
from sklearn.linear_model import LinearRegression


def test_TypeError_when_estimator_is_not_valid_stats_model_single_estimator():
    """
    Raise TypeError if estimator is not one of the valid statistical model types
    when initializing the forecaster with a single estimator.
    """
    estimator = LinearRegression()

    valid_estimator_types = (
        'skforecast.stats._arima.Arima',
        'skforecast.stats._arar.Arar',
        'skforecast.stats._ets.Ets',
        'skforecast.stats._sarimax.Sarimax',
        'aeon.forecasting.stats._arima.ARIMA',
        'aeon.forecasting.stats._ets.ETS',
        'sktime.forecasting.arima._pmdarima.ARIMA'
    )

    err_msg = re.escape(
        f"Estimator at index 0 must be an instance of type "
        f"{valid_estimator_types}. Got '{type(estimator)}'."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterStats(estimator=estimator)


def test_TypeError_when_estimator_is_not_valid_stats_model_in_list():
    """
    Raise TypeError if one estimator in a list is not one of the valid 
    statistical model types when initializing the forecaster.
    """
    valid_estimator = Sarimax(order=(1, 0, 1))
    invalid_estimator = LinearRegression()

    valid_estimator_types = (
        'skforecast.stats._arima.Arima',
        'skforecast.stats._arar.Arar',
        'skforecast.stats._ets.Ets',
        'skforecast.stats._sarimax.Sarimax',
        'aeon.forecasting.stats._arima.ARIMA',
        'aeon.forecasting.stats._ets.ETS',
        'sktime.forecasting.arima._pmdarima.ARIMA'
    )

    err_msg = re.escape(
        f"Estimator at index 1 must be an instance of type "
        f"{valid_estimator_types}. Got '{type(invalid_estimator)}'."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterStats(estimator=[valid_estimator, invalid_estimator])


def test_ValueError_when_estimator_list_is_empty():
    """
    Raise ValueError if estimator list is empty.
    """
    err_msg = re.escape("`estimator` list cannot be empty.")
    with pytest.raises(ValueError, match=err_msg):
        ForecasterStats(estimator=[])


def test_estimators_stored_as_list_when_single_estimator():
    """
    Check that a single estimator is stored as a list internally.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    assert isinstance(forecaster.estimators, list)
    assert len(forecaster.estimators) == 1
    assert forecaster.n_estimators == 1


def test_estimators_stored_correctly_when_multiple_estimators():
    """
    Check that multiple estimators are stored correctly.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1)), Ets()]
    forecaster = ForecasterStats(estimator=estimators)
    
    assert isinstance(forecaster.estimators, list)
    assert len(forecaster.estimators) == 3
    assert forecaster.n_estimators == 3


def test_estimator_ids_generated_correctly():
    """
    Check that estimator ids are generated correctly.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    assert forecaster.estimator_ids == ['skforecast.Sarimax']


def test_estimator_ids_unique_for_duplicate_estimators():
    """
    Check that estimator ids are made unique when duplicates exist.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Sarimax(order=(1, 0, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    assert forecaster.estimator_ids[0] != forecaster.estimator_ids[1]
    assert forecaster.estimator_ids[0] == 'skforecast.Sarimax'
    assert forecaster.estimator_ids[1] == 'skforecast.Sarimax_2'


def test_estimator_typesstored_correctly():
    """
    Check that estimator types are stored correctly as a list.
    """
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    
    assert isinstance(forecaster.estimator_types, list)
    assert forecaster.estimator_types == [
        'skforecast.stats._sarimax.Sarimax',
        'skforecast.stats._arima.Arima'
    ]


def test_fit_kwargs_is_ignored():
    """
    Check that fit_kwargs is ignored and set to None during initialization.
    """
    forecaster = ForecasterStats(
        estimator=Sarimax(order=(1, 0, 1)),
        fit_kwargs={'warning': 1}
    )
    
    assert forecaster.fit_kwargs is None
