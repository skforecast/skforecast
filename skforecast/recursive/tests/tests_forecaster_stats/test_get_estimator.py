# Unit test get_estimator ForecasterStats
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.stats import Sarimax, Arima
from skforecast.recursive import ForecasterStats


def test_NotFittedError_when_forecaster_not_fitted():
    """
    Raise NotFittedError if get_estimator is called before fitting.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    
    err_msg = re.escape(
        "This ForecasterStats instance is not fitted yet. "
        "Call `fit` with appropriate arguments before using "
        "this method."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.get_estimator(name='Sarimax(1,0,1)(0,0,0)[0]')


def test_KeyError_when_estimator_name_not_found():
    """
    Raise KeyError if the requested estimator name does not exist.
    """
    y = pd.Series(data=np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "No estimator named 'NonExistent'. "
        "Available estimators: ['Sarimax(1,0,1)(0,0,0)[0]']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.get_estimator(name='NonExistent')


def test_get_estimator_returns_correct_estimator_single():
    """
    Check that get_estimator returns the correct fitted estimator when 
    using a single estimator.
    """
    y = pd.Series(data=np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    
    estimator = forecaster.get_estimator(name='Sarimax(1,0,1)(0,0,0)[0]')
    
    assert estimator is forecaster.estimators_[0]


def test_get_estimator_returns_correct_estimator_multiple():
    """
    Check that get_estimator returns the correct fitted estimator when 
    using multiple estimators.
    """
    y = pd.Series(data=np.arange(50), name='y')
    estimators = [Sarimax(order=(1, 0, 1)), Arima(order=(1, 1, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=y)
    
    # Get first estimator
    est_sarimax = forecaster.get_estimator(name='Sarimax(1,0,1)(0,0,0)[0]')
    assert est_sarimax is forecaster.estimators_[0]
    
    # Get second estimator
    est_arima = forecaster.get_estimator(name='Arima(1,1,1)')
    assert est_arima is forecaster.estimators_[1]


def test_get_estimator_with_duplicate_names():
    """
    Check that get_estimator correctly retrieves estimators with duplicate 
    base names (with numeric suffixes).
    """
    y = pd.Series(data=np.arange(50), name='y')
    estimators = [Sarimax(order=(1, 0, 1)), Sarimax(order=(1, 0, 1))]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=y)
    
    # First estimator (no suffix)
    est_1 = forecaster.get_estimator(name='Sarimax(1,0,1)(0,0,0)[0]')
    assert est_1 is forecaster.estimators_[0]
    
    # Second estimator (with suffix _2)
    est_2 = forecaster.get_estimator(name='Sarimax(1,0,1)(0,0,0)[0]_2')
    assert est_2 is forecaster.estimators_[1]
    
    # They should be different objects
    assert est_1 is not est_2


def test_get_estimator_returns_fitted_estimator():
    """
    Check that the estimator returned by get_estimator is the fitted version,
    not the original unfitted one.
    """
    y = pd.Series(data=np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Arima(order=(1, 1, 1)))
    forecaster.fit(y=y)
    
    estimator = forecaster.get_estimator(name='Arima(1,1,1)')
    
    # estimators_ contains fitted estimators, estimators contains unfitted
    assert estimator is forecaster.estimators_[0]
    assert estimator is not forecaster.estimators[0]
