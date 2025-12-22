# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
import pandas as pd
import numpy as np
from skforecast.stats import Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression


def test_TypeError_when_estimator_is_not_valid_stats_model_when_initialization():
    """
    Raise TypeError if estimator is not one of the valid statistical model types
    when initializing the forecaster.
    """
    estimator = LinearRegression()

    err_msg = re.escape(
        (f"`estimator` must be an instance of type ['skforecast.stats._sarimax.Sarimax', "
         f"'skforecast.stats._arar.Arar', 'skforecast.stats._ets.Ets', "
         f"'aeon.forecasting.stats._arima.ARIMA', 'aeon.forecasting.stats._ets.ETS']. "
         f"Got '{type(estimator)}'.")
    )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterStats(estimator = estimator)


def test_params_are_stored_when_initialization():
    """
    Check `params` are stored in the forecaster when using a statistical model.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    expected_params = Sarimax(order=(1, 0, 1)).get_params(deep=True)

    assert forecaster.params == expected_params


def test_IgnoredArgumentWarning_when_fit_kwargs_with_skforecast_stats_model():
    """
    Test IgnoredArgumentWarning is raised when `fit_kwargs` is used with Sarimax estimator during fit.
    """ 
    y = pd.Series(data=np.arange(10), name='y')
    forecaster = ForecasterStats(
                     estimator  = Sarimax(order=(1, 0, 1)),
                     fit_kwargs = {'warning': 1}
                 )
    
    # fit_kwargs should be stored during initialization
    assert forecaster.fit_kwargs == {'warning': 1}
    
    warn_msg = re.escape(
        ("When using the skforecast Sarimax estimator, fit kwargs should be passed "
         "using the parameter `sm_fit_kwargs` during estimator initialization, "
         "not via ForecasterStats `fit_kwargs`. The provided `fit_kwargs` will be ignored.")
    )
    # Warning should be raised during fit
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        forecaster.fit(y=y)
    
    # fit_kwargs remain stored after fit (but were ignored)
    assert forecaster.fit_kwargs == {'warning': 1}
