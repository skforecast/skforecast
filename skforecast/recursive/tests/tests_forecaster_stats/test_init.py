# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
from skforecast.stats import Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression


def test_TypeError_when_estimator_is_not_Sarimax_when_initialization():
    """
    Raise TypeError if estimator is not of type 
    skforecast.sarimax.Sarimax when initializing the forecaster.
    """
    estimator = LinearRegression()

    err_msg = re.escape(
        (f"`estimator` must be an instance of type ['skforecast.stats._sarimax.Sarimax', "
         f"'skforecast.stats._arar.Arar', 'aeon.forecasting.stats._arima.ARIMA', "
         f"'aeon.forecasting.stats._ets.ETS']. Got '{type(estimator)}'.")
    )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterStats(estimator = estimator)


def test_skforecast_Sarimax_params_are_stored_when_initialization():
    """
    Check `params` are stored in the forecaster.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    expected_params = Sarimax(order=(1, 0, 1)).get_params(deep=True)

    assert forecaster.params == expected_params


def test_IgnoredArgumentWarning_when_skforecast_Sarimax_and_fit_kwargs():
    """
    Test IgnoredArgumentWarning is raised when `fit_kwargs` is not None when
    using the skforecast Sarimax model.
    """ 
    warn_msg = re.escape(
        ("When using the skforecast Sarimax model, the fit kwargs should "
         "be passed using the model parameter `sm_fit_kwargs`.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        forecaster = ForecasterStats(
                         estimator  = Sarimax(order=(1, 0, 1)),
                         fit_kwargs = {'warning': 1}
                     )
    
    assert forecaster.fit_kwargs == {}
