# Unit test __init__ ForecasterSarimax
# ==============================================================================
import re
import pytest
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax

# Fixtures
from .fixtures_forecaster_sarimax import y


def test_ForecasterSarimax_get_info_criteria_ValueError_criteria_invalid_value():
    """
    Test ForecasterSarimax get_info_criteria ValueError when `criteria` is an invalid value.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    criteria = 'not_valid'

    err_msg = re.escape(
        ("Invalid value for `criteria`. Valid options are 'aic', 'bic', "
         "and 'hqic'.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(criteria=criteria)


def test_ForecasterSarimax_get_info_criteria_ValueError_method_invalid_value():
    """
    Test ForecasterSarimax get_info_criteria ValueError when `method` is an invalid value.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    method = 'not_valid'

    err_msg = re.escape(
        ("Invalid value for `method`. Valid options are 'standard' and "
         "'lutkepohl'.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(method=method)


def test_Sarimax_get_info_criteria_skforecast():
    """
    Test ForecasterSarimax get_info_criteria after fit `y` with skforecast.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    results = forecaster.get_info_criteria(criteria='aic', method='standard')
    expected = -56.80222086732

    assert results == pytest.approx(expected)
