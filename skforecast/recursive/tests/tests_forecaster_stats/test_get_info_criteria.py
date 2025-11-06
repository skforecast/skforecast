# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
from skforecast.stats import Sarimax
from skforecast.recursive import ForecasterStats

# Fixtures
from .fixtures_forecaster_stats import y


def test_ForecasterStats_get_info_criteria_ValueError_criteria_invalid_value():
    """
    Test ForecasterStats get_info_criteria ValueError when `criteria` is an invalid value.
    """
    forecaster = ForecasterStats(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    criteria = 'not_valid'

    err_msg = re.escape(
        ("Invalid value for `criteria`. Valid options are 'aic', 'bic', "
         "and 'hqic'.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(criteria=criteria)


def test_ForecasterStats_get_info_criteria_ValueError_method_invalid_value():
    """
    Test ForecasterStats get_info_criteria ValueError when `method` is an invalid value.
    """
    forecaster = ForecasterStats(regressor=Sarimax(order=(1, 1, 1)))
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
    Test ForecasterStats get_info_criteria after fit `y` with skforecast.
    """
    forecaster = ForecasterStats(regressor=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    results = forecaster.get_info_criteria(criteria='aic', method='standard')
    expected = -56.80222086732

    assert results == pytest.approx(expected)
