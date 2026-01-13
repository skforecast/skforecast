# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.stats import Sarimax, Arar, Ets
from skforecast.recursive import ForecasterStats

# Fixtures
from .fixtures_forecaster_stats import y


def test_ForecasterStats_get_info_criteria_ValueError_criteria_invalid_value():
    """
    Test ForecasterStats get_info_criteria ValueError when `criteria` is an invalid value.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    criteria = 'not_valid'

    err_msg = re.escape(
        "Invalid value for `criteria`. Valid options are 'aic', 'bic', "
        "and 'hqic'."
    )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(criteria=criteria)


def test_ForecasterStats_get_info_criteria_ValueError_method_invalid_value():
    """
    Test ForecasterStats get_info_criteria ValueError when `method` is an invalid value.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)

    method = 'not_valid'

    err_msg = re.escape(
        "Invalid value for `method`. Valid options are 'standard' and "
        "'lutkepohl'."
    )
    with pytest.raises(ValueError, match = err_msg): 
        forecaster.get_info_criteria(method=method)


def test_Sarimax_get_info_criteria_skforecast():
    """
    Test ForecasterStats get_info_criteria after fit `y` with skforecast Sarimax.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y)
    results = forecaster.get_info_criteria(criteria='aic', method='standard')
    expected = -56.80222086732

    assert results == pytest.approx(expected)


def test_Arar_get_info_criteria():
    """
    Test ForecasterStats get_info_criteria with Arar estimator.
    """    
    forecaster = ForecasterStats(estimator=Arar())
    forecaster.fit(y=y)
    
    # Test AIC
    aic_result = forecaster.get_info_criteria(criteria='aic', method='standard')
    assert isinstance(aic_result, (float, np.floating))
    # Handle NaN case properly
    if np.isnan(aic_result):
        assert np.isnan(forecaster.estimator.aic_)
    else:
        assert aic_result == forecaster.estimator.aic_
    
    # Test BIC
    bic_result = forecaster.get_info_criteria(criteria='bic', method='standard')
    assert isinstance(bic_result, (float, np.floating))
    # Handle NaN case properly
    if np.isnan(bic_result):
        assert np.isnan(forecaster.estimator.bic_)
    else:
        assert bic_result == forecaster.estimator.bic_


def test_Arar_get_info_criteria_ValueError_invalid_criteria():
    """
    Test ForecasterStats get_info_criteria ValueError when criteria='hqic' with Arar.
    """    
    forecaster = ForecasterStats(estimator=Arar())
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "Invalid value for `criteria`. Valid options are 'aic' and 'bic' "
        "for ARAR model."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.get_info_criteria(criteria='hqic')


def test_Arar_get_info_criteria_ValueError_invalid_method():
    """
    Test ForecasterStats get_info_criteria ValueError when method='lutkepohl' with Arar.
    """
    forecaster = ForecasterStats(estimator=Arar())
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "Invalid value for `method`. Only 'standard' is supported for "
        "ARAR model."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.get_info_criteria(criteria='aic', method='lutkepohl')


def test_Ets_get_info_criteria():
    """
    Test ForecasterStats get_info_criteria with Ets estimator.
    """    
    forecaster = ForecasterStats(estimator=Ets(model='ANN'))
    forecaster.fit(y=y)
    
    # Test AIC
    aic_result = forecaster.get_info_criteria(criteria='aic', method='standard')
    assert isinstance(aic_result, float)
    assert aic_result == forecaster.estimator.model_.aic
    
    # Test BIC
    bic_result = forecaster.get_info_criteria(criteria='bic', method='standard')
    assert isinstance(bic_result, float)
    assert bic_result == forecaster.estimator.model_.bic


def test_Ets_get_info_criteria_ValueError_invalid_criteria():
    """
    Test ForecasterStats get_info_criteria ValueError when criteria='hqic' with Ets.
    """    
    forecaster = ForecasterStats(estimator=Ets(model='ANN'))
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "Invalid value for `criteria`. Valid options are 'aic' and 'bic' "
        "for ETS model."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.get_info_criteria(criteria='hqic')


def test_Ets_get_info_criteria_ValueError_invalid_method():
    """
    Test ForecasterStats get_info_criteria ValueError when method='lutkepohl' with Ets.
    """    
    forecaster = ForecasterStats(estimator=Ets(model='ANN'))
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "Invalid value for `method`. Only 'standard' is supported for "
        "ETS model."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.get_info_criteria(criteria='aic', method='lutkepohl')
