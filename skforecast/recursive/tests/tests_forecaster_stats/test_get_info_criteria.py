# Unit test __init__ ForecasterStats
# ==============================================================================
import re
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.stats import Sarimax, Arar, Ets, Arima
from skforecast.recursive import ForecasterStats

# Fixtures
from .fixtures_forecaster_stats import y


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_info_criteria() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `get_info_criteria()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):         
        forecaster.get_info_criteria()


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
    expected = pd.DataFrame({
        'criteria': 'aic',
        'value': -56.80222086732
    }, index=[0])

    pd.testing.assert_frame_equal(results, expected)


def test_Arar_get_info_criteria():
    """
    Test ForecasterStats get_info_criteria with Arar estimator.
    """    
    forecaster = ForecasterStats(estimator=Arar())
    forecaster.fit(y=y)
    results = forecaster.get_info_criteria(criteria='aic')
    expected_aic = pd.DataFrame({
        'criteria': 'aic',
        'value': -75.467178
    }, index=[0])
    pd.testing.assert_frame_equal(results, expected_aic)

    results = forecaster.get_info_criteria(criteria='bic')
    expected_bic = pd.DataFrame({
        'criteria': 'bic',
        'value': -67.918599
    }, index=[0])
    pd.testing.assert_frame_equal(results, expected_bic)


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


def test_Ets_get_info_criteria():
    """
    Test ForecasterStats get_info_criteria with Ets estimator.
    """    
    forecaster = ForecasterStats(estimator=Ets(model='ANN'))
    forecaster.fit(y=y)
    
    # Test AIC
    aic_result = forecaster.get_info_criteria(criteria='aic')
    assert aic_result['value'].item() == forecaster.estimators_[0].model_.aic
    
    # Test BIC
    bic_result = forecaster.get_info_criteria(criteria='bic')
    assert bic_result['value'].item() == forecaster.estimators_[0].model_.bic


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


def test_ForecasterStats_get_info_criteria_multiple_estimators():
    """
    Test ForecasterStats get_info_criteria with multiple estimators.
    """
    forecaster = ForecasterStats(
        estimator=[
            Arima(order=(1, 0, 1)),
            Arar(),
            Ets(model='ANN')
        ]
    )
    forecaster.fit(y=y)
    
    results = forecaster.get_info_criteria(criteria='aic')
    
    # Get expected values from individual estimators
    forecaster_sarimax = ForecasterStats(estimator=Arima(order=(1, 0, 1)))
    forecaster_sarimax.fit(y=y)
    expected_sarimax = forecaster_sarimax.get_info_criteria(criteria='aic')['value'].iloc[0]
    
    forecaster_arar = ForecasterStats(estimator=Arar())
    forecaster_arar.fit(y=y)
    expected_arar = forecaster_arar.get_info_criteria(criteria='aic')['value'].iloc[0]
    
    forecaster_ets = ForecasterStats(estimator=Ets(model='ANN'))
    forecaster_ets.fit(y=y)
    expected_ets = forecaster_ets.get_info_criteria(criteria='aic')['value'].iloc[0]
    
    expected = pd.DataFrame({
        'estimator_id': ['skforecast.Arima', 'skforecast.Arar', 'skforecast.Ets'],
        'criteria': ['aic', 'aic', 'aic'],
        'value': [expected_sarimax, expected_arar, expected_ets]
    })
    
    pd.testing.assert_frame_equal(results, expected)
