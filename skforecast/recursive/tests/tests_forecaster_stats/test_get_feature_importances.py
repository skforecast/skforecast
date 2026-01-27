# Unit test get_feature_importances ForecasterStats
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats import Sarimax, Arar, Ets
from aeon.forecasting.stats import ARIMA
from skforecast.recursive import ForecasterStats
from sklearn.exceptions import NotFittedError

# Fixtures
from .fixtures_forecaster_stats import y
from .fixtures_forecaster_stats import exog


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `get_feature_importances()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_ForecasterStats_with_Sarimax_estimator():
    """
    Test output of get_feature_importances ForecasterStats using Sarimax as
    estimator.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order= (1, 1, 1), maxiter=1000, method='cg', disp=False)
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['exog', 'ar.L1', 'ma.L1', 'sigma2'],
                   'importance': np.array([0.9690539855149568, 0.4666537980992382, 
                                           -0.5263430267037418, 0.7862622654382363])
               })

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importances_ForecasterStats_with_Arar_estimator():
    """
    Test output of get_feature_importances ForecasterStats using Arar as
    estimator.
    """
    forecaster = ForecasterStats(estimator = Arar(max_ar_depth=26, max_lag=40)) 
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame(
                    {
                        "feature": {0: "lag_1", 1: "lag_10", 2: "lag_11", 3: "lag_12"},
                        "importance": {
                            0: 0.37984509460831944,
                            1: -0.208652315125212,
                            2: 0.23459105611032688,
                            3: -0.40956956673453687,
                        },
                    }
                )
    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importances_ForecasterStats_with_ARIMA_estimator():
    """
    Test output of get_feature_importances ForecasterStats using ARIMA as
    estimator.
    """
    forecaster = ForecasterStats(estimator = ARIMA(p=4, d=1, q=1))
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=True)

    expected = pd.DataFrame(
                {'feature': {0: 'lag_1',
                            1: 'lag_3',
                            2: 'intercept',
                            3: 'lag_4',
                            4: 'lag_2',
                            5: 'ma'
                        },
                'importance': {0: 0.6648894126315568,
                                1: 0.1478647645869075,
                                2: 0.0,
                                3: -0.23108776151763405,
                                4: -0.2444320861918562,
                                5: -0.8696258072896246
                        }
                }
            )
    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importances_ForecasterStats_with_Ets_estimator_ANN():
    """
    Test output of get_feature_importances ForecasterStats using Ets as
    estimator with ANN model (Additive error, No trend, No seasonality).
    """
    forecaster = ForecasterStats(estimator=Ets(model='ANN', m=1))
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=False)

    assert results.shape[0] == 1  # Only alpha for ANN model
    assert results['feature'].iloc[0] == 'alpha (level)'
    assert 0 < results['importance'].iloc[0] < 1  # Alpha should be between 0 and 1


def test_output_get_feature_importances_ForecasterStats_with_Ets_estimator_AAA():
    """
    Test output of get_feature_importances ForecasterStats using Ets as
    estimator with AAA model (Additive error, Additive trend, Additive seasonality).
    """
    forecaster = ForecasterStats(estimator=Ets(model='AAA', m=12))
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=False)

    assert results.shape[0] == 3  # alpha, beta, gamma for AAA model
    assert list(results['feature']) == ['alpha (level)', 'beta (trend)', 'gamma (seasonal)']
    assert all(0 < imp < 1 for imp in results['importance'])  # All should be between 0 and 1


def test_output_get_feature_importances_ForecasterStats_with_Ets_estimator_AAdA():
    """
    Test output of get_feature_importances ForecasterStats using Ets as
    estimator with damped trend model.
    """
    forecaster = ForecasterStats(estimator=Ets(model='AAA', m=12, damped=True))
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=False)

    assert results.shape[0] == 4  # alpha, beta, gamma, phi for damped model
    assert list(results['feature']) == ['alpha (level)', 'beta (trend)', 'gamma (seasonal)', 'phi (damping)']
    assert all(0 < imp < 1 for imp in results['importance'])  # All should be between 0 and 1


def test_output_get_feature_importances_ForecasterStats_with_Ets_estimator_sorted():
    """
    Test output of get_feature_importances ForecasterStats using Ets with
    sort_importance=True.
    """
    forecaster = ForecasterStats(estimator=Ets(model='AAA', m=12))
    forecaster.fit(y=y)
    results = forecaster.get_feature_importances(sort_importance=True)

    # Check that results are sorted in descending order
    assert all(results['importance'].iloc[i] >= results['importance'].iloc[i+1] 
               for i in range(len(results)-1))
