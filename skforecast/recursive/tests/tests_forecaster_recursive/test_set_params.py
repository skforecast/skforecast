# Unit test set_params ForecasterRecursive
# ==============================================================================
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression


def test_set_params():
    """
    """
    forecaster = ForecasterRecursive(LinearRegression(fit_intercept=True), lags=3)
    new_params = {'fit_intercept': False}
    forecaster.set_params(new_params)
    expected = {'copy_X': True,
                'fit_intercept': False,
                'n_jobs': None,
                'normalize': 'deprecated', #For sklearn < 1.2
                'positive': False
               }

    results = forecaster.regressor.get_params()
    results.update({'normalize': 'deprecated'})
    assert results == expected