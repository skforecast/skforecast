# Unit test set_params ForecasterRecursive
# ==============================================================================
import sklearn
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression


def test_set_params():
    """
    """
    forecaster = ForecasterRecursive(
        LinearRegression(fit_intercept=True), lags=3
    )
    new_params = {'fit_intercept': False}
    forecaster.set_params(new_params)
    expected = {
        'copy_X': True,
        'fit_intercept': False,
        'n_jobs': None,
        'positive': False
    }
    if sklearn.__version__ >= '1.6':
        expected.update({'tol': 1e-06})

    results = forecaster.regressor.get_params()
    
    assert results == expected