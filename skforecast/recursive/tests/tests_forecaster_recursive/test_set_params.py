# Unit test set_params ForecasterRecursive
# ==============================================================================
import sklearn
from packaging import version
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
    if version.parse(sklearn.__version__) >= version.parse("1.7.0"):
        expected.update({'tol': 1e-06})

    results = forecaster.regressor.get_params()
    
    assert results == expected