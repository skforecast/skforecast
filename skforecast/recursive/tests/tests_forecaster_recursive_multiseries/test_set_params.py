# Unit test set_params ForecasterRecursiveMultiSeries
# ==============================================================================
import sklearn
from packaging import version
from sklearn.linear_model import LinearRegression
from ....recursive import ForecasterRecursiveMultiSeries


def test_set_params():
    """
    """
    forecaster = ForecasterRecursiveMultiSeries(
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
