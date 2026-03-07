# Unit test set_params ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
import sklearn
from packaging import version
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import LinearRegression


def test_set_params():
    """
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(fit_intercept=True), lags=3, steps=3
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

    results = forecaster.estimator.get_params()
    
    assert results == expected


def test_set_params_sets_is_fitted_to_false():
    """
    Test that set_params sets is_fitted to False after a forecaster has been fitted.
    """
    y = pd.Series(np.arange(10), name='y')
    forecaster = ForecasterDirect(
        estimator=LinearRegression(fit_intercept=True), lags=3, steps=3
    )
    forecaster.fit(y=y)
    assert forecaster.is_fitted is True
    
    new_params = {'fit_intercept': False}
    forecaster.set_params(new_params)
    
    assert forecaster.is_fitted is False