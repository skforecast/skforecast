# Unit test set_params ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
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
    
    results = forecaster.estimator.get_params()
    
    assert results == expected


def test_set_params_sets_is_fitted_to_false():
    """
    Test that set_params sets is_fitted to False after a forecaster has been fitted.
    """
    series = pd.DataFrame(
        {'l1': np.arange(10), 'l2': np.arange(10)},
        index=pd.date_range('2020-01-01', periods=10, freq='D')
    )
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(fit_intercept=True), lags=3
    )
    forecaster.fit(series=series)
    assert forecaster.is_fitted is True
    
    new_params = {'fit_intercept': False}
    forecaster.set_params(new_params)
    
    assert forecaster.is_fitted is False
