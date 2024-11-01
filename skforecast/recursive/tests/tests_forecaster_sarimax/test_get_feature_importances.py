# Unit test get_feature_importances ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from sklearn.exceptions import NotFittedError

# Fixtures
from .fixtures_forecaster_sarimax import y
from .fixtures_forecaster_sarimax import exog


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))

    err_msg = re.escape(
        ("This forecaster is not fitted yet. Call `fit` with appropriate "
         "arguments before using `get_feature_importances()`.")
    )
    with pytest.raises(NotFittedError, match=err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_ForecasterSarimax_skforecast():
    """
    Test output of get_feature_importances ForecasterSarimax skforecast.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order= (1, 1, 1), maxiter=1000, method='cg', disp=False)
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['exog', 'ar.L1', 'ma.L1', 'sigma2'],
                   'importance': np.array([0.9690539855149568, 0.4666537980992382, 
                                           -0.5263430267037418, 0.7862622654382363])
               })

    pd.testing.assert_frame_equal(expected, results)
