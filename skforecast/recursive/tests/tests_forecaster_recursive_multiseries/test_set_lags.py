# Unit test set_lags ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_set_lags_ValueError_when_lags_set_to_None_and_window_features_is_None():
    """
    Test ValueError is raised when lags is set to None and window_features is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, window_features=None
    )

    err_msg = re.escape(
        "At least one of the arguments `lags` or `window_features` "
        "must be different from None. This is required to create the "
        "predictors used in training the forecaster."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_lags(lags=None)


@pytest.mark.parametrize("lags", 
                         [3, [1, 2, 3], np.array([1, 2, 3]), range(1, 4)],
                         ids = lambda lags: f'lags: {lags}')
def test_set_lags_with_different_inputs(lags):
    """
    Test how lags and max_lag attributes change with lags argument of different types.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.set_lags(lags=lags)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3']
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


def test_set_lags_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes differentiation.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 3,
                     differentiation = 1
                 )
    
    forecaster.set_lags(lags=5)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5 + 1


def test_set_lags_to_None():
    """
    Test how lags and max_lag attributes change when lags is set to None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=3)
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = rolling
                 )
    
    forecaster.set_lags(lags=None)

    assert forecaster.lags is None
    assert forecaster.lags_names is None
    assert forecaster.max_lag is None
    assert forecaster.max_size_window_features == 3
    assert forecaster.window_size == 3
