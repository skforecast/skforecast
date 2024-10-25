# Unit test set_lags ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate


@pytest.mark.parametrize("lags", 
                         [None, {'l1': None, 'l2': None}], 
                         ids = lambda lags: f'lags: {lags}')
def test_set_lags_ValueError_when_lags_set_to_None_and_window_features_is_None(lags):
    """
    Test ValueError is raised when lags is set to None and window_features is None.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = 3,
                     window_features = None
                 )
    err_msg = re.escape(
        ("At least one of the arguments `lags` or `window_features` "
         "must be different from None. This is required to create the "
         "predictors used in training the forecaster.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_lags(lags=lags)


@pytest.mark.parametrize("lags", 
                         [3, [1, 2, 3], np.array([1, 2, 3]), range(1, 4)],
                         ids = lambda lags: f'lags: {lags}')
def test_set_lags_with_different_inputs(lags):
    """
    Test how lags and max_lag attributes change with lags argument of different types.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=2
    )
    forecaster.set_lags(lags=lags)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3']
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


def test_set_lags_when_lags_argument_is_a_dict():
    """
    Test how lags and max_lag attributes change when lags argument is a dict.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags={'l1': 3, 'l2': [1, 5]})
    
    np.testing.assert_array_almost_equal(forecaster.lags['l1'], np.array([1, 2, 3]))
    np.testing.assert_array_almost_equal(forecaster.lags['l2'], np.array([1, 5]))
    assert forecaster.lags_names['l1'] == ['l1_lag_1', 'l1_lag_2', 'l1_lag_3']
    assert forecaster.lags_names['l2'] == ['l2_lag_1', 'l2_lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5


def test_set_lags_when_lags_argument_is_a_dict_with_None():
    """
    Test how lags and max_lag attributes change when lags argument is a dict 
    containing None.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags={'l1': 3, 'l2': None})
    
    np.testing.assert_array_almost_equal(forecaster.lags['l1'], np.array([1, 2, 3]))
    assert forecaster.lags['l2'] is None
    assert forecaster.lags_names['l1'] == ['l1_lag_1', 'l1_lag_2', 'l1_lag_3']
    assert forecaster.lags_names['l2'] is None
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


@pytest.mark.parametrize("lags, max_lag",
                         [(3, 3), 
                          ({'l1': 3, 'l2': 4}, 4), 
                          ({'l1': None, 'l2': 5}, 5)],
                         ids = lambda value: f'lags: {value}')
def test_set_lags_max_lag_stored(lags, max_lag):
    """
    Test max_lag is equal to the maximum lag.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=1, steps=2
    )
    forecaster.set_lags(lags=lags)
    
    assert forecaster.max_lag == max_lag
    assert forecaster.window_size == max_lag


def test_set_lags_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes 
    differentiation.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     lags            = 3,
                     steps           = 2,
                     differentiation = 1
                 )
    
    forecaster.set_lags(lags=5)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5 + 1


def test_set_lags_when_window_features():
    """
    Test how `window_size` is also updated when the forecaster includes
    window_features.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=9, window_features=rolling
    )
    
    forecaster.set_lags(lags=5)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_size == 6


def test_set_lags_to_None():
    """
    Test how lags and max_lag attributes change when lags is set to None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=3)
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=3, lags=5, window_features=rolling
    )
    
    forecaster.set_lags(lags=None)

    assert forecaster.lags is None
    assert forecaster.lags_names is None
    assert forecaster.max_lag is None
    assert forecaster.max_size_window_features == 3
    assert forecaster.window_size == 3
