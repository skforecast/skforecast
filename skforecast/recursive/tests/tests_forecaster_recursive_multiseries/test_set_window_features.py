# Unit test set_window_features ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_set_window_features_ValueError_when_window_features_set_to_None_and_lags_is_None():
    """
    Test ValueError is raised when window_features is set to None and lags is None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=None, window_features=rolling
    )

    err_msg = re.escape(
        "At least one of the arguments `lags` or `window_features` "
        "must be different from None. This is required to create the "
        "predictors used in training the forecaster."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_window_features(window_features=None)


@pytest.mark.parametrize("wf", 
                         [RollingFeatures(stats='mean', window_sizes=6),
                          [RollingFeatures(stats='mean', window_sizes=6)]],
                         ids = lambda wf: f'window_features: {type(wf)}')
def test_set_window_features_with_different_inputs(wf):
    """
    Test how attributes change with window_features argument.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.set_window_features(window_features=wf)

    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_features_names == ['roll_mean_6']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 6


def test_set_window_features_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes 
    differentiation.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 3,
                     window_features = RollingFeatures(stats='median', window_sizes=2),
                     differentiation = 1
                 )
    
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster.set_window_features(window_features=rolling)

    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3']
    assert forecaster.max_lag == 3
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_features_names == ['roll_mean_6']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 6 + 1


def test_set_window_features_when_lags():
    """
    Test how `window_size` is also updated when the forecaster includes
    lags.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=10)
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 9,
                     window_features = rolling
                 )
    
    rolling = RollingFeatures(stats='median', window_sizes=5)
    forecaster.set_window_features(window_features=rolling)

    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                                     'lag_6', 'lag_7', 'lag_8', 'lag_9']
    assert forecaster.max_lag == 9
    assert forecaster.max_size_window_features == 5
    assert forecaster.window_features_names == ['roll_median_5']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 9


def test_set_window_features_to_None():
    """
    Test how attributes change when window_features is set to None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = rolling
                 )
    
    forecaster.set_window_features(window_features=None)
    
    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features is None
    assert forecaster.window_features_names is None
    assert forecaster.window_features_class_names is None
    assert forecaster.window_size == 5
