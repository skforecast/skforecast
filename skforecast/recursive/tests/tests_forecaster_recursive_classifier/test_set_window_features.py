# Unit test set_window_features ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
from sklearn.linear_model import LogisticRegression 
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier


def test_set_window_features_ValueError_when_window_features_set_to_None_and_lags_is_None():
    """
    Test ValueError is raised when window_features is set to None and lags is None.
    """
    rolling = RollingFeaturesClassification(stats='mode', window_sizes=6)
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=None, window_features=rolling
    )

    err_msg = re.escape(
        "At least one of the arguments `lags` or `window_features` "
        "must be different from None. This is required to create the "
        "predictors used in training the forecaster."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_window_features(window_features=None)


@pytest.mark.parametrize("wf", 
                         [RollingFeaturesClassification(stats='mode', window_sizes=6),
                          [RollingFeaturesClassification(stats='mode', window_sizes=6)]],
                         ids = lambda wf: f'window_features: {type(wf)}')
def test_set_window_features_with_different_inputs(wf):
    """
    Test how attributes change with window_features argument.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    forecaster.set_window_features(window_features=wf)

    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_features_names == ['roll_mode_6']
    assert forecaster.window_features_class_names == ['RollingFeaturesClassification']
    assert forecaster.window_size == 6


def test_set_window_features_when_lags():
    """
    Test how `window_size` is also updated when the forecaster includes
    lags.
    """
    rolling = RollingFeaturesClassification(stats='mode', window_sizes=10)
    forecaster = ForecasterRecursiveClassifier(
                     regressor       = LogisticRegression(),
                     lags            = 9,
                     window_features = rolling
                 )
    
    rolling = RollingFeaturesClassification(stats='entropy', window_sizes=5)
    forecaster.set_window_features(window_features=rolling)

    assert forecaster.lags_names == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                                     'lag_6', 'lag_7', 'lag_8', 'lag_9']
    assert forecaster.max_lag == 9
    assert forecaster.max_size_window_features == 5
    assert forecaster.window_features_names == ['roll_entropy_5']
    assert forecaster.window_features_class_names == ['RollingFeaturesClassification']
    assert forecaster.window_size == 9


def test_set_window_features_to_None():
    """
    Test how attributes change when window_features is set to None.
    """
    rolling = RollingFeaturesClassification(stats='mode', window_sizes=6)
    forecaster = ForecasterRecursiveClassifier(
                     regressor       = LogisticRegression(),
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
