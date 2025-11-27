# Unit test _recursive_predict ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y
from .fixtures_forecaster_recursive_classifier import exog
from .fixtures_forecaster_recursive_classifier import exog_predict


def test_recursive_predict_output_when_estimator_is_LogisticRegression():
    """
    Test _recursive_predict output when using LogisticRegression as estimator.
    """
    y_dummy = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']), 
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )
    
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y_dummy)

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([0., 1., 2., 0., 1.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_estimator_is_LogisticRegression_and_exog():
    """
    Test _recursive_predict output when using LogisticRegression as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1., 1., 1., 1., 1.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features():
    """
    Test _recursive_predict output with window features.
    """
    rolling = RollingFeaturesClassification(
        stats=['proportion', 'entropy'], window_sizes=4
    )
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1), lags=3, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1., 1., 0., 0., 2., 1., 1., 1., 0., 0.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_two_window_features():
    """
    Test _recursive_predict output with 2 window features.
    """
    rolling = RollingFeaturesClassification(stats=['proportion'], window_sizes=4)
    rolling_2 = RollingFeaturesClassification(stats=['entropy'], window_sizes=4)
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1), lags=3, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1., 1., 0., 0., 2., 1., 1., 1., 0., 0.])
    
    np.testing.assert_array_almost_equal(predictions, expected)
