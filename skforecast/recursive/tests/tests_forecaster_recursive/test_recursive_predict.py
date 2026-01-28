# Unit test _recursive_predict ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


def test_recursive_predict_output_when_estimator_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as estimator.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_estimator_is_Ridge_StandardScaler():
    """
    Test _recursive_predict output when using Ridge as estimator and
    StandardScaler.
    """
    forecaster = ForecasterRecursive(
                     estimator     = Ridge(random_state=123),
                     lags          = [1, 5],
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1.745476, 1.803196, 1.865844, 1.930923, 1.997202])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_LGBMRegressor():
    """
    Test _recursive_predict output with window features.
    """
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1), lags=3, window_features=rolling
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
    
    expected = np.array(
                   [0.584584, 0.487441, 0.483098, 0.483098, 0.580241, 
                    0.584584, 0.584584, 0.487441, 0.483098, 0.483098]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_XGBRegressor():
    """
    Test _recursive_predict output with window features.
    """
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        XGBRegressor(random_state=123, verbosity=0), lags=3, window_features=rolling
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
    
    expected = np.array(
                   [0.537775, 0.520587, 0.611477, 0.683535, 0.685327, 
                    0.70897 , 0.605805, 0.503862, 0.612269, 0.68454 ]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_two_window_features():
    """
    Test _recursive_predict output with 2 window features.
    """
    rolling = RollingFeatures(stats=['mean'], window_sizes=4)
    rolling_2 = RollingFeatures(stats=['median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1), lags=3, window_features=[rolling, rolling_2]
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
    
    expected = np.array(
                   [0.584584, 0.487441, 0.483098, 0.483098, 0.580241, 
                    0.584584, 0.584584, 0.487441, 0.483098, 0.483098]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)



