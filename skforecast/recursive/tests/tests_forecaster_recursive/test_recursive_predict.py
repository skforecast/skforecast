# Unit test _recursive_predict ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


def test_recursive_predict_output_when_regressor_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as regressor.
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


def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler():
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler.
    """
    forecaster = ForecasterRecursive(
                     regressor     = Ridge(random_state=123),
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


def test_recursive_predict_output_with_window_features():
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


def test_recursive_predict_output_with_residuals_zero():
    """
    Test _recursive_predict output with residuals when all residuals are zero.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0], [0], [0], [0], [0]])
    predictions = forecaster._recursive_predict(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      residuals            = residuals,
                      use_binned_residuals = False
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals_last_step():
    """
    Test _recursive_predict output with residuals when all residuals are zero 
    except the last step.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0], [0], [0], [0], [100]])
    predictions = forecaster._recursive_predict(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      residuals            = residuals,
                      use_binned_residuals = False
                  )
    
    expected = np.array([50., 51., 52., 53., 154.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals():
    """
    Test _recursive_predict output with residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[10], [20], [30], [40], [50]])
    predictions = forecaster._recursive_predict(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      residuals            = residuals,
                      use_binned_residuals = False,
                  )
    
    expected = np.array([60., 74.333333, 93.111111, 117.814815, 147.08642])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_binned_residuals():
    """
    Test _recursive_predict output with binned residuals.
    """
    rng = np.random.default_rng(12345)
    steps = 10
    forecaster = ForecasterRecursive(LGBMRegressor(verbose=-1), lags=3)
    forecaster.fit(y=y, exog=exog)
    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=steps, exog=exog_predict)
    )

    sampled_residuals = {
                    k: v[rng.integers(low=0, high=len(v), size=steps)]
                    for k, v in forecaster.in_sample_residuals_by_bin_.items()
                }

    predictions = forecaster._recursive_predict(
                        steps                = steps,
                        last_window_values   = last_window_values,
                        exog_values          = exog_values,
                        residuals            = sampled_residuals,
                        use_binned_residuals = True,
                    )

    expected = np.array(
        [
            0.722443,
            0.849432,
            0.611024,
            0.893993,
            0.612895,
            0.223093,
            0.642686,
            0.68483,
            0.321592,
            0.499459,
        ]
    )

    np.testing.assert_array_almost_equal(predictions, expected)
