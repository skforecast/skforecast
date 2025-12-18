# Unit test _recursive_predict_bootstrapping ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


def test_recursive_predict_bootstrapping_output_with_residuals_zero():
    """
    Test _recursive_predict_bootstrapping output when all residuals are zero.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    # Create 2D array with sampled residuals: (steps, n_boot)
    n_boot = 1
    sampled_residuals = np.zeros((5, n_boot))
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])

    np.testing.assert_array_almost_equal(predictions.ravel(), expected)


def test_recursive_predict_bootstrapping_output_with_residuals_last_step():
    """
    Test _recursive_predict_bootstrapping output when all residuals are zero 
    except the last step.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    # Create 2D array with sampled residuals: (steps, n_boot)
    n_boot = 1
    sampled_residuals = np.array([[0], [0], [0], [0], [100]])
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([50., 51., 52., 53., 154.])
    
    np.testing.assert_array_almost_equal(predictions.ravel(), expected)


def test_recursive_predict_bootstrapping_output_with_residuals():
    """
    Test _recursive_predict_bootstrapping output with residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    # Create 2D array with sampled residuals: (steps, n_boot)
    n_boot = 1
    sampled_residuals = np.array([[10], [20], [30], [40], [50]])
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([60., 74.333333, 93.111111, 117.814815, 147.08642])
    
    np.testing.assert_array_almost_equal(predictions.ravel(), expected)


def test_recursive_predict_bootstrapping_output_with_binned_residuals():
    """
    Test _recursive_predict_bootstrapping output with binned residuals.
    """
    rng = np.random.default_rng(12345)
    steps = 10
    n_boot = 1
    forecaster = ForecasterRecursive(LGBMRegressor(verbose=-1), lags=3)
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=steps, exog=exog_predict)
    )

    # Create 3D array with sampled residuals: (n_bins, steps, n_boot)
    n_bins = len(forecaster.in_sample_residuals_by_bin_)
    sampled_residuals = np.stack(
        [
            forecaster.in_sample_residuals_by_bin_[k][
                rng.integers(
                    low=0,
                    high=len(forecaster.in_sample_residuals_by_bin_[k]),
                    size=(steps, n_boot),
                )
            ]
            for k in range(n_bins)
        ],
        axis=0,
    )

    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = steps,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = True,
                      n_boot               = n_boot
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

    np.testing.assert_array_almost_equal(predictions.ravel(), expected)


def test_recursive_predict_bootstrapping_output_with_multiple_boots():
    """
    Test _recursive_predict_bootstrapping output with multiple bootstrap iterations.
    Check actual predicted values using LinearRegression for reproducibility.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    n_boot = 3
    rng = np.random.default_rng(123)
    sampled_residuals = rng.normal(loc=0, scale=5, size=(5, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([
        [45.05439325, 48.16106674, 56.43962631],
        [50.32133651, 54.98817675, 56.03206106],
        [46.94292502, 55.4261756,  54.24091853],
        [47.82760601, 55.34430962, 49.9412166],
        [56.32478637, 53.89743895, 60.40607916]
    ])
    
    assert predictions.shape == (5, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_window_features():
    """
    Test _recursive_predict_bootstrapping output with window features.
    Check actual predicted values using LinearRegression for reproducibility.
    """
    from skforecast.preprocessing import RollingFeatures
    
    rolling = RollingFeatures(
        stats=['mean', 'std'],
        window_sizes=[3, 3]
    )
    
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=rolling
    )
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    n_boot = 2
    rng = np.random.default_rng(456)
    sampled_residuals = rng.normal(loc=0, scale=3, size=(5, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([
        [52.68811986, 43.62815438],
        [56.81665138, 48.77012927],
        [47.02662812, 48.56256546],
        [52.18480309, 47.74284148],
        [53.1320252,  56.1935267]
    ])
    
    assert predictions.shape == (5, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_different_lags():
    """
    Test _recursive_predict_bootstrapping output with different lag configurations.
    Check actual predicted values using LinearRegression for reproducibility.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=3)
    )
    n_boot = 2
    rng = np.random.default_rng(789)
    sampled_residuals = rng.normal(loc=0, scale=2, size=(3, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([
        [51.32042224, 46.46825658],
        [51.10328569, 51.14123342],
        [52.20587016, 55.04185412]
    ])
    
    assert predictions.shape == (3, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_exog():
    """
    Test _recursive_predict_bootstrapping output with exogenous variables.
    Check actual predicted values using LinearRegression for reproducibility.
    """
    exog_data = pd.Series(np.arange(100, 150), index=range(50), name='exog_1')
    exog_pred = pd.Series(np.arange(150, 157), index=range(50, 57), name='exog_1')
    
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)), exog=exog_data)

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=7, exog=exog_pred)
    )
    n_boot = 3
    rng = np.random.default_rng(321)
    sampled_residuals = rng.normal(loc=0, scale=4, size=(7, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 7,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected = np.array([
        [47.32443518, 51.6873766,  44.16660538],
        [44.37393249, 49.71736736, 49.01239056],
        [51.47513879, 48.59828704, 56.7180888],
        [47.05777305, 54.90394171, 51.95209034],
        [49.91496586, 54.53672891, 54.85925442],
        [53.79830813, 56.11653652, 54.09907472],
        [52.54511391, 55.52356279, 62.4301072]
    ])
    
    assert predictions.shape == (7, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_values_consistency():
    """
    Test _recursive_predict_bootstrapping output consistency: verify that with 
    identical residuals across boots, all bootstrap predictions are the same.
    Check actual predicted values using LinearRegression for reproducibility.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _ = (
        forecaster._create_predict_inputs(steps=4)
    )
    n_boot = 4
    residuals_single = np.array([5.0, 10.0, -3.0, 8.0])
    sampled_residuals = np.tile(residuals_single[:, np.newaxis], (1, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 4,
                      last_window_values   = last_window_values,
                      exog_values          = exog_values,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      n_boot               = n_boot
                  )
    
    expected_single_boot = np.array([55., 62.66666667, 54.55555556, 67.40740741])
    
    assert predictions.shape == (4, n_boot)
    for boot in range(n_boot):
        np.testing.assert_array_almost_equal(predictions[:, boot], expected_single_boot)
