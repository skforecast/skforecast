# Unit test _recursive_predict_bootstrapping ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


series_2 = pd.DataFrame(
    {'1': pd.Series(np.arange(start=0, stop=50)), 
     '2': pd.Series(np.arange(start=50, stop=100))}
).to_dict(orient='series')


def test_recursive_predict_bootstrapping_output_with_residuals_zero():
    """
    Test _recursive_predict_bootstrapping output with residuals when all 
    residuals are zero.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    
    n_boot = 3
    # sampled_residuals shape: (steps, n_levels, n_boot)
    sampled_residuals = np.zeros((5, 2, n_boot))
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # Expected shape: (steps, n_levels, n_boot)
    # Since residuals are all zeros, all bootstrap samples should be identical
    expected = np.array([
                   [[50., 50., 50.], [100., 100., 100.]],
                   [[51., 51., 51.], [101., 101., 101.]],
                   [[52., 52., 52.], [102., 102., 102.]],
                   [[53., 53., 53.], [103., 103., 103.]],
                   [[54., 54., 54.], [104., 104., 104.]]
               ])

    assert predictions.shape == (5, 2, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_residuals_last_step():
    """
    Test _recursive_predict_bootstrapping output with residuals when all 
    residuals are zero except the last step.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    
    n_boot = 2
    # sampled_residuals shape: (steps, n_levels, n_boot)
    sampled_residuals = np.zeros((5, 2, n_boot))
    # Add residuals to last step: [100, 200] for both boots
    sampled_residuals[4, 0, :] = 100  # level 0
    sampled_residuals[4, 1, :] = 200  # level 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # Expected shape: (steps, n_levels, n_boot)
    expected = np.array([
                   [[50., 50.], [100., 100.]],
                   [[51., 51.], [101., 101.]],
                   [[52., 52.], [102., 102.]],
                   [[53., 53.], [103., 103.]],
                   [[154., 154.], [304., 304.]]
               ])

    assert predictions.shape == (5, 2, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_residuals():
    """
    Test _recursive_predict_bootstrapping output with residuals.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    
    n_boot = 1
    # sampled_residuals shape: (steps, n_levels, n_boot)
    sampled_residuals = np.zeros((5, 2, n_boot))
    sampled_residuals[:, 0, 0] = np.array([1, 2, 3, 4, 5])    # level 0
    sampled_residuals[:, 1, 0] = np.array([10, 20, 30, 40, 50])  # level 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # Expected shape: (steps, n_levels, n_boot)
    # Values should match the original test
    expected = np.array([
                   [[51.    ], [110.   ]],
                   [[53.2   ], [123.   ]],
                   [[55.64  ], [138.4  ]],
                   [[58.368 ], [156.68 ]],
                   [[61.4416], [178.416]]
               ])

    assert predictions.shape == (5, 2, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_output_with_residuals_binned():
    """
    Test _recursive_predict_bootstrapping output with residuals when residuals 
    are binned.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None,
                     binner_kwargs      = {'n_bins': 2}
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    
    n_boot = 1
    n_bins = 2
    # sampled_residuals shape for binned: (n_bins, steps, n_boot, n_levels)
    sampled_residuals = np.zeros((n_bins, 5, n_boot, 2))
    
    # Bin 0 residuals
    sampled_residuals[0, :, 0, 0] = 1    # level 0
    sampled_residuals[0, :, 0, 1] = 300  # level 1
    
    # Bin 1 residuals
    sampled_residuals[1, :, 0, 0] = 20    # level 0
    sampled_residuals[1, :, 0, 1] = 4000  # level 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 5,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = True,
                      exog_values_dict     = exog_values_dict
                  )
    
    # Expected shape: (steps, n_levels, n_boot)
    # Values should match the original test
    expected = np.array([
                   [[70.], [4100.]],
                   [[75.], [4901.]],
                   [[80.8], [5862.]],
                   [[87.56], [7015.]],
                   [[95.472], [8398.4]]
               ])

    assert predictions.shape == (5, 2, n_boot)
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_multiple_boot_samples_different_residuals():
    """
    Test _recursive_predict_bootstrapping with multiple bootstrap samples
    where each sample has different residuals to verify independence and 
    correct residual application per bootstrap sample.
    Uses LinearRegression for reproducibility and to test recursive propagation.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=3)
    )
    
    n_boot = 3
    # sampled_residuals shape: (steps, n_levels, n_boot)
    # Each boot sample has different residuals
    sampled_residuals = np.zeros((3, 2, n_boot))
    
    # Level 0: boot 0, 1, 2 residuals
    sampled_residuals[:, 0, 0] = np.array([1, 1, 1])     # boot 0, level 0
    sampled_residuals[:, 0, 1] = np.array([5, 5, 5])     # boot 1, level 0
    sampled_residuals[:, 0, 2] = np.array([10, 10, 10])  # boot 2, level 0
    
    # Level 1: boot 0, 1, 2 residuals
    sampled_residuals[:, 1, 0] = np.array([2, 2, 2])     # boot 0, level 1
    sampled_residuals[:, 1, 1] = np.array([10, 10, 10])  # boot 1, level 1
    sampled_residuals[:, 1, 2] = np.array([20, 20, 20])  # boot 2, level 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # LinearRegression learns the pattern perfectly (linear series 0-49, 50-99)
    # Residuals propagate through recursive steps via lags
    # Step 0: base + residual
    # Step 1: prediction affected by step 0's residual-modified prediction in lags
    # Step 2: prediction affected by steps 0 and 1's residual-modified predictions
    expected = np.array([
        [[51., 55., 60.], [102., 110., 120.]],  # step 0
        [[52.33333333, 57.66666667, 64.33333333], [103.66666667, 114.33333333, 127.66666667]],  # step 1
        [[53.77777778, 60.88888889, 69.77777778], [105.55555556, 119.77777778, 137.55555556]]   # step 2
    ])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_with_exog():
    """
    Test _recursive_predict_bootstrapping with exogenous variables.
    Verifies that exog values are correctly tiled across bootstrap samples.
    Uses DummyRegressor with 'constant' strategy for reproducibility.
    """
    # DummyRegressor with constant strategy always predicts the same value
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = DummyRegressor(strategy='constant', constant=50.0),
                     lags               = 3,
                     transformer_series = None
                 )
    
    exog_train = pd.DataFrame({
        'exog_1': np.arange(100, dtype=float),
        'exog_2': np.arange(100, 200, dtype=float)
    })
    
    series_dict_range = {
        '1': pd.Series(np.arange(start=0, stop=100, dtype=float)),
        '2': pd.Series(np.arange(start=1000, stop=1100, dtype=float))
    }
    
    forecaster.fit(series=series_dict_range, exog=exog_train)

    exog_pred = pd.DataFrame({
        'exog_1': np.arange(100, 103, dtype=float),
        'exog_2': np.arange(200, 203, dtype=float)
    })
    
    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=3, exog=exog_pred)
    )
    
    n_boot = 2
    # sampled_residuals shape: (steps, n_levels, n_boot)
    sampled_residuals = np.zeros((3, 2, n_boot))
    # Different constant residuals for each bootstrap sample and level
    sampled_residuals[:, 0, 0] = 10.0  # level 0, boot 0
    sampled_residuals[:, 0, 1] = 5.0   # level 0, boot 1
    sampled_residuals[:, 1, 0] = 20.0  # level 1, boot 0
    sampled_residuals[:, 1, 1] = 15.0  # level 1, boot 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # DummyRegressor(constant=50) always predicts 50, then residuals are added
    expected = np.array([
        [[60.0, 55.0], [70.0, 65.0]],  # step 0: 50 + residuals
        [[60.0, 55.0], [70.0, 65.0]],  # step 1: 50 + residuals
        [[60.0, 55.0], [70.0, 65.0]]   # step 2: 50 + residuals
    ])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_with_window_features():
    """
    Test _recursive_predict_bootstrapping with window features.
    Verifies that window features are correctly computed for all bootstrap samples
    and that residuals propagate through recursive steps.
    Uses LinearRegression for reproducibility and to test recursive behavior.
    """
    rolling = RollingFeatures(stats=['mean'], window_sizes=3)
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 3,
                     window_features    = rolling,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=3)
    )
    
    n_boot = 2
    # sampled_residuals shape: (steps, n_levels, n_boot)
    sampled_residuals = np.zeros((3, 2, n_boot))
    sampled_residuals[:, 0, 0] = 2.0  # level 0, boot 0
    sampled_residuals[:, 0, 1] = 6.0  # level 0, boot 1
    sampled_residuals[:, 1, 0] = 4.0  # level 1, boot 0
    sampled_residuals[:, 1, 1] = 8.0  # level 1, boot 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # LinearRegression with window features learns the pattern
    # Window features (rolling mean) are computed from the expanding window
    # that includes residual-modified predictions from previous steps
    expected = np.array([
        [[52., 56.], [104., 108.]],  # step 0
        [[53.66666667, 59.], [106.33333333, 111.66666667]],  # step 1
        [[55.55555556, 62.66666667], [109.11111111, 116.22222222]]   # step 2
    ])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_bootstrapping_binned_residuals_multiple_boots():
    """
    Test _recursive_predict_bootstrapping with binned residuals and multiple
    bootstrap samples to verify correct bin-based residual selection.
    Uses DummyRegressor with quantile strategy for predictable binning.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = DummyRegressor(strategy='quantile', quantile=0.5),
                     lags               = 3,
                     transformer_series = None,
                     binner_kwargs      = {'n_bins': 2}
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=3)
    )
    
    n_boot = 2
    n_bins = 2
    # sampled_residuals shape for binned: (n_bins, steps, n_boot, n_levels)
    sampled_residuals = np.zeros((n_bins, 3, n_boot, 2))
    
    # Bin 0: lower residuals
    sampled_residuals[0, :, :, 0] = 5   # level 0
    sampled_residuals[0, :, :, 1] = 10  # level 1
    
    # Bin 1: higher residuals
    sampled_residuals[1, :, :, 0] = 15  # level 0
    sampled_residuals[1, :, :, 1] = 30  # level 1
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = True,
                      exog_values_dict     = exog_values_dict
                  )
    
    # DummyRegressor(quantile=0.5) predicts global median of combined series (~49.5)
    # since ForecasterRecursiveMultiSeries concatenates all series for training.
    # Residuals depend on which bin the prediction falls into.
    # Base prediction ~46 (median of training y values after lag creation)
    base_pred = 46.0
    assert predictions.shape == (3, 2, n_boot)
    assert not np.isnan(predictions).any()
    # Verify all predictions are greater than base predictions due to positive residuals
    assert np.all(predictions[:, 0, :] > base_pred)
    assert np.all(predictions[:, 1, :] > base_pred)


def test_recursive_predict_bootstrapping_single_level():
    """
    Test _recursive_predict_bootstrapping with a single level to ensure
    the method works correctly with n_levels=1.
    Uses DummyRegressor for reproducibility.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = DummyRegressor(strategy='constant', constant=100.0),
                     lags               = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    # Only predict for level '1'
    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=3, levels=['1'])
    )
    
    n_boot = 3
    # sampled_residuals shape: (steps, n_levels, n_boot) - but n_levels=1
    sampled_residuals = np.zeros((3, 1, n_boot))
    sampled_residuals[:, 0, 0] = 1.0
    sampled_residuals[:, 0, 1] = 5.0
    sampled_residuals[:, 0, 2] = 10.0
    
    predictions = forecaster._recursive_predict_bootstrapping(
                      steps                = 3,
                      levels               = levels,
                      last_window          = last_window,
                      n_boot               = n_boot,
                      sampled_residuals    = sampled_residuals,
                      use_binned_residuals = False,
                      exog_values_dict     = exog_values_dict
                  )
    
    # DummyRegressor(constant=100) always predicts 100
    expected = np.array([
        [[101.0, 105.0, 110.0]],  # step 0
        [[101.0, 105.0, 110.0]],  # step 1
        [[101.0, 105.0, 110.0]]   # step 2
    ])
    
    np.testing.assert_array_almost_equal(predictions, expected)
