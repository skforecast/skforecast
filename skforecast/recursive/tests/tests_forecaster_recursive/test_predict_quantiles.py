# Unit test predict_quantiles ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     binner_kwargs    = {'n_bins': 15}
                 )

    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  exog                    = exog_predict,
                  quantiles               = [0.05, 0.55, 0.95],
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.25836446, 0.571056, 0.71651764],
                                       [0.10646288, 0.32036708, 0.64980398]]),
                   columns = ['q_0.05', 'q_0.55', 'q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     binner_kwargs    = {'n_bins': 15}
                 )
    
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  exog                    = exog_predict,
                  quantiles               = (0.05, 0.55, 0.95),
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.25836446, 0.571056  , 0.71651764],
                                       [0.10646288, 0.32036708, 0.64980398]]),
                   columns = ['q_0.05', 'q_0.55', 'q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)
