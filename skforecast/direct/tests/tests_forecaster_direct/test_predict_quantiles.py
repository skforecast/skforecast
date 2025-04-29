# Unit test predict_quantiles ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog
from .fixtures_forecaster_direct import exog_predict


def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )

    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  exog                    = exog_predict,
                  quantiles               = [0.05, 0.55, 0.95],
                  n_boot                  = 4,
                  use_in_sample_residuals = True,
                  use_binned_residuals    = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.64316119, 0.7436322 , 0.82292603],
                                        [0.11683137, 0.37420031, 0.50180502]]),
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
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  exog                    = exog_predict,
                  quantiles               = (0.05, 0.55, 0.95),
                  n_boot                  = 4,
                  use_in_sample_residuals = False,
                  use_binned_residuals    = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.64316119, 0.7436322 , 0.82292603],
                                       [0.11683137, 0.37420031, 0.50180502]]),
                   columns = ['q_0.05', 'q_0.55', 'q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)
