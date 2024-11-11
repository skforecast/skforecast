# Unit test predict_interval ForecasterDirect
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


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_interval(
                  steps                   = 2,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.67523588, 0.2541421877507869, 0.7256141639758971],
                                       [0.38024988, 0.16009135981052192, 0.48878124019421915]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_interval(
                  steps                   = 2,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.67523588, 0.2541421877507869, 0.7256141639758971],
                                       [0.38024988, 0.16009135981052192, 0.48878124019421915]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)