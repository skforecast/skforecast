# Unit test predict_quantiles ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ....recursive import ForecasterRecursiveMultiSeries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize("level", 
                         ['1', ['1']], 
                         ids=lambda lvl: f'level: {lvl}')
def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(level):
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed. Single level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  quantiles               = [0.05, 0.55, 0.95],
                  levels                  = level,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = True,
                  suppress_warnings       = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.12837500, 0.33920342, 0.47157640],
                                       [0.09385929, 0.32079042, 0.62315962]]),
                   columns = ['1_q_0.05', '1_q_0.55', '1_q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("levels", 
                         [['1', '2'], None], 
                         ids=lambda lvl: f'levels: {lvl}')
def test_predict_quantiles_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer(levels):
    """
    Test output of predict_quantiles when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed. Multiple levels.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    
    forecaster.fit(series=series, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_quantiles(
                  steps                   = 2,
                  quantiles               = [0.05, 0.55, 0.95],
                  levels                  = levels,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.12837500, 0.33920342, 0.47157640, 0.16153615, 0.65521692, 0.92455144],
                                       [0.09385929, 0.32079042, 0.62315962, 0.07374959, 0.10749931, 0.21846331]]),
                   columns = ['1_q_0.05', '1_q_0.55', '1_q_0.95', '2_q_0.05', '2_q_0.55', '2_q_0.95'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(results, expected)
