# Unit test predict_dist ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ....recursive import ForecasterRecursiveMultiSeries
from scipy.stats import norm
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
def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(level):
    """
    Test output of predict_dist when regressor is LinearRegression,
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
    results = forecaster.predict_dist(
                  steps                   = 2,
                  distribution            = norm,
                  levels                  = level,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = True,
                  suppress_warnings       = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.30718046, 0.14355782],
                                       [0.33695529, 0.21900963]]),
                   columns = ['1_loc', '1_scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("levels", 
                         [['1', '2'], None], 
                         ids=lambda lvl: f'levels: {lvl}')
def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer(levels):
    """
    Test output of predict_dist when regressor is LinearRegression,
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
    results = forecaster.predict_dist(
                  steps                   = 2,
                  distribution            = norm,
                  levels                  = levels,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.30718046, 0.14355782, 0.58121270, 0.31737888],
                                       [0.33695529, 0.21900963, 0.12968874, 0.06418038]]),
                   columns = ['1_loc', '1_scale', '2_loc', '2_scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(results, expected)
