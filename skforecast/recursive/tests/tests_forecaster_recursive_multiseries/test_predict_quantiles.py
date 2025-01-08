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
        {'level': ['1', '1'],
         'q_0.05': [0.1283750019253314, 0.09385929028369558],
         'q_0.55': [0.3392034161273868, 0.3207904249631374],
         'q_0.95': [0.47157639833964976, 0.6231596160709784]},
        index = pd.RangeIndex(start=50, stop=52)
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
                  quantiles               = (0.05, 0.55, 0.95),
                  levels                  = levels,
                  exog                    = exog_predict,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
        {
            "level": ["1", "2", "1", "2"],
            "q_0.05": [
                0.1283750019253314,
                0.1615361493231256,
                0.09385929028369558,
                0.07374959117551036,
            ],
            "q_0.55": [
                0.3392034161273868,
                0.6552169189586099,
                0.3207904249631374,
                0.10749930737109713,
            ],
            "q_0.95": [
                0.47157639833964976,
                0.9245514385384845,
                0.6231596160709784,
                0.2184633069802528,
            ],
        },
        index=pd.Index([50, 50, 51, 51]),
    )

    pd.testing.assert_frame_equal(results, expected)
