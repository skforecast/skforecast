# Unit test get_feature_importances ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series as series_2
from .fixtures_forecaster_direct_multivariate import exog as exog_2

series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                       'l2': pd.Series(np.arange(10, dtype=float))})
exog = pd.DataFrame({'exog_1': np.arange(10, dtype=float),
                     'exog_2': np.arange(100, 110, dtype=float)})


def test_TypeError_is_raised_when_step_is_not_int():
    """
    Test TypeError is raised when calling get_feature_importances() and step is 
    not an int.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    step = 'not_an_int'

    err_msg = re.escape(f"`step` must be an integer. Got {type(step)}.")
    with pytest.raises(TypeError, match = err_msg):         
        forecaster.get_feature_importances(step=step)


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importances(step=1)


@pytest.mark.parametrize("step", [0, 2], ids=lambda step: f'step: {step}')
def test_ValueError_is_raised_when_step_is_greater_than_forecaster_steps(step):
    """
    Test ValueError is raised when calling get_feature_importances() and step is 
    less than 1 or greater than the forecaster.steps.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    err_msg = re.escape(
                (f"The step must have a value from 1 to the maximum number of steps "
                 f"({forecaster.steps}). Got {step}.")
            )
    with pytest.raises(ValueError, match = err_msg):         
        forecaster.get_feature_importances(step=step)


def test_output_get_feature_importances_when_regressor_is_RandomForestRegressor_lags_3_step_1():
    """
    Test output of get_feature_importances for step 1, when regressor is 
    RandomForestRegressor with lags=3.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    results = forecaster.get_feature_importances(step=1)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.17522374, 0.17408016, 0.17918514, 
                                           0.13738555, 0.17994996, 0.15417544])
               }).sort_values(by='importance', ascending=False)
    
    pd.testing.assert_frame_equal(results, expected)
  
    
def test_output_get_feature_importances_when_regressor_is_RandomForestRegressor_lags_3_step_2_exog_included():
    """
    Test output of get_feature_importances for step 2, when regressor is 
    RandomForestRegressor with lags=3 and exog.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l2',
                     lags      = 3,
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)

    results = forecaster.get_feature_importances(step=2)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 
                               'exog_1', 'exog_2'],
                   'importance': np.array([0.08793949, 0.11408644, 0.16284783, 
                                           0.15234993, 0.12713124, 0.08502362, 
                                           0.16050297, 0.11011847])
               }).sort_values(by='importance', ascending=False)
    
    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_3_step_1():
    """
    Test output of get_feature_importances for step 1, when regressor is 
    LinearRegression with lags=3.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l2',
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.16666667, 0.16666667, 0.16666667, 
                                           0.16666667, 0.16666667, 0.16666667])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_3_step_2_exog_included():
    """
    Test output of get_feature_importances for step 2, when regressor is 
    LinearRegression with lags=3 and exog.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l1',
                     lags      = 3,
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)

    results = forecaster.get_feature_importances(step=2, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 
                               'exog_1', 'exog_2'],
                   'importance': np.array([0.04444444, 0.04444444, 0.04444444,
                                           0.04444444, 0.04444444, 0.04444444, 
                                           0.12765695, 0.12765695])
               })
    
    pd.testing.assert_frame_equal(results, expected)
    

def test_output_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when regressor is MLPRegressor with lags=5
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = MLPRegressor(solver='lbfgs', max_iter=50, random_state=123),
                     level     = 'l1',
                     lags      = 5,
                     steps     = 1
                 )
    forecaster.fit(series=series)

    estimator = forecaster.regressor
    expected = None

    warn_msg = re.escape(
            (f"Impossible to access feature importances for regressor of type "
             f"{type(estimator)}. This method is only valid when the "
             f"regressor stores internally the feature importances in the "
             f"attribute `feature_importances_` or `coef_`.")
        )
    with pytest.warns(UserWarning, match = warn_msg):
        results = forecaster.get_feature_importances(step=1)
        assert results is expected


def test_output_get_feature_importances_when_pipeline_LinearRegression():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3).
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = make_pipeline(StandardScaler(), 
                                                        LinearRegression()),
                     level              = 'l1',
                     lags               = 3,
                     steps              = 1,
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.3333333333333332, 0.33333333333333337, 0.33333333333333326, 
                                           0.33333333333333326, 0.33333333333333326, 0.33333333333333326])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_pipeline_RandomForestRegressor():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + RandomForestRegressor with lags=3).
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = make_pipeline(StandardScaler(), 
                                                        RandomForestRegressor(random_state=123)),
                     level              = 'l1',
                     lags               = 3,
                     steps              = 1,
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(step=1)
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
                   'importance': np.array([0.17522374, 0.17408016, 0.17918514, 
                                           0.13738555, 0.17994996, 0.15417544])
               }).sort_values(by='importance', ascending=False)
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_dict_step_2_exog_included():
    """
    Test output of get_feature_importances for step 2, when regressor is 
    LinearRegression with lags as dict and exog.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l1',
                     lags      = {'l1': None, 'l2': 3},
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)

    results = forecaster.get_feature_importances(step=2, sort_importance=False)
    expected = pd.DataFrame({
        'feature': ['l2_lag_1', 'l2_lag_2', 'l2_lag_3', 
                    'exog_1', 'exog_2'],
        'importance': np.array([0.05128205128205132, 0.05128205128205135, 0.05128205128205134,
                                0.14729647811635985, 0.14729647811635985])
    })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_dict_None_step_2_exog_included():
    """
    Test output of get_feature_importances for step 2, when regressor is 
    LinearRegression with lags as dict with None and exog.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l1',
                     lags      = {'l1': 3, 'l2': None},
                     steps     = 2
                 )
    forecaster.fit(series=series, exog=exog)

    results = forecaster.get_feature_importances(step=2, sort_importance=False)
    expected = pd.DataFrame({
        'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                    'exog_1', 'exog_2'],
        'importance': np.array([0.05128205128205132, 0.05128205128205135, 0.05128205128205134,
                                0.14729647811635985, 0.14729647811635985])
    })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_window_features():
    """
    Test output of get_feature_importances when regressor is LGMBRegressor with 
    lags=3 and window features.
    """

    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LGBMRegressor(verbose=-1, random_state=123),
                     level           = 'l1',
                     steps           = 2,
                     lags            = 3,
                     window_features = rolling
                 )
    forecaster.fit(series=series_2, exog=exog_2['exog_1'])

    results = forecaster.get_feature_importances(step=2, sort_importance=False)
    results = results.astype({'importance': float})
    expected = pd.DataFrame({
                   'feature': ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
                               'l1_roll_mean_3', 'l1_roll_sum_5',
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                               'l2_roll_mean_3', 'l2_roll_sum_5',
                               'exog_1'],
                   'importance': np.array([0, 18, 11,  0, 19, 12,  9,  8,  7,  3, 13])
               }).astype({'importance': float})
    
    pd.testing.assert_frame_equal(results, expected)
