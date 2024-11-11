# Unit test get_feature_importances ForecasterDirect
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
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog


def test_TypeError_is_raised_when_step_is_not_int():
    """
    Test TypeError is raised when calling get_feature_importances() and step is 
    not an int.
    """
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))

    step = 'not_an_int'

    err_msg = re.escape(f"`step` must be an integer. Got {type(step)}.")
    with pytest.raises(TypeError, match = err_msg):         
        forecaster.get_feature_importances(step=step)


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
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
def test_exception_is_raised_when_step_is_greater_than_forecaster_steps(step):
    """
    Test exception is raised when calling get_feature_importances() and step is 
    less than 1 or greater than the forecaster.steps.
    """
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))

    err_msg = re.escape(
                (f"The step must have a value from 1 to the maximum number of steps "
                 f"({forecaster.steps}). Got {step}.")
            )
    with pytest.raises(ValueError, match = err_msg):         
        forecaster.get_feature_importances(step=step)


def test_output_get_feature_importances_when_regressor_is_RandomForestRegressor_lags_3_step_1():
    """
    Test output of get_feature_importances for step 1, when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))

    results = forecaster.get_feature_importances(step=1)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.3902439024390244, 0.3170731707317073, 0.2926829268292683])
               }).sort_values(by='importance', ascending=False)
    
    pd.testing.assert_frame_equal(results, expected)
  
    
def test_output_get_feature_importances_when_regressor_is_RandomForestRegressor_lags_3_step_2_exog_included():
    """
    Test output of get_feature_importances for step 2, when regressor is 
    RandomForestRegressor with lags=3, steps 3 and it is trained with 
    y pandas Series and exog is pandas DataFrame.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float),
                         'exog_2': np.arange(1000, 1010, dtype=float)})
    
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=123),
                     lags      = 3,
                     steps     = 3
                 )
    forecaster.fit(y=y, exog=exog)

    results = forecaster.get_feature_importances(step=2)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2'],
                   'importance': np.array([0.16428571, 0.2, 0.20446429, 0.23333333, 0.19791667])
               }).sort_values(by='importance', ascending=False)
    
    pd.testing.assert_frame_equal(results, expected)
    
    
def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_3_step_1():
    """
    Test output of get_feature_importances for step 1, when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)
    forecaster.fit(y=pd.Series(np.arange(5)))

    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.33333333, 0.33333333, 0.33333333])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_lags_3_step_1_exog_included():
    """
    Test output of get_feature_importances for step 1, when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and
    exog=pd.Series(np.arange(5), name='exog').
    """
    forecaster = ForecasterDirect(regressor=LinearRegression(), lags=3, steps=1)
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(5), name='exog'))

    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                   'importance': np.array([0.25, 0.25, 0.25, 0.25])
               })
    
    pd.testing.assert_frame_equal(results, expected)
    

def test_output_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when regressor is MLPRegressor with lags=5
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a warning is raised`
    """
    forecaster = ForecasterDirect(
                     regressor = MLPRegressor(solver='lbfgs', max_iter=50, random_state=123),
                     lags      = 5,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))

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
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterDirect(
                     regressor = make_pipeline(StandardScaler(), LinearRegression()),
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))
    
    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.166667, 0.166667, 0.166667])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_pipeline_RandomForestRegressor():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + RandomForestRegressor with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterDirect(
                     regressor = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=5, max_depth=2, random_state=123)),
                     lags      = 3,
                     steps     = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))
    
    results = forecaster.get_feature_importances(step=1, sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.5, 0.5, 0.0])
               })
    
    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_window_features():
    """
    Test output of get_feature_importances when regressor is LGMBRegressor with 
    lags=3 and window features.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y_datetime), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog_datetime), freq='D')
    
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirect(
                     regressor       = LGBMRegressor(verbose=-1, random_state=123),
                     steps           = 3,
                     lags            = 3,
                     window_features = rolling
                 )
    forecaster.fit(y=y_datetime, exog=exog_datetime)

    results = forecaster.get_feature_importances(step=2, sort_importance=False)
    results = results.astype({'importance': float})
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'roll_sum_5', 'exog'],
                   'importance': np.array([13., 23., 1., 17., 33., 13.])
               }).astype({'importance': float})
    
    pd.testing.assert_frame_equal(results, expected)
