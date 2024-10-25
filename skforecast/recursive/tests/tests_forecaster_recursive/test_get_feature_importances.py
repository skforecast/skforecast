# Unit test get_feature_importances ForecasterRecursive
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
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags = 3,
                 )

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
              )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_when_regressor_is_RandomForest():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)).
    """
    forecaster = ForecasterRecursive(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.94766355, 0., 0.05233645])
               }).sort_values(by='importance', ascending=False)

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)) and a exogenous variable
    exog=pd.Series(np.arange(10, 20), name='exog').
    """
    forecaster = ForecasterRecursive(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), exog=pd.Series(np.arange(10, 20), name='exog'))
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                   'importance': np.array([0.94766355, 0.05233645, 0., 0.])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression():
    """
    Test output of get_feature_importances when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.33333333, 0.33333333, 0.33333333])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_with_exog():
    """
    Test output of get_feature_importances when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and a exogenous variable
    exog=pd.Series(np.arange(10, 15), name='exog').
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(10, 15), name='exog'))
    
    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                   'importance': np.array([0.25, 0.25, 0.25, 0.25])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_and_UserWarning_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when regressor is MLPRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(10)). Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a UserWarning is issues.
    """
    forecaster = ForecasterRecursive(MLPRegressor(solver = 'lbfgs', max_iter= 50, random_state=123), lags=3)
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
        results = forecaster.get_feature_importances()
        assert results is expected


def test_output_get_feature_importances_when_pipeline_LinearRegression():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    """
    forecaster = ForecasterRecursive(
                     regressor = make_pipeline(StandardScaler(), LinearRegression()),
                     lags      = 3
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))

    results = forecaster.get_feature_importances(sort_importance=False)
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
    forecaster = ForecasterRecursive(
                     regressor = make_pipeline(StandardScaler(), 
                                               RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123)),
                     lags      = 3
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.94766355, 0., 0.05233645])
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
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)

    results = forecaster.get_feature_importances(sort_importance=False)
    results = results.astype({'importance': float})
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'roll_sum_5', 'exog'],
                   'importance': np.array([0., 16., 7., 27., 45., 5.])
               }).astype({'importance': float})
    
    pd.testing.assert_frame_equal(results, expected)
