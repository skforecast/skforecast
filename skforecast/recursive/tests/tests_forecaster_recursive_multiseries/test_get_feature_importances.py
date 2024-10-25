# Unit test get_feature_importances ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                       '2': pd.Series(np.arange(10))})


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 3,
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
    and it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                   'importance': np.array([0.9446366782006932, 0.0, 0.05536332179930687, 0.0, 0.0])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable
    exog=pd.Series(np.arange(10, 20), name='exog').
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 20), name='exog'))
    results = forecaster.get_feature_importances()

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2', 'exog'],
                   'importance': np.array([0.73269896, 0., 0.21193772, 0., 0., 0.05536332])
               }).sort_values(by='importance', ascending=False)

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression():
    """
    Test output of get_feature_importances when regressor is LinearRegression with lags=3
    and it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                   'importance': np.array([3.33333333e-01, 3.33333333e-01, 3.33333333e-01, 
                                           -1.48164367e-16, 1.48164367e-16])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LinearRegression_with_exog():
    """
    Test output of get_feature_importances when regressor is LinearRegression with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable
    exog=pd.Series(np.arange(10, 15), name='exog').
    """
    series_2 = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                             '2': pd.Series(np.arange(10))})

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, exog=pd.Series(np.arange(10, 20), name='exog'))
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2', 'exog'],
                   'importance': np.array([2.50000000e-01,  2.50000000e-01,  2.50000000e-01,  
                                           -2.97120907e-17,  2.97120907e-17, 2.50000000e-01])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_and_UserWarning_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when regressor is MLPRegressor with lags=3
    and it is trained with series pandas DataFrame. Since MLPRegressor hasn't attributes
    `feature_importances_` or `coef_, results = None and a UserWarning is issues.
    """
    series_2 = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                             '2': pd.Series(np.arange(5))})
    forecaster = ForecasterRecursiveMultiSeries(MLPRegressor(solver = 'lbfgs', max_iter= 50, random_state=123), lags=3)
    forecaster.fit(series=series_2)

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
    it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = make_pipeline(StandardScaler(), LinearRegression()),
                     lags               = 3,
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                   'importance': np.array([6.66666667e-01, 6.66666667e-01, 6.66666667e-01, 
                                           -7.40821837e-17, 7.40821837e-17])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_pipeline_RandomForestRegressor():
    """
    Test output of get_feature_importances when regressor is pipeline,
    (StandardScaler() + RandomForestRegressor with lags=3),
    it is trained with series pandas DataFrame.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = make_pipeline(
                                     StandardScaler(), 
                                     RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123)
                                 ),
                     lags               = 3,
                     encoding           = "onehot",
                     transformer_series = None,
                 )
    forecaster.fit(series=series)
    results = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '1', '2'],
                   'importance': np.array([0.9446366782006932, 0.0, 0.05536332179930687, 0.0, 0.0])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog_ordinal():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable 
    with encoding='ordinal'.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     lags      = 3,
                     encoding  = 'ordinal'
                 )
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 20), name='exog'))
    results = forecaster.get_feature_importances()

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'],
                   'importance': np.array([0.944637, 0., 0.055363, 0., 0.])
               }).sort_values(by='importance', ascending=False)

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog_ordinal_category():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable 
    with encoding='ordinal_category'.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     lags      = 3,
                     encoding  = 'ordinal_category'
                 )
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 20), name='exog'))
    results = forecaster.get_feature_importances()

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'],
                   'importance': np.array([0.944637, 0., 0.055363, 0., 0.])
               }).sort_values(by='importance', ascending=False)

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog_None_encoding():
    """
    Test output of get_feature_importances when regressor is RandomForestRegressor with lags=3
    and it is trained with series pandas DataFrame and a exogenous variable 
    with encoding=None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                     lags      = 3,
                     encoding  = None
                 )
    forecaster.fit(series=series, exog=pd.Series(np.arange(10, 20), name='exog'))
    results = forecaster.get_feature_importances()

    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                   'importance': np.array([0.732699, 0.211938, 0.055363, 0.])
               }).sort_values(by='importance', ascending=False)

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_window_features():
    """
    Test output of get_feature_importances when regressor is LGMBRegressor with 
    lags=3 and window features.
    """    
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(series=series_dict, exog=exog_dict)

    results = forecaster.get_feature_importances(sort_importance=False)
    results = results.astype({'importance': float})
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'roll_sum_5', '_level_skforecast', 
                               'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                   'importance': np.array([480.0, 361.0, 323.0, 251.0, 430.0, 
                                           41.0, 175.0, 105.0, 468.0, 366.0])
               }).astype({'importance': float})
    
    pd.testing.assert_frame_equal(results, expected)
