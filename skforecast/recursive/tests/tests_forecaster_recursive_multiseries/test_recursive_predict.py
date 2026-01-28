# Unit test _recursive_predict ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_dict_range,
    exog_wide_range,
    exog_pred_wide_range
)

series_2 = pd.DataFrame(
    {'1': pd.Series(np.arange(start=0, stop=50)), 
     '2': pd.Series(np.arange(start=50, stop=100))}
).to_dict(orient='series')


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot", None],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_estimator_is_LinearRegression(encoding):
    """
    Test _recursive_predict output when using LinearRegression as estimator.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = encoding,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )
    
    expected = np.array([
                   [50., 100.],
                   [51., 101.],
                   [52., 102.],
                   [53., 103.],
                   [54., 104.]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot"],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_estimator_is_Ridge_StandardScaler(encoding):
    """
    Test _recursive_predict output when using Ridge as estimator and
    StandardScaler.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = encoding,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )
    
    expected = np.array([
                   [1.75618727, 1.75618727],
                   [1.81961906, 1.81961906],
                   [1.88553079, 1.88553079],
                   [1.95267405, 1.95267405],
                   [2.02042886, 2.02042886]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_estimator_is_Ridge_StandardScaler_encoding_None():
    """
    Test _recursive_predict output when using Ridge as estimator and
    StandardScaler with encoding=None.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = None,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )
    
    expected = np.array([
                   [0.01772315, 1.73981623],
                   [0.05236473, 1.76946476],
                   [0.08680601, 1.801424  ],
                   [0.12114773, 1.83453189],
                   [0.15543994, 1.86821077]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_recursive_predict_output_when_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test _recursive_predict output when using LinearRegression as estimator, StandardScaler
    as transformer_series and transformer_exog as transformer_exog with series 
    of different lengths.
    """
    series_dict_range_nan = {
        'l1': series_dict_range['l1'].copy(),
        'l2': series_dict_range['l2'].copy()
    }
    series_dict_range_nan['l2'].iloc[:10] = np.nan

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_range_nan, exog=exog_wide_range)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_pred_wide_range)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )

    expected = np.array([
                   [ 9.28117103e-02,  5.17490723e-05],
                   [-3.22132190e-01,  1.40710229e-01],
                   [-5.50366938e-02,  4.43021934e-01],
                   [ 2.24101497e-01,  4.50750363e-01],
                   [ 9.30561169e-02,  1.78713706e-01]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_LGBMRegressor():
    """
    Test _recursive_predict output with window features.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LGBMRegressor(verbose=-1),
                     lags               = 5,
                     window_features    = rolling,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_range, exog=exog_wide_range)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_pred_wide_range)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )

    expected = np.array([
                   [ 0.52368935,  0.88399212],
                   [-0.13413796, -0.81145254],
                   [ 0.26257161,  0.40407821],
                   [ 0.93852676,  0.93243637],
                   [ 0.06002111,  0.16197356]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_XGBRegressor():
    """
    Test _recursive_predict output with window features.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = XGBRegressor(random_state=123, verbosity=0),
                     lags               = 5,
                     window_features    = rolling,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_range, exog=exog_wide_range)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_pred_wide_range)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )

    expected = np.array([
                   [ 0.28408778,  0.83336014],
                   [-1.00499725, -0.26915601],
                   [ 0.87588722,  0.08214854],
                   [ 1.05633557,  1.28230059],
                   [-0.15949866,  0.64663899]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_two_window_features():
    """
    Test _recursive_predict output with 2 window features.
    """
    rolling = RollingFeatures(stats=['mean'], window_sizes=4)
    rolling_2 = RollingFeatures(stats=['median'], window_sizes=4)
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LGBMRegressor(verbose=-1),
                     lags               = 5,
                     window_features    = [rolling, rolling_2],
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_range, exog=exog_wide_range)

    last_window, exog_values_dict, levels, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_pred_wide_range)
    )
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict
                  )

    expected = np.array([
                   [ 0.52368935,  0.88399212],
                   [-0.13413796, -0.81145254],
                   [ 0.26257161,  0.40407821],
                   [ 0.93852676,  0.93243637],
                   [ 0.06002111,  0.16197356]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)
