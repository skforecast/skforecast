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
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict

series_2 = pd.DataFrame(
               {'1': pd.Series(np.arange(start=0, stop=50)), 
                '2': pd.Series(np.arange(start=50, stop=100))}
           )


@pytest.mark.parametrize("encoding",
                         ["ordinal", "ordinal_category", "onehot", None],
                         ids=lambda dt: f"encoding: {dt}")
def test_recursive_predict_output_when_regressor_is_LinearRegression(encoding):
    """
    Test _recursive_predict output when using LinearRegression as regressor.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = encoding,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
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
def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler(encoding):
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = encoding,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
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


def test_recursive_predict_output_when_regressor_is_Ridge_StandardScaler_encoding_None():
    """
    Test _recursive_predict output when using Ridge as regressor and
    StandardScaler with encoding=None.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = [1, 5],
                     encoding           = None,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
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
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_recursive_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test _recursive_predict output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog with series 
    of different lengths.
    """
    new_series = series.copy()
    new_series.iloc[:10, 1] = np.nan

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=new_series, exog=exog)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_predict)
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


def test_recursive_predict_output_with_window_features():
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
                     regressor          = LGBMRegressor(verbose=-1),
                     lags               = 5,
                     window_features    = rolling,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_predict)
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
                     regressor          = LGBMRegressor(verbose=-1),
                     lags               = 5,
                     window_features    = [rolling, rolling_2],
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5, exog=exog_predict)
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


def test_recursive_predict_output_with_residuals_zero():
    """
    Test _recursive_predict output with residuals when all residuals are zero.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict,
                      residuals        = residuals
                  )
    
    expected = np.array([
                   [50., 100.],
                   [51., 101.],
                   [52., 102.],
                   [53., 103.],
                   [54., 104.]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals_last_step():
    """
    Test _recursive_predict output with residuals when all residuals are zero 
    except the last step.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [100, 200]])
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict,
                      residuals        = residuals
                  )
    
    expected = np.array([
                   [50., 100.],
                   [51., 101.],
                   [52., 102.],
                   [53., 103.],
                   [154., 304.]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_residuals():
    """
    Test _recursive_predict output with residuals.
    """

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)

    last_window, exog_values_dict, levels, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    residuals = np.full(
                    shape      = (5, len(levels)),
                    fill_value = np.nan,
                    dtype      = float
                )
    residuals[:, 0] = np.array([1, 2, 3, 4, 5])
    residuals[:, 1] = np.array([10, 20, 30, 40, 50])
    predictions = forecaster._recursive_predict(
                      steps            = 5,
                      levels           = levels,
                      last_window      = last_window,
                      exog_values_dict = exog_values_dict,
                      residuals        = residuals
                  )
    
    expected = np.array([
                   [51.    , 110.   ],
                   [53.2   , 123.   ],
                   [55.64  , 138.4  ],
                   [58.368 , 156.68 ],
                   [61.4416, 178.416]]
               )

    np.testing.assert_array_almost_equal(predictions, expected)
