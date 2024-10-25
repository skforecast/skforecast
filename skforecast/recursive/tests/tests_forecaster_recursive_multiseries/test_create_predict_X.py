# Unit test create_predict_X ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

from ....utils import transform_numpy
from ....preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict

THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}

series_2 = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50, dtype=float)), 
                         '2': pd.Series(np.arange(start=50, stop=100, dtype=float))})


def test_create_predict_X_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_X(steps=5)


def test_output_create_predict_X_when_regressor_is_LinearRegression():
    """
    Test output create_predict_X when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.fit(series=series_2)
    results = forecaster.create_predict_X(steps=5)

    expected = {
        '1': pd.DataFrame(
                 data = {
                     'lag_1': [49., 50., 51., 52., 53.],
                     'lag_2': [48., 49., 50., 51., 52.],
                     'lag_3': [47., 48., 49., 50., 51.],
                     'lag_4': [46., 47., 48., 49., 50.],
                     'lag_5': [45., 46., 47., 48., 49.],
                     '_level_skforecast': [0., 0., 0., 0., 0.]
                 },
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             ),
        '2': pd.DataFrame(
                 data = {
                     'lag_1': [99., 100., 101., 102., 103.],
                     'lag_2': [98., 99., 100., 101., 102.],
                     'lag_3': [97., 98., 99., 100., 101.],
                     'lag_4': [96., 97., 98., 99., 100.],
                     'lag_5': [95., 96., 97., 98., 99.],
                     '_level_skforecast': [1., 1., 1., 1., 1.]
                 },
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_when_regressor_is_LinearRegression_and_StandardScaler():
    """
    Test output create_predict_X when using LinearRegression and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=StandardScaler())
    forecaster.fit(series=series_2)
    results = forecaster.create_predict_X(steps=5, levels='1')

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [1.69774938, 1.62845348, 1.55915759, 1.4898617 , 1.4205658 ,
                      0.        ],
                     [1.76704527, 1.69774938, 1.62845348, 1.55915759, 1.4898617 ,
                      0.        ],
                     [1.83634116, 1.76704527, 1.69774938, 1.62845348, 1.55915759,
                      0.        ],
                     [1.90563705, 1.83634116, 1.76704527, 1.69774938, 1.62845348,
                      0.        ],
                     [1.97493295, 1.90563705, 1.83634116, 1.76704527, 1.69774938,
                      0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', '_level_skforecast'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test create_predict_X output when using LinearRegression as regressor 
    and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=5, levels='1')

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       1.        ,  0.        ],
                     [ 0.11208288,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       1.        ,  0.        ],
                     [-0.24415873,  0.11208288,  0.47762884,  0.07582436,  2.08066403,
                       1.        ,  0.        ],
                     [-0.34454241, -0.24415873,  0.11208288,  0.47762884,  0.07582436,
                       1.        ,  0.        ],
                     [-0.09026999, -0.34454241, -0.24415873,  0.11208288,  0.47762884,
                       1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', '1', '2'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(transformer_series):
    """
    Test create_predict_X output when using LinearRegression as regressor, 
    StandardScaler as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.create_predict_X(steps=5, levels='1', exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       0.        , -0.09362908,  1.        ,  0.        ],
                     [ 0.13255401,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       0.        ,  0.45144522,  1.        ,  0.        ],
                     [-0.24552086,  0.13255401,  0.47762884,  0.07582436,  2.08066403,
                       0.        , -1.53968887,  1.        ,  0.        ],
                     [ 0.10296929, -0.24552086,  0.13255401,  0.47762884,  0.07582436,
                       0.        , -1.45096055,  1.        ,  0.        ],
                     [ 0.30994137,  0.10296929, -0.24552086,  0.13255401,  0.47762884,
                       0.        , -0.77240468,  1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test create_predict_X output when using LinearRegression as regressor, StandardScaler
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
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       0.        , -0.09575486,  1.        ,  0.        ],
                     [ 0.09281171,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       0.        ,  0.44024531,  1.        ,  0.        ],
                     [-0.32213219,  0.09281171,  0.47762884,  0.07582436,  2.08066403,
                       0.        , -1.51774136,  1.        ,  0.        ],
                     [-0.05503669, -0.32213219,  0.09281171,  0.47762884,  0.07582436,
                       0.        , -1.43049014,  1.        ,  0.        ],
                     [ 0.2241015 , -0.05503669, -0.32213219,  0.09281171,  0.47762884,
                       0.        , -0.76323054,  1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [-5.73884683e-01, -9.60902713e-01, -3.64833761e-01,
                      -1.29936753e+00,  7.39733788e-01,  1.00000000e+00,
                      -9.57548591e-02,  1.00000000e+00,  0.00000000e+00],
                     [ 5.17490723e-05, -5.73884683e-01, -9.60902713e-01,
                      -3.64833761e-01, -1.29936753e+00,  1.00000000e+00,
                       4.40245313e-01,  1.00000000e+00,  0.00000000e+00],
                     [ 1.40710229e-01,  5.17490723e-05, -5.73884683e-01,
                      -9.60902713e-01, -3.64833761e-01,  1.00000000e+00,
                      -1.51774136e+00,  1.00000000e+00,  0.00000000e+00],
                     [ 4.43021934e-01,  1.40710229e-01,  5.17490723e-05,
                      -5.73884683e-01, -9.60902713e-01,  1.00000000e+00,
                      -1.43049014e+00,  1.00000000e+00,  0.00000000e+00],
                     [ 4.50750363e-01,  4.43021934e-01,  1.40710229e-01,
                       5.17490723e-05, -5.73884683e-01,  1.00000000e+00,
                      -7.63230542e-01,  1.00000000e+00,  0.00000000e+00]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog['exog_1'],
         'exog_2': ['a', 'b'] * 25,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    categorical_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               categorical_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     lags               = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=df_exog)
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                      0.        , 0.        , 0.        , 0.51312815],
                     [0.55845607, 0.61289453, 0.51948512, 0.98555979, 0.48303426,
                      0.        , 1.        , 1.        , 0.66662455],
                     [0.39788933, 0.55845607, 0.61289453, 0.51948512, 0.98555979,
                      0.        , 0.        , 2.        , 0.10590849],
                     [0.55897239, 0.39788933, 0.55845607, 0.61289453, 0.51948512,
                      0.        , 1.        , 3.        , 0.13089495],
                     [0.78325771, 0.55897239, 0.39788933, 0.55845607, 0.61289453,
                      0.        , 0.        , 4.        , 0.32198061],
                     [0.56944748, 0.78325771, 0.55897239, 0.39788933, 0.55845607,
                      0.        , 1.        , 0.        , 0.66156434],
                     [0.38053867, 0.56944748, 0.78325771, 0.55897239, 0.39788933,
                      0.        , 0.        , 1.        , 0.84650623],
                     [0.5081955 , 0.38053867, 0.56944748, 0.78325771, 0.55897239,
                      0.        , 1.        , 2.        , 0.55325734],
                     [0.4990876 , 0.5081955 , 0.38053867, 0.56944748, 0.78325771,
                      0.        , 0.        , 3.        , 0.85445249],
                     [0.47413047, 0.4990876 , 0.5081955 , 0.38053867, 0.56944748,
                      0.        , 1.        , 4.        , 0.38483781]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_2', 'exog_3', 'exog_1'],
                 index = pd.RangeIndex(start=50, stop=60, step=1)
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                      1.        , 0.        , 0.        , 0.51312815],
                     [0.49028507, 0.34345601, 0.2408559 , 0.39887629, 0.15112745,
                      1.        , 1.        , 1.        , 0.66662455],
                     [0.57728083, 0.49028507, 0.34345601, 0.2408559 , 0.39887629,
                      1.        , 0.        , 2.        , 0.10590849],
                     [0.6263173 , 0.57728083, 0.49028507, 0.34345601, 0.2408559 ,
                      1.        , 1.        , 3.        , 0.13089495],
                     [0.69634096, 0.6263173 , 0.57728083, 0.49028507, 0.34345601,
                      1.        , 0.        , 4.        , 0.32198061],
                     [0.70693932, 0.69634096, 0.6263173 , 0.57728083, 0.49028507,
                      1.        , 1.        , 0.        , 0.66156434],
                     [0.50129985, 0.70693932, 0.69634096, 0.6263173 , 0.57728083,
                      1.        , 0.        , 1.        , 0.84650623],
                     [0.52123367, 0.50129985, 0.70693932, 0.69634096, 0.6263173 ,
                      1.        , 1.        , 2.        , 0.55325734],
                     [0.3447972 , 0.52123367, 0.50129985, 0.70693932, 0.69634096,
                      1.        , 0.        , 3.        , 0.85445249],
                     [0.44095994, 0.3447972 , 0.52123367, 0.50129985, 0.70693932,
                      1.        , 1.        , 4.        , 0.38483781]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_2', 'exog_3', 'exog_1'],
                 index = pd.RangeIndex(start=50, stop=60, step=1)
             )
    }

    for k in results.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X method when 
    series and exog are dictionaries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    results = forecaster.create_predict_X(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                           [-1.77580056, -1.07810218, -0.49618465, -0.457091  , -0.3525861 ,
                             0.        ,  0.00821644,  1.42962482,      np.nan,      np.nan],
                           [-0.05501833, -1.77580056, -1.07810218, -0.49618465, -0.457091  ,
                             0.        ,  1.11220226,  0.89634375,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -1.77580056, -1.07810218, -0.49618465,
                             0.        ,  1.38486425, -0.30192795,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -0.05501833, -1.77580056, -1.07810218,
                             0.        ,  0.62088235, -1.26286725,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -0.05501833, -0.05501833, -1.77580056,
                             0.        , -0.60444947, -1.26286725,      np.nan,      np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                           [ 1.1523137 ,  1.16332175,  0.67896814,  0.36936645,  0.21800529,
                             1.        ,  0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [ 0.10401967,  1.1523137 ,  1.16332175,  0.67896814,  0.36936645,
                             1.        ,  1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [-0.05501833,  0.10401967,  1.1523137 ,  1.16332175,  0.67896814,
                             1.        ,  1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [-0.05501833, -0.05501833,  0.10401967,  1.1523137 ,  1.16332175,
                             1.        ,  0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833, -0.05501833,  0.10401967,  1.1523137 ,
                             1.        , -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                           [ 3.39980134, -0.37837738, -1.03823091, -0.74685959, -0.62651976,
                             3.        ,  0.00821644,      np.nan,  1.11141113, -0.87943526],
                           [ 0.18846847,  3.39980134, -0.37837738, -1.03823091, -0.74685959,
                             3.        ,  1.11220226,      np.nan,  1.1327558 ,  0.0058948 ],
                           [ 0.03948954,  0.18846847,  3.39980134, -0.37837738, -1.03823091,
                             3.        ,  1.38486425,      np.nan,  1.1775869 , -0.3532584 ],
                           [-0.05501833,  0.03948954,  0.18846847,  3.39980134, -0.37837738,
                             3.        ,  0.62088235,      np.nan,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833,  0.03948954,  0.18846847,  3.39980134,
                             3.        , -0.60444947,      np.nan,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                           [-0.75073762, -0.24309812,  0.6719899 ,  0.8322971 ,  0.61495753,
                             4.        ,  0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [-0.05501833, -0.75073762, -0.24309812,  0.6719899 ,  0.8322971 ,
                             4.        ,  1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [-0.05501833, -0.05501833, -0.75073762, -0.24309812,  0.6719899 ,
                             4.        ,  1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [-0.05501833, -0.05501833, -0.05501833, -0.75073762, -0.24309812,
                             4.        ,  0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833, -0.05501833, -0.05501833, -0.75073762,
                             4.        , -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_series():
    """
    Test create_predict_X output when using LinearRegression as regressor and differentiation=1,
    and transformer_series is StandardScaler.
    """
    end_train = '2003-01-30 23:59:00'

    # Data scaled and differentiated
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2003-01-01', periods=len(series), freq='D')
    series_dict_datetime = {
        "1": series_datetime['1'].loc[:end_train],
        "2": series_datetime['2'].loc[:end_train]
    }
    
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(series)), name='exog'
    )
    exog.index = pd.date_range(start='2003-01-01', periods=len(series), freq='D')
    exog_dict_datetime = {
        "1": exog.loc[:end_train],
        "2": exog.loc[:end_train]
    }
    exog_pred = {
        '1': exog.loc[end_train:],
        '2': exog.loc[end_train:]
    }

    steps = len(series_datetime.loc[end_train:])

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(), 
                     lags               = 3, 
                     transformer_series = StandardScaler(),    
                     differentiation    = 1
                 )
    forecaster.fit(series=series_dict_datetime, exog=exog_dict_datetime)
    results = forecaster.create_predict_X(steps=steps, exog=exog_pred)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 1.51888875,  0.2947634 , -0.60134326,  0.        , -0.78916779],
                     [-0.77061816,  1.51888875,  0.2947634 ,  0.        ,  0.07769117],
                     [-0.27080147, -0.77061816,  1.51888875,  0.        ,  0.27283009],
                     [ 0.18942457, -0.27080147, -0.77061816,  0.        ,  0.35075195],
                     [ 0.22221668,  0.18942457, -0.27080147,  0.        ,  0.31597336],
                     [-0.18989524,  0.22221668,  0.18942457,  0.        ,  0.14091339],
                     [ 0.0178716 , -0.18989524,  0.22221668,  0.        ,  1.28618144],
                     [-0.27726617,  0.0178716 , -0.18989524,  0.        ,  0.09164865],
                     [ 0.32136982, -0.27726617,  0.0178716 ,  0.        , -0.50744682],
                     [ 0.16307991,  0.32136982, -0.27726617,  0.        ,  0.01522573],
                     [-0.12259515,  0.16307991,  0.32136982,  0.        ,  0.82813767],
                     [-0.25245926, -0.12259515,  0.16307991,  0.        ,  0.15290495],
                     [ 0.25699022, -0.25245926, -0.12259515,  0.        ,  0.61532804],
                     [-0.11076989,  0.25699022, -0.25245926,  0.        , -0.06287136],
                     [ 0.13325561, -0.11076989,  0.25699022,  0.        , -1.1896156 ],
                     [ 0.35393851,  0.13325561, -0.11076989,  0.        ,  0.28674823],
                     [-0.29503577,  0.35393851,  0.13325561,  0.        , -0.64581528],
                     [ 0.28534431, -0.29503577,  0.35393851,  0.        ,  0.20879998],
                     [-0.12204731,  0.28534431, -0.29503577,  0.        ,  1.80029302],
                     [-0.44119326, -0.12204731,  0.28534431,  0.        ,  0.14269745]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'], 
                 index = pd.date_range(start='2003-01-31', periods=steps, freq='D')
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [ 2.09399597, -0.02273313, -1.53568303,  1.        , -0.78916779],
                     [-0.81250198,  2.09399597, -0.02273313,  1.        ,  0.07769117],
                     [-0.52333604, -0.81250198,  2.09399597,  1.        ,  0.27283009],
                     [ 0.16874655, -0.52333604, -0.81250198,  1.        ,  0.35075195],
                     [ 0.3256144 ,  0.16874655, -0.52333604,  1.        ,  0.31597336],
                     [-0.24368079,  0.3256144 ,  0.16874655,  1.        ,  0.14091339],
                     [-0.05501668, -0.24368079,  0.3256144 ,  1.        ,  1.28618144],
                     [-0.28643196, -0.05501668, -0.24368079,  1.        ,  0.09164865],
                     [ 0.32144844, -0.28643196, -0.05501668,  1.        , -0.50744682],
                     [ 0.1270445 ,  0.32144844, -0.28643196,  1.        ,  0.01522573],
                     [-0.15555442,  0.1270445 ,  0.32144844,  1.        ,  0.82813767],
                     [-0.27066946, -0.15555442,  0.1270445 ,  1.        ,  0.15290495],
                     [ 0.23674045, -0.27066946, -0.15555442,  1.        ,  0.61532804],
                     [-0.13847369,  0.23674045, -0.27066946,  1.        , -0.06287136],
                     [ 0.1078768 , -0.13847369,  0.23674045,  1.        , -1.1896156 ],
                     [ 0.33150703,  0.1078768 , -0.13847369,  1.        ,  0.28674823],
                     [-0.31874052,  0.33150703,  0.1078768 ,  1.        , -0.64581528],
                     [ 0.26029126, -0.31874052,  0.33150703,  1.        ,  0.20879998],
                     [-0.14626729,  0.26029126, -0.31874052,  1.        ,  1.80029302],
                     [-0.46492117, -0.14626729,  0.26029126,  1.        ,  0.14269745]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'], 
                 index = pd.date_range(start='2003-01-31', periods=steps, freq='D')
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_encoding_None():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X method when 
    series and exog are dictionaries and encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    results = forecaster.create_predict_X(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                                  [-0.98327068, -0.88248605, -0.79842631, -0.79277912, -0.77768307,
                                    0.00821644,  1.42962482,      np.nan,      np.nan],
                                  [-0.12514786, -0.98327068, -0.88248605, -0.79842631, -0.79277912,
                                    1.11220226,  0.89634375,      np.nan,      np.nan],
                                  [-0.12514786, -0.12514786, -0.98327068, -0.88248605, -0.79842631,
                                    1.38486425, -0.30192795,      np.nan,      np.nan],
                                  [-0.07645311, -0.12514786, -0.12514786, -0.98327068, -0.88248605,
                                    0.62088235, -1.26286725,      np.nan,      np.nan],
                                  [-0.07645311, -0.07645311, -0.12514786, -0.12514786, -0.98327068,
                                   -0.60444947, -1.26286725,      np.nan,      np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                                  [-0.10075438, -0.09727905, -0.25019327, -0.34793695, -0.39572285,
                                    0.00821644,  1.42962482,  1.11141113, -0.87943526],
                                  [ 0.01290134, -0.10075438, -0.09727905, -0.25019327, -0.34793695,
                                    1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                                  [-0.07645311,  0.01290134, -0.10075438, -0.09727905, -0.25019327,
                                    1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                                  [ 0.01290134, -0.07645311,  0.01290134, -0.10075438, -0.09727905,
                                    0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                                  [ 0.01290134,  0.01290134, -0.07645311,  0.01290134, -0.10075438,
                                   -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                                  [ 0.22739054, -0.51757641, -0.64768383, -0.59023234, -0.56650418,
                                    0.00821644,      np.nan,  1.11141113, -0.87943526],
                                  [-0.12514786,  0.22739054, -0.51757641, -0.64768383, -0.59023234,
                                    1.11220226,      np.nan,  1.1327558 ,  0.0058948 ],
                                  [ 0.01290134, -0.12514786,  0.22739054, -0.51757641, -0.64768383,
                                    1.38486425,      np.nan,  1.1775869 , -0.3532584 ],
                                  [-0.07645311,  0.01290134, -0.12514786,  0.22739054, -0.51757641,
                                    0.62088235,      np.nan,  1.0428337 ,  0.84287284],
                                  [ 0.01290134, -0.07645311,  0.01290134, -0.12514786,  0.22739054,
                                   -0.60444947,      np.nan,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                                  [1.22899343,  1.65819658,  2.4318926 ,  2.56743042,  2.38367241,
                                   0.00821644,  1.42962482,  1.11141113, -0.87943526],
                                  [0.12663112,  1.22899343,  1.65819658,  2.4318926 ,  2.56743042,
                                   1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                                  [0.01290134,  0.12663112,  1.22899343,  1.65819658,  2.4318926 ,
                                   1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                                  [0.01290134,  0.01290134,  0.12663112,  1.22899343,  1.65819658,
                                   0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                                  [0.01290134,  0.01290134,  0.01290134,  0.12663112,  1.22899343,
        -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X method when 
    series and exog are dictionaries and unknown level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    results = forecaster.create_predict_X(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                           [-1.77580056, -1.07810218, -0.49618465, -0.457091  , -0.3525861 ,
                             0.        ,  0.00821644,  1.42962482,      np.nan,      np.nan],
                           [-0.05501833, -1.77580056, -1.07810218, -0.49618465, -0.457091  ,
                             0.        ,  1.11220226,  0.89634375,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -1.77580056, -1.07810218, -0.49618465,
                             0.        ,  1.38486425, -0.30192795,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -0.05501833, -1.77580056, -1.07810218,
                             0.        ,  0.62088235, -1.26286725,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -0.05501833, -0.05501833, -1.77580056,
                             0.        , -0.60444947, -1.26286725,      np.nan,      np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                           [ 1.1523137 ,  1.16332175,  0.67896814,  0.36936645,  0.21800529,
                             1.        ,  0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [ 0.10401967,  1.1523137 ,  1.16332175,  0.67896814,  0.36936645,
                             1.        ,  1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [-0.05501833,  0.10401967,  1.1523137 ,  1.16332175,  0.67896814,
                             1.        ,  1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [-0.05501833, -0.05501833,  0.10401967,  1.1523137 ,  1.16332175,
                             1.        ,  0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833, -0.05501833,  0.10401967,  1.1523137 ,
                             1.        , -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                           [ 3.39980134, -0.37837738, -1.03823091, -0.74685959, -0.62651976,
                             3.        ,  0.00821644,      np.nan,  1.11141113, -0.87943526],
                           [ 0.18846847,  3.39980134, -0.37837738, -1.03823091, -0.74685959,
                             3.        ,  1.11220226,      np.nan,  1.1327558 ,  0.0058948 ],
                           [ 0.03948954,  0.18846847,  3.39980134, -0.37837738, -1.03823091,
                             3.        ,  1.38486425,      np.nan,  1.1775869 , -0.3532584 ],
                           [-0.05501833,  0.03948954,  0.18846847,  3.39980134, -0.37837738,
                             3.        ,  0.62088235,      np.nan,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833,  0.03948954,  0.18846847,  3.39980134,
                             3.        , -0.60444947,      np.nan,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                           [-0.75073762, -0.24309812,  0.6719899 ,  0.8322971 ,  0.61495753,
                             4.        ,  0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [-0.05501833, -0.75073762, -0.24309812,  0.6719899 ,  0.8322971 ,
                             4.        ,  1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [-0.05501833, -0.05501833, -0.75073762, -0.24309812,  0.6719899 ,
                             4.        ,  1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [-0.05501833, -0.05501833, -0.05501833, -0.75073762, -0.24309812,
                             4.        ,  0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [-0.05501833, -0.05501833, -0.05501833, -0.05501833, -0.75073762,
                             4.        , -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1005': pd.DataFrame(
                       data = np.array([
                           [ 1.22899343,  1.65819658,  2.4318926 ,  2.56743042,  2.38367241,
                                 np.nan,      np.nan,      np.nan,      np.nan,      np.nan],
                           [ 0.10401967,  1.22899343,  1.65819658,  2.4318926 ,  2.56743042,
                                 np.nan,      np.nan,      np.nan,      np.nan,      np.nan],
                           [-0.05501833,  0.10401967,  1.22899343,  1.65819658,  2.4318926 ,
                                 np.nan,      np.nan,      np.nan,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833,  0.10401967,  1.22899343,  1.65819658,
                                 np.nan,      np.nan,      np.nan,      np.nan,      np.nan],
                           [-0.05501833, -0.05501833, -0.05501833,  0.10401967,  1.22899343,
                                 np.nan,      np.nan,      np.nan,      np.nan,      np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X method when 
    series and exog are dictionaries and encoding is None and unknown level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1004']
    results = forecaster.create_predict_X(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                           [-0.98327068, -0.88248605, -0.79842631, -0.79277912, -0.77768307,
                             0.00821644,  1.42962482,      np.nan,      np.nan],
                           [-0.12514786, -0.98327068, -0.88248605, -0.79842631, -0.79277912,
                             1.11220226,  0.89634375,      np.nan,      np.nan],
                           [-0.12514786, -0.12514786, -0.98327068, -0.88248605, -0.79842631,
                             1.38486425, -0.30192795,      np.nan,      np.nan],
                           [-0.07645311, -0.12514786, -0.12514786, -0.98327068, -0.88248605,
                             0.62088235, -1.26286725,      np.nan,      np.nan],
                           [-0.07645311, -0.07645311, -0.12514786, -0.12514786, -0.98327068,
                            -0.60444947, -1.26286725,      np.nan,      np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                           [-0.10075438, -0.09727905, -0.25019327, -0.34793695, -0.39572285,
                             0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [ 0.01290134, -0.10075438, -0.09727905, -0.25019327, -0.34793695,
                             1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [-0.07645311,  0.01290134, -0.10075438, -0.09727905, -0.25019327,
                             1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [ 0.01290134, -0.07645311,  0.01290134, -0.10075438, -0.09727905,
                             0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [ 0.01290134,  0.01290134, -0.07645311,  0.01290134, -0.10075438,
                            -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                           [0.22739054, -0.51757641, -0.64768383, -0.59023234, -0.56650418,
                             0.00821644,      np.nan,  1.11141113, -0.87943526],
                           [-0.12514786,  0.22739054, -0.51757641, -0.64768383, -0.59023234,
                             1.11220226,      np.nan,  1.1327558 ,  0.0058948 ],
                           [ 0.01290134, -0.12514786,  0.22739054, -0.51757641, -0.64768383,
                             1.38486425,      np.nan,  1.1775869 , -0.3532584 ],
                           [-0.07645311,  0.01290134, -0.12514786,  0.22739054, -0.51757641,
                             0.62088235,      np.nan,  1.0428337 ,  0.84287284],
                           [ 0.01290134, -0.07645311,  0.01290134, -0.12514786,  0.22739054,
                            -0.60444947,      np.nan,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                           [1.22899343,  1.65819658,  2.4318926 ,  2.56743042,  2.38367241,
                             0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [ 0.12663112,  1.22899343,  1.65819658,  2.4318926 ,  2.56743042,
                             1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [ 0.01290134,  0.12663112,  1.22899343,  1.65819658,  2.4318926 ,
                             1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [ 0.01290134,  0.01290134,  0.12663112,  1.22899343,  1.65819658,
                             0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [ 0.01290134,  0.01290134,  0.01290134,  0.12663112,  1.22899343,
                            -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1005': pd.DataFrame(
                       data = np.array([
                           [1.22899343,  1.65819658,  2.4318926 ,  2.56743042,  2.38367241,
                             0.00821644,  1.42962482,  1.11141113, -0.87943526],
                           [ 0.12663112,  1.22899343,  1.65819658,  2.4318926 ,  2.56743042,
                             1.11220226,  0.89634375,  1.1327558 ,  0.0058948 ],
                           [ 0.01290134,  0.12663112,  1.22899343,  1.65819658,  2.4318926 ,
                             1.38486425, -0.30192795,  1.1775869 , -0.3532584 ],
                           [ 0.01290134,  0.01290134,  0.12663112,  1.22899343,  1.65819658,
                             0.62088235, -1.26286725,  1.0428337 ,  0.84287284],
                           [ 0.01290134,  0.01290134,  0.01290134,  0.12663112,  1.22899343,
                            -0.60444947, -1.26286725,  1.00599776, -0.62314633]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_same_predictions_as_predict():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X matrix returns 
    the same predictions as predict method when passing to the regressor predict 
    method.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = [1, 5],
        window_features    = [rolling, rolling_2],
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = None,
        transformer_exog   = None,
        differentiation    = None
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    steps = 5
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    X_predict = forecaster.create_predict_X(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    )
    results = np.full(
        shape=(steps, len(levels)), fill_value=np.nan, order='F', dtype=float
    )
    for i, level in enumerate(levels):
        results[:, i] = forecaster.regressor.predict(X_predict[level])

    expected = forecaster.predict(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    ).to_numpy()
    
    np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X matrix returns 
    the same predictions as predict method when passing to the regressor predict 
    method and transformers are used.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = [1, 5],
        window_features    = [rolling, rolling_2],
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
        differentiation    = None
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    steps = 5
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    X_predict = forecaster.create_predict_X(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    )
    results = np.full(
        shape=(steps, len(levels)), fill_value=np.nan, order='F', dtype=float
    )
    for i, level in enumerate(levels):
        results[:, i] = transform_numpy(
            array             = forecaster.regressor.predict(X_predict[level]),
            transformer       = forecaster.transformer_series_.get(level, forecaster.transformer_series_['_unknown_level']),
            fit               = False,
            inverse_transform = True
        )

    expected = forecaster.predict(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    ).to_numpy()
    
    np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers_diff():
    """
    Test output ForecasterRecursiveMultiSeries create_predict_X matrix returns 
    the same predictions as predict method when passing to the regressor predict 
    method and transformers are used.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = [1, 5],
        # window_features    = [rolling, rolling_2],
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
        differentiation    = 1
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    steps = 5
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    X_predict = forecaster.create_predict_X(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    )
    results = np.full(
        shape=(steps, len(levels)), fill_value=np.nan, order='F', dtype=float
    )
    for i, level in enumerate(levels):
        results[:, i] = forecaster.regressor.predict(X_predict[level])
        results[:, i] = (
            forecaster
            .differentiator_[level]
            .inverse_transform_next_window(results[:, i])
        )
        results[:, i] = transform_numpy(
            array             = results[:, i],
            transformer       = forecaster.transformer_series_.get(level, forecaster.transformer_series_['_unknown_level']),
            fit               = False,
            inverse_transform = True
        )

    expected = forecaster.predict(
        steps=steps, levels=levels, last_window=last_window, exog=exog_dict_test
    ).to_numpy()
    
    np.testing.assert_array_almost_equal(results, expected, decimal=7)
