# Unit test _create_predict_inputs ForecasterRecursiveMultiSeries
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

from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict

THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')

for k in series_dict.keys():
    series_dict[k].index.name = None
for k in exog_dict.keys():
    exog_dict[k].index.name = None

end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}

series_2 = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50, dtype=float)), 
                         '2': pd.Series(np.arange(start=50, stop=100, dtype=float))})


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster._create_predict_inputs(steps=5)


def test_output_create_predict_inputs_when_regressor_is_LinearRegression():
    """
    Test output _create_predict_inputs when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.fit(series=series_2)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        pd.DataFrame(
            {'1': np.array([45., 46., 47., 48., 49.]),
             '2': np.array([95., 96., 97., 98., 99.])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        None,
        ['1', '2'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test _create_predict_inputs output when using LinearRegression as regressor 
    and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=5, levels='1')
    
    expected = (
        pd.DataFrame(
            {'1': np.array([
                      -1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                       0.07582436346198025, 0.4776288428854555])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        None,
        ['1'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(transformer_series):
    """
    Test _create_predict_inputs output when using LinearRegression as regressor, 
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
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster._create_predict_inputs(steps=5, levels='1', exog=exog_predict)
    
    expected = (
        pd.DataFrame(
            {'1': np.array([
                      -1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                       0.07582436346198025, 0.4776288428854555])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[-0.09362908,  1., 0.]]),
         2: np.array([[0.45144522,  1., 0.]]),
         3: np.array([[-1.53968887,  1., 0.]]),
         4: np.array([[-1.45096055,  1., 0.]]),
         5: np.array([[-0.77240468,  1., 0.]])
        },
        ['1'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test _create_predict_inputs output when using LinearRegression as regressor, StandardScaler
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
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    
    expected = (
        pd.DataFrame(
            {'1': np.array([-1.08141835, -0.08097053, 2.08066403, 0.07582436, 0.47762884]),
             '2': np.array([0.73973379, -1.29936753, -0.36483376, -0.96090271, -0.57388468])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[-0.09575486,  1., 0.],
                      [-0.09575486,  1., 0.]]),
         2: np.array([[0.44024531,  1., 0.],
                      [0.44024531,  1., 0.]]),
         3: np.array([[-1.51774136,  1., 0.],
                      [-1.51774136,  1., 0.]]),
         4: np.array([[-1.43049014,  1., 0.],
                      [-1.43049014,  1., 0.]]),
         5: np.array([[-0.76323054,  1., 0.],
                      [-0.76323054,  1., 0.]])
        },
        ['1', '2'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog['exog_1'],
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
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
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    
    expected = (
        pd.DataFrame(
            {'1': np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
             '2': np.array([0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[0, 0, 0.51312815],
                      [0, 0, 0.51312815]]),
         2: np.array([[1, 1, 0.66662455],
                      [1, 1, 0.66662455]]),
         3: np.array([[2, 2, 0.10590849],
                      [2, 2, 0.10590849]]),
         4: np.array([[3, 3, 0.13089495],
                      [3, 3, 0.13089495]]),
         5: np.array([[4, 4, 0.32198061],
                      [4, 4, 0.32198061]]),
         6: np.array([[0, 0, 0.66156434],
                      [0, 0, 0.66156434]]),
         7: np.array([[1, 1, 0.84650623],
                      [1, 1, 0.84650623]]),
         8: np.array([[2, 2, 0.55325734],
                      [2, 2, 0.55325734]]),
         9: np.array([[3, 3, 0.85445249],
                      [3, 3, 0.85445249]]),
         10: np.array([[4, 4, 0.38483781],
                       [4, 4, 0.38483781]])

        },
        ['1', '2'],
        pd.RangeIndex(start=50, stop=60, step=1),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
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
    results = forecaster._create_predict_inputs(steps=5, exog=exog_dict_test)
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.3525861, -0.457091, -0.49618465, -1.07810218, -1.77580056]),
             'id_1001': np.array([0.21800529, 0.36936645, 0.67896814, 1.16332175, 1.1523137]),
             'id_1003': np.array([-0.62651976, -0.74685959, -1.03823091, -0.37837738, 3.39980134]),
             'id_1004': np.array([0.61495753, 0.8322971, 0.6719899, -0.24309812, -0.75073762])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633]])
        },
        ['id_1000', 'id_1001', 'id_1003', 'id_1004'],
        pd.date_range(start='2016-08-01', periods=5, freq='D'),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_series():
    """
    Test _create_predict_inputs output when using LinearRegression as regressor and differentiation=1,
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
                     lags               = 15, 
                     transformer_series = StandardScaler(),    
                     differentiation    = 1
                 )
    forecaster.fit(series=series_dict_datetime, exog=exog_dict_datetime)
    results = forecaster._create_predict_inputs(
        steps=steps, exog=exog_pred, predict_boot=True
    )
    
    expected = (
        pd.DataFrame(
            {'1': np.array([
                      np.nan, 1.53099926e+00, -2.50175863e+00, -3.17051107e-02,
                      1.60372524e+00, 1.24398054e-03, 4.61947999e-01, 9.68409847e-01,
                      -5.62842287e-01, -5.10849917e-01, 5.01788973e-01, -1.79911269e+00,
                      1.74873121e-01, -6.01343264e-01, 2.94763400e-01, 1.51888875e+00]
                  ),
             '2': np.array([
                      np.nan, -2.93045496, 2.6268138, -2.00736345, -0.19086106,
                      1.46005589, -1.8403251, 3.04806289, -0.99623011, 0.3712246,
                      -2.73026424, 2.23235938, -0.14532345, -1.53568303, -0.02273313,
                      2.09399597]
                  )},
            index = pd.date_range(start='2003-01-15', periods=16, freq='D')
        ),
        {1: np.array([[-0.78916779],
                      [-0.78916779]]),
         2: np.array([[0.07769117],
                      [0.07769117]]),
         3: np.array([[0.27283009],
                      [0.27283009]]),
         4: np.array([[0.35075195],
                      [0.35075195]]),
         5: np.array([[0.31597336],
                      [0.31597336]]),
         6: np.array([[0.14091339],
                      [0.14091339]]),
         7: np.array([[1.28618144],
                      [1.28618144]]),
         8: np.array([[0.09164865],
                      [0.09164865]]),
         9: np.array([[-0.50744682],
                      [-0.50744682]]),
         10: np.array([[0.01522573],
                       [0.01522573]]),
         11: np.array([[0.82813767],
                       [0.82813767]]),
         12: np.array([[0.15290495],
                       [0.15290495]]),
         13: np.array([[0.61532804],
                       [0.61532804]]),
         14: np.array([[-0.06287136],
                       [-0.06287136]]),
         15: np.array([[-1.1896156],
                       [-1.1896156]]),
         16: np.array([[0.28674823],
                       [0.28674823]]),
         17: np.array([[-0.64581528],
                       [-0.64581528]]),
         18: np.array([[0.20879998],
                       [0.20879998]]),
         19: np.array([[1.80029302],
                       [1.80029302]]),
         20: np.array([[0.14269745],
                       [0.14269745]])
        },
        ['1', '2'],
        pd.date_range(start='2003-01-31', periods=steps, freq='D'),
        {'1': np.array([0.11250873, -0.45215401,  0.50931909, -1.2515849 ,  0.32346492,
                        -0.16591436,  0.61386783,  0.49003355,  0.32481062, -0.050827  ,
                        -0.46246282, -0.20198051, -0.22783566,  0.43875451]),
         '2': np.array([-0.11905336, -0.96448195,  0.20875269,  0.7334418 , -0.45287812,
                        0.20410656,  0.27482526,  0.12415931, -1.10570249, -0.08341689,
                        0.83327134,  0.37868888, -0.69352672,  0.66181368])}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    for k in expected[4].keys():
        np.testing.assert_array_almost_equal(results[4][k], expected[4][k])


def test_create_predict_inputs_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
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
        transformer_exog   = StandardScaler(),
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
    results = forecaster._create_predict_inputs(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2
    )
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.3525861, -0.457091, -0.49618465, -1.07810218, -1.77580056]),
             'id_1001': np.array([0.21800529, 0.36936645, 0.67896814, 1.16332175, 1.1523137]),
             'id_1003': np.array([-0.62651976, -0.74685959, -1.03823091, -0.37837738, 3.39980134]),
             'id_1004': np.array([0.61495753, 0.8322971, 0.6719899, -0.24309812, -0.75073762]),
             'id_1005': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633]])
        },
        ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005'],
        pd.date_range(start='2016-08-01', periods=5, freq='D'),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_series_and_exog_dict_unknown_level_encoding_None():
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
    series and exog are dictionaries and unknown level with encoding=None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    results = forecaster._create_predict_inputs(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test
    )
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.77768307, -0.79277912, -0.79842631, -0.88248605, -0.98327068]),
             'id_1001': np.array([-0.39572285, -0.34793695, -0.25019327, -0.09727905, -0.10075438]),
             'id_1003': np.array([-0.56650418, -0.59023234, -0.64768383, -0.51757641,  0.22739054]),
             'id_1004': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343]),
             'id_1005': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [np.nan, np.nan, np.nan, np.nan]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [np.nan, np.nan, np.nan, np.nan]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [np.nan, np.nan, np.nan, np.nan]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [np.nan, np.nan, np.nan, np.nan]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [np.nan, np.nan, np.nan, np.nan]])
        },
        ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005'],
        pd.date_range(start='2016-08-01', periods=5, freq='D'),
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]