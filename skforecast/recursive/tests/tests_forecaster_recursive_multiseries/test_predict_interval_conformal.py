# Unit test _predict_interval_conformal ForecasterRecursiveMultiSeries
# ==============================================================================
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series, exog, exog_predict

THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}
series_2 = pd.DataFrame({
    'l1': pd.Series(np.arange(10)),
    'l2': pd.Series(np.arange(10))
})


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series_2)
    forecaster.in_sample_residuals_ = {
        'l1': np.array([10] * 10),
        'l2': np.array([20] * 10),
        '_unknown_level': np.array([20] * 10)
    }
    results = forecaster._predict_interval_conformal(steps=1, use_in_sample_residuals=True)

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [10., -10., 30.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.Index([10, 10])
               )
    expected.insert(0, 'level', np.array(['l1', 'l2']))
    
    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series_2)
    forecaster.in_sample_residuals_ = {
        'l1': np.array([10] * 10),
        'l2': np.array([20] * 10),
        '_unknown_level': np.array([20] * 10)
    }
    results = forecaster._predict_interval_conformal(steps=2, use_in_sample_residuals=True)

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [10., -10., 30.],
                                       [11., 1., 21.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.Index([10, 10, 11, 11])
               )
    expected.insert(0, 'level', np.array(['l1', 'l2', 'l1', 'l2']))

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series_2)
    forecaster.out_sample_residuals_ = {
        'l1': np.array([10] * 10),
        'l2': np.array([20] * 10),
        '_unknown_level': np.array([20] * 10)
    }
    results = forecaster._predict_interval_conformal(steps=2, use_in_sample_residuals=False)

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [10., -10., 30.],
                                       [11., 1., 21.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.Index([10, 10, 11, 11])
               )
    expected.insert(0, 'level', np.array(['l1', 'l2', 'l1', 'l2']))

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test _predict_interval_conformal output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster._predict_interval_conformal(steps=5)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.52791431,  0.0809377 ,  0.97489092],
                              [0.52235108,  0.0802589 ,  0.96444326],
                              [0.44509712, -0.00187949,  0.89207373],
                              [0.58157238,  0.1394802 ,  1.02366456],
                              [0.42176045, -0.02521616,  0.86873706],
                              [0.55987796,  0.11778578,  1.00197014],
                              [0.48087237,  0.03389576,  0.92784898],
                              [0.56344784,  0.12135566,  1.00554002],
                              [0.48268008,  0.03570347,  0.92965669],
                              [0.52752391,  0.08543173,  0.96961608]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.Index([50, 50, 51, 51, 52, 52, 53, 53, 54, 54])
               )
    expected.insert(0, 'level', np.array(['1', '2'] * 5))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test predict_interval output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog.
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
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster._predict_interval_conformal(steps=5, levels=['1', '2'], exog=exog_predict)
    
    expected = pd.DataFrame(
                   data = np.array([
                              [0.53267333, 0.11562736, 0.9497193 ],
                              [0.55496412, 0.09211358, 1.01781466],
                              [0.44478046, 0.02773449, 0.86182643],
                              [0.57787982, 0.11502928, 1.04073036],
                              [0.52579563, 0.10874966, 0.9428416 ],
                              [0.66389117, 0.20104063, 1.12674171],
                              [0.57391142, 0.15686545, 0.99095739],
                              [0.65789846, 0.19504792, 1.12074899],
                              [0.54633594, 0.12928997, 0.96338191],
                              [0.5841187 , 0.12126817, 1.04696924]]
                          ),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.Index([50, 50, 51, 51, 52, 52, 53, 53, 54, 54])
               )
    expected.insert(0, 'level', np.array(['1', '2'] * 5))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries predict_interval method when series and 
    exog are dictionaries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    predictions = forecaster._predict_interval_conformal(
        steps=5, exog=exog_dict_test, suppress_warnings=True, n_boot=10, interval=[5, 95]
    )

    expected = pd.DataFrame(
        data=np.array([
            [1438.14154717, 1151.25387091, 1834.64267811, 2090.79352613,
             1352.80566771, 2850.70364521, 2166.9832933 , 1915.71031656,
             2538.88007772, 7285.52781428, 5289.04209733, 8604.79589441],
            [1438.14154717,  754.44282535, 1708.80983557, 2089.11038884,
             933.86328276, 2878.42686855, 2074.55994929, 1491.6401476 ,
             2447.16553138, 7488.18398744, 5588.65514279, 9175.53503076],
            [1438.14154717, 1393.20049911, 1786.2144554 , 2089.11038884,
             934.76328276, 2852.8076229 , 2035.99448247, 1478.62225697,
             2202.14689944, 7488.18398744, 5330.31083389, 9272.09463487],
            [1403.93625654, 1097.60654368, 1655.32766412, 2089.11038884,
             975.32975784, 2909.31686272, 2035.99448247, 1530.74471247,
             2284.17651415, 7488.18398744, 4379.32842459, 8725.84690603],
            [1403.93625654, 1271.81039554, 1712.6314556 , 2089.11038884,
             900.67253042, 2775.32496318, 2035.99448247, 1352.22100994,
             2023.10287794, 7488.18398744, 4576.01555597, 9477.77316645]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound'
        ]
    )
    expected = expected_df_to_long_format(expected, method='interval')

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries predict_interval method when series and 
    exog are dictionaries and unknown_level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     lags               = 14,
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
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    predictions = forecaster.predict_interval(
        steps=5, levels=levels, exog=exog_dict_test_2, last_window=last_window,
        suppress_warnings=True, n_boot=10, interval=[5, 95]
    )

    expected = pd.DataFrame(
        data=np.array([
            [1330.53853595,  1193.18439697,  1520.46886991,  2655.95253058,
             2117.99790867,  3164.46774918,  2645.09087689,  2413.68255828,
             2714.93144634,  7897.51938494,  6897.54816773,  8408.75899232,
             4890.22840888,  4032.76806221,  5839.5001305 ],
            [1401.63085157,  1102.76792249,  1496.73904158,  2503.75247961,
             1696.30950851,  3054.85705656,  2407.17525054,  2103.19159494,
             2551.12420784,  8577.09840856,  7649.53221732,  9121.38815062,
             4756.81020006,  4196.95852991,  6296.71335467],
            [1387.26572882,  1283.49595363,  1535.34888996,  2446.28038665,
             1379.74372794,  3132.71196912,  2314.08602238,  1824.59807565,
             2319.72840767,  8619.98311729,  7643.13451622, 10237.69172218,
             4947.44052717,  2995.3670943 ,  5639.58423386],
            [1310.82275942,  1263.98299475,  1418.08104408,  2389.3764241 ,
             1665.80863511,  3283.65732497,  2245.05149747,  1690.30171463,
             2286.47286188,  8373.80334337,  7925.08454873,  9170.19662943,
             4972.50918694,  3854.22592844,  6543.05424315],
            [1279.37274512,  1166.68391264,  1336.93180134,  2185.06104284,
             1363.74911381,  2889.04815824,  2197.45288166,  1495.02913524,
             2195.10669302,  8536.31820994,  7008.65077106,  9540.86051966,
             5213.7612468 ,  4296.64990694,  6414.60074985]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound',
            'id_1005',
            'id_1005_lower_bound',
            'id_1005_upper_bound'
        ]
    )
    expected = expected_df_to_long_format(expected, method='interval')

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries predict_interval method when series and 
    exog are dictionaries, encoding is None and unknown_level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     lags               = 14,
                     encoding           = None,
                     dropna_from_series = False,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = 1
                 )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    predictions = forecaster.predict_interval(
        steps=5, levels=levels, exog=exog_dict_test_2, last_window=last_window,
        suppress_warnings=True, n_boot=10, interval=(5, 95)
    )
    
    expected = pd.DataFrame(
        data=np.array([
            [1261.93265537,  -54.92223394, 1663.74382259, 3109.36774743,
             2834.05294715, 3433.86345449, 3565.43804407, 2010.48988467,
             6228.58127953, 7581.0124551 , 6286.22433874, 8126.77691062,
             6929.60563584, 5701.61445078, 7461.52760124],
            [1312.20749816,  846.82193474, 1979.47123353, 3370.63276557,
             2829.80711162, 4181.07740596, 3486.84974947, 2355.89401381,
             6040.01056807, 7877.71418945, 6949.09892306, 8598.60429504,
             7226.30737019, 5831.8370949 , 8032.64993796],
            [1269.60061174,  533.03651126, 2202.6484267 , 3451.58214186,
             2695.91254302, 4378.96352476, 3265.50308765,  993.09722549,
             5076.21524805, 7903.88998388, 6987.79918911, 8618.47845871,
             7211.07145676, 5560.79494367, 8245.41102809],
            [1216.71296132,  708.30263058, 3257.15272095, 3420.93162585,
             2573.32473893, 4242.48762932, 3279.93748551, 1749.1552736 ,
             6080.64740059, 7895.69977262, 7141.16482696, 8777.31500381,
             7260.90982474, 5196.83217466, 8943.71308608],
            [1199.80671909, 1068.148691  , 3785.98433563, 3410.88134138,
             2669.37848396, 5273.44882377, 3385.66459202, 1843.03033853,
             6001.13573768, 7915.94534006, 7115.27808438, 9998.45490231,
             7281.15539217, 5039.18009578, 8656.83019612]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound',
            'id_1005',
            'id_1005_lower_bound',
            'id_1005_upper_bound'
        ]
    )
    expected = expected_df_to_long_format(expected, method='interval')

    pd.testing.assert_frame_equal(predictions, expected)
