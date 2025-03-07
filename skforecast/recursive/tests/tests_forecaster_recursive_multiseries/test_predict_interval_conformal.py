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
    results = forecaster._predict_interval_conformal(
        steps=1, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

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
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

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
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

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
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

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


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test _predict_interval_conformal output when using LinearRegression as regressor, StandardScaler
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
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        steps=5, levels=['1', '2'], exog=exog_predict, nominal_coverage=0.95,
        use_in_sample_residuals=True, use_binned_residuals=False
    )
    
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


def test_predict_interval_conformal_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries _predict_interval_conformal method when series and 
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=True, suppress_warnings=True
    )
    results = forecaster._predict_interval_conformal(
        steps=5, exog=exog_dict_test, nominal_coverage=0.95, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
        data={
            "level": [
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
            ],
            "pred": [
                1438.141547171955,
                2090.793526125921,
                2166.9832932984505,
                7285.527814283447,
                1438.141547171955,
                2089.110388842724,
                2074.5599492877495,
                7488.18398743828,
                1438.141547171955,
                2089.110388842724,
                2035.994482471084,
                7488.18398743828,
                1403.9362565363836,
                2089.110388842724,
                2035.994482471084,
                7488.18398743828,
                1403.9362565363836,
                2089.110388842724,
                2035.994482471084,
                7488.18398743828,
            ],
            "lower_bound": [
                930.4222807518261,
                957.6735882668283,
                1500.578936864964,
                4188.404660735961,
                930.4222807518261,
                955.990450983632,
                1408.155592854263,
                4391.060833890795,
                930.4222807518261,
                955.990450983632,
                1369.5901260375977,
                4391.060833890795,
                896.2169901162547,
                955.990450983632,
                1369.5901260375977,
                4391.060833890795,
                896.2169901162547,
                955.990450983632,
                1369.5901260375977,
                4391.060833890795,
            ],
            "upper_bound": [
                1945.8608135920836,
                3223.9134639850135,
                2833.3876497319366,
                10382.650967830932,
                1945.8608135920836,
                3222.230326701817,
                2740.964305721236,
                10585.307140985766,
                1945.8608135920836,
                3222.230326701817,
                2702.3988389045703,
                10585.307140985766,
                1911.6555229565124,
                3222.230326701817,
                2702.3988389045703,
                10585.307140985766,
                1911.6555229565124,
                3222.230326701817,
                2702.3988389045703,
                10585.307140985766,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
            ]
        ),
    )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries _predict_interval_conformal method when series and 
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=True, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    results = forecaster._predict_interval_conformal(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
        data={
            "level": [
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
            ],
            "pred": [
                1330.5385359512072,
                2655.9525305801844,
                2645.090876890802,
                7897.5193849420975,
                4890.228408881278,
                1401.6308515684586,
                2503.7524796063676,
                2407.1752505407007,
                8577.098408560274,
                4756.810200061664,
                1387.265728819639,
                2446.2803866452205,
                2314.0860223813092,
                8619.983117285357,
                4947.440527174498,
                1310.8227594194243,
                2389.3764241040913,
                2245.0514974677576,
                8373.803343371283,
                4972.509186937418,
                1279.3727451240927,
                2185.0610428355444,
                2197.452881661525,
                8536.318209938987,
                5213.761246797846,
            ],
            "lower_bound": [
                1121.6681311586203,
                1892.4609387760195,
                2266.817051341049,
                6480.183045274484,
                2664.5123312578003,
                1192.7604467758715,
                1740.2608878022022,
                2028.9014249909478,
                7159.762068892662,
                2531.094122438187,
                1178.395324027052,
                1682.7887948410553,
                1935.8121968315563,
                7202.646777617744,
                2721.7244495510213,
                1101.9523546268374,
                1625.884832299926,
                1866.777671918005,
                6956.46700370367,
                2746.7931093139414,
                1070.5023403315058,
                1421.569451031379,
                1819.1790561117725,
                7118.981870271374,
                2988.045169174369,
            ],
            "upper_bound": [
                1539.4089407437941,
                3419.4441223843496,
                3023.3647024405545,
                9314.85572460971,
                7115.944486504754,
                1610.5012563610455,
                3267.244071410533,
                2785.4490760904537,
                9994.434748227888,
                6982.526277685141,
                1596.136133612226,
                3209.7719784493856,
                2692.359847931062,
                10037.31945695297,
                7173.156604797975,
                1519.6931642120112,
                3152.868015908257,
                2623.3253230175105,
                9791.139683038895,
                7198.225264560895,
                1488.2431499166796,
                2948.5526346397096,
                2575.726707211278,
                9953.6545496066,
                7439.477324421323,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
            ]
        ),
    )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_series_and_exog_dict_unknown_level_binned_residuals():
    """
    Test output ForecasterRecursiveMultiSeries _predict_interval_conformal method when series and 
    exog are dictionaries, unknown_level and binned_residuals.
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=True, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    results = forecaster._predict_interval_conformal(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
        data={
            "level": [
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
            ],
            "pred": [
                1330.5385359512072,
                2655.9525305801844,
                2645.090876890802,
                7897.5193849420975,
                4890.228408881278,
                1401.6308515684586,
                2503.7524796063676,
                2407.1752505407007,
                8577.098408560274,
                4756.810200061664,
                1387.265728819639,
                2446.2803866452205,
                2314.0860223813092,
                8619.983117285357,
                4947.440527174498,
                1310.8227594194243,
                2389.3764241040913,
                2245.0514974677576,
                8373.803343371283,
                4972.509186937418,
                1279.3727451240927,
                2185.0610428355444,
                2197.452881661525,
                8536.318209938987,
                5213.761246797846,
            ],
            "lower_bound": [
                1197.854135185738,
                2575.3596492481447,
                1588.3625904237701,
                7196.496574366325,
                3020.1032767294423,
                1296.2417306030477,
                2085.93781658185,
                2111.676283753576,
                7786.668390595367,
                2886.685067909829,
                1281.8766078542283,
                2028.465723620703,
                2107.2848247817155,
                7829.553099320449,
                3077.3153950226633,
                1178.138358653955,
                2157.885950586338,
                1836.1513462801986,
                7589.904515477171,
                3102.3840547855834,
                1004.3867556017174,
                2028.7130586061166,
                1752.6486289281029,
                7745.888191974078,
                1791.6062838083303,
            ],
            "upper_bound": [
                1463.2229367166765,
                2736.5454119122246,
                3701.8191633578335,
                8598.54219551787,
                6760.353541033112,
                1507.0199725338693,
                2921.567142630885,
                2702.6742173278253,
                9367.528426525183,
                6626.935332213499,
                1492.65484978505,
                2864.0950496697383,
                2520.887219980903,
                9410.413135250265,
                6817.565659326334,
                1443.5071601848936,
                2620.8668976218446,
                2653.9516486553166,
                9157.702171265395,
                6842.634319089253,
                1554.358734646468,
                2341.4090270649717,
                2642.2571343949476,
                9326.748227903894,
                8635.916209787363,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
            ]
        ),
    )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries _predict_interval_conformal method when series and 
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=True, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    results = forecaster._predict_interval_conformal(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
        data={
            "level": [
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
            ],
            "pred": [
                1261.9326553725798,
                3109.3677474297733,
                3565.438044069866,
                7581.012455102697,
                6929.605635844412,
                1312.2074981587975,
                3370.6327655712835,
                3486.849749467782,
                7877.71418945036,
                7226.307370192074,
                1269.6006117354937,
                3451.582141864614,
                3265.503087649355,
                7903.889983876083,
                7211.071456755368,
                1216.7129613241045,
                3420.9316258462322,
                3279.9374855113547,
                7895.699772623197,
                7260.909824740005,
                1199.8067190914946,
                3410.881341378549,
                3385.6645920219025,
                7915.945340057939,
                7281.155392174747,
            ],
            "lower_bound": [
                -458.30422144921704,
                1389.1308706079765,
                1845.201167248069,
                5860.775578280901,
                5209.368759022615,
                -2128.266255484796,
                -69.84098807230976,
                46.375995824188976,
                4437.240435806765,
                3785.83361654848,
                -3891.1100187298966,
                -1709.1284886007757,
                -1895.2075428160347,
                2743.179353410692,
                2050.360826289977,
                -5664.2345459630815,
                -3460.0158814409547,
                -3601.0100217758322,
                1014.7522653360097,
                379.9623174528183,
                -7401.377665017488,
                -5190.303042730435,
                -5215.519792087081,
                -685.2390440510458,
                -1320.0289919342372,
            ],
            "upper_bound": [
                2982.1695321943766,
                4829.6046242515695,
                5285.674920891663,
                9301.249331924495,
                8649.842512666208,
                4752.681251802391,
                6811.106519214877,
                6927.323503111376,
                11318.187943093952,
                10666.781123835668,
                6430.3112422008835,
                8612.292772330005,
                8426.213718114745,
                13064.600614341474,
                12371.78208722076,
                8097.660468611291,
                10301.87913313342,
                10160.884992798543,
                14776.647279910385,
                14141.857332027193,
                9800.991103200478,
                12012.065725487535,
                11986.848976130887,
                16517.129724166924,
                15882.339776283732,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
            ]
        ),
    )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_series_and_exog_dict_encoding_None_unknown_level_binned_residuals():
    """
    Test output ForecasterRecursiveMultiSeries _predict_interval_conformal method when series and 
    exog are dictionaries, encoding is None, unknown_level and binned_residuals.
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=True, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    results = forecaster._predict_interval_conformal(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
        data={
            "level": [
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
                "id_1000",
                "id_1001",
                "id_1003",
                "id_1004",
                "id_1005",
            ],
            "pred": [
                1261.9326553725798,
                3109.3677474297733,
                3565.438044069866,
                7581.012455102697,
                6929.605635844412,
                1312.2074981587975,
                3370.6327655712835,
                3486.849749467782,
                7877.71418945036,
                7226.307370192074,
                1269.6006117354937,
                3451.582141864614,
                3265.503087649355,
                7903.889983876083,
                7211.071456755368,
                1216.7129613241045,
                3420.9316258462322,
                3279.9374855113547,
                7895.699772623197,
                7260.909824740005,
                1199.8067190914946,
                3410.881341378549,
                3385.6645920219025,
                7915.945340057939,
                7281.155392174747,
            ],
            "lower_bound": [
                -98.64486847220087,
                2548.8495539617647,
                2677.877796699927,
                4708.300372249577,
                4056.8935529912915,
                -785.7588306649855,
                1449.5370482584945,
                1711.7292547279053,
                3644.424582752458,
                2993.0177634941733,
                -1371.2502775914295,
                969.9682310838166,
                907.0743466377785,
                2933.211572199178,
                2618.2910190880207,
                -1967.0224885059574,
                396.4331545622954,
                184.11993952077592,
                2565.5305299768465,
                1930.7405820936567,
                -2343.4195617080136,
                26.89203912516632,
                -270.67114743668526,
                1848.3872924325863,
                1213.5973445493962,
            ],
            "upper_bound": [
                2622.5101792173605,
                3669.8859408977814,
                4452.998291439804,
                10453.724537955817,
                9802.317718697532,
                3410.1738269825805,
                5291.728482884073,
                5261.97024420766,
                12111.003796148261,
                11459.596976889974,
                3910.4515010624164,
                5933.196052645412,
                5623.931828660932,
                12874.568395552986,
                11803.851894422714,
                4400.448411154167,
                6445.430097130169,
                6375.755031501934,
                13225.869015269547,
                12591.079067386356,
                4743.032999891004,
                6794.870643631933,
                7042.000331480489,
                13983.503387683291,
                13348.7134398001,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-01",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-02",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-03",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-04",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
                "2016-08-05",
            ]
        ),
    )

    pd.testing.assert_frame_equal(results, expected)
