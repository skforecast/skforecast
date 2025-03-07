# Unit test predict_interval ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict
from .fixtures_forecaster_recursive_multiseries import expected_df_to_long_format

THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}
series_2 = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                         '2': pd.Series(np.arange(10))})


def test_check_interval_ValueError_when_method_is_not_valid_method():
    """
    Check ValueError is raised when `method` is not 'bootstrapping' or 'conformal'.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series_2)

    method = 'not_valid_method'
    err_msg = re.escape(
        f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, method=method)


@pytest.fixture(params=[('1', np.array([[10., 20., 20.]])), 
                        (['2'], np.array([[10., 30., 30.]])),
                        (['1', '2'], np.array([[10., 20., 20., 10., 30., 30.]]))
                        ],
                        ids=lambda d: f'levels: {d[0]}, preds: {d[1]}')
def expected_pandas_dataframe(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    levels = request.param[0]
    levels_names = [levels] if isinstance(levels, str) else levels

    column_names = []
    for level in levels_names:
        _ = [f'{level}', f'{level}_lower_bound', f'{level}_upper_bound']
        column_names.append(_)
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = np.concatenate(column_names),
                      index   = pd.RangeIndex(start=10, stop=11, step=1)
                  )
    expected_df = expected_df_to_long_format(expected_df, method='interval')

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_steps_is_1_in_sample_residuals_is_True_with_fixture(expected_pandas_dataframe):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10)
    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20)

    predictions = forecaster.predict_interval(
        steps=1, levels=expected_pandas_dataframe[0], method='bootstrapping',
        use_in_sample_residuals=True, use_binned_residuals=False, suppress_warnings=True
    )
    
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    data = np.array([[10., 20., 20.]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_1 = forecaster.predict_interval(
        steps=1, levels='1', method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    data = np.array([[10., 30., 30.]]),
                    columns = ['2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_2 = forecaster.predict_interval(
        steps=1, levels=['2'], method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected_3 = pd.DataFrame(
                    data = np.array([[10., 20., 20., 10., 30., 30.]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_3 = forecaster.predict_interval(
        steps=1, levels=None, method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = [expected_1, expected_2, expected_3]
    for i, df in enumerate(expected):
        expected[i] = expected_df_to_long_format(df, method='interval')

    pd.testing.assert_frame_equal(results_1, expected[0])
    pd.testing.assert_frame_equal(results_2, expected[1])
    pd.testing.assert_frame_equal(results_3, expected[2])


@pytest.fixture(params=[('1', np.array([[10., 20., 20.],
                                          [11., 24.33333333, 24.33333333]])), 
                        (['2'], np.array([[10., 30., 30.],
                                          [11., 37.66666666666667, 37.66666666666667]])),
                        (['1', '2'], np.array([[10., 20., 20., 10., 30., 30.],
                                               [11., 24.33333333, 24.33333333, 11., 37.66666666666667, 37.66666666666667]]))
                        ],
                        ids=lambda d: f'levels: {d[0]}, preds: {d[1]}')
def expected_pandas_dataframe_2(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    levels = request.param[0]
    levels_names = [levels] if isinstance(levels, str) else levels

    column_names = []
    for level in levels_names:
        _ = [f'{level}', f'{level}_lower_bound', f'{level}_upper_bound']
        column_names.append(_)
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = np.concatenate(column_names),
                      index   = pd.RangeIndex(start=10, stop=12, step=1)
                  )
    expected_df = expected_df_to_long_format(expected_df, method='interval')

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_steps_is_2_in_sample_residuals_is_True_with_fixture(expected_pandas_dataframe_2):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10)
    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20)

    predictions = forecaster.predict_interval(
        steps=2, levels=expected_pandas_dataframe_2[0], method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = expected_pandas_dataframe_2[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    np.array([[10., 20.        , 20.],
                              [11., 24.33333333, 24.33333333]
                             ]),
                    columns = ['1', '1_lower_bound', '1_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_1 = forecaster.predict_interval(
        steps=2, levels='1', method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    np.array([[10., 30.              , 30.],
                              [11., 37.66666666666667, 37.66666666666667]
                             ]),
                    columns = ['2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_2 = forecaster.predict_interval(
        steps=2, levels='2', method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected_3 = pd.DataFrame(
                    data = np.array([[10., 20.        , 20.        , 10., 30.              , 30.              ],
                                     [11., 24.33333333, 24.33333333, 11., 37.66666666666667, 37.66666666666667]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_3 = forecaster.predict_interval(
        steps=2, levels=['1', '2'], method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = [expected_1, expected_2, expected_3]
    for i, df in enumerate(expected):
        expected[i] = expected_df_to_long_format(df, method='interval')

    pd.testing.assert_frame_equal(results_1, expected[0])
    pd.testing.assert_frame_equal(results_2, expected[1])
    pd.testing.assert_frame_equal(results_3, expected[2])


def test_predict_output_when_regressor_is_LinearRegression_steps_is_1_in_sample_residuals_is_False_with_fixture(expected_pandas_dataframe):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        '1': np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10), 
        '2': np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20),
        '_unknown_level': np.full(shape=20, fill_value=10)
    }

    predictions = forecaster.predict_interval(
        steps=1, levels=expected_pandas_dataframe[0], method='bootstrapping', 
        use_in_sample_residuals=False, use_binned_residuals=False
    )
    
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_steps_is_2_in_sample_residuals_is_False_with_fixture(expected_pandas_dataframe_2):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        '1': np.full_like(forecaster.in_sample_residuals_['1'], fill_value=10), 
        '2': np.full_like(forecaster.in_sample_residuals_['2'], fill_value=20),
        '_unknown_level': np.full(shape=20, fill_value=10)
    }
    predictions = forecaster.predict_interval(
        steps=2, levels=expected_pandas_dataframe_2[0], method='bootstrapping', 
        use_in_sample_residuals=False, use_binned_residuals=False
    )
    expected = expected_pandas_dataframe_2[1]

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("interval", 
                         [0.90, [5, 95], (5, 95)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series(interval):
    """
    Test predict_interval output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(series=series)
    predictions = forecaster.predict_interval(
        steps=5, levels='1', method='bootstrapping', interval=interval,
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([[0.52791431, 0.16188966, 0.93871914],
                                    [0.44509712, 0.09277709, 0.86784697],
                                    [0.42176045, 0.03764527, 0.8287668 ],
                                    [0.48087237, 0.10818534, 0.90310365],
                                    [0.48268008, 0.15967278, 0.9054366 ]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound']
               )
    expected = expected_df_to_long_format(expected, method='interval')
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict_interval output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler(), '_unknown_level': StandardScaler()}
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(
        steps=5, levels=['1'], method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([[0.59619193, 0.18343628, 0.99595007],
                                    [0.46282914, 0.11901931, 0.87279397],
                                    [0.41738496, 0.04689358, 0.82876371],
                                    [0.48522676, 0.07637607, 0.89258319],
                                    [0.47525733, 0.1192259 , 0.90741363]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound']
               )
    expected = expected_df_to_long_format(expected, method='interval')
    
    pd.testing.assert_frame_equal(predictions, expected)


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
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=5, levels=['1', '2'], exog=exog_predict, method='bootstrapping', 
        use_in_sample_residuals=True, use_binned_residuals=False
    )
    
    expected = pd.DataFrame(
                   data = np.array([
                              [0.53267333, 0.14385946, 0.93785138, 0.55496412, 0.13040389, 0.86143235],
                              [0.44478046, 0.07076074, 0.8549911 , 0.57787982, 0.14137988, 0.945465  ],
                              [0.52579563, 0.11961453, 0.92964449, 0.66389117, 0.24814764, 1.00640437],
                              [0.57391142, 0.18111008, 0.97045191, 0.65789846, 0.22574071, 1.02971185],
                              [0.54633594, 0.21019872, 0.96136894, 0.5841187 , 0.12771706, 0.94734674]]
                          ),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound']
               )
    expected = expected_df_to_long_format(expected, method='interval')
    
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
        series=series_dict_train, exog=exog_dict_train, 
        store_in_sample_residuals=False, suppress_warnings=True
    )
    forecaster.set_in_sample_residuals(series=series_dict_train, exog=exog_dict_train)
    predictions = forecaster.predict_interval(
        steps=5, exog=exog_dict_test, method='bootstrapping', 
        interval=[5, 95], n_boot=10,  use_in_sample_residuals=True, 
        use_binned_residuals=False, suppress_warnings=True
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
    results = forecaster.predict_interval(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        method='bootstrapping', interval=[5, 95], n_boot=10, 
        use_in_sample_residuals=True, use_binned_residuals=False, suppress_warnings=True
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

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_series_and_exog_dict_unknown_level_binned_residuals():
    """
    Test output ForecasterRecursiveMultiSeries predict_interval method when series and 
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
    results = forecaster.predict_interval(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        method='bootstrapping', interval=[5, 95], n_boot=10, 
        use_in_sample_residuals=True, use_binned_residuals=True, suppress_warnings=True
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
                1280.691397121706,
                2705.9255940079975,
                2351.2169926457227,
                7456.30218140561,
                4518.09207378455,
                1338.5256579919173,
                2658.8038574432203,
                2010.1128086433266,
                8286.016756370773,
                4695.361794129111,
                1324.4177375758195,
                2750.3765780123113,
                1767.304578121676,
                8039.73491219328,
                3859.7437041937233,
                1299.5531237560697,
                2600.229892806877,
                1771.885509411404,
                8229.019039149467,
                3258.6866240603604,
                1210.2559241669333,
                2312.550433131647,
                1682.340227568467,
                8194.857096861739,
                4927.714549206342,
            ],
            "upper_bound": [
                1439.4553800244555,
                2738.156981275605,
                3741.8972784855855,
                8241.938477634612,
                6843.515887200728,
                1549.4628248207773,
                2935.3967892197093,
                3175.981247149604,
                9460.836936220336,
                8273.075290042703,
                1480.350727438289,
                2983.6422355070913,
                2579.8385939039645,
                9155.923184853164,
                6261.057395164305,
                1475.6791491359868,
                2876.822824583366,
                2408.3597533104735,
                9434.945472173118,
                8060.945697475116,
                1438.0084071244983,
                2896.1999567914872,
                2368.450760903573,
                9517.69363475937,
                9349.46495485382,
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
    results = forecaster.predict_interval(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        method='bootstrapping', interval=(5, 95), n_boot=10, 
        use_in_sample_residuals=True, use_binned_residuals=False, suppress_warnings=True
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

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_series_and_exog_dict_encoding_None_unknown_level_binned_residuals():
    """
    Test output ForecasterRecursiveMultiSeries predict_interval method when series and 
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
    results = forecaster.predict_interval(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2,
        method='bootstrapping', interval=(5, 95), n_boot=10, 
        use_in_sample_residuals=True, use_binned_residuals=True, suppress_warnings=True
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
                819.9337017726947,
                2852.086286493839,
                2185.68371683204,
                7067.312499958239,
                6320.866291742634,
                949.0080907131693,
                2724.6933488070995,
                2003.2928963661257,
                7823.917981568088,
                6941.5707091646045,
                848.4003232265973,
                2792.388424948016,
                2693.226527319288,
                7838.632252990579,
                6802.268112262388,
                927.6414047150448,
                2762.397454966053,
                2646.5312785810665,
                7751.7006663018965,
                6713.968070171806,
                891.4148355515429,
                2708.8445868510776,
                2564.509291435976,
                7752.98706209402,
                6748.509077723544,
            ],
            "upper_bound": [
                2455.8939978169306,
                3562.781729860123,
                4075.8215920217,
                9974.785265402967,
                8120.058251346585,
                2443.379060932464,
                4332.945765188091,
                4497.779920141797,
                11355.693731059055,
                8154.384764301869,
                2466.846650769211,
                4621.470221829929,
                4277.453581581279,
                11248.104970351964,
                8092.99775924399,
                2261.1891131225707,
                4601.085503786943,
                4423.109145176862,
                11186.082951284985,
                8264.17998610828,
                2138.779067144402,
                4625.038744331387,
                4725.130044655679,
                11094.120630362502,
                8244.660999361791,
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


@pytest.mark.parametrize("interval", 
                         [0.95, (2.5, 97.5)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_series_and_exog_dict(interval):
    """
    Test output ForecasterRecursiveMultiSeries predict_interval conformal method 
    when series and exog are dictionaries.
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
    results = forecaster.predict_interval(
        steps=5, exog=exog_dict_test, interval=interval, method='conformal',
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
