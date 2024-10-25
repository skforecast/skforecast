# Unit test predict ForecasterRecursiveMultiSeries
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
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

series_2 = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                         '2': pd.Series(np.arange(start=50, stop=100))})


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_IgnoredArgumentWarning_when_not_available_self_last_window_for_some_levels():
    """
    Test IgnoredArgumentWarning is raised when last_window is not available for 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2, store_last_window=['1'])

    warn_msg = re.escape(
        ("Levels {'2'} are excluded from prediction "
         "since they were not stored in `last_window_` attribute "
         "during training. If you don't want to retrain the "
         "Forecaster, provide `last_window` as argument.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        predictions = forecaster.predict(steps=5, levels=['1', '2'], last_window=None)

    expected = pd.DataFrame(
                   data    = np.array([50., 51., 52., 53., 54.]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("store_last_window",
                         [['1'], False],
                         ids=lambda slw: f"store_last_window: {slw}")
def test_predict_ValueError_when_not_available_self_last_window_for_levels(store_last_window):
    """
    Test ValueError is raised when last_window is not available for all 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2, store_last_window=store_last_window)

    err_msg = re.escape(
        ("No series to predict. None of the series {'2'} are present in "
         "`last_window_` attribute. Provide `last_window` as argument "
         "in predict method.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(steps=5, levels=['2'], last_window=None)


def test_predict_IgnoredArgumentWarning_when_levels_is_list_and_different_last_index_in_self_last_window_DatetimeIndex():
    """
    Test IgnoredArgumentWarning is raised when levels is a list and have 
    different last index in last_window attribute using a DatetimeIndex.
    """
    series_3 = {
        '1': series_2['1'].copy(),
        '2': series_2['2'].iloc[:30].copy(),
    }
    series_3['1'].index = pd.date_range(start='2020-01-01', periods=50)
    series_3['2'].index = pd.date_range(start='2020-01-01', periods=30)
    
    exog_2 = {
        '1': exog['exog_1'].copy(),
        '2': exog['exog_1'].iloc[:30].copy()
    }
    exog_2['1'].index = pd.date_range(start='2020-01-01', periods=50)
    exog_2['2'].index = pd.date_range(start='2020-01-01', periods=30)
    exog_2_pred = {
        '1': exog_predict['exog_1'].copy()
    }
    exog_2_pred['1'].index = pd.date_range(start='2020-02-20', periods=50)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_3, exog=exog_2)

    warn_msg = re.escape(
        ("Only series whose last window ends at the same index "
         "can be predicted together. Series that do not reach the "
         "maximum index, '2020-02-19 00:00:00', are excluded "
         "from prediction: {'2'}.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        predictions = forecaster.predict(steps=5, levels=['1', '2'], last_window=None,
                                         exog = exog_2_pred,)

    expected = pd.DataFrame(
                   data    = np.array([50., 51., 52., 53., 54.]),
                   index   = pd.date_range(start='2020-02-20', periods=5),
                   columns = ['1']
               )

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.fixture(params=[('1'  , [50., 51., 52., 53., 54.]), 
                        (['2'], [100., 101., 102., 103., 104.]),
                        (['1', '2'], [[50., 100.],
                                      [51., 101.],
                                      [52., 102.],
                                      [53., 103.],
                                      [54., 104.]])
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
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = levels_names,
                      index   = pd.RangeIndex(start=50, stop=55, step=1)
                  )

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_with_fixture(expected_pandas_dataframe):
    """
    Test predict output when using LinearRegression as regressor with pytest fixture.
    This test is equivalent to the next one.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions = forecaster.predict(steps=5, levels=expected_pandas_dataframe[0])

    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression():
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions_1 = forecaster.predict(steps=5, levels='1')
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'])
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=None)
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_last_window():
    """
    Test predict output when using LinearRegression as regressor and last_window.
    """
    last_window = pd.DataFrame(
                      {'1': [45, 46, 47, 48, 49], 
                       '2': [95, 96, 97, 98, 99], 
                       '3': [1, 2, 3, 4, 5]}, 
                      index = pd.RangeIndex(start=45, stop=50, step=1)
                  )

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_2)
    predictions_1 = forecaster.predict(steps=5, levels='1', last_window=last_window)
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'], last_window=last_window)
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=['1', '2'], last_window=last_window)
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, levels='1')

    expected = pd.DataFrame(
                   data    = np.array([0.52791431, 0.44509712, 0.42176045, 0.48087237, 0.48268008]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler(), '_unknown_level': StandardScaler()}
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, levels=['1'], suppress_warnings=True)

    expected = pd.DataFrame(
                   data    = np.array([0.59619193, 0.46282914, 0.41738496, 0.48522676, 0.47525733]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(transformer_series):
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
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
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=5, levels='1', exog=exog_predict)

    expected = pd.DataFrame(
                   data    = np.array([0.53267333, 0.44478046, 0.52579563, 0.57391142, 0.54633594]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
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
    predictions = forecaster.predict(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
                   data    = np.array([[0.52343425, 0.49560898],
                                       [0.42697026, 0.53289813],
                                       [0.48906322, 0.61304224],
                                       [0.55395581, 0.61509108],
                                       [0.52349107, 0.54297301]]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '2']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test predict output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
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
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.55674244, 0.54564144],
                                    [0.39860559, 0.53774558],
                                    [0.53719969, 0.63668042],
                                    [0.7759386 , 0.66871809],
                                    [0.56654221, 0.80858451],
                                    [0.43373393, 0.39998025],
                                    [0.49776999, 0.64896715],
                                    [0.47797067, 0.43578494],
                                    [0.49149785, 0.37087107],
                                    [0.68189581, 0.52863857]]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1', '2']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor():
    """
    Test predict output when using LGBMRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
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
                     regressor          = LGBMRegressor(random_state=123),
                     lags               = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.6211422 , 0.73248662],
                                    [0.42283865, 0.45391191],
                                    [0.49351412, 0.661996  ],
                                    [0.71846231, 0.58080355],
                                    [0.55089719, 0.59780378],
                                    [0.39224631, 0.40316854],
                                    [0.46827996, 0.51400857],
                                    [0.67707084, 0.42834292],
                                    [0.45119292, 0.4503941 ],
                                    [0.61998977, 0.65552498]]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1', '2']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor_auto():
    """
    Test predict output when using LGBMRegressor and categorical variables with 
    categorical_features='auto'.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    pipeline_categorical = make_pipeline(
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               FunctionTransformer(
                                   func=lambda x: x.astype('category'),
                                   feature_names_out= 'one-to-one'
                               )
                           )
    transformer_exog = make_column_transformer(
                            (
                                pipeline_categorical,
                                make_column_selector(dtype_exclude=np.number)
                            ),
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(random_state=123),
                     lags               = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.6211422 , 0.73248662],
                                    [0.42283865, 0.45391191],
                                    [0.49351412, 0.661996  ],
                                    [0.71846231, 0.58080355],
                                    [0.55089719, 0.59780378],
                                    [0.39224631, 0.40316854],
                                    [0.46827996, 0.51400857],
                                    [0.67707084, 0.42834292],
                                    [0.45119292, 0.4503941 ],
                                    [0.61998977, 0.65552498]]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1', '2']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_window_features():
    """
    Test output of predict when regressor is LGBMRegressor and window features.
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
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.62360244, 0.72875212],
                              [0.47067417, 0.29193681],
                              [0.56289914, 0.60510683],
                              [0.72004167, 0.74123332],
                              [0.51581126, 0.54273085]]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '2']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries predict method when series and 
    exog are dictionaries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=2, random_state=123, verbose=-1, max_depth=2
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
    predictions = forecaster.predict(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )
    expected = pd.DataFrame(
        data=np.array(
            [
                [1438.14154717, 2090.79352613, 2166.9832933, 7285.52781428],
                [1438.14154717, 2089.11038884, 2074.55994929, 7488.18398744],
                [1438.14154717, 2089.11038884, 2035.99448247, 7488.18398744],
                [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
                [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
            ]
        ),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=["id_1000", "id_1001", "id_1003", "id_1004"],
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1():
    """
    Test predict output when using LinearRegression as regressor and differentiation=1.
    """
    end_train = 30
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    series_1_diff = diferenciator.fit_transform(series['1'].to_numpy())
    series_2_diff = diferenciator.fit_transform(series['2'].to_numpy())
    data_diff = pd.DataFrame(
        {'1': series_1_diff, '2': series_2_diff}
    ).dropna()
    data_diff.index = series.index[1:]

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(series)), index=series.index, name='exog'
    )
    exog_diff = exog.iloc[1:]

    steps = len(data_diff.loc[end_train + 1:])

    forecaster_1 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15,
                                                transformer_series=None)
    forecaster_1.fit(series=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_diff.loc[end_train + 1:])
    # Revert the differentiation
    last_value_train = series.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15, 
                                                differentiation=1, transformer_series=None)
    forecaster_2.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train + 1:])

    pd.testing.assert_frame_equal(predictions_1, predictions_2)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_series():
    """
    Test predict output when using LinearRegression as regressor and differentiation=1,
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

    scaler_dict = {
        "1": StandardScaler(),
        "2": StandardScaler()
    }

    series_dict_diff = {}
    df_last_value_train = pd.DataFrame()
    for k, v in series_dict_datetime.items():

        scaler_dict[k].fit(v.to_numpy().reshape(-1, 1))
        data_scaled = scaler_dict[k].transform(v.copy().to_numpy().reshape(-1, 1))
        data_scaled = pd.Series(data_scaled.ravel(), index=v.index)
        last_value_train = data_scaled.iloc[[-1]]
        df_last_value_train[k] = last_value_train

        data_scaled_diff = TimeSeriesDifferentiator(order=1).fit_transform(data_scaled.to_numpy())
        data_scaled_diff = pd.Series(data_scaled_diff).dropna()
        data_scaled_diff.index = v.index[1:]
        series_dict_diff[k] = data_scaled_diff
    
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
    exog_diff = {
        '1': exog_dict_datetime['1'].iloc[1:],
        '2': exog_dict_datetime['2'].iloc[1:]
    }
    exog_pred = {
        '1': exog.loc[end_train:],
        '2': exog.loc[end_train:]
    }

    steps = len(series_datetime.loc[end_train:])

    forecaster_1 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15,
                                                transformer_series = None)
    forecaster_1.fit(series=series_dict_diff, exog=exog_diff)
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_pred)
    # Revert the differentiation
    predictions_1 = pd.concat([df_last_value_train, predictions_diff]).cumsum()[1:]
    # Revert the scaling
    for k in scaler_dict.keys():
        predictions_1[k] = scaler_dict[k].inverse_transform(predictions_1[k].to_numpy().reshape(-1, 1))
    
    forecaster_2 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15, 
                                                differentiation=1, transformer_series = StandardScaler())
    forecaster_2.fit(series=series_dict_datetime, exog=exog_dict_datetime)
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog_pred)

    pd.testing.assert_frame_equal(predictions_1.asfreq('D'), predictions_2)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_2():
    """
    Test predict output when using LinearRegression as regressor and differentiation=2.
    """
    series_dt = series.copy()
    series_dt.index = pd.date_range(start='2003-01-01', periods=len(series), freq='D')

    # Data differentiated
    df_diff_1 = pd.DataFrame()
    df_diff_2 = pd.DataFrame()
    for col in series_dt.columns:

        diferenciator_1 = TimeSeriesDifferentiator(order=1)
        diferenciator_2 = TimeSeriesDifferentiator(order=2)

        data_diff_1 = diferenciator_1.fit_transform(series_dt[col].to_numpy())
        data_diff_1 = pd.Series(data_diff_1).dropna()
        df_diff_1[col] = data_diff_1

        data_diff_2 = diferenciator_2.fit_transform(series_dt[col].to_numpy())
        data_diff_2 = pd.Series(data_diff_2).dropna()
        df_diff_2[col] = data_diff_2

    df_diff_1.index = series_dt.index[1:]
    df_diff_2.index = series_dt.index[2:]

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(series_dt)), index=series_dt.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]

    end_train = '2003-01-30 23:59:00'
    steps = len(series_dt.loc[end_train:])

    forecaster_1 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15,
                                                transformer_series=None)
    forecaster_1.fit(series=df_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(steps=steps, exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = df_diff_1.loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = series_dt.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterRecursiveMultiSeries(regressor=LinearRegression(), lags=15, 
                                                differentiation=2, transformer_series=None)
    forecaster_2.fit(series=series_dt.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_frame_equal(predictions_1, predictions_2)


def test_predict_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries predict method when series and 
    exog are dictionaries and encoding is None and unknown level with no exog.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     lags               = [1, 7, 14],
                     encoding           = None,
                     dropna_from_series = False,
                     differentiation    = 1,
                     transformer_series = None,
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
    predictions = forecaster.predict(
        steps=5, levels=levels, last_window=last_window,
        exog=exog_dict_test, suppress_warnings=True
    )

    expected = pd.DataFrame(
        data=np.array([
            [1351.44874045, 3267.13419659, 3843.06135374, 7220.15909652, 5776.91782212],
            [1411.73786344, 3537.21728977, 3823.777024  , 7541.54119992, 6083.61461454],
            [1367.56233886, 3595.37617537, 3861.96220571, 7599.93110962, 6018.65643062],
            [1349.98221742, 3730.36025071, 3900.14738742, 7997.86353537, 6213.61087632],
            [1330.33321825, 3752.7395927 , 3850.39111741, 7923.44398373, 6128.67898851]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=["id_1000", "id_1001", "id_1003", "id_1004", "id_1005"],
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_series_and_exog_dict_encoding_None_transformer_series_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries predict method when series and 
    exog are dictionaries and encoding is None and transformer_series is StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     lags               = 14,
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
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    predictions = forecaster.predict(
        steps=5, levels=levels, last_window=last_window,
        exog=exog_dict_test_2, suppress_warnings=True
    )
    expected = pd.DataFrame(
        data=np.array([
            [1540.61279683, 3160.95721541, 3121.57888505, 7369.19869868, 7011.48712882],
            [1533.43616081, 3157.30206473, 2976.43108071, 8430.97835738, 8401.89546128],
            [1466.94536267, 3215.9383148 , 2991.20843378, 8565.22794876, 8506.18236246],
            [1427.46483497, 3512.06387649, 2806.20216519, 8343.01317975, 8561.27068389],
            [1411.73980543, 3275.61791117, 1959.1294473 , 8360.01068926, 7633.82955233]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=["id_1000", "id_1001", "id_1003", "id_1004", "id_1005"],
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterRecursiveMultiSeries predict method when series and 
    exog are dictionaries and unknown level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     lags               = [1, 7, 14],
                     encoding           = 'onehot',
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
    predictions = forecaster.predict(
        steps=5, levels=levels, last_window=last_window,
        exog=exog_dict_test_2, suppress_warnings=True
    )
    expected = pd.DataFrame(
        data=np.array([
            [1444.89638131, 2608.52179991, 2594.53128524, 7607.65586239, 5178.02908923],
            [1426.33025646, 2522.90927921, 2240.71166036, 7993.7648681 , 4793.03739895],
            [1419.2856951 , 2451.51448204, 2183.50817345, 8473.84457748, 4721.84643207],
            [1368.78988907, 2477.40763584, 2180.37278483, 8528.07173563, 4559.15070243],
            [1324.15676752, 2566.91100046, 2186.03652396, 8767.76813577, 4966.9971064 ]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=["id_1000", "id_1001", "id_1003", "id_1004", "id_1005"],
    )

    pd.testing.assert_frame_equal(predictions, expected)