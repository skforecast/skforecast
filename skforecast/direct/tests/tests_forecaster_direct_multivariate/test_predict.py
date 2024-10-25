# Unit test predict ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
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

from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
from .fixtures_forecaster_direct_multivariate import exog
from .fixtures_forecaster_direct_multivariate import exog_predict
from .fixtures_forecaster_direct_multivariate import data  # to test results when using differentiation

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_TypeError_when_steps_list_contain_floats(steps):
    """
    Test predict TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.predict(steps=steps)


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_output_when_regressor_is_LinearRegression(steps):
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=steps)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417, 0.33255977]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_list_interspersed():
    """
    Test predict output when using LinearRegression as regressor and steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=3, steps=5)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=[1, 4])
    expected = pd.DataFrame(
                   data    = np.array([0.61048324, 0.53962565]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1)[[0, 3]],
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_different_lags():
    """
    Test predict output when using LinearRegression as regressor and different
    lags configuration for each series.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': [1, 7]}, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.58053278, 0.43052971, 0.60582844]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_lags_dict_with_None_in_level_lags():
    """
    Test predict output when using LinearRegression as regressor when lags is a 
    dict and level has None lags configuration.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': None}, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.59695170, 0.49497765, 0.51970720]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_lags_dict_with_None_but_no_in_level():
    """
    Test predict output when using LinearRegression as regressor when lags is a 
    dict with None values.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 5, 'l2': None}, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.61119488, 0.48858659, 0.46753222]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_last_window():
    """
    Test predict output when using LinearRegression as regressor and last_window.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    last_window = pd.DataFrame(
                      data    = np.array([[0.98555979, 0.39887629],
                                          [0.51948512, 0.2408559 ],
                                          [0.61289453, 0.34345601]]),
                      index   = pd.RangeIndex(start=47, stop=50, step=1),
                      columns = ['l1', 'l2']
                  )
    results = forecaster.predict(steps=[1, 2], last_window=last_window)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417]),
                   index   = pd.RangeIndex(start=50, stop=52, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_exog():
    """
    Test predict output when using LinearRegression as regressor and exog.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series.iloc[:40,], exog=exog.iloc[:40, 0])
    results = forecaster.predict(steps=None, exog=exog.iloc[40:43, 0])
    expected = pd.DataFrame(
                   data    = np.array([0.57243323, 0.56121924, 0.38879785]),
                   index   = pd.RangeIndex(start=40, stop=43, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.predict()
    expected = pd.DataFrame(
                   data    = np.array([0.60056539, 0.42924504, 0.34173573, 0.44231236, 0.40133213]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l2',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = {'l1': StandardScaler(), 'l2': MinMaxScaler()}
                 )
    forecaster.fit(series=series)
    results = forecaster.predict()
    expected = pd.DataFrame(
                   data    = np.array([0.65049981, 0.57548048, 0.64278726, 0.54421867, 0.7851753 ]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(n_jobs):
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                     n_jobs             = n_jobs
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict(steps=[1, 2, 3, 4, 5], exog=exog_predict)
    expected = pd.DataFrame(
                   data    = np.array([0.61043227, 0.46658137, 0.54994519, 0.52561227, 0.46596527]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_and_weight_func():
    """
    Test predict output when using LinearRegression as regressor and custom_weights.
    """
    def custom_weights(index):
        """
        Return 1 for all elements in index
        """
        weights = np.ones_like(index)

        return weights
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3, weight_func=custom_weights)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417, 0.33255977]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


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
    
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.4668392 , 
                                    0.47754412, 0.47360906, 0.47749396, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor():
    """
    Test predict output when using LGBMRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)})
    
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
    
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LGBMRegressor(random_state=123),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919,
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
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
    
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LGBMRegressor(random_state=123),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919,
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_1():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=1 and steps=1.
    """

    arr = data.to_numpy(copy=True)
    series_2 = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    series_diff = pd.DataFrame(
        {'l1': diferenciator.fit_transform(arr),
         'l2': diferenciator.fit_transform(arr * 1.6)},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, transformer_series=None
    )
    forecaster_1.fit(series=series_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = series_2[['l1']].loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, transformer_series=None, differentiation=1
    )
    forecaster_2.fit(series=series_2.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_frame_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_10():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=1 and steps=10.
    """

    arr = data.to_numpy(copy=True)
    series_dt = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    series_diff = pd.DataFrame(
        {'l1': diferenciator.fit_transform(arr),
         'l2': diferenciator.fit_transform(arr * 1.6)},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog_dt = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog_dt.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, transformer_series=None
    )
    forecaster_1.fit(series=series_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(exog=exog_diff.loc[end_train:])
    
    # Revert the differentiation
    last_value_train = series_dt[['l1']].loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, transformer_series=None, differentiation=1
    )
    forecaster_2.fit(series=series_dt.loc[:end_train], exog=exog_dt.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog_dt.loc[end_train:])

    pd.testing.assert_frame_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_2():
    """
    Test predict output when using LinearRegression as regressor and differentiation=2.
    """

    arr = data.to_numpy(copy=True)
    series_dt = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

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
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, transformer_series=None
    )
    forecaster_1.fit(series=df_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = df_diff_1[['l1']].loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = series_dt[['l1']].loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, 
        transformer_series=None, differentiation=2
    )
    forecaster_2.fit(series=series_dt.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_frame_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_2_steps_10():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=2 and steps=10.
    """

    arr = data.to_numpy(copy=True)
    series_dt = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

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
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, transformer_series=None
    )
    forecaster_1.fit(series=df_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = df_diff_1[['l1']].loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = series_dt[['l1']].loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, 
        transformer_series=None, differentiation=2
    )
    forecaster_2.fit(series=series_dt.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_frame_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_window_features_steps_1():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=1.
    """

    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(verbose=-1, random_state=123), level='l1', 
        steps=1, lags=5, window_features=rolling
    )
    forecaster.fit(series=series, exog=exog['exog_1'])
    predictions = forecaster.predict(exog=exog_predict['exog_1'])

    expected = pd.DataFrame(
                   data = np.array([0.6005080563629931]),
                   index   = pd.RangeIndex(start=50, stop=51, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_window_features_steps_10():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=10.
    """

    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(verbose=-1, random_state=123), level='l1', 
        steps=10, lags=15, window_features=rolling
    )
    forecaster.fit(series=series, exog=exog['exog_1'])
    predictions = forecaster.predict(exog=exog_predict['exog_1'])

    expected = pd.DataFrame(
                   data = np.array([[0.50449652],
                                    [0.48055894],
                                    [0.48574331],
                                    [0.49495002],
                                    [0.50782532],
                                    [0.49700331],
                                    [0.49118152],
                                    [0.49641721],
                                    [0.48853374],
                                    [0.4886057 ]]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)
