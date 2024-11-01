# Unit test predict ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
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
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y as y_categorical
from .fixtures_forecaster_direct import exog as exog_categorical
from .fixtures_forecaster_direct import exog_predict as exog_predict_categorical
from .fixtures_forecaster_direct import data  # to test results when using differentiation


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_TypeError_when_steps_list_contain_floats(steps):
    """
    Test predict TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(10)))

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(50), 
                   index=pd.date_range(start='2020-01-01', periods=50, freq='D')))
    results = forecaster.predict(steps=steps)

    expected = pd.Series(
                   data  = np.array([50., 51., 52.]),
                   index = pd.date_range(start='2020-02-20', periods=3, freq='D'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_list_interspersed():
    """
    Test predict output when using LinearRegression as regressor and steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(50)))
    results = forecaster.predict(steps=[1, 4])
    
    expected = pd.Series(
                   data  = np.array([50., 53.]),
                   index = pd.RangeIndex(start=50, stop=54, step=1)[[0, 3]],
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_last_window():
    """
    Test predict output when using LinearRegression as regressor and last_window.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(50)))
    last_window = pd.Series(data  = [47, 48, 49], 
                            index = pd.RangeIndex(start=47, stop=50, step=1))
    results = forecaster.predict(steps=[1, 2, 3, 4], last_window=last_window)

    expected = pd.Series(
                   data  = np.array([50., 51., 52., 53.]),
                   index = pd.RangeIndex(start=50, stop=54, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_exog():
    """
    Test predict output when using LinearRegression as regressor and exog.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(
        y=pd.Series(np.arange(50)),
        exog=pd.Series(np.arange(start=100, stop=150, step=1), name="exog")
    )
    results = forecaster.predict(
                  steps = 5, 
                  exog  = pd.Series(np.arange(start=25, stop=50, step=0.5),
                                    index=pd.RangeIndex(start=50, stop=100),
                                    name="exog")
              )
    
    expected = pd.Series(
                   data  = np.array([18.750, 19.625, 20.500, 21.375, 22.250]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )

    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()

    forecaster = ForecasterDirect(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     steps         = 5,
                     transformer_y = transformer_y,
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict()

    expected = pd.Series(
                   data  = np.array([0.27498792, 0.1134674 , 0.3824246 , 0.62852197, 0.44001725]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_predict_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog(n_jobs):
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    exog = pd.DataFrame({
                'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 60.3, 87.2,
                          7.5, 60.4, 50.3, 57.3, 24.7, 87.4, 87.2, 60.4, 50.7, 7.5],
                'col_2': ['a'] * 10 + ['b'] * 10}
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=20, stop=40)

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     steps            = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                     n_jobs           = n_jobs
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = pd.Series(
                   data  = np.array([1.10855119, -0.83442443, 0.9434436 , 0.6676508 , 0.58666266]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


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
    
    forecaster = ForecasterDirect(
                     regressor   = LinearRegression(), 
                     lags        = 3, 
                     steps       = 3, 
                     weight_func = custom_weights
                 )
    forecaster.fit(y=pd.Series(np.arange(50)))
    results = forecaster.predict(steps=3)

    expected = pd.Series(
                   data  = np.array([50., 51., 52.]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test predict output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
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
    
    forecaster = ForecasterDirect(
                     regressor        = HistGradientBoostingRegressor(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     lags             = 5,
                     steps            = 10, 
                     transformer_y    = None,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.4668392 , 
                                    0.47754412, 0.47360906, 0.47749396, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor():
    """
    Test predict output when using LGBMRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
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
    
    forecaster = ForecasterDirect(
                     regressor        = LGBMRegressor(random_state=123),
                     lags             = 5,
                     steps            = 10, 
                     transformer_y    = None,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919, 
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor_auto():
    """
    Test predict output when using LGBMRegressor and categorical variables with 
    categorical_features='auto'.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
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
    
    forecaster = ForecasterDirect(
                     regressor        = LGBMRegressor(random_state=123),
                     lags             = 5,
                     steps            = 10, 
                     transformer_y    = None,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919, 
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_1():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=1 and steps=1.
    """

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_10():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=1 and steps=10.
    """

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_2():
    """
    Test predict output when using LinearRegression as regressor and differentiation=2.
    """

    # Data differentiated
    diferenciator_1 = TimeSeriesDifferentiator(order=1)
    diferenciator_2 = TimeSeriesDifferentiator(order=2)
    data_diff_1 = diferenciator_1.fit_transform(data.to_numpy())
    data_diff_1 = pd.Series(data_diff_1, index=data.index).dropna()
    data_diff_2 = diferenciator_2.fit_transform(data.to_numpy())
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15)
    forecaster_1.fit(y=data_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = data_diff_1.loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15, differentiation=2)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_2_steps_10():
    """
    Test predict output when using LinearRegression as regressor and 
    differentiation=2 and steps=10.
    """

    # Data differentiated
    diferenciator_1 = TimeSeriesDifferentiator(order=1)
    diferenciator_2 = TimeSeriesDifferentiator(order=2)
    data_diff_1 = diferenciator_1.fit_transform(data.to_numpy())
    data_diff_1 = pd.Series(data_diff_1, index=data.index).dropna()
    data_diff_2 = diferenciator_2.fit_transform(data.to_numpy())
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'
    
    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15)
    forecaster_1.fit(y=data_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = data_diff_1.loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15, differentiation=2)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_window_features_steps_1():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=1.
    """
    y_datetime = y_categorical.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y_datetime), freq='D')
    exog_datetime = exog_categorical.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog_datetime), freq='D')
    exog_predict_datetime = exog_predict_categorical.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict_datetime), freq='D')
    
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirect(
        LGBMRegressor(verbose=-1, random_state=123), steps=1, lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(exog=exog_predict_datetime)

    expected = pd.Series(
                   data = np.array([0.5326654111553376]),
                   index = pd.date_range(start='2001-02-20', periods=1, freq='D'),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_window_features_steps_10():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=10.
    """
    y_datetime = y_categorical.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y_datetime), freq='D')
    exog_datetime = exog_categorical.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog_datetime), freq='D')
    exog_predict_datetime = exog_predict_categorical.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict_datetime), freq='D')
    
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirect(
        LGBMRegressor(verbose=-1, random_state=123), steps=10, lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(exog=exog_predict_datetime)

    expected = pd.Series(
                   data = np.array([0.5013105906546116, 0.49276925561328727, 
                                    0.4743392908324798, 0.4668391936769088, 
                                    0.47754411751197445, 0.47360905777249074, 
                                    0.47749395275281536, 0.4846192327224546, 
                                    0.48686681294606793, 0.50223394172887]),
                   index = pd.date_range(start='2001-02-20', periods=10, freq='D'),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
