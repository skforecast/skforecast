# Unit test predict ForecasterRecursive
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
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y as y_categorical
from .fixtures_forecaster_recursive import exog as exog_categorical
from .fixtures_forecaster_recursive import exog_predict as exog_predict_categorical
from .fixtures_forecaster_recursive import data  # to test results when using differentiation


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_output_when_regressor_is_LinearRegression():
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data = np.array([50., 51., 52., 53., 54.]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)

        
def test_predict_output_when_regressor_is_LinearRegression_with_exog():
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50), name='y'), exog=pd.Series(np.arange(50, 150, 2), name='exog'))
    exog_pred = pd.Series(np.arange(100, 105), index=pd.RangeIndex(start=50, stop=55), name='exog')
    predictions = forecaster.predict(steps=5, exog=exog_pred)

    expected = pd.Series(
                   data = np.array([35.71428571428572, 34.38775510204082, 32.72886297376094,
                                    30.69012911286965, 30.258106741238777]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


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
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags = 5,
                    transformer_y = transformer_y,
                )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)

    expected = pd.Series(
                data = np.array([-0.1578203 , -0.18459942, -0.13711051, -0.01966358, -0.03228613]),
                index = pd.RangeIndex(start=20, stop=25, step=1),
                name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59, 0.02, -0.9, 1.09, -3.61, 0.72, -0.11, -0.4])
        )
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']}
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.50619336, -0.09630298,  0.05254973,  0.12281153,  0.00221741]),
                   index = pd.RangeIndex(start=8, stop=13, step=1),
                   name = 'pred'
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

    forecaster = ForecasterRecursive(LinearRegression(), lags=3, weight_func=custom_weights)
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster.predict(steps=5)

    expected = pd.Series(
                   data = np.array([50., 51., 52., 53., 54.]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test predict output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
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
    
    forecaster = ForecasterRecursive(
                     regressor        = HistGradientBoostingRegressor(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     lags             = 5,
                     transformer_y    = None,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.61187012, 0.42274801, 0.43214802, 0.4923281 , 
                                    0.53073262, 0.48004443, 0.56731689, 0.53956024, 
                                    0.47670124, 0.43896242]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor():
    """
    Test predict output when using LGBMRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
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
    
    forecaster = ForecasterRecursive(
                     regressor        = LGBMRegressor(random_state=123),
                     lags             = 5,
                     transformer_y    = None,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.5857033 , 0.3894503 , 0.45053399, 0.49686551, 
                                    0.45887492, 0.51481068, 0.5857033 , 0.3894503 , 
                                    0.45053399, 0.48818584]),
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
                            'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)})
    
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
    
    forecaster = ForecasterRecursive(
                     regressor        = LGBMRegressor(random_state=123),
                     lags             = 5,
                     transformer_y    = None,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.5857033 , 0.3894503 , 0.45053399, 0.49686551, 
                                    0.45887492, 0.51481068, 0.5857033 , 0.3894503 , 
                                    0.45053399, 0.48818584]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1():
    """
    Test predict output when using LinearRegression as regressor and differentiation=1.
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
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(regressor=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterRecursive(regressor=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_y():
    """
    Test predict output when using LinearRegression as regressor and differentiation=1,
    and transformer_y is StandardScaler.
    """

    end_train = '2003-03-01 23:59:00'
    # Data scaled and differentiated
    scaler = StandardScaler()
    scaler.fit(data.loc[:end_train].to_numpy().reshape(-1, 1))
    data_scaled = scaler.transform(data.to_numpy().reshape(-1, 1))
    data_scaled = pd.Series(data_scaled.ravel(), index=data.index)
    data_scaled_diff = TimeSeriesDifferentiator(order=1).fit_transform(data_scaled.to_numpy())
    data_scaled_diff = pd.Series(data_scaled_diff, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(regressor=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_scaled_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data_scaled.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]
    # Revert the scaling
    predictions_1 = scaler.inverse_transform(predictions_1.to_numpy().reshape(-1, 1))
    predictions_1 = pd.Series(predictions_1.ravel(), index=data.loc[end_train:].index)

    forecaster_2 = ForecasterRecursive(regressor=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

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
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(regressor=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(steps=steps, exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = data_diff_1.loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterRecursive(regressor=LinearRegression(), lags=15, differentiation=2)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


@pytest.mark.parametrize("steps", 
                         [10, '2001-03-01', pd.to_datetime('2001-03-01')], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_output_when_window_features(steps):
    """
    Test output of predict when regressor is LGBMRegressor and window features.
    """
    y_datetime = y_categorical.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y_datetime), freq='D')
    exog_datetime = exog_categorical.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog_datetime), freq='D')
    exog_predict_datetime = exog_predict_categorical.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict_datetime), freq='D')
    
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(steps=steps, exog=exog_predict_datetime)

    expected = pd.Series(
                   data = np.array([0.5326654111553376, 0.5050280233102159, 
                                    0.5050280233102159, 0.5050280233102159, 
                                    0.5326654111553376, 0.5326654111553376, 
                                    0.5326654111553376, 0.5050280233102159, 
                                    0.5050280233102159, 0.5326654111553376]),
                   index = pd.date_range(start='2001-02-20', periods=10, freq='D'),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
