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
from skforecast.exceptions import MissingValuesWarning

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
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_output_when_estimator_is_LinearRegression():
    """
    Test predict output when using LinearRegression as estimator.
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

        
def test_predict_output_when_with_exog():
    """
    Test predict output when using LinearRegression as estimator.
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


def test_predict_output_when_with_transform_y():
    """
    Test predict output when using LinearRegression as estimator and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()
    forecaster = ForecasterRecursive(
                    estimator = LinearRegression(),
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


def test_predict_output_when_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as estimator, StandardScaler
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
                     estimator        = LinearRegression(),
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


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "lags": 5},
        {"estimator": LinearRegression(), "lags": 5,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=3)},
        {"estimator": LinearRegression(), "lags": 5,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=3),
         "transformer_y": StandardScaler(), "transformer_exog": StandardScaler()},
        {"estimator": LinearRegression(), "lags": 5,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=3),
         "transformer_y": StandardScaler(), "transformer_exog": StandardScaler(),
         "differentiation": 1},
    ],
    ids=["base", "window_features", "transformers", "differentiation"]
)
def test_predict_does_not_modify_y_exog(forecaster_kwargs):
    """
    Test forecaster.predict does not modify y, exog, exog_predict or last_window.
    """
    y_local = y_categorical.copy()
    exog_local = exog_categorical.copy()
    exog_predict_local = exog_predict_categorical.copy()
    last_window_local = y_local.iloc[-6:].copy()

    y_copy = y_local.copy()
    exog_copy = exog_local.copy()
    exog_predict_copy = exog_predict_local.copy()
    last_window_copy = last_window_local.copy()

    forecaster = ForecasterRecursive(**forecaster_kwargs)
    forecaster.fit(y=y_local, exog=exog_local)
    _ = forecaster.predict(steps=5, exog=exog_predict_local, last_window=last_window_local)

    pd.testing.assert_series_equal(y_local, y_copy)
    pd.testing.assert_series_equal(exog_local, exog_copy)
    pd.testing.assert_series_equal(last_window_local, last_window_copy)
    pd.testing.assert_series_equal(exog_predict_local, exog_predict_copy)


def test_predict_output_when_and_weight_func():
    """
    Test predict output when using LinearRegression as estimator and custom_weights.
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


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_HistGradientBoostingRegressor(categorical_features):
    """
    Test predict output when using HistGradientBoostingRegressor and categorical variables.
    Native implementation of categorical features in HistGradientBoostingRegressor
    should return the same predictions as the one obtained when using the Forecaster 
    to encode the categorical features.
    """
    df_exog = pd.DataFrame({'exog_1': exog_categorical,
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    cat_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               cat_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    # No categorical features managed by the forecaster.
    # make_column_transformer reorders columns to ['exog_2', 'exog_3', 'exog_1']
    # so categorical indices in X_train_step (5 lags + 3 exog) are [5, 6].
    # HistGradientBoostingRegressor requires integer indices when X is numpy.
    forecaster = ForecasterRecursive(
                     estimator             = HistGradientBoostingRegressor(
                                                 categorical_features = [5, 6],
                                                 random_state         = 123
                                             ),
                     lags                  = 5,
                     transformer_y         = None,
                     transformer_exog      = transformer_exog,
                     categorical_features  = None
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)
    
    # Categorical features managed by the forecaster
    forecaster_2 = ForecasterRecursive(
                       estimator             = HistGradientBoostingRegressor(
                                                   random_state = 123
                                               ),
                       lags                  = 5,
                       transformer_y         = None,
                       transformer_exog      = None,
                       categorical_features  = categorical_features
                   )
    forecaster_2.fit(y=y_categorical, exog=df_exog)
    predictions_2 = forecaster_2.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.61187012, 0.42274801, 0.43214802, 0.4923281 , 
                                    0.53073262, 0.48004443, 0.56731689, 0.53956024, 
                                    0.47670124, 0.43896242]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
    pd.testing.assert_series_equal(predictions_2, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_LGBMRegressor(categorical_features):
    """
    Test predict output when using LGBMRegressor and categorical variables.
    Native implementation of categorical features in LGBMRegressor
    should return the same predictions as the one obtained when using the Forecaster 
    to encode the categorical features.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_categorical,
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    cat_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               cat_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    # make_column_transformer reorders columns to ['exog_2', 'exog_3', 'exog_1']
    # so categorical indices in X_train_step (5 lags + 3 exog) are [5, 6].
    # LGBMRegressor requires integer indices when X is numpy.
    forecaster = ForecasterRecursive(
                     estimator            = LGBMRegressor(random_state=123, verbose=-1),
                     lags                 = 5,
                     transformer_y        = None,
                     transformer_exog     = transformer_exog,
                     categorical_features = None,
                     fit_kwargs           = {'categorical_feature': [5, 6]}  
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)
    
    forecaster_2 = ForecasterRecursive(
                       estimator            = LGBMRegressor(random_state=123, verbose=-1),
                       lags                 = 5,
                       transformer_y        = None,
                       transformer_exog     = transformer_exog,
                       categorical_features = categorical_features
                   )
    forecaster_2.fit(y=y_categorical, exog=df_exog)
    predictions_2 = forecaster_2.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.5857033 , 0.3894503 , 0.45053399, 0.49686551, 
                                    0.45887492, 0.51481068, 0.5857033 , 0.3894503 , 
                                    0.45053399, 0.48818584]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
    pd.testing.assert_series_equal(predictions_2, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_LGBMRegressor_auto(categorical_features):
    """
    Test predict output when using LGBMRegressor and categorical variables with 
    categorical_features='auto'.
    Native implementation of categorical features in LGBMRegressor
    should return the same predictions as the one obtained when using the Forecaster 
    to encode the categorical features.
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
    
    forecaster = ForecasterRecursive(
                     estimator            = LGBMRegressor(random_state=123, verbose=-1),
                     lags                 = 5,
                     transformer_y        = None,
                     transformer_exog     = transformer_exog,
                     categorical_features = None,
                     fit_kwargs           = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)
    
    forecaster_2 = ForecasterRecursive(
                       estimator            = LGBMRegressor(random_state=123, verbose=-1),
                       lags                 = 5,
                       transformer_y        = None,
                       transformer_exog     = transformer_exog,
                       categorical_features = categorical_features
                   )
    forecaster_2.fit(y=y_categorical, exog=df_exog)
    predictions_2 = forecaster_2.predict(steps=10, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([0.5857033 , 0.3894503 , 0.45053399, 0.49686551, 
                                    0.45887492, 0.51481068, 0.5857033 , 0.3894503 , 
                                    0.45053399, 0.48818584]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
    pd.testing.assert_series_equal(predictions_2, expected)


def test_predict_output_when_with_exog_and_differentiation_is_1():
    """
    Test predict output when using LinearRegression as estimator and differentiation=1.
    """

    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=1)
    data_diff = differentiator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(estimator=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]

    forecaster_2 = ForecasterRecursive(estimator=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_with_exog_differentiation_is_1_and_transformer_y():
    """
    Test predict output when using LinearRegression as estimator and differentiation=1,
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

    forecaster_1 = ForecasterRecursive(estimator=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_scaled_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    predictions_diff = forecaster_1.predict(steps=steps, exog=exog_diff.loc[end_train:])
    # Revert the differentiation
    last_value_train = data_scaled.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff]).cumsum()[1:]
    # Revert the scaling
    predictions_1 = scaler.inverse_transform(predictions_1.to_numpy().reshape(-1, 1))
    predictions_1 = pd.Series(predictions_1.ravel(), index=data.loc[end_train:].index)

    forecaster_2 = ForecasterRecursive(estimator=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


def test_predict_output_when_with_exog_and_differentiation_is_2():
    """
    Test predict output when using LinearRegression as estimator and differentiation=2.
    """

    # Data differentiated
    differentiator_1 = TimeSeriesDifferentiator(order=1)
    differentiator_2 = TimeSeriesDifferentiator(order=2)
    data_diff_1 = differentiator_1.fit_transform(data.to_numpy())
    data_diff_1 = pd.Series(data_diff_1, index=data.index).dropna()
    data_diff_2 = differentiator_2.fit_transform(data.to_numpy())
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(estimator=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff_2.loc[:end_train], exog=exog_diff_2.loc[:end_train])
    predictions_diff_2 = forecaster_1.predict(steps=steps, exog=exog_diff_2.loc[end_train:])
    
    # Revert the differentiation
    last_value_train_diff = data_diff_1.loc[:end_train].iloc[[-1]]
    predictions_diff_1 = pd.concat([last_value_train_diff, predictions_diff_2]).cumsum()[1:]
    last_value_train = data.loc[:end_train].iloc[[-1]]
    predictions_1 = pd.concat([last_value_train, predictions_diff_1]).cumsum()[1:]

    forecaster_2 = ForecasterRecursive(estimator=LinearRegression(), lags=15, differentiation=2)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    predictions_2 = forecaster_2.predict(steps=steps, exog=exog.loc[end_train:])

    pd.testing.assert_series_equal(predictions_1.asfreq('MS'), predictions_2, check_names=False)


@pytest.mark.parametrize("steps", 
                         [10, '2001-03-01', pd.to_datetime('2001-03-01')], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_output_when_window_features(steps):
    """
    Test output of predict when estimator is LGBMRegressor and window features.
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


def test_predict_output_when_last_window_stored_has_NaN():
    """
    Test predict output when the stored last_window_ contains NaN values.
    Estimator: HistGradientBoostingRegressor (supports NaN natively).
    """
    y_nan = pd.Series(
        data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0],
        name='y'
    )
    forecaster = ForecasterRecursive(
                     estimator=HistGradientBoostingRegressor(random_state=123),
                     lags=3,
                     dropna_from_series=True
                 )

    warn_msg = re.escape(
        "NaNs detected in `X_train`. They have been dropped."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        forecaster.fit(y=y_nan)

    assert forecaster.last_window_.isna().any().any()

    warn_msg = re.escape(
        "`last_window` has missing values."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        predictions = forecaster.predict(steps=3)

    expected = pd.Series(
                   data=np.array([5.5, 5.5, 5.5]),
                   index=pd.RangeIndex(start=10, stop=13, step=1),
                   name='pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_last_window_argument_has_NaN():
    """
    Test predict output when a custom last_window with NaN values is passed
    to the predict method. Estimator: HistGradientBoostingRegressor.
    """
    y = pd.Series(data=np.arange(1.0, 21.0), name='y')
    forecaster = ForecasterRecursive(
                     estimator=HistGradientBoostingRegressor(random_state=123),
                     lags=3,
                     dropna_from_series=False
                 )
    forecaster.fit(y=y)

    last_window_nan = pd.Series(
        data=[np.nan, 19.0, 20.0],
        index=pd.RangeIndex(start=17, stop=20)
    )

    warn_msg = re.escape(
        "`last_window` has missing values."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        predictions = forecaster.predict(steps=3, last_window=last_window_nan)

    expected = pd.Series(
                   data=np.array([12., 12., 12.]),
                   index=pd.RangeIndex(start=20, stop=23, step=1),
                   name='pred'
               )

    pd.testing.assert_series_equal(predictions, expected)
