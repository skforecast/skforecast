# Unit test create_predict_X ForecasterAutoreg
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
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Fixtures
from .fixtures_ForecasterAutoreg import y as y_categorical
from .fixtures_ForecasterAutoreg import exog as exog_categorical
from .fixtures_ForecasterAutoreg import data  # to test results when using differentiation


def test_create_predict_X_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5
                 )

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_X(steps=5)


def test_create_predict_X_when_regressor_is_LinearRegression():
    """
    Test create_predict_X when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float), name='y'))
    results = forecaster.create_predict_X(steps=5)

    expected = pd.DataFrame(
        data = {
            'lag_1': [49., 50., 51., 52., 53.],
            'lag_2': [48., 49., 50., 51., 52.],
            'lag_3': [47., 48., 49., 50., 51.],
            'lag_4': [46., 47., 48., 49., 50.],
            'lag_5': [45., 46., 47., 48., 49.]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test create_predict_X when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88]),
            name = 'y'
        )

    forecaster = ForecasterAutoreg(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster.create_predict_X(steps=5)

    expected = pd.DataFrame(
        data = np.array([
            [-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608 ],
            [-0.1578203 , -0.52297655, -0.28324197, -0.45194408, -0.30987914],
            [-0.18459942, -0.1578203 , -0.52297655, -0.28324197, -0.45194408],
            [-0.13711051, -0.18459942, -0.1578203 , -0.52297655, -0.28324197],
            [-0.01966358, -0.13711051, -0.18459942, -0.1578203 , -0.52297655]]),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        index = pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_series():
    """
    Test create_predict_X when using LinearRegression as regressor, StandardScaler
    as transformer_y and StandardScaler as transformer_exog.
    """
    y = pd.Series(np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4]))
    exog = pd.Series(np.array([7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4]), name='exog')
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler()
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([
            [ 0.0542589 ,  0.27129451,  0.89246539, -2.34810076,  1.16937289, -1.76425513],
            [ 0.50619336,  0.0542589 ,  0.27129451,  0.89246539, -2.34810076, -1.00989936],
            [-0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.89246539,  0.59254869],
            [ 0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.45863938],
            [ 0.12281153,  0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.1640389 ]]
        ),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        index = pd.RangeIndex(start=8, stop=13, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_df():
    """
    Test create_predict_X when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9,  1.09, -3.61,  0.72, -0.11, -0.4])
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
    
    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([[ 0.0542589 ,  0.27129451,  0.89246539, -2.34810076,  1.16937289,
                           -1.76425513,  1.        ,  0.        ],
                         [ 0.50619336,  0.0542589 ,  0.27129451,  0.89246539, -2.34810076,
                           -1.00989936,  1.        ,  0.        ],
                         [-0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.89246539,
                           0.59254869,  1.        ,  0.        ],
                         [ 0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.27129451,
                           0.45863938,  1.        ,  0.        ],
                         [ 0.12281153,  0.05254973, -0.09630298,  0.50619336,  0.0542589 ,
                           0.1640389 ,  0.        ,  1.        ]]),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1', 'col_2_a', 'col_2_b'],
        index = pd.RangeIndex(start=8, stop=13, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_categorical,
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
    
    forecaster = ForecasterAutoreg(
                     regressor        = HistGradientBoostingRegressor(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     lags             = 5,
                     transformer_y    = None,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([
                [0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.        , 0.        , 0.12062867],
                [0.61187012, 0.61289453, 0.51948512, 0.98555979, 0.48303426,
                    1.        , 1.        , 0.8263408 ],
                [0.42274801, 0.61187012, 0.61289453, 0.51948512, 0.98555979,
                    2.        , 2.        , 0.60306013],
                [0.43214802, 0.42274801, 0.61187012, 0.61289453, 0.51948512,
                    3.        , 3.        , 0.54506801],
                [0.4923281 , 0.43214802, 0.42274801, 0.61187012, 0.61289453,
                    4.        , 4.        , 0.34276383],
                [0.53073262, 0.4923281 , 0.43214802, 0.42274801, 0.61187012,
                    0.        , 0.        , 0.30412079],
                [0.48004443, 0.53073262, 0.4923281 , 0.43214802, 0.42274801,
                    1.        , 1.        , 0.41702221],
                [0.56731689, 0.48004443, 0.53073262, 0.4923281 , 0.43214802,
                    2.        , 2.        , 0.68130077],
                [0.53956024, 0.56731689, 0.48004443, 0.53073262, 0.4923281 ,
                    3.        , 3.        , 0.87545684],
                [0.47670124, 0.53956024, 0.56731689, 0.48004443, 0.53073262,
                    4.        , 4.        , 0.51042234]]),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_2', 'exog_3', 'exog_1'],
        index = pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_y():
    """
    Test create_predict_X when using LinearRegression as regressor and differentiation=1,
    and transformer_y is StandardScaler.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    steps = 4

    forecaster = ForecasterAutoreg(
                     regressor       = LinearRegression(),
                     lags            = [1, 5],
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster.create_predict_X(steps=steps, exog=exog.loc[end_train:])

    expected = pd.DataFrame(
        data = np.array([
            [ 0.07503713, -0.01018012,  1.16172882],
            [ 2.04678814,  0.10597971,  0.29468848],
            [ 2.0438193 , -0.01463081, -0.4399757 ],
            [ 2.06140742, -0.48984868,  1.25008389]]
        ),
        columns = ['lag_1', 'lag_5', 'exog'],
        index = pd.date_range(start='2003-04-01', periods=steps, freq='MS')
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_window_features():
    """
    Test the output of create_predict_X when using window_features and exog 
    with datetime index.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    exog_datetime_pred = pd.Series(
        np.arange(115, 120), index=pd.date_range('2000-01-16', periods=5, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterAutoreg(
        LinearRegression(), lags=5, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.create_predict_X(steps=5, exog=exog_datetime_pred)

    expected = pd.DataFrame(
        data = np.array([
                    [14., 13., 12., 11., 10., 12., 12., 69., 115.],
                    [15., 14., 13., 12., 11., 13., 13., 75., 116.],
                    [16., 15., 14., 13., 12., 14., 14., 81., 117.],
                    [17., 16., 15., 14., 13., 15., 15., 87., 118.],
                    [18., 17., 16., 15., 14., 16., 16., 93., 119.]]),
        index   = pd.date_range('2000-01-16', periods=5, freq='D'),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                   'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_window_features_and_lags_None():
    """
    Test the output of create_predict_X when using window_features and exog 
    with datetime index and lags=None.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    exog_datetime_pred = pd.Series(
        np.arange(115, 120), index=pd.date_range('2000-01-16', periods=5, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterAutoreg(
        LinearRegression(), lags=None, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.create_predict_X(steps=5, exog=exog_datetime_pred)

    expected = pd.DataFrame(
        data = np.array([
                    [12., 12., 69., 115.],
                    [13., 13., 75., 116.],
                    [14., 14., 81., 117.],
                    [15., 15., 87., 118.],
                    [16., 16., 93., 119.]]),
        index   = pd.date_range('2000-01-16', periods=5, freq='D'),
        columns = ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
    )

    pd.testing.assert_frame_equal(results, expected)