# Unit test _create_predict_inputs ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.recursive import ForecasterRecursive
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# Fixtures
from .fixtures_forecaster_recursive import y as y_categorical
from .fixtures_forecaster_recursive import exog as exog_categorical
from .fixtures_forecaster_recursive import data  # to test results when using differentiation


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5
                 )

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster._create_predict_inputs(steps=5)


def test_create_predict_inputs_when_regressor_is_LinearRegression():
    """
    Test _create_predict_inputs when using LinearRegression as regressor.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float), name='y'))
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        np.array([45., 46., 47., 48., 49.]),
        None,
        pd.RangeIndex(start=50, stop=55, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] is None
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test _create_predict_inputs when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9,  1.09, -3.61,  0.72, -0.11, -0.4,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8,
                      -0.61, -0.88]),
            name = 'y'
        )

    forecaster = ForecasterRecursive(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        np.array([-0.1056608, -0.30987914, -0.45194408, -0.28324197, -0.52297655]),
        None,
        pd.RangeIndex(start=20, stop=25, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] is None
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_series():
    """
    Test _create_predict_inputs when using LinearRegression as regressor, StandardScaler
    as transformer_y and StandardScaler as transformer_exog.
    """
    y = pd.Series(np.array([-0.59,  0.02, -0.9,  1.09, -3.61,  0.72, -0.11, -0.4]))
    exog = pd.Series(np.array([7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4]), name='exog')
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler()
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    expected = (
        np.array([1.16937289, -2.34810076, 0.89246539, 0.27129451, 0.0542589]),
        np.array([[-1.76425513], [-1.00989936], [0.59254869], [0.45863938], [0.1640389]]),
        pd.RangeIndex(start=8, stop=13, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_df():
    """
    Test _create_predict_inputs when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4])
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
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    
    expected = (
        np.array([1.16937289, -2.34810076, 0.89246539,  0.27129451,  0.0542589]),
        np.array([[-1.76425513,  1.        ,  0.        ],
                  [-1.00989936,  1.        ,  0.        ],
                  [ 0.59254869,  1.        ,  0.        ],
                  [ 0.45863938,  1.        ,  0.        ],
                  [ 0.1640389 ,  0.        ,  1.        ]]),
        pd.RangeIndex(start=8, stop=13, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs when using HistGradientBoostingRegressor and categorical variables.
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
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    
    expected = (
        np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
        np.array([[0.        , 0.        , 0.12062867],
                  [1.        , 1.        , 0.8263408 ],
                  [2.        , 2.        , 0.60306013],
                  [3.        , 3.        , 0.54506801],
                  [4.        , 4.        , 0.34276383],
                  [0.        , 0.        , 0.30412079],
                  [1.        , 1.        , 0.41702221],
                  [2.        , 2.        , 0.68130077],
                  [3.        , 3.        , 0.87545684],
                  [4.        , 4.        , 0.51042234]]),
        pd.RangeIndex(start=50, stop=60, step=1),
        10
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_regressor_is_LinearRegression_with_exog_differentiation_is_1():
    """
    Test _create_predict_inputs when using LinearRegression as regressor 
    and differentiation=1.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    steps = data.index[-1]

    forecaster = ForecasterRecursive(
                     regressor       = LinearRegression(),
                     lags            = 15,
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster._create_predict_inputs(steps=steps, exog=exog.loc[end_train:])
    
    expected = (
        np.array(
            [np.nan, 0.14355438, -0.56028323,  0.07558021, 0.04869748,
             0.09807633, -0.00584249,  0.17596768,  0.01630394,  0.09883014,
             0.02377842, -0.01018012,  0.10597971, -0.01463081, -0.48984868,
             0.07503713]
        ),
        np.array([[ 1.16172882], [ 0.29468848], [-0.4399757 ], [ 1.25008389],
                  [ 1.37496887], [-0.41673182], [ 0.32732157], [ 1.57848827],
                  [ 0.40402941], [ 1.34466867], [-1.0724777 ], [ 0.14619469],
                  [-0.78475177], [-2.97987806], [-1.46735738], [ 0.84295053],
                  [ 0.37485   ], [ 0.03500746], [-1.37307705], [ 1.43220391],
                  [ 0.76682773], [ 0.50114842], [ 0.839902  ], [-1.0012572 ],
                  [-1.67885896], [-0.81587204], [-1.33508966], [-0.42014975], 
                  [ 0.98579761], [-0.17262001], [-0.02161134], [-0.62664312],
                  [-0.91997057], [-1.14759504], [ 0.18661806], [ 0.15897422], 
                  [-3.20047182], [-0.5236335 ], [-1.00620967], [ 0.12763016], 
                  [-0.54508051], [ 0.13153937], [-0.14105938], [-0.16935134], 
                  [ 0.06279071], [-0.55075912], [-0.22425219], [ 2.00991015], 
                  [ 0.79230622], [-0.55605221], [-0.27088044], [-0.39179496], 
                  [-1.34347494], [-0.07230096], [-1.51146602], [-0.66840335], 
                  [-0.78148407], [-0.27354003], [-0.27128144], [-0.6389055 ], 
                  [ 0.19573233], [-0.67321672], [ 1.1559056 ]]
        ),
        pd.date_range(start='2003-04-01', periods=63, freq='MS'),
        63
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]
