# Unit test _create_predict_inputs ForecasterDirect
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
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y as y_categorical
from .fixtures_forecaster_direct import exog as exog_categorical
from .fixtures_forecaster_direct import data  # to test results when using differentiation


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_inputs_TypeError_when_steps_list_contain_floats(steps):
    """
    Test _create_predict_inputs TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=5
    )
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        f"`steps` argument must be an int, a list of ints or `None`. "
        f"Got {type(steps)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_predict_inputs(steps=steps)


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=5
    )

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster._create_predict_inputs(steps=5)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_inputs_output(steps):
    """
    Test _create_predict_inputs output.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )
    forecaster.fit(
        y=pd.Series(
            np.arange(50, dtype=float), index=pd.date_range(start='2020-01-01', periods=50, freq='D')
        )
    )
    results = forecaster._create_predict_inputs(steps=steps)

    expected = (
        [np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]])],
        ['lag_1', 'lag_2', 'lag_3'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_when_with_list_interspersed():
    """
    Test _create_predict_inputs output when steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=5
    )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    results = forecaster._create_predict_inputs(steps=[1, 4])

    expected = (
        [np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]])],
        ['lag_1', 'lag_2', 'lag_3'],
        [1, 4],
        pd.RangeIndex(start=50, stop=54, step=1)[[0, 3]]
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_when_last_window():
    """
    Test _create_predict_inputs output when external last_window.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=5
    )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    last_window = pd.Series(
        data  = [47, 48, 49], 
        index = pd.RangeIndex(start=47, stop=50, step=1)
    )
    results = forecaster._create_predict_inputs(steps=[1, 2, 3, 4], last_window=last_window)

    expected = (
        [np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]]),
         np.array([[49., 48., 47.]])],
        ['lag_1', 'lag_2', 'lag_3'],
        [1, 2, 3, 4],
        pd.RangeIndex(start=50, stop=54, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_when_exog():
    """
    Test _create_predict_inputs output when exog.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=5
    )
    forecaster.fit(
        y=pd.Series(np.arange(50, dtype=float)),
        exog=pd.Series(np.arange(start=100, stop=150, step=1), name="exog")
    )
    results = forecaster._create_predict_inputs(
                  steps = 5, 
                  exog  = pd.Series(np.arange(start=25, stop=50, step=0.5, dtype=float),
                                    index=pd.RangeIndex(start=50, stop=100),
                                    name="exog")
               )

    expected = (
        [np.array([[49., 48., 47., 25.]]),
         np.array([[49., 48., 47., 25.5]]),
         np.array([[49., 48., 47., 26.]]),
         np.array([[49., 48., 47., 26.5]]),
         np.array([[49., 48., 47., 27.]])],
        ['lag_1', 'lag_2', 'lag_3', 'exog'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_with_transform_y():
    """
    Test _create_predict_inputs output when StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()

    forecaster = ForecasterDirect(
                     estimator     = LinearRegression(),
                     lags          = 5,
                     steps         = 5,
                     transformer_y = transformer_y,
                 )
    forecaster.fit(y=y)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_create_predict_inputs_output_with_transform_y_and_transform_exog(n_jobs):
    """
    Test _create_predict_inputs output when StandardScaler
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
                     estimator        = LinearRegression(),
                     lags             = 5,
                     steps            = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                     n_jobs           = n_jobs
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster._create_predict_inputs(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = (
        [np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608,
                    -1.7093071, 1., 0.,]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608,
                    -1.0430105, 1., 0.,]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608,
                    0.372377, 1., 0.,]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608,
                    0.2540995, 1., 0.,]]),
         np.array([[-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608,
                    -0.006111, 1., 0.,]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1', 'col_2_a', 'col_2_b'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs output when using HistGradientBoostingRegressor 
    and categorical variables.
    """
    df_exog = pd.DataFrame({
        'exog_1': exog_categorical,
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
    
    forecaster = ForecasterDirect(
                     estimator        = HistGradientBoostingRegressor(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     lags             = 5,
                     steps            = 10, 
                     transformer_y    = None,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0., 0., 0.12062867]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    1., 1., 0.8263408 ]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    2., 2., 0.60306013]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    3., 3., 0.54506801]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    4., 4., 0.34276383]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0., 0., 0.30412079]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    1., 1., 0.41702221]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    2., 2., 0.68130077]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    3., 3., 0.87545684]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    4., 4., 0.51042234]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_2', 'exog_3', 'exog_1'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_when_with_exog_differentiation_is_1():
    """
    Test _create_predict_inputs when using LinearRegression as estimator 
    and differentiation=1.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )

    forecaster = ForecasterDirect(
                     estimator       = LinearRegression(),
                     steps           = 5,
                     lags            = 15,
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster._create_predict_inputs(exog=exog.loc[end_train:])
    
    expected = (
        [np.array([[ 0.07503713, -0.48984868, -0.01463081,  0.10597971, -0.01018012,
                0.02377842,  0.09883014,  0.01630394,  0.17596768, -0.00584249,
                0.09807633,  0.04869748,  0.07558021, -0.56028323,  0.14355438,
                1.16172882]]),
        np.array([[ 0.07503713, -0.48984868, -0.01463081,  0.10597971, -0.01018012,
                0.02377842,  0.09883014,  0.01630394,  0.17596768, -0.00584249,
                0.09807633,  0.04869748,  0.07558021, -0.56028323,  0.14355438,
                0.29468848]]),
        np.array([[ 0.07503713, -0.48984868, -0.01463081,  0.10597971, -0.01018012,
                0.02377842,  0.09883014,  0.01630394,  0.17596768, -0.00584249,
                0.09807633,  0.04869748,  0.07558021, -0.56028323,  0.14355438,
                -0.4399757 ]]),
        np.array([[ 0.07503713, -0.48984868, -0.01463081,  0.10597971, -0.01018012,
                0.02377842,  0.09883014,  0.01630394,  0.17596768, -0.00584249,
                0.09807633,  0.04869748,  0.07558021, -0.56028323,  0.14355438,
                1.25008389]]),
        np.array([[ 0.07503713, -0.48984868, -0.01463081,  0.10597971, -0.01018012,
                0.02377842,  0.09883014,  0.01630394,  0.17596768, -0.00584249,
                0.09807633,  0.04869748,  0.07558021, -0.56028323,  0.14355438,
                1.37496887]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
         'lag_9', 'lag_10', 'lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15', 'exog'],
        [1, 2, 3, 4, 5],
        pd.date_range(start='2003-04-01', periods=5, freq='MS')
    )
    
    for step in range(len(results[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert isinstance(results[4], TimeSeriesDifferentiator)
    assert results[4] is not forecaster.differentiator


def test_create_predict_inputs_output_window_features():
    """
    Test _create_predict_inputs output with window_features.
    """
    y_datetime = pd.Series(
        np.arange(50), index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[4, 5, 6]
    )
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=3, lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[49., 48., 47., 47.5, 47., 279.]]),
         np.array([[49., 48., 47., 47.5, 47., 279.]]),
         np.array([[49., 48., 47., 47.5, 47., 279.]])],
        ['lag_1', 'lag_2', 'lag_3', 'roll_mean_4', 'roll_median_5', 'roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_with_2_window_features():
    """
    Test _create_predict_inputs output with 2 window_features.
    """
    y_datetime = pd.Series(
        np.arange(50), index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y', dtype=float
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[4, 5])
    rolling_2 = RollingFeatures(stats=['sum'], window_sizes=6)
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=3, lags=3, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[49., 48., 47., 47.5, 47., 279.]]),
         np.array([[49., 48., 47., 47.5, 47., 279.]]),
         np.array([[49., 48., 47., 47.5, 47., 279.]])],
        ['lag_1', 'lag_2', 'lag_3', 'roll_mean_4', 'roll_median_5', 'roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_output_window_features_and_no_lags():
    """
    Test _create_predict_inputs output with window_features and no lags.
    """
    y_datetime = pd.Series(
        np.arange(50), index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[4, 5, 6]
    )
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=3, lags=None, window_features=rolling
    )
    forecaster.fit(y=y_datetime)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[47.5, 47., 279.]]),
         np.array([[47.5, 47., 279.]]),
         np.array([[47.5, 47., 279.]])],
        ['roll_mean_4', 'roll_median_5', 'roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_does_not_mutate_differentiator():
    """
    Test that _create_predict_inputs does not mutate the forecaster's
    differentiator when differentiation is applied.
    """
    end_train = '2003-03-01 23:59:00'

    forecaster = ForecasterDirect(
                     estimator       = LinearRegression(),
                     steps           = 5,
                     lags            = 15,
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train])

    differentiator_initial_values = forecaster.differentiator.initial_values.copy()
    differentiator_last_values = forecaster.differentiator.last_values.copy()

    # Call _create_predict_inputs twice to verify no side effects
    results_1 = forecaster._create_predict_inputs()
    results_2 = forecaster._create_predict_inputs()

    np.testing.assert_array_equal(
        forecaster.differentiator.initial_values,
        differentiator_initial_values
    )
    np.testing.assert_array_equal(
        forecaster.differentiator.last_values,
        differentiator_last_values
    )
    # Both calls should produce identical results
    for step in range(len(results_1[0])):
        np.testing.assert_almost_equal(results_1[0][step], results_2[0][step])


def test_create_predict_inputs_output_with_transform_y_and_transform_exog_series():
    """
    Test _create_predict_inputs output when StandardScaler as transformer_y
    and StandardScaler as transformer_exog with exog as Series.
    """
    y = pd.Series(
            np.array([-0.59, 0.02, -0.9, 1.09, -3.61, 0.72, -0.11, -0.4])
        )
    exog = pd.Series(
               np.array([7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4]),
               name='exog'
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     lags             = 3,
                     steps            = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler()
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster._create_predict_inputs(steps=3, exog=exog_predict)

    expected = (
        [np.array([[ 0.0542589 ,  0.27129451,  0.89246539, -1.76425513]]),
         np.array([[ 0.0542589 ,  0.27129451,  0.89246539, -1.00989936]]),
         np.array([[ 0.0542589 ,  0.27129451,  0.89246539,  0.59254869]])],
        ['lag_1', 'lag_2', 'lag_3', 'exog'],
        [1, 2, 3],
        pd.RangeIndex(start=8, stop=11, step=1)
    )

    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_predict_inputs_when_categorical_features_auto_and_explicit_no_transformer_exog(
    categorical_features,
):
    """
    Test _create_predict_inputs when using internal categorical encoding
    (`categorical_features='auto'` and explicit list) without `transformer_exog`.
    This exercises the copy guard branch (`transformer_exog is None`).
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_categorical,
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )

    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    forecaster = ForecasterDirect(
                     estimator            = LinearRegression(),
                     lags                 = 5,
                     steps                = 10,
                     transformer_y        = None,
                     transformer_exog     = None,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.12062867, 0., 0.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.8263408, 1., 1.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.60306013, 2., 2.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.54506801, 3., 3.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34276383, 4., 4.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.30412079, 0., 0.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.41702221, 1., 1.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.68130077, 2., 2.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.87545684, 3., 3.]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.51042234, 4., 4.]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2', 'exog_3'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        pd.RangeIndex(start=50, stop=60, step=1)
    )

    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None


def test_create_predict_inputs_when_categorical_features_auto_with_transformer_exog():
    """
    Test _create_predict_inputs when using internal categorical encoding
    (`categorical_features='auto'`) together with `transformer_exog`
    (StandardScaler on numeric columns). This exercises the branch where
    copy is NOT needed because `transformer_exog` already returns a new
    DataFrame.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_categorical,
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )

    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    transformer_exog = make_column_transformer(
                           (StandardScaler(), make_column_selector(dtype_include=np.number)),
                           remainder='passthrough',
                           verbose_feature_names_out=False,
                       ).set_output(transform='pandas')

    forecaster = ForecasterDirect(
                     estimator            = LinearRegression(),
                     lags                 = 5,
                     steps                = 10,
                     transformer_y        = None,
                     transformer_exog     = transformer_exog,
                     categorical_features = 'auto'
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)

    expected = (
        [np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                    -1.47636391,  0.,  0.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     1.26277054,  1.,  1.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     0.3961342,   2.,  2.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     0.17104495,  3.,  3.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                    -0.61417373,  4.,  4.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                    -0.76416192,  0.,  0.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                    -0.325949,    1.,  1.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     0.69981558,  2.,  2.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     1.45340838,  3.,  3.]]),
         np.array([[ 0.61289453,  0.51948512,  0.98555979,  0.48303426,  0.25045537,
                     0.03657206,  4.,  4.]])],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2', 'exog_3'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        pd.RangeIndex(start=50, stop=60, step=1)
    )

    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] is None
