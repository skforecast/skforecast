# Unit test _create_predict_inputs ForecasterDirectMultiVariate
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
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.preprocessing import RollingFeatures
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
def test_create_predict_inputs_TypeError_when_steps_list_contain_floats(steps):
    """
    Test _create_predict_inputs TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_predict_inputs(steps=steps)


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
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
        forecaster._create_predict_inputs(steps=5)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_inputs_output(steps):
    """
    Test _create_predict_inputs output.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2020-01-01', periods=50, freq='D')
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3, transformer_series=None)
    forecaster.fit(series=series_datetime)
    results = forecaster._create_predict_inputs(steps=steps)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_list_interspersed():
    """
    Test _create_predict_inputs output when steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=3, steps=5, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=[1, 4])

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
        [1, 4],
        pd.RangeIndex(start=50, stop=54, step=1)[[0, 3]]
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_different_lags():
    """
    Test _create_predict_inputs output when different
    lags configuration for each series.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': [1, 7]}, 
                                               steps=3, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_7'],
        [1, 2, 3],
        pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_lags_dict_with_None_in_level_lags():
    """
    Test _create_predict_inputs output when lags is a 
    dict and level has None lags configuration.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5'],
        [1, 2, 3],
        pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_lags_dict_with_None_but_no_in_level():
    """
    Test _create_predict_inputs output when lags is a 
    dict with None values.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5'],
        [1, 2, 3],
        pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_last_window():
    """
    Test _create_predict_inputs output when external last_window.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    last_window = pd.DataFrame(
        data    = np.array([[0.98555979, 0.39887629],
                            [0.51948512, 0.2408559 ],
                            [0.61289453, 0.34345601]]),
        index   = pd.RangeIndex(start=47, stop=50, step=1),
        columns = ['l1', 'l2']
    )
    results = forecaster._create_predict_inputs(steps=[1, 2], last_window=last_window)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
        [1, 2],
        pd.RangeIndex(start=50, stop=52, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_exog():
    """
    Test _create_predict_inputs output when exog.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series.iloc[:40,], exog=exog.iloc[:40, 0])
    results = forecaster._create_predict_inputs(steps=None, exog=exog.iloc[40:43, 0])

    expected = (
        [np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.09332671]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.29686078]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.92758424]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'exog_1'],
        [1, 2, 3],
        pd.RangeIndex(start=40, stop=43, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_with_transform_series():
    """
    Test _create_predict_inputs output when StandardScaler.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_with_transform_series_as_dict():
    """
    Test _create_predict_inputs output when transformer_series
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
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_create_predict_inputs_output_with_transform_series_and_transform_exog(n_jobs):
    """
    Test _create_predict_inputs output when StandardScaler
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
    results = forecaster._create_predict_inputs(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -0.02551075,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                     0.51927383,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -1.470801945,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -1.38212079,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -0.70392558,  1.,  0.]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1', 'exog_2_a', 'exog_2_b'],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs output when using HistGradientBoostingRegressor 
    and categorical variables.
    """
    df_exog = pd.DataFrame({
        'exog_1': exog['exog_1'],
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
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    0.        , 0.        , 0.51312815]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    1.        , 1.        , 0.66662455]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    2.        , 2.        , 0.10590849]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    3.        , 3.        , 0.13089495]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    4.        , 4.        , 0.32198061]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    0.        , 0.        , 0.66156434]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    1.        , 1.        , 0.84650623]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    2.        , 2.        , 0.55325734]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    3.        , 3.        , 0.85445249]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    4.        , 4.        , 0.38483781]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_2', 'exog_3', 'exog_1'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_when_regressor_is_LinearRegression_with_exog_differentiation_is_1():
    """
    Test _create_predict_inputs when using LinearRegression as regressor 
    and differentiation=1.
    """

    end_train = '2003-03-01 23:59:00'
    arr = data.to_numpy(copy=True)
    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr},
        index=data.index
    )

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 5,
                     lags               = 15,
                     differentiation    = 1,
                     transformer_series = StandardScaler()
                )
    forecaster.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster._create_predict_inputs(exog=exog.loc[end_train:])
    
    expected = ([
        np.array([[ 0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    1.16172882]]),
        np.array([[ 0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.29468848]]),
        np.array([[ 0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    -0.4399757 ]]),
        np.array([[ 0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    1.25008389]]),
        np.array([[ 0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    0.1371526 , -0.89534369, -0.02674214,  0.19370934, -0.01860719,
                    0.04346211,  0.18064138,  0.02980028,  0.3216331 , -0.01067888,
                    0.17926357,  0.08900908,  0.13814524, -1.02408371,  0.26238819,
                    1.37496887]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l1_lag_6', 'l1_lag_7', 'l1_lag_8', 'l1_lag_9', 'l1_lag_10',
         'l1_lag_11', 'l1_lag_12', 'l1_lag_13', 'l1_lag_14', 'l1_lag_15',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'l2_lag_6', 'l2_lag_7', 'l2_lag_8', 'l2_lag_9', 'l2_lag_10',
         'l2_lag_11', 'l2_lag_12', 'l2_lag_13', 'l2_lag_14', 'l2_lag_15',
         'exog'],
        [1, 2, 3, 4, 5],
        pd.date_range(start='2003-04-01', periods=5, freq='MS')
    )
    
    for step in range(len(results[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_window_features():
    """
    Test _create_predict_inputs output with window_features.
    """
    series = pd.DataFrame(
        {'l1': np.arange(50),
         'l2': np.arange(50, 100)},
        index = pd.date_range('2020-01-01', periods=50, freq='D')
    )
    
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[4, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     window_features    = rolling,
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]]),
         np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]]),
         np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l1_roll_mean_4', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
         'l2_roll_mean_4', 'l2_roll_median_5', 'l2_roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_with_2_window_features():
    """
    Test _create_predict_inputs output with 2 window_features.
    """
    series = pd.DataFrame(
        {'l1': np.arange(50),
         'l2': np.arange(50, 100)},
        index = pd.date_range('2020-01-01', periods=50, freq='D')
    )
    
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[4, 5])
    rolling_2 = RollingFeatures(stats=['sum'], window_sizes=6)

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     window_features    = [rolling, rolling_2],
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]]),
         np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]]),
         np.array([[49., 48., 47., 47.5, 47., 279., 99., 98., 97., 97.5, 97., 579.]])],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l1_roll_mean_4', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
         'l2_roll_mean_4', 'l2_roll_median_5', 'l2_roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])


def test_create_predict_inputs_output_window_features_and_no_lags():
    """
    Test _create_predict_inputs output with window_features and no lags.
    """
    series = pd.DataFrame(
        {'l1': np.arange(50),
         'l2': np.arange(50, 100)},
        index = pd.date_range('2020-01-01', periods=50, freq='D')
    )
    
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[4, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = None,
                     window_features    = rolling,
                     transformer_series = None
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[47.5, 47., 279., 97.5, 97., 579.]]),
         np.array([[47.5, 47., 279., 97.5, 97., 579.]]),
         np.array([[47.5, 47., 279., 97.5, 97., 579.]])],
        ['l1_roll_mean_4', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_roll_mean_4', 'l2_roll_median_5', 'l2_roll_sum_6'],
        [1, 2, 3],
        pd.date_range(start='2020-02-20', periods=3, freq='D')
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])