# Unit test create_predict_X ForecasterDirectMultiVariate
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
from skforecast.utils import transform_numpy
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
def test_create_predict_X_TypeError_when_steps_list_contain_floats(steps):
    """
    Test create_predict_X TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_predict_X(steps=steps)


def test_create_predict_X_NotFittedError_when_fitted_is_False():
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
        forecaster.create_predict_X(steps=5)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_X_output(steps):
    """
    Test create_predict_X output.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=steps)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979, 0.98555979],
            'l2_lag_1': [0.34345601, 0.34345601, 0.34345601],
            'l2_lag_2': [0.2408559, 0.2408559, 0.2408559],
            'l2_lag_3': [0.39887629, 0.39887629, 0.39887629]
        },
        index = pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_list_interspersed():
    """
    Test create_predict_X output when steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=3, steps=5, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=[1, 4])

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979],
            'l2_lag_1': [0.34345601, 0.34345601],
            'l2_lag_2': [0.2408559, 0.2408559],
            'l2_lag_3': [0.39887629, 0.39887629]
        },
        index = pd.Index([50, 53], dtype=int)
    )
    expected.index = expected.index.astype(results.index.dtype)
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_different_lags():
    """
    Test create_predict_X output when different
    lags configuration for each series.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': [1, 7]}, 
                                               steps=3, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=3)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979, 0.98555979],
            'l1_lag_4': [0.48303426, 0.48303426, 0.48303426],
            'l1_lag_5': [0.25045537, 0.25045537, 0.25045537],
            'l2_lag_1': [0.34345601, 0.34345601, 0.34345601],
            'l2_lag_7': [0.76254781, 0.76254781, 0.76254781]
        },
        index = pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_lags_dict_with_None_in_level_lags():
    """
    Test create_predict_X output when lags is a 
    dict and level has None lags configuration.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=3)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979, 0.98555979],
            'l1_lag_4': [0.48303426, 0.48303426, 0.48303426],
            'l1_lag_5': [0.25045537, 0.25045537, 0.25045537]
        },
        index = pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_lags_dict_with_None_but_no_in_level():
    """
    Test create_predict_X output when lags is a 
    dict with None values.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=3)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979, 0.98555979],
            'l1_lag_4': [0.48303426, 0.48303426, 0.48303426],
            'l1_lag_5': [0.25045537, 0.25045537, 0.25045537]
        },
        index = pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_last_window():
    """
    Test create_predict_X output when external last_window.
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
    results = forecaster.create_predict_X(steps=[1, 2], last_window=last_window)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979],
            'l2_lag_1': [0.34345601, 0.34345601],
            'l2_lag_2': [0.2408559, 0.2408559],
            'l2_lag_3': [0.39887629, 0.39887629]
        },
        index = pd.RangeIndex(start=50, stop=52, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_exog():
    """
    Test create_predict_X output when exog.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series.iloc[:40,], exog=exog.iloc[:40, 0])
    results = forecaster.create_predict_X(steps=None, exog=exog.iloc[40:43, 0])

    expected = (
        [np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.09332671]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.29686078]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.92758424]])],
        [1, 2, 3],
        pd.RangeIndex(start=37, stop=40, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'exog_1']
    )

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.50183668, 0.50183668, 0.50183668],
            'l1_lag_2': [0.94416002, 0.94416002, 0.94416002],
            'l1_lag_3': [0.89338916, 0.89338916, 0.89338916],
            'l2_lag_1': [0.39818568, 0.39818568, 0.39818568],
            'l2_lag_2': [0.30476807, 0.30476807, 0.30476807],
            'l2_lag_3': [0.04359146, 0.04359146, 0.04359146],
            'exog_1':   [0.09332671, 0.29686078, 0.92758424]
        },
        index = pd.RangeIndex(start=40, stop=43, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_with_transform_series():
    """
    Test create_predict_X output when StandardScaler.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.create_predict_X()

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.47762884, 0.47762884, 0.47762884, 0.47762884, 0.47762884],
            'l1_lag_2': [0.07582436, 0.07582436, 0.07582436, 0.07582436, 0.07582436],
            'l1_lag_3': [2.08066403, 2.08066403, 2.08066403, 2.08066403, 2.08066403],
            'l1_lag_4': [-0.08097053, -0.08097053, -0.08097053, -0.08097053, -0.08097053],
            'l1_lag_5': [-1.08141835, -1.08141835, -1.08141835, -1.08141835, -1.08141835],
            'l2_lag_1': [-0.61148712, -0.61148712, -0.61148712, -0.61148712, -0.61148712],
            'l2_lag_2': [-1.00971677, -1.00971677, -1.00971677, -1.00971677, -1.00971677],
            'l2_lag_3': [-0.39638015, -0.39638015, -0.39638015, -0.39638015, -0.39638015],
            'l2_lag_4': [-1.35798666, -1.35798666, -1.35798666, -1.35798666, -1.35798666],
            'l2_lag_5': [0.74018589, 0.74018589, 0.74018589, 0.74018589, 0.74018589]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_with_transform_series_as_dict():
    """
    Test create_predict_X output when transformer_series
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
    results = forecaster.create_predict_X()

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.47762884, 0.47762884, 0.47762884, 0.47762884, 0.47762884],
            'l1_lag_2': [0.07582436, 0.07582436, 0.07582436, 0.07582436, 0.07582436],
            'l1_lag_3': [2.08066403, 2.08066403, 2.08066403, 2.08066403, 2.08066403],
            'l1_lag_4': [-0.08097053, -0.08097053, -0.08097053, -0.08097053, -0.08097053],
            'l1_lag_5': [-1.08141835, -1.08141835, -1.08141835, -1.08141835, -1.08141835],
            'l2_lag_1': [0.33426983, 0.33426983, 0.33426983, 0.33426983, 0.33426983],
            'l2_lag_2': [0.22949344, 0.22949344, 0.22949344, 0.22949344, 0.22949344],
            'l2_lag_3': [0.39086564, 0.39086564, 0.39086564, 0.39086564, 0.39086564],
            'l2_lag_4': [0.13786173, 0.13786173, 0.13786173, 0.13786173, 0.13786173],
            'l2_lag_5': [0.68990237, 0.68990237, 0.68990237, 0.68990237, 0.68990237]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_create_predict_X_output_with_transform_series_and_transform_exog(n_jobs):
    """
    Test create_predict_X output when StandardScaler
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
    results = forecaster.create_predict_X(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.47762884, 0.47762884, 0.47762884, 0.47762884, 0.47762884],
            'l1_lag_2': [0.07582436, 0.07582436, 0.07582436, 0.07582436, 0.07582436],
            'l1_lag_3': [2.08066403, 2.08066403, 2.08066403, 2.08066403, 2.08066403],
            'l1_lag_4': [-0.08097053, -0.08097053, -0.08097053, -0.08097053, -0.08097053],
            'l1_lag_5': [-1.08141835, -1.08141835, -1.08141835, -1.08141835, -1.08141835],
            'l2_lag_1': [-0.61148712, -0.61148712, -0.61148712, -0.61148712, -0.61148712],
            'l2_lag_2': [-1.00971677, -1.00971677, -1.00971677, -1.00971677, -1.00971677],
            'l2_lag_3': [-0.39638015, -0.39638015, -0.39638015, -0.39638015, -0.39638015],
            'l2_lag_4': [-1.35798666, -1.35798666, -1.35798666, -1.35798666, -1.35798666],
            'l2_lag_5': [0.74018589, 0.74018589, 0.74018589, 0.74018589, 0.74018589],
            'exog_1': [-0.02551075, 0.51927383, -1.470801945, -1.38212079, -0.70392558],
            'exog_2_a': [1., 1., 1., 1., 1.],
            'exog_2_b': [0., 0., 0., 0., 0.]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X output when using HistGradientBoostingRegressor 
    and categorical variables.
    """
    df_exog = pd.DataFrame({
        'exog_1': exog['exog_1'],
        'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
        'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)}
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
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
        data = {
            'l1_lag_1': [0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453,
                         0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453],
            'l1_lag_2': [0.51948512, 0.51948512, 0.51948512, 0.51948512, 0.51948512,
                         0.51948512, 0.51948512, 0.51948512, 0.51948512, 0.51948512],
            'l1_lag_3': [0.98555979, 0.98555979, 0.98555979, 0.98555979, 0.98555979,
                         0.98555979, 0.98555979, 0.98555979, 0.98555979, 0.98555979],
            'l1_lag_4': [0.48303426, 0.48303426, 0.48303426, 0.48303426, 0.48303426,
                         0.48303426, 0.48303426, 0.48303426, 0.48303426, 0.48303426],
            'l1_lag_5': [0.25045537, 0.25045537, 0.25045537, 0.25045537, 0.25045537,
                         0.25045537, 0.25045537, 0.25045537, 0.25045537, 0.25045537],
            'l2_lag_1': [0.34345601, 0.34345601, 0.34345601, 0.34345601, 0.34345601,
                         0.34345601, 0.34345601, 0.34345601, 0.34345601, 0.34345601],
            'l2_lag_2': [0.2408559 , 0.2408559 , 0.2408559 , 0.2408559 , 0.2408559 ,
                         0.2408559 , 0.2408559 , 0.2408559 , 0.2408559 , 0.2408559 ],
            'l2_lag_3': [0.39887629, 0.39887629, 0.39887629, 0.39887629, 0.39887629,
                         0.39887629, 0.39887629, 0.39887629, 0.39887629, 0.39887629],
            'l2_lag_4': [0.15112745, 0.15112745, 0.15112745, 0.15112745, 0.15112745,
                         0.15112745, 0.15112745, 0.15112745, 0.15112745, 0.15112745],
            'l2_lag_5': [0.6917018 , 0.6917018 , 0.6917018 , 0.6917018 , 0.6917018 ,
                         0.6917018 , 0.6917018 , 0.6917018 , 0.6917018 , 0.6917018 ],
            'exog_2': [0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
            'exog_3': [0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
            'exog_1': [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                       0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781]
        },
        index = pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_same_predictions_as_predict():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method.
    """

    end_train = '2003-03-01 23:59:00'
    arr = data.to_numpy(copy=True)
    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = [1, 5],
                     window_features    = [rolling, rolling_2],
                     steps              = 6,
                     transformer_series = None,
                     transformer_exog   = None,
                     differentiation    = None
                 )
    forecaster.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(range(1, forecaster.steps + 1)):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy().item()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method with transformation.
    """

    end_train = '2003-03-01 23:59:00'
    arr = data.to_numpy(copy=True)
    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = [1, 5],
                     window_features    = [rolling, rolling_2],
                     steps              = 6,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = None
                 )
    forecaster.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(range(1, forecaster.steps + 1)):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        results = transform_numpy(
                      array             = results,
                      transformer       = forecaster.transformer_series_[forecaster.level],
                      fit               = False,
                      inverse_transform = True
                  )
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy().item()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers_diff():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method with transformation and differentiation.
    """

    end_train = '2003-03-01 23:59:00'
    arr = data.to_numpy(copy=True)
    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = [1, 5],
                     window_features    = [rolling, rolling_2],
                     steps              = 6,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = 1
                 )
    forecaster.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(range(1, forecaster.steps + 1)):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        results = forecaster.differentiator_[forecaster.level].inverse_transform_next_window(results)
        results = transform_numpy(
                      array             = results,
                      transformer       = forecaster.transformer_series_[forecaster.level],
                      fit               = False,
                      inverse_transform = True
                  )
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy().item()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)
