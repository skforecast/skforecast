# Unit test create_predict_X ForecasterDirect
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
from skforecast.utils import transform_numpy
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y as y_categorical
from .fixtures_forecaster_direct import exog as exog_categorical
from .fixtures_forecaster_direct import data  # to test results when using differentiation


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_X_TypeError_when_steps_list_contain_floats(steps):
    """
    Test create_predict_X TypeError when steps is a list with floats.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(10)))

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    results = forecaster.create_predict_X(steps=steps)

    expected = pd.DataFrame(
        data = {
            'lag_1': [49., 49., 49.],
            'lag_2': [48., 48., 48.],
            'lag_3': [47., 47., 47.]
        },
        index = pd.RangeIndex(start=50, stop=53, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_regressor_is_LinearRegression_with_list_interspersed():
    """
    Test create_predict_X output when steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    results = forecaster.create_predict_X(steps=[1, 4])

    expected = pd.DataFrame(
        data = {
            'lag_1': [49., 49.],
            'lag_2': [48., 48.],
            'lag_3': [47., 47.]
        },
        index = pd.Index([50, 53], dtype=int)
    )
    expected.index = expected.index.astype(results.index.dtype)
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_regressor_is_LinearRegression_using_last_window():
    """
    Test create_predict_X output when external last_window.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    last_window = pd.Series(
        data  = [47., 48., 49.], 
        index = pd.RangeIndex(start=47, stop=50, step=1)
    )
    results = forecaster.create_predict_X(steps=[1, 2, 3, 4], last_window=last_window)

    expected = pd.DataFrame(
        data = {
            'lag_1': [49., 49., 49., 49.],
            'lag_2': [48., 48., 48., 48.],
            'lag_3': [47., 47., 47., 47.]
        },
        index = pd.RangeIndex(start=50, stop=54, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_inputs_error_when_pred_steps_are_not_in_train_steps():
    """
    Test create_predict_X output when pred steps aren't in init steps
    """
    train_steps = [1, 3, 5]
    pred_steps = [1, 4]
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=train_steps)
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float)))
    err_msg = re.escape(
        (f"'steps' should match with steps defined when "
         f"initializing the forecaster."
         f"Got {pred_steps}, but available steps are {np.array(train_steps)}")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_predict_X(steps=pred_steps)


def test_create_predict_X_output_when_exog():
    """
    Test create_predict_X output when exog.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    forecaster.fit(
        y=pd.Series(np.arange(50, dtype=float)),
        exog=pd.Series(np.arange(start=100, stop=150, step=1), name="exog")
    )
    results = forecaster.create_predict_X(
                  steps = 5, 
                  exog  = pd.Series(np.arange(start=25, stop=50, step=0.5, dtype=float),
                                    index=pd.RangeIndex(start=50, stop=100),
                                    name="exog")
               )

    expected = pd.DataFrame(
        data = {
            'lag_1': [49., 49., 49., 49., 49.],
            'lag_2': [48., 48., 48., 48., 48.],
            'lag_3': [47., 47., 47., 47., 47.],
            'exog': [25, 25.5, 26, 26.5, 27]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_inputs_output_when_exog_with_train_steps_interspersed_and_pred_steps_by_default():
    """
    Test create_predict_X output when exog, with steps as interspersed list and no prediction
    steps defined.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=[1,3,5])
    forecaster.fit(
        y=pd.Series(np.arange(50, dtype=float)),
        exog=pd.Series(np.arange(start=100, stop=150, step=1), name="exog")
    )
    results = forecaster.create_predict_X(
                  exog  = pd.Series(np.arange(start=25, stop=50, step=0.5, dtype=float),
                                    index=pd.RangeIndex(start=50, stop=100),
                                    name="exog")
    )
    expected = pd.DataFrame(
        data={
            'lag_1': [49., 49., 49.],
            'lag_2': [48., 48., 48.],
            'lag_3': [47., 47., 47.],
            'exog': [25., 26., 27.]
        },
        index=pd.RangeIndex(start=50, stop=55, step=2)
    )
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_with_transform_y():
    """
    Test create_predict_X output when StandardScaler.
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
    results = forecaster.create_predict_X()

    expected = pd.DataFrame(
        data = {
            'lag_1': [-0.52297655, -0.52297655, -0.52297655, -0.52297655, -0.52297655],
            'lag_2': [-0.28324197, -0.28324197, -0.28324197, -0.28324197, -0.28324197],
            'lag_3': [-0.45194408, -0.45194408, -0.45194408, -0.45194408, -0.45194408],
            'lag_4': [-0.30987914, -0.30987914, -0.30987914, -0.30987914, -0.30987914],
            'lag_5': [-0.1056608, -0.1056608, -0.1056608, -0.1056608, -0.1056608]
        },
        index = pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_create_predict_X_output_with_transform_y_and_transform_exog(n_jobs):
    """
    Test create_predict_X output when StandardScaler
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
    results = forecaster.create_predict_X(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = pd.DataFrame(
        data = {
            'lag_1': [-0.52297655, -0.52297655, -0.52297655, -0.52297655, -0.52297655],
            'lag_2': [-0.28324197, -0.28324197, -0.28324197, -0.28324197, -0.28324197],
            'lag_3': [-0.45194408, -0.45194408, -0.45194408, -0.45194408, -0.45194408],
            'lag_4': [-0.30987914, -0.30987914, -0.30987914, -0.30987914, -0.30987914],
            'lag_5': [-0.1056608, -0.1056608, -0.1056608, -0.1056608, -0.1056608],
            'col_1': [-1.7093071, -1.0430105, 0.372377, 0.2540995, -0.006111],
            'col_2_a': [1., 1., 1., 1., 1.],
            'col_2_b': [0., 0., 0., 0., 0.]
        },
        index = pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X output when using HistGradientBoostingRegressor 
    and categorical variables.
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
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
        data = {
            'lag_1': [0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453,
                      0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453],
            'lag_2': [0.51948512, 0.51948512, 0.51948512, 0.51948512, 0.51948512,
                      0.51948512, 0.51948512, 0.51948512, 0.51948512, 0.51948512],
            'lag_3': [0.98555979, 0.98555979, 0.98555979, 0.98555979, 0.98555979,
                      0.98555979, 0.98555979, 0.98555979, 0.98555979, 0.98555979],
            'lag_4': [0.48303426, 0.48303426, 0.48303426, 0.48303426, 0.48303426,
                      0.48303426, 0.48303426, 0.48303426, 0.48303426, 0.48303426],
            'lag_5': [0.25045537, 0.25045537, 0.25045537, 0.25045537, 0.25045537,
                      0.25045537, 0.25045537, 0.25045537, 0.25045537, 0.25045537],
            'exog_2': [0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
            'exog_3': [0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
            'exog_1': [0.12062867, 0.8263408, 0.60306013, 0.54506801, 0.34276383,
                       0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234]
        },
        index = pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_y_steps_1():
    """
    Test create_predict_X when using LinearRegression as regressor and differentiation=1,
    and transformer_y is StandardScaler, steps=1.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )

    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
                     lags            = [1, 5],
                     steps           = 1,
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster.create_predict_X(exog=exog.loc[end_train:])

    expected = pd.DataFrame(
        data = np.array([
            [0.07503713, -0.01018012,  1.16172882]]
        ),
        columns = ['lag_1', 'lag_5', 'exog'],
        index = pd.date_range(start='2003-04-01', periods=1, freq='MS')
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_y_steps_5():
    """
    Test create_predict_X when using LinearRegression as regressor and differentiation=1,
    and transformer_y is StandardScaler, steps=5.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )

    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
                     lags            = [1, 5],
                     steps           = 5,
                     differentiation = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster.create_predict_X(exog=exog.loc[end_train:])

    expected = pd.DataFrame(
        data = np.array([
            [0.07503713, -0.01018012,  1.16172882],
            [0.07503713, -0.01018012,  0.29468848],
            [0.07503713, -0.01018012, -0.4399757 ],
            [0.07503713, -0.01018012,  1.25008389],
            [0.07503713, -0.01018012,  1.37496887]]
        ),
        columns = ['lag_1', 'lag_5', 'exog'],
        index = pd.date_range(start='2003-04-01', periods=5, freq='MS')
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_window_features_steps_1():
    """
    Test the output of create_predict_X when using window_features and exog 
    with datetime index and steps=1.
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

    forecaster = ForecasterDirect(
        LinearRegression(), steps=1, lags=5, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.create_predict_X(exog=exog_datetime_pred)

    expected = pd.DataFrame(
        data = np.array([
                    [14., 13., 12., 11., 10., 12., 12., 69., 115.]]),
        index   = pd.date_range('2000-01-16', periods=1, freq='D'),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                   'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_window_features_steps_10():
    """
    Test the output of create_predict_X when using window_features and exog 
    with datetime index and steps=10.
    """
    y_datetime = pd.Series(
        np.arange(20), index=pd.date_range('2000-01-01', periods=20, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.DataFrame(
        {'exog_1': np.arange(100, 120), 
         'exog_2': np.arange(200, 220)},
        index=pd.date_range('2000-01-01', periods=20, freq='D')
    )
    exog_datetime_pred = pd.DataFrame(
        {'exog_1': np.arange(120, 130),
         'exog_2': np.arange(220, 230)},
        index=pd.date_range('2000-01-21', periods=10, freq='D')
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
        LinearRegression(), steps=10, lags=5, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.create_predict_X(exog=exog_datetime_pred)

    expected = pd.DataFrame(
        data = np.array([
                    [19., 18., 17., 16., 15., 17., 17., 99., 120., 220.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 121., 221.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 122., 222.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 123., 223.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 124., 224.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 125., 225.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 126., 226.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 127., 227.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 128., 228.],
                    [19., 18., 17., 16., 15., 17., 17., 99., 129., 229.]]),
        index   = pd.date_range('2000-01-21', periods=10, freq='D'),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                   'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_1', 'exog_2']
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_window_features_and_lags_None_steps_10():
    """
    Test the output of create_predict_X when using window_features and exog 
    with datetime index, lags=None and steps=10.
    """
    y_datetime = pd.Series(
        np.arange(20), index=pd.date_range('2000-01-01', periods=20, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.DataFrame(
        {'exog_1': np.arange(100, 120), 
         'exog_2': np.arange(200, 220)},
        index=pd.date_range('2000-01-01', periods=20, freq='D')
    )
    exog_datetime_pred = pd.DataFrame(
        {'exog_1': np.arange(120, 130),
         'exog_2': np.arange(220, 230)},
        index=pd.date_range('2000-01-21', periods=10, freq='D')
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
        LinearRegression(), steps=10, lags=None, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.create_predict_X(exog=exog_datetime_pred)

    expected = pd.DataFrame(
        data = np.array([
                    [17., 17., 99., 120., 220.],
                    [17., 17., 99., 121., 221.],
                    [17., 17., 99., 122., 222.],
                    [17., 17., 99., 123., 223.],
                    [17., 17., 99., 124., 224.],
                    [17., 17., 99., 125., 225.],
                    [17., 17., 99., 126., 226.],
                    [17., 17., 99., 127., 227.],
                    [17., 17., 99., 128., 228.],
                    [17., 17., 99., 129., 229.]]),
        index   = pd.date_range('2000-01-21', periods=10, freq='D'),
        columns = ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_1', 'exog_2']
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_same_predictions_as_predict():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     lags             = [1, 5],
                     window_features  = [rolling, rolling_2],
                     steps            = 6,
                     transformer_y    = None,
                     transformer_exog = None,
                     differentiation  = None
                 )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(forecaster.steps):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method with transformation.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     lags             = [1, 5],
                     window_features  = [rolling, rolling_2],
                     steps            = 6,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     differentiation  = None
                 )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(forecaster.steps):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        results = transform_numpy(
                      array             = results,
                      transformer       = forecaster.transformer_y,
                      fit               = False,
                      inverse_transform = True
                  )
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)


def test_create_predict_X_same_predictions_as_predict_transformers_diff():
    """
    Test create_predict_X matrix returns the same predictions as predict method
    when passing to the regressor predict method with transformation and differentiation.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     lags             = [1, 5],
                     window_features  = [rolling, rolling_2],
                     steps            = 6,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     differentiation  = 1
                 )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    X_predict = forecaster.create_predict_X(exog=exog.loc[end_train:])

    for i, step in enumerate(forecaster.steps):
        results = forecaster.regressors_[step].predict(X_predict.iloc[[i]])
        results = forecaster.differentiator.inverse_transform_next_window(results)
        results = transform_numpy(
                      array             = results,
                      transformer       = forecaster.transformer_y,
                      fit               = False,
                      inverse_transform = True
                  )
        expected = forecaster.predict(steps=[step], exog=exog.loc[end_train:]).to_numpy()
        np.testing.assert_array_almost_equal(results, expected, decimal=7)
