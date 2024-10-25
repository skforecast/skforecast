# Unit test _create_train_X_y ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.exceptions import MissingValuesWarning
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import data  # to test results when using differentiation


def test_create_train_X_y_ValueError_when_len_y_less_than_window_size():
    """
    Test ValueError is raised when len(y) <= window_size.
    """
    y = pd.Series(np.arange(5))

    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    err_msg = re.escape(
        ("Length of `y` must be greater than the maximum window size "
         "needed by the forecaster.\n"
         "    Length `y`: 5.\n"
         "    Max window size: 5.\n"
         "    Lags window size: 5.\n"
         "    Window features window size: None.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=6)
    forecaster = ForecasterRecursive(LinearRegression(), lags=2, window_features=rolling)
    err_msg = re.escape(
        ("Length of `y` must be greater than the maximum window size "
         "needed by the forecaster.\n"
         "    Length `y`: 5.\n"
         "    Max window size: 6.\n"
         "    Lags window size: 2.\n"
         "    Window features window size: 6.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    y = pd.Series(np.arange(3))
    exog = pd.Series(['A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterRecursive(LinearRegression(), lags=2)

    err_msg = re.escape(
        ("Categorical dtypes in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_MissingValuesWarning_when_exog_has_missing_values():
    """
    Test _create_train_X_y is issues a MissingValuesWarning when exog has missing values.
    """
    y = pd.Series(np.arange(4))
    exog = pd.Series([1, 2, 3, np.nan], name='exog')
    forecaster = ForecasterRecursive(LinearRegression(), lags=2)

    warn_msg = re.escape(
        ("`exog` has missing values. Most machine learning models do "
         "not allow missing values. Fitting the forecaster may fail.")  
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


@pytest.mark.parametrize("y                        , exog", 
                         [(pd.Series(np.arange(50), name='y'), pd.Series(np.arange(10), name='exog')), 
                          (pd.Series(np.arange(10), name='y'), pd.Series(np.arange(50), name='exog')), 
                          (pd.Series(np.arange(10), name='y'), pd.DataFrame(np.arange(50).reshape(25, 2), columns=['exog_1', 'exog_2']))])
def test_create_train_X_y_ValueError_when_len_y_is_different_from_len_exog(y, exog):
    """
    Test ValueError is raised when length of y is not equal to length exog.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `y`. "
         f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)

  
def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index():
    """
    Test ValueError is raised when y and exog have different index.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("Different index for `y` and `exog`. They must be equal "
         "to ensure the correct alignment of values.")  
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y    = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'), name='y'),
            exog = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1), name='exog')
        )


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_None():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is None.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = None
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert isinstance(results[2], type(None))
    assert isinstance(results[3], type(None))
    assert isinstance(results[4], type(None))
    assert results[5] == expected[5]
    assert isinstance(results[6], type(None))


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of _create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    y = pd.Series(np.arange(10), index=np.arange(3, 13), dtype=float)
    exog = pd.Series(np.arange(100, 110), index=np.arange(3, 13), 
                     name='exog', dtype=float)
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with 2 columns of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2']
        ).astype({'exog_1': float, 'exog_2': float}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog=exog_values * 5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({
               'exog_1': v_exog_1 * 10,
               'exog_2': v_exog_2 * 10,
           })
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1=v_exog_1 * 5, exog_2=v_exog_2 * 5).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog=pd.Categorical(range(5, 10), categories=range(10))),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            exog_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with 3 columns of float, int, category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).astype({'exog_1': float, 
                  'exog_2': int}
        ).assign(exog_3=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        ['exog_1', 'exog_2', 'exog_3'],
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_y_is_series_10_and_transformer_y_is_StandardScaler():
    """
    Test the output of _create_train_X_y when exog is None and transformer_y
    is StandardScaler.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_y = StandardScaler()
                 )
    
    results = forecaster._create_train_X_y(y=pd.Series(np.arange(10), dtype=float))
    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ],
                             [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                             [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                             [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert isinstance(results[2], type(None))
    assert isinstance(results[3], type(None))
    assert isinstance(results[4], type(None))
    assert results[5] == expected[5]
    assert isinstance(results[6], type(None))


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of _create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_exog = StandardScaler()
                 )
    
    results = forecaster._create_train_X_y(y=pd.Series(np.arange(10), dtype=float))
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None
    )
    
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert isinstance(results[2], type(None))
    assert isinstance(results[3], type(None))
    assert isinstance(results[4], type(None))
    assert results[5] == expected[5]
    assert isinstance(results[6], type(None))


def test_create_train_X_y_output_when_transformer_y_and_transformer_exog():
    """
    Test the output of _create_train_X_y when using transformer_y and transformer_exog.
    """
    y = pd.Series(np.arange(8), dtype = float)
    y.index = pd.date_range("1990-01-01", periods=8, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']},
               index = pd.date_range("1990-01-01", periods=8, freq='D')
           )

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
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[0.21821789, -0.21821789, -0.65465367, -1.09108945,
                              -1.52752523, -0.25107995,  0.        ,  1.        ],
                             [0.65465367,  0.21821789, -0.21821789, -0.65465367,
                              -1.09108945, 1.79326881,  0.        ,  1.        ],
                             [1.09108945,  0.65465367,  0.21821789, -0.21821789,
                              -0.65465367, 0.01673866,  0.        ,  1.        ]]),
            index   = pd.date_range("1990-01-06", periods=3, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1',
                       'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([0.65465367, 1.09108945, 1.52752523]),
            index = pd.date_range("1990-01-06", periods=3, freq='D'),
            name  = 'y',
            dtype = float
        ),
        ['col_1', 'col_2'],
        None,
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1', 'col_2_a', 'col_2_b'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert isinstance(results[3], type(None))
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda fitted: f'fit_forecaster: {fitted}')
def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_1(fit_forecaster):
    """
    Test the output of _create_train_X_y when using differentiation=1. Comparing 
    the matrix created with and without differentiating the series.
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

    forecaster_1 = ForecasterRecursive(LinearRegression(), lags=5)
    forecaster_2 = ForecasterRecursive(LinearRegression(), lags=5, differentiation=1)
    
    if fit_forecaster:
        forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])

    output_1 = forecaster_1._create_train_X_y(
                   y    = data_diff.loc[:end_train],
                   exog = exog_diff.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   y    = data.loc[:end_train],
                   exog = exog.loc[:end_train]
               )
    
    pd.testing.assert_frame_equal(output_1[0], output_2[0], check_names=True)
    pd.testing.assert_series_equal(output_1[1], output_2[1], check_names=True)
    assert output_1[2] == output_2[2]
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    for k in output_1[6].keys():
        assert output_1[6][k] == output_2[6][k]


def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_2():
    """
    Test the output of _create_train_X_y when using differentiation=2. Comparing 
    the matrix created with and without differentiating the series.
    """

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=2)
    data_diff_2 = diferenciator.fit_transform(data.to_numpy())
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterRecursive(LinearRegression(), lags=5)
    forecaster_2 = ForecasterRecursive(LinearRegression(), lags=5, differentiation=2)

    output_1 = forecaster_1._create_train_X_y(
                   y    = data_diff_2.loc[:end_train],
                   exog = exog_diff_2.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   y    = data.loc[:end_train],
                   exog = exog.loc[:end_train]
               )
    
    pd.testing.assert_frame_equal(output_1[0], output_2[0], check_names=True)
    pd.testing.assert_series_equal(output_1[1], output_2[1], check_names=True)
    assert output_1[2] == output_2[2]
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    for k in output_1[6].keys():
        assert output_1[6][k] == output_2[6][k]


def test_create_train_X_y_output_when_window_features_and_exog():
    """
    Test the output of _create_train_X_y when using window_features and exog 
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
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursive(LinearRegression(), lags=5, window_features=rolling)
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 107.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 108.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 109.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 110.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 111.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 112.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 113.],
                             [13., 12., 11., 10., 9., 11., 11., 63., 114.]]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog'],
        {'exog': exog_datetime.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_two_window_features_and_exog():
    """
    Test the output of _create_train_X_y when using 2 window_features and exog 
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
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursive(
        LinearRegression(), lags=5, window_features=[rolling, rolling_2]
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 107.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 108.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 109.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 110.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 111.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 112.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 113.],
                             [13., 12., 11., 10., 9., 11., 11., 63., 114.]]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog'],
        {'exog': exog_datetime.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_window_features_lags_None_and_exog():
    """
    Test the output of _create_train_X_y when using window_features and exog 
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
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursive(LinearRegression(), lags=None, window_features=rolling)
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        pd.DataFrame(
            data = np.array([[3., 3., 15., 106.],
                             [4., 4., 21., 107.],
                             [5., 5., 27., 108.],
                             [6., 6., 33., 109.],
                             [7., 7., 39., 110.],
                             [8., 8., 45., 111.],
                             [9., 9., 51., 112.],
                             [10., 10., 57., 113.],
                             [11., 11., 63., 114.]]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        ['exog'],
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        ['exog'],
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog'],
        {'exog': exog_datetime.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]


def test_create_train_X_y_output_when_window_features_and_exog_transformers_diff():
    """
    Test the output of _create_train_X_y when using window_features, exog, 
    transformers and differentiation.
    """
    y_datetime = pd.Series(
        [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='y', dtype=float
    )
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
               'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
               index = pd.date_range('2000-01-01', periods=10, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )

    forecaster = ForecasterRecursive(
                     LinearRegression(), 
                     lags             = [1, 5], 
                     window_features  = rolling,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                     differentiation  = 2
                 )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog)
    
    expected = (
        pd.DataFrame(
            data = np.array([[-1.56436158, -0.14173746, -0.89489489, -0.27035108,  0.04040264,
                               0.        ,  1.        ],
                             [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472, -1.32578962,
                               0.        ,  1.        ],
                             [-0.24672817, -0.49870587, -0.83943662,  0.75068358,  1.12752513,
                               0.        ,  1.        ]]),
            index   = pd.date_range('2000-01-08', periods=3, freq='D'),
            columns = ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4',
                       'col_1', 'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([1.8635851, -0.24672817, -4.60909217]),
            index = pd.date_range('2000-01-08', periods=3, freq='D'),
            name  = 'y',
            dtype = float
        ),
        ['col_1', 'col_2'],
        ['roll_ratio_min_max_4', 'roll_median_4'],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4',
         'col_1', 'col_2_a', 'col_2_b'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    for k in results[6].keys():
        assert results[6][k] == expected[6][k]
