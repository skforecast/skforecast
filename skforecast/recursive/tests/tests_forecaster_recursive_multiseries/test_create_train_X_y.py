# Unit test _create_train_X_y ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(4)),  
                           '2': pd.Series(np.arange(4))})
    exog = pd.Series(['A', 'B', 'C', 'D'], name='exog', dtype='category')
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)

    err_msg = re.escape(
        ("Categorical dtypes in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_and_different_columns_names():
    """
    Test ValueError is raised when the forecaster is fitted and the columns names
    of the series are different from the columns names used to fit the forecaster.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    forecaster.series_names_in_ = ['l1', 'l2']

    new_series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l4': pd.Series(np.arange(10))
    })

    err_msg = re.escape(
        ("Once the Forecaster has been trained, `series` must contain "
         "the same series names as those used during training:\n"
         " Got      : ['l1', 'l4']\n"
         " Expected : ['l1', 'l2']")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=new_series)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_without_exog_and_exog_is_not_None():
    """
    Test ValueError is raised when the forecaster was fitted without exog and
    exog is not None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    forecaster.series_names_in_ = ['l1', 'l2']
    forecaster.exog_names_in_ = None

    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })
    exog = pd.Series(np.arange(10), name='exog')

    err_msg = re.escape(
        ("Once the Forecaster has been trained, `exog` must be `None` "
         "because no exogenous variables were added during training.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_and_different_exog_columns_names():
    """
    Test ValueError is raised when the forecaster is fitted and the columns names
    of exog are different from the columns names used to fit the forecaster.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    forecaster.series_names_in_ = ['l1', 'l2']
    forecaster.exog_names_in_ = ['exog']

    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })
    new_exog = pd.Series(np.arange(10), name='exog2')

    err_msg = re.escape(
        ("Once the Forecaster has been trained, `exog` must contain "
         "the same exogenous variables as those used during training:\n" 
         " Got      : ['exog2']\n"
         " Expected : ['exog']")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=new_exog)


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        transformer_series=StandardScaler(),
        lags=3,
        encoding="onehot"
    )

    results = forecaster._create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 1., 0.],
                             [ 0. , -0.5, -1. , 1., 0.],
                             [ 0.5,  0. , -0.5, 1., 0.],
                             [ 1. ,  0.5,  0. , 1., 0.],
                             [-0.5, -1. , -1.5, 0., 1.],
                             [ 0. , -0.5, -1. , 0., 1.],
                             [ 0.5,  0. , -0.5, 0., 1.],
                             [ 1. ,  0.5,  0. , 0., 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ).astype({'1': int, '2': int}),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=7, step=1),
         '2': pd.RangeIndex(start=0, stop=7, step=1)},
        ['1', '2'],
        ['1', '2'],
        None,
        None,
        None,
        None,
        {'1': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.RangeIndex(start=4, stop=7, step=1),
                  name  = '1',
                  dtype = float
              ),
         '2': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.RangeIndex(start=4, stop=7, step=1),
                  name  = '2',
                  dtype = float
              )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert isinstance(results[5], type(None))
    assert isinstance(results[6], type(None))
    assert isinstance(results[7], type(None))
    assert isinstance(results[8], type(None))
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("encoding, dtype", 
                         [('ordinal'         , int), 
                          ('ordinal_category', 'category'),
                          (None              , int)], 
                         ids = lambda dt: f'encoding, dtype: {dt}')
def test_create_train_X_y_output_when_series_and_exog_is_None_ordinal_encoding(encoding, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    transformer_series=StandardScaler(),
                    lags=3,
                    encoding=encoding
                )

    results = forecaster._create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 0.],
                             [ 0. , -0.5, -1. , 0.],
                             [ 0.5,  0. , -0.5, 0.],
                             [ 1. ,  0.5,  0. , 0.],
                             [-0.5, -1. , -1.5, 1.],
                             [ 0. , -0.5, -1. , 1.],
                             [ 0.5,  0. , -0.5, 1.],
                             [ 1. ,  0.5,  0. , 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast'],
        ).astype({'_level_skforecast': int}).astype({'_level_skforecast': dtype}),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=7, step=1),
         '2': pd.RangeIndex(start=0, stop=7, step=1)},
        ['1', '2'],
        ['1', '2'],
        None,
        None,
        None,
        None,
        {'1': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.RangeIndex(start=4, stop=7, step=1),
                  name  = '1',
                  dtype = float
              ),
         '2': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.RangeIndex(start=4, stop=7, step=1),
                  name  = '2',
                  dtype = float
              )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert isinstance(results[5], type(None))
    assert isinstance(results[6], type(None))
    assert isinstance(results[7], type(None))
    assert isinstance(results[8], type(None))
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_when_series_and_exog_no_pandas_index():
    """
    Test the output of _create_train_X_y when series and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    
    warn_msg = re.escape(
        ("Series {'l3'} are not present in `series`. No last window is stored for them.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg): 
        results = forecaster._create_train_X_y(series=series, exog=exog,
                                               store_last_window=['l1', 'l2', 'l3'])

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 1., 0., 106.],
                             [6., 5., 4., 3., 2., 1., 0., 107.],
                             [7., 6., 5., 4., 3., 1., 0., 108.],
                             [8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("encoding, dtype", 
                         [('ordinal'         , int), 
                          ('ordinal_category', 'category'),
                          (None              , int)], 
                         ids = lambda dt: f'encoding, dtype: {dt}')
def test_create_train_X_y_output_when_series_and_exog_no_pandas_index_ordinal_encoding(encoding, dtype):
    """
    Test the output of _create_train_X_y when series and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(
                     LinearRegression(),
                     lags=5,
                     encoding=encoding,
                     transformer_series=None
                 )
    
    warn_msg = re.escape(
        ("Series {'l3'} are not present in `series`. No last window is stored for them.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg): 
        results = forecaster._create_train_X_y(series=series, exog=exog,
                                               store_last_window=['l1', 'l2', 'l3'])

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 0., 105.],
                             [5., 4., 3., 2., 1., 0., 106.],
                             [6., 5., 4., 3., 2., 0., 107.],
                             [7., 6., 5., 4., 3., 0., 108.],
                             [8., 7., 6., 5., 4., 0., 109.],
                             [4., 3., 2., 1., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 1., 106.],
                             [6., 5., 4., 3., 2., 1., 107.],
                             [7., 6., 5., 4., 3., 1., 108.],
                             [8., 7., 6., 5., 4., 1., 109.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast', 'exog']
        ).astype({'_level_skforecast': int}).astype({'_level_skforecast': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas series of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=['1'])

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 1., 0., 106.],
                             [6., 5., 4., 3., 2., 1., 0., 107.],
                             [7., 6., 5., 4., 3., 1., 0., 108.],
                             [8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '1', '2', 'exog']
        ).astype({'1': int, '2': int, 'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=10, step=1),
         '2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['1', '2'],
        ['1', '2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'1': pd.Series(
                  data  = np.array([5., 6., 7., 8., 9.]),
                  index = pd.RangeIndex(start=5, stop=10, step=1),
                  name  = '1',
                  dtype = float
              )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='ordinal_category',
                                              transformer_series=None)
    
    warn_msg = re.escape(
        ("Series {'3'} are not present in `series`. No last window is stored for them.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster._create_train_X_y(series=series, exog=exog,
                                               store_last_window=['3'])    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 0., 106., 1006.],
                             [6., 5., 4., 3., 2., 0., 107., 1007.],
                             [7., 6., 5., 4., 3., 0., 108., 1008.],
                             [8., 7., 6., 5., 4., 0., 109., 1009.],
                             [4., 3., 2., 1., 0., 1., 105., 1005.],
                             [5., 4., 3., 2., 1., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 1., 107., 1007.],
                             [7., 6., 5., 4., 3., 1., 108., 1008.],
                             [8., 7., 6., 5., 4., 1., 109., 1009.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast', 'exog_1', 'exog_2']
        ).astype(
            {'_level_skforecast': int,
             'exog_1': dtype, 'exog_2': dtype}
        ).astype(
            {'_level_skforecast': 'category'}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=10, step=1),
         '2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['1', '2'],
        ['1', '2'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=False)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            l1   = [1.] * 5 + [0.] * 5, 
            l2   = [0.] * 5 + [1.] * 5,
            exog = exog_values * 5 + exog_values * 5
        ).astype(
            {'l1': int, 'l2': int, 'exog': dtype}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='ordinal_category',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=False)    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            _level_skforecast = [0] * 5 + [1] * 5,
            exog_1 = v_exog_1 * 5 + v_exog_1 * 5, 
            exog_2 = v_exog_2 * 5 + v_exog_2 * 5
        ).astype({'_level_skforecast': int}
        ).astype(
            {'_level_skforecast': 'category', 
             'exog_1': dtype, 'exog_2': dtype}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_category():
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=False)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            l1   = [1.]*5 + [0.]*5, 
            l2   = [0.]*5 + [1.]*5,
            exog = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10))
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=False)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            l1     = [1.] * 5 + [0.] * 5, 
            l2     = [0.] * 5 + [1.] * 5,
            exog_1 = pd.Categorical([5, 6, 7, 8, 9] * 2, categories=range(10)),
            exog_2 = pd.Categorical([105, 106, 107, 108, 109] * 2, categories=range(100, 110))
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=False)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 1., 0., 106., 1006.],
                             [6., 5., 4., 3., 2., 1., 0., 107., 1007.],
                             [7., 6., 5., 4., 3., 1., 0., 108., 1008.],
                             [8., 7., 6., 5., 4., 1., 0., 109., 1009.],
                             [4., 3., 2., 1., 0., 0., 1., 105., 1005.],
                             [5., 4., 3., 2., 1., 0., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 0., 1., 107., 1007.],
                             [7., 6., 5., 4., 3., 0., 1., 108., 1008.],
                             [8., 7., 6., 5., 4., 0., 1., 109., 1009.]],
                             dtype=float),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog_1', 'exog_2']
        ).assign(
            exog_3 = pd.Categorical([105, 106, 107, 108, 109] * 2, categories=range(100, 110)), 
        ).astype(
            {'l1': int, 'l2': int, 
             'exog_1': float, 'exog_2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2', 'exog_3'],
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes, 
         'exog_2': exog['exog_2'].dtypes,
         'exog_3': exog['exog_3'].dtypes},
         None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


@pytest.mark.parametrize("encoding, dtype", 
                         [('ordinal'         , int), 
                          ('ordinal_category', 'category'),
                          (None              , int),], 
                         ids = lambda dt: f'encoding, dtype: {dt}')
def test_create_train_X_y_output_when_series_and_exog_is_dataframe_datetime_index(encoding, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    series = pd.DataFrame({'1': np.arange(7, dtype=float), 
                           '2': np.arange(7, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=7, freq='D'))
    exog = pd.DataFrame({'exog_1': np.arange(100, 107, dtype=float),
                         'exog_2': np.arange(1000, 1007, dtype=float)},
                        index = pd.date_range("1990-01-01", periods=7, freq='D'))
                         
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog,
                                           store_last_window=True)

    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 0., 103., 1003.],
                             [3.0, 2.0, 1.0, 0., 104., 1004.],
                             [4.0, 3.0, 2.0, 0., 105., 1005.],
                             [5.0, 4.0, 3.0, 0., 106., 1006.],
                             [2.0, 1.0, 0.0, 1., 103., 1003.],
                             [3.0, 2.0, 1.0, 1., 104., 1004.],
                             [4.0, 3.0, 2.0, 1., 105., 1005.],
                             [5.0, 4.0, 3.0, 1., 106., 1006.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                               '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 
                       '_level_skforecast', 'exog_1', 'exog_2']
        ).astype({'_level_skforecast': int}
        ).astype({'_level_skforecast': dtype}
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                             '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.date_range("1990-01-01", periods=7, freq='D'),
         '2': pd.date_range("1990-01-01", periods=7, freq='D')},
        ['1', '2'],
        ['1', '2'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 
         'exog_2': exog['exog_2'].dtypes},
        {'1': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.date_range("1990-01-05", periods=3, freq='D'),
                  name  = '1',
                  dtype = float
              ),
         '2': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.date_range("1990-01-05", periods=3, freq='D'),
                  name  = '2',
                  dtype = float
              )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of _create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
                    regressor          = LinearRegression(),
                    lags               = 5,
                    encoding           = 'onehot',
                    transformer_series = StandardScaler()
                )
    results = forecaster._create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 1.,  0.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 1.,  0.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 1.,  0.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 1.,  0.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 1.,  0.],
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 0.,  1.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 0.,  1.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 0.,  1.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 0.,  1.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 0.,  1.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'l1', 'l2']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ,
                              0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        None,
        None,
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of _create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = 'onehot',
                     transformer_series = None,
                     transformer_exog   = StandardScaler()
                 )

    results = forecaster._create_train_X_y(series=series, store_last_window=False)
    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 1., 0.],
                             [3.0, 2.0, 1.0, 1., 0.],
                             [4.0, 3.0, 2.0, 1., 0.],
                             [5.0, 4.0, 3.0, 1., 0.],
                             [2.0, 1.0, 0.0, 0., 1.],
                             [3.0, 2.0, 1.0, 0., 1.],
                             [4.0, 3.0, 2.0, 0., 1.],
                             [5.0, 4.0, 3.0, 0., 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ).astype({'1': int, '2': int}
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=7, step=1),
         '2': pd.RangeIndex(start=0, stop=7, step=1)},
        ['1', '2'],
        ['1', '2'],
        None,
        None,
        None,
        None,
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    assert results[9] == expected[9]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog(transformer_series):
    """
    Test the output of _create_train_X_y when using transformer_series and 
    transformer_exog.
    """
    series = pd.DataFrame({'1': np.arange(10, dtype=float), 
                           '2': np.arange(10, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=10, freq='D'))
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = 'onehot',
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.87038828, -1.21854359, -1.5666989 , 1., 0.,  0.49084060, 1., 0.],
                       [-0.52223297, -0.87038828, -1.21854359, 1., 0.,  0.16171381, 1., 0.],
                       [-0.17407766, -0.52223297, -0.87038828, 1., 0., -0.30205575, 0., 1.],
                       [ 0.17407766, -0.17407766, -0.52223297, 1., 0., -1.14980658, 0., 1.],
                       [ 0.52223297,  0.17407766, -0.17407766, 1., 0.,  1.98188469, 0., 1.],
                       [ 0.87038828,  0.52223297,  0.17407766, 1., 0., -0.00284958, 0., 1.],
                       [ 1.21854359,  0.87038828,  0.52223297, 1., 0., -1.17972719, 0., 1.],
                       [-0.87038828, -1.21854359, -1.5666989 , 0., 1.,  0.49084060, 1., 0.],
                       [-0.52223297, -0.87038828, -1.21854359, 0., 1.,  0.16171381, 1., 0.],
                       [-0.17407766, -0.52223297, -0.87038828, 0., 1., -0.30205575, 0., 1.],
                       [ 0.17407766, -0.17407766, -0.52223297, 0., 1., -1.14980658, 0., 1.],
                       [ 0.52223297,  0.17407766, -0.17407766, 0., 1.,  1.98188469, 0., 1.],
                       [ 0.87038828,  0.52223297,  0.17407766, 0., 1., -0.00284958, 0., 1.],
                       [ 1.21854359,  0.87038828,  0.52223297, 0., 1., -1.17972719, 0., 1.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                               '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07',
                               '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2', 
                       'exog_1', 'exog_2_a', 'exog_2_b']
        ).astype({'1': int, '2': int}
        ),
        pd.Series(
            data  = np.array([-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828,
                               1.21854359,  1.5666989 , -0.52223297, -0.17407766,  0.17407766,
                               0.52223297,  0.87038828,  1.21854359,  1.5666989 ]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                               '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07',
                               '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.date_range("1990-01-01", periods=10, freq='D'),
         '2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['1', '2'],
        ['1', '2'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2_a', 'exog_2_b'],
        {'exog_1': exog['exog_1'].dtypes, 
         'exog_2': exog['exog_2'].dtypes},
        {'1': pd.Series(
                  data  = np.array([7., 8., 9.]),
                  index = pd.date_range("1990-01-08", periods=3, freq='D'),
                  name  = '1',
                  dtype = float
              ),
         '2': pd.Series(
                  data  = np.array([7., 8., 9.]),
                  index = pd.date_range("1990-01-08", periods=3, freq='D'),
                  name  = '2',
                  dtype = float
              )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_when_series_different_length_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when series has 2 columns with different 
    lengths and exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 1., 0., 106., 1006.],
                             [6., 5., 4., 3., 2., 1., 0., 107., 1007.],
                             [7., 6., 5., 4., 3., 1., 0., 108., 1008.],
                             [8., 7., 6., 5., 4., 1., 0., 109., 1009.],
                             [6., 5., 4., 3., 2., 0., 1., 107., 1007.],
                             [7., 6., 5., 4., 3., 0., 1., 108., 1008.],
                             [8., 7., 6., 5., 4., 0., 1., 109., 1009.]],
                             dtype=float),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10', 
                               '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog_1', 'exog_2']
        ).assign(exog_3 = pd.Categorical([105, 106, 107, 108, 109, 
                                          107, 108, 109], categories=range(100, 110))
        ).astype({
            'l1': int, 'l2': int, 
            'exog_1': float, 'exog_2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 7, 8, 9]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10', 
                               '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2', 'exog_3'],
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes, 
         'exog_2': exog['exog_2'].dtypes,
         'exog_3': exog['exog_3'].dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), 'l3': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_with_different_series_lengths(transformer_series):
    """
    Test the output of _create_train_X_y when using transformer_series and 
    transformer_exog with series with different lengths.
    """
    series = pd.DataFrame({'l1': np.arange(10, dtype=float), 
                           'l2': pd.Series([np.nan, np.nan, 
                                            2., 3., 4., 5., 6., 7., 8., 9.]), 
                           'l3': pd.Series([np.nan, np.nan, np.nan, np.nan, 
                                            4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = 'onehot',
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog, store_last_window=False)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.8703882797784892,  -1.2185435916898848,  -1.5666989036012806,  1.0, 0.0, 0.0,  0.42685655,  0.0, 1.0],
                       [-0.5222329678670935,  -0.8703882797784892,  -1.2185435916898848,  1.0, 0.0, 0.0,  0.13481233,  1.0, 0.0],
                       [-0.17407765595569785, -0.5222329678670935,  -0.8703882797784892,  1.0, 0.0, 0.0, -0.27670452,  0.0, 1.0],
                       [ 0.17407765595569785, -0.17407765595569785, -0.5222329678670935,  1.0, 0.0, 0.0, -1.02893962,  1.0, 0.0],
                       [ 0.5222329678670935,   0.17407765595569785, -0.17407765595569785, 1.0, 0.0, 0.0,  1.74990535,  0.0, 1.0],
                       [ 0.8703882797784892,   0.5222329678670935,   0.17407765595569785, 1.0, 0.0, 0.0, -0.01120978,  1.0, 0.0],
                       [ 1.2185435916898848,   0.8703882797784892,   0.5222329678670935,  1.0, 0.0, 0.0, -1.05548910,  0.0, 1.0],
                       [-0.6546536707079772,  -1.091089451179962,   -1.5275252316519468,  0.0, 1.0, 0.0, -0.27670452,  0.0, 1.0],
                       [-0.2182178902359924,  -0.6546536707079772,  -1.091089451179962,   0.0, 1.0, 0.0, -1.02893962,  1.0, 0.0],
                       [ 0.2182178902359924,  -0.2182178902359924,  -0.6546536707079772,  0.0, 1.0, 0.0,  1.74990535,  0.0, 1.0],
                       [ 0.6546536707079772,   0.2182178902359924,  -0.2182178902359924,  0.0, 1.0, 0.0, -0.01120978,  1.0, 0.0],
                       [ 1.091089451179962,    0.6546536707079772,   0.2182178902359924,  0.0, 1.0, 0.0, -1.05548910,  0.0, 1.0],
                       [-0.29277002188455997, -0.8783100656536799,  -1.4638501094227998,  0.0, 0.0, 1.0,  1.74990535,  0.0, 1.0],
                       [ 0.29277002188455997, -0.29277002188455997, -0.8783100656536799,  0.0, 0.0, 1.0, -0.01120978,  1.0, 0.0],
                       [ 0.8783100656536799,   0.29277002188455997, -0.29277002188455997, 0.0, 0.0, 1.0, -1.05548910,  0.0, 1.0]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-08', '1990-01-09', '1990-01-10',]
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'l1', 'l2', 'l3',
                       'exog_1', 'exog_2_a', 'exog_2_b']
        ).astype({'l1': int, 'l2': int, 'l3': int}
        ),
        pd.Series(
            data  = np.array([-0.5222329678670935, -0.17407765595569785, 0.17407765595569785, 0.5222329678670935, 0.8703882797784892, 1.2185435916898848, 1.5666989036012806, 
                              -0.2182178902359924, 0.2182178902359924, 0.6546536707079772, 1.091089451179962, 1.5275252316519468, 
                              0.29277002188455997, 0.8783100656536799, 1.4638501094227998]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-08', '1990-01-09', '1990-01-10',]
                          )
                      ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l3': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2_a', 'exog_2_b'],
        {'exog_1': exog['exog_1'].dtypes, 
         'exog_2': exog['exog_2'].dtypes},
         None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_y_train():
    """
    Test the output of _create_train_X_y when series is a DataFrame and y_train
    has NaNs. Also test the MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[5, 'l1'] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=False)
    
    warn_msg = re.escape(
        ("NaNs detected in `y_train`. They have been dropped because the "
         "target variable cannot have NaN values. Same rows have been "
         "dropped from `X_train` to maintain alignment. This is caused by "
         "series with interspersed NaNs.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):    
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[np.nan, 4., 3., 2., 1., 1., 0., 106.],
                             [6., np.nan, 4., 3., 2., 1., 0., 107.],
                             [7., 6., np.nan, 4., 3., 1., 0., 108.],
                             [8., 7., 6., np.nan, 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([np.nan, 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_y_train_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and y_train
    has NaNs with datetime index. Also test the MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[5, 'l1'] = np.nan
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=False)
    
    warn_msg = re.escape(
        ("NaNs detected in `y_train`. They have been dropped because the "
         "target variable cannot have NaN values. Same rows have been "
         "dropped from `X_train` to maintain alignment. This is caused by "
         "series with interspersed NaNs.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):    
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[np.nan, 4., 3., 2., 1., 1., 0., 106.],
                             [6., np.nan, 4., 3., 2., 1., 0., 107.],
                             [7., 6., np.nan, 4., 3., 1., 0., 108.],
                             [8., 7., 6., np.nan, 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([np.nan, 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_True():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=True`. Also test the MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[3, 'l1'] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=True)
    
    warn_msg = re.escape(
        ("NaNs detected in `X_train`. They have been dropped. If "
         "you want to keep them, set `forecaster.dropna_from_series = False`. " 
         "Same rows have been removed from `y_train` to maintain alignment. "
         "This caused by series with interspersed NaNs.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):    
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([9, 5, 6, 7, 8, 9]),
            index = pd.Index([9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5, 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_True_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=True` with datetime index. Also test the 
    MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[3, 'l1'] = np.nan
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=True)
    
    warn_msg = re.escape(
        ("NaNs detected in `X_train`. They have been dropped. If "
         "you want to keep them, set `forecaster.dropna_from_series = False`. " 
         "Same rows have been removed from `y_train` to maintain alignment. "
         "This caused by series with interspersed NaNs.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):    
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([9, 5, 6, 7, 8, 9]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-10',
                             '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_False():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=False`. Also test the MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[3, 'l1'] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=False)
    
    warn_msg = re.escape(
        ("NaNs detected in `X_train`. Some regressors do not allow "
         "NaN values during training. If you want to drop them, "
         "set `forecaster.dropna_from_series = True`.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):    
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., np.nan, 2., 1., 0., 1., 0., 105.],
                             [5., 4., np.nan, 2., 1., 1., 0., 106.],
                             [6., 5., 4., np.nan, 2., 1., 0., 107.],
                             [7., 6., 5., 4., np.nan, 1., 0., 108.],
                             [8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.RangeIndex(start=5, stop=10, step=1),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_False_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=False` with datetime index. Also test the 
    MissingValuesWarning message.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.loc[3, 'l1'] = np.nan
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=False)
    
    warn_msg = re.escape(
        ("NaNs detected in `X_train`. Some regressors do not allow "
         "NaN values during training. If you want to drop them, "
         "set `forecaster.dropna_from_series = True`.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., np.nan, 2., 1., 0., 1., 0., 105.],
                             [5., 4., np.nan, 2., 1., 1., 0., 106.],
                             [6., 5., 4., np.nan, 2., 1., 0., 107.],
                             [7., 6., 5., 4., np.nan, 1., 0., 108.],
                             [8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ).astype({'l1': int, 'l2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_ValueError_create_train_X_series_DataFrame_exog_dict_and_empty_X_train_drop_nan_True():
    """
    Test ValueError is raised when series is a DataFrame and exog dict is used
    and all samples have been removed due to NaNs in exog.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')

    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float),
                         'exog_2': np.arange(200, 210, dtype=float)})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')

    exog_dict = {
        'l1': exog['exog_1'].copy(),
        'l2': exog[['exog_2']].copy()
    }
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              encoding='onehot',
                                              dropna_from_series=True)
    
    error_msg = re.escape(
        ("All samples have been removed due to NaNs. Set "
         "`forecaster.dropna_from_series = False` or review `exog` values.")
    )
    with pytest.raises(ValueError, match = error_msg):
        forecaster._create_train_X_y(series=series, exog=exog_dict)


def test_create_train_X_y_output_series_dict_and_exog_dict():
    """
    Test the output of _create_train_X_y when series is a dict and exog is a
    dict.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(15, 20, dtype=float)),
        'l3': pd.Series(np.arange(20, 25, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    series['l2'].loc[2] = np.nan
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-05", periods=5, freq='D')
    series['l3'].index = pd.date_range("1990-01-03", periods=5, freq='D')

    exog = {
        'l1': pd.Series(np.arange(100, 110), name='exog_1', dtype=float),
        'l2': None,
        'l3': pd.DataFrame({'exog_1': np.arange(203, 207, dtype=float),
                            'exog_2': ['a', 'b', 'a', 'b']})
    }
    exog['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog['l3'].index = pd.date_range("1990-01-03", periods=4, freq='D')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding='onehot',
                                              transformer_series=None,
                                              dropna_from_series=False)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[np.nan, 2., 1., 1., 0., 0., 104., np.nan],
                             [4., np.nan, 2., 1., 0., 0., 105., np.nan],
                             [5., 4., np.nan, 1., 0., 0., 106., np.nan],
                             [6., 5., 4., 1., 0., 0., 107., np.nan],
                             [7., 6., 5., 1., 0., 0., 108., np.nan],
                             [8., 7., 6., 1., 0., 0., 109., np.nan],
                             [np.nan, 16., 15., 0., 1., 0., np.nan, np.nan],
                             [18., np.nan, 16., 0., 1., 0., np.nan, np.nan],
                             [22., 21., 20., 0., 0., 1., 206., 'b'],
                             [23., 22., 21., 0., 0., 1., np.nan, np.nan]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                              '1990-01-09', '1990-01-10',
                              '1990-01-08', '1990-01-09', 
                              '1990-01-06', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'l1', 'l2', 'l3', 
                       'exog_1', 'exog_2']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float, 'l1': float, 
                  'l2': float, 'l3': float, 'exog_1': float, 'exog_2': object}
        ).astype({'l1': int, 'l2': int, 'l3': int}
        ),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9., 18., 19., 23., 24.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08',
                             '1990-01-09', '1990-01-10',
                             '1990-01-08', '1990-01-09', 
                             '1990-01-06', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=5, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=5, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['l1'].dtypes,
         'exog_2': exog['l3'].dtypes},
        {'l1': pd.Series(
                   data  = np.array([7., 8., 9.]),
                   index = pd.date_range("1990-01-08", periods=3, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([np.nan, 18., 19.]),
                   index = pd.date_range("1990-01-07", periods=3, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([22., 23., 24.]),
                   index = pd.date_range("1990-01-05", periods=3, freq='D'),
                   name  = 'l3',
                   dtype = float
               )
        }
    )
    expected[0].iloc[[0, 1, 2, 3, 4, 5, 6, 7, 9], -1] = np.nan

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize(
    "encoding, dtype",
    [("ordinal", int), 
     ("ordinal_category", "category"),
     (None, int)],
    ids=lambda dt: f"encoding, dtype: {dt}",
)
def test_create_train_X_y_output_series_dict_and_exog_dict_ordinal_encoding(
    encoding, dtype
):
    """
    Test the output of _create_train_X_y when series is a dict and exog is a
    dict with ordinal encoding.
    """
    series = {
        "l1": pd.Series(np.arange(10, dtype=float)),
        "l2": pd.Series(np.arange(15, 20, dtype=float)),
        "l3": pd.Series(np.arange(20, 25, dtype=float)),
    }
    series["l1"].loc[3] = np.nan
    series["l2"].loc[2] = np.nan
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=5, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=5, freq="D")

    exog = {
        "l1": pd.Series(np.arange(100, 110), name="exog_1", dtype=float),
        "l3": pd.DataFrame(
            {"exog_1": np.arange(203, 207, dtype=float), "exog_2": ["a", "b", "a", "b"]}
        ),
    }
    exog["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    exog["l3"].index = pd.date_range("1990-01-03", periods=4, freq="D")

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, transformer_series=None,
        dropna_from_series=False
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data=np.array(
                [
                    [np.nan, 2.0, 1.0, 0, 104.0, np.nan],
                    [4.0, np.nan, 2.0, 0, 105.0, np.nan],
                    [5.0, 4.0, np.nan, 0, 106.0, np.nan],
                    [6.0, 5.0, 4.0, 0, 107.0, np.nan],
                    [7.0, 6.0, 5.0, 0, 108.0, np.nan],
                    [8.0, 7.0, 6.0, 0, 109.0, np.nan],
                    [np.nan, 16.0, 15.0, 1, np.nan, np.nan],
                    [18.0, np.nan, 16.0, 1, np.nan, np.nan],
                    [22.0, 21.0, 20.0, 2, 206.0, "b"],
                    [23.0, 22.0, 21.0, 2, np.nan, np.nan],
                ]
            ),
            index=pd.Index(
                pd.DatetimeIndex(
                    [
                        "1990-01-05",
                        "1990-01-06",
                        "1990-01-07",
                        "1990-01-08",
                        "1990-01-09",
                        "1990-01-10",
                        "1990-01-08",
                        "1990-01-09",
                        "1990-01-06",
                        "1990-01-07",
                    ]
                )
            ),
            columns=[
                "lag_1",
                "lag_2",
                "lag_3",
                "_level_skforecast",
                "exog_1",
                "exog_2",
            ],
        )
        .astype(
            {
                "lag_1": float,
                "lag_2": float,
                "lag_3": float,
                "_level_skforecast": int,
                "exog_1": float,
                "exog_2": object,
            }
        )
        .astype({"_level_skforecast": dtype}),
        pd.Series(
            data=np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 18.0, 19.0, 23.0, 24.0]),
            index=pd.Index(
                pd.DatetimeIndex(
                    [
                        "1990-01-05",
                        "1990-01-06",
                        "1990-01-07",
                        "1990-01-08",
                        "1990-01-09",
                        "1990-01-10",
                        "1990-01-08",
                        "1990-01-09",
                        "1990-01-06",
                        "1990-01-07",
                    ]
                )
            ),
            name="y",
            dtype=float,
        ),
        {
            "l1": pd.date_range("1990-01-01", periods=10, freq="D"),
            "l2": pd.date_range("1990-01-05", periods=5, freq="D"),
            "l3": pd.date_range("1990-01-03", periods=5, freq="D"),
        },
        ["l1", "l2", "l3"],
        ["l1", "l2", "l3"],
        ["exog_1", "exog_2"],
        None,
        ['exog_1', 'exog_2'],
        {"exog_1": exog["l1"].dtypes, "exog_2": exog["l3"].dtypes},
        {
            "l1": pd.Series(
                data=np.array([7.0, 8.0, 9.0]),
                index=pd.date_range("1990-01-08", periods=3, freq="D"),
                name="l1",
                dtype=float,
            ),
            "l2": pd.Series(
                data=np.array([np.nan, 18.0, 19.0]),
                index=pd.date_range("1990-01-07", periods=3, freq="D"),
                name="l2",
                dtype=float,
            ),
            "l3": pd.Series(
                data=np.array([22.0, 23.0, 24.0]),
                index=pd.date_range("1990-01-05", periods=3, freq="D"),
                name="l3",
                dtype=float,
            ),
        },
    )
    expected[0].iloc[[0, 1, 2, 3, 4, 5, 6, 7, 9], -1] = np.nan

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("encoding, encoding_mapping_", 
                         [('ordinal'         , {'1': 0, '2': 1}), 
                          ('ordinal_category', {'1': 0, '2': 1}),
                          ('onehot'          , {'1': 0, '2': 1}),
                          (None              , {'1': 0, '2': 1})], 
                         ids = lambda dt: f'encoding, mapping: {dt}')
def test_create_train_X_y_encoding_mapping(encoding, encoding_mapping_):
    """
    Test the encoding mapping of _create_train_X_y.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=3,
        encoding=encoding,
    )
    _ = forecaster._create_train_X_y(series=series)
    
    assert forecaster.encoding_mapping_ == encoding_mapping_


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
def test_create_train_X_y_output_when_series_and_exog_and_differentitation_1_and_already_trained(fit_forecaster):
    """
    Test the output of _create_train_X_y when differentiation=1 and already 
    trained forecaster.
    """
    series = {
        "l1": pd.Series(np.arange(10, dtype=float)),
        "l2": pd.Series(np.arange(15, 20, dtype=float)),
        "l3": pd.Series(np.arange(20, 25, dtype=float)),
    }
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=5, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=5, freq="D")
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding           = 'ordinal',
                                              transformer_series = StandardScaler(),
                                              differentiation    = 1)
    
    if fit_forecaster:
        forecaster.fit(series=series)

    results = forecaster._create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[0.34815531, 0.34815531, 0.34815531, 0],
                             [0.34815531, 0.34815531, 0.34815531, 0],
                             [0.34815531, 0.34815531, 0.34815531, 0],
                             [0.34815531, 0.34815531, 0.34815531, 0],
                             [0.34815531, 0.34815531, 0.34815531, 0],
                             [0.34815531, 0.34815531, 0.34815531, 0],
                             [0.70710678, 0.70710678, 0.70710678, 1],
                             [0.70710678, 0.70710678, 0.70710678, 2]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                               '1990-01-09', '1990-01-10',
                               '1990-01-09', 
                               '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'_level_skforecast': int}
        ),
        pd.Series(
            data  = np.array([0.34815531, 0.34815531, 0.34815531, 0.34815531,
                              0.34815531, 0.34815531, 0.70710678, 0.70710678]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08',
                             '1990-01-09', '1990-01-10',
                             '1990-01-09', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=5, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=5, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        None,
        None,
        None,
        None,
        {'l1': pd.Series(
                   data  = np.array([6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-07", periods=4, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([16., 17., 18., 19.]),
                   index = pd.date_range("1990-01-06", periods=4, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([21., 22., 23., 24.]),
                   index = pd.date_range("1990-01-04", periods=4, freq='D'),
                   name  = 'l3',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]    
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
def test_create_train_X_y_output_when_series_and_exog_and_already_trained_encoding_None(fit_forecaster):
    """
    Test the output of _create_train_X_y when encoding None and already 
    trained forecaster.
    """
    series = {
        "l1": pd.Series(np.arange(10, dtype=float)),
        "l2": pd.Series(np.arange(15, 20, dtype=float)),
        "l3": pd.Series(np.arange(20, 25, dtype=float)),
    }
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=5, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=5, freq="D")
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding           = None,
                                              transformer_series = StandardScaler())
    
    if fit_forecaster:
        forecaster.fit(series=series)

    results = forecaster._create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[-1.24514561, -1.36966017, -1.49417474, 0.],
                             [-1.12063105, -1.24514561, -1.36966017, 0.],
                             [-0.99611649, -1.12063105, -1.24514561, 0.],
                             [-0.87160193, -0.99611649, -1.12063105, 0.],
                             [-0.74708737, -0.87160193, -0.99611649, 0.],
                             [-0.62257281, -0.74708737, -0.87160193, 0.],
                             [-0.49805825, -0.62257281, -0.74708737, 0.],
                             [ 0.62257281,  0.49805825,  0.37354368, 1.],
                             [ 0.74708737,  0.62257281,  0.49805825, 1.],
                             [ 1.24514561,  1.12063105,  0.99611649, 2.],
                             [ 1.36966017,  1.24514561,  1.12063105, 2.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                               '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-08', '1990-01-09', 
                               '1990-01-06', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'_level_skforecast': int}
        ),
        pd.Series(
            data  = np.array([
                        -1.12063105, -0.99611649, -0.87160193, -0.74708737, -0.62257281,
                        -0.49805825, -0.37354368,  0.74708737,  0.87160193,  1.36966017,
                        1.49417474]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-04', '1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-08', '1990-01-09', 
                             '1990-01-06', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=5, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=5, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        None,
        None,
        None,
        None,
        {'l1': pd.Series(
                   data  = np.array([7., 8., 9.]),
                   index = pd.date_range("1990-01-08", periods=3, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([17., 18., 19.]),
                   index = pd.date_range("1990-01-07", periods=3, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([22., 23., 24.]),
                   index = pd.date_range("1990-01-05", periods=3, freq='D'),
                   name  = 'l3',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
def test_create_train_X_y_output_when_series_and_exog_and_differentitation_1_and_already_trained_encoding_None(fit_forecaster):
    """
    Test the output of _create_train_X_y when differentiation=1,
    encoding None and already trained forecaster.
    """
    series = {
        "l1": pd.Series(np.arange(10, dtype=float)),
        "l2": pd.Series(np.arange(15, 20, dtype=float)),
        "l3": pd.Series(np.arange(20, 25, dtype=float)),
    }
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=5, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=5, freq="D")
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding           = None,
                                              transformer_series = StandardScaler(),
                                              differentiation    = 1)
    
    if fit_forecaster:
        forecaster.fit(series=series)

    results = forecaster._create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 0],
                             [0.12451456, 0.12451456, 0.12451456, 1],
                             [0.12451456, 0.12451456, 0.12451456, 2]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                               '1990-01-09', '1990-01-10',
                               '1990-01-09', 
                               '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'_level_skforecast': int}
        ),
        pd.Series(
            data  = np.array([0.12451456, 0.12451456, 0.12451456, 0.12451456,
                              0.12451456, 0.12451456, 0.12451456, 0.12451456]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08',
                             '1990-01-09', '1990-01-10',
                             '1990-01-09', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=5, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=5, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        None,
        None,
        None,
        None,
        {'l1': pd.Series(
                   data  = np.array([6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-07", periods=4, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([16., 17., 18., 19.]),
                   index = pd.date_range("1990-01-06", periods=4, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([21., 22., 23., 24.]),
                   index = pd.date_range("1990-01-04", periods=4, freq='D'),
                   name  = 'l3',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_series_dict_and_exog_dict_window_features():
    """
    Test the output of _create_train_X_y when series is a dict and exog is a
    dict with window features.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(15, 20, dtype=float)),
        'l3': pd.Series(np.arange(20, 25, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    series['l2'].loc[2] = np.nan
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-05", periods=5, freq='D')
    series['l3'].index = pd.date_range("1990-01-03", periods=5, freq='D')

    exog = {
        'l1': pd.Series(np.arange(100, 110), name='exog_1', dtype=float),
        'l2': None,
        'l3': pd.DataFrame({'exog_1': np.arange(203, 209, dtype=float),
                            'exog_2': ['a', 'b', 'a', 'b', 'a', 'b']})
    }
    exog['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog['l3'].index = pd.date_range("1990-01-03", periods=6, freq='D')

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[3, 3])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[4])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3,
        encoding='onehot',
        window_features=[rolling, rolling_2],
        transformer_series=None,
        dropna_from_series=False
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[np.nan, 2., 1., np.nan, np.nan, np.nan, 1., 0., 0., 104., np.nan],
                             [4., np.nan, 2., np.nan, np.nan, np.nan, 1., 0., 0., 105., np.nan],
                             [5., 4., np.nan, np.nan, np.nan, np.nan, 1., 0., 0., 106., np.nan],
                             [6., 5., 4., 5.0, 5.0, np.nan, 1., 0., 0., 107., np.nan],
                             [7., 6., 5., 6.0, 6.0, 22.0, 1., 0., 0., 108., np.nan],
                             [8., 7., 6., 7.0, 7.0, 26.0, 1., 0., 0., 109., np.nan],
                             [18., np.nan, 16., np.nan, np.nan, np.nan, 0., 1., 0., np.nan, np.nan],
                             [23., 22., 21., 22.0, 22.0, 86.0, 0., 0., 1., 207., 'a']]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                               '1990-01-09', '1990-01-10',
                               '1990-01-09', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'roll_median_3', 'roll_sum_4',
                       'l1', 'l2', 'l3', 'exog_1', 'exog_2']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float, 
                  'roll_mean_3': float, 'roll_median_3': float, 'roll_sum_4': float, 
                  'l1': float, 'l2': float, 'l3': float, 'exog_1': float, 'exog_2': object}
        ).astype({'l1': int, 'l2': int, 'l3': int}
        ),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9., 19., 24.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08',
                             '1990-01-09', '1990-01-10',
                             '1990-01-09', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=5, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=5, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        ['exog_1', 'exog_2'],
        ['roll_mean_3', 'roll_median_3', 'roll_sum_4'],
        ['exog_1', 'exog_2'],
        {'exog_1': exog['l1'].dtypes,
         'exog_2': exog['l3'].dtypes},
        {'l1': pd.Series(
                   data  = np.array([6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-07", periods=4, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([16., np.nan, 18., 19.]),
                   index = pd.date_range("1990-01-06", periods=4, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([21., 22., 23., 24.]),
                   index = pd.date_range("1990-01-04", periods=4, freq='D'),
                   name  = 'l3',
                   dtype = float
               )
        }
    )
    expected[0].iloc[[0, 1, 2, 3, 4, 5, 6], -1] = np.nan

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])


def test_create_train_X_y_output_when_series_and_exog_with_window_features_no_lags():
    """
    Test the output of _create_train_X_y when series and exog with window
    features but no lags.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, 20, dtype=float))})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[3, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[4])
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=None, window_features=[rolling, rolling_2]
    )
    
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[3., 2., 10., 0., 105.],
                             [4., 3., 14., 0., 106.],
                             [5., 4., 18., 0., 107.],
                             [6., 5., 22., 0., 108.],
                             [7., 6., 26., 0., 109.],
                             [13., 12., 50., 1., 105.],
                             [14., 13., 54., 1., 106.],
                             [15., 14., 58., 1., 107.],
                             [16., 15., 62., 1., 108.],
                             [17., 16., 66., 1., 109.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['roll_mean_3', 'roll_median_5', 'roll_sum_4', 
                       '_level_skforecast', 'exog']
        ).astype({'_level_skforecast': int}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9., 15., 16., 17., 18., 19.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-01", periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['roll_mean_3', 'roll_median_5', 'roll_sum_4'],
        ['exog'],
        {'exog': exog.dtypes},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([15., 16., 17., 18., 19.]),
                   index = pd.date_range("1990-01-06", periods=5, freq='D'),
                   name  = 'l2',
                   dtype = float
               )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    for k in results[9].keys():
        pd.testing.assert_series_equal(results[9][k], expected[9][k])
