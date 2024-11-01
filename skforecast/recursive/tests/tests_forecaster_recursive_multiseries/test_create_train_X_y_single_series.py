# Unit test _create_train_X_y_single_series ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_create_train_X_y_single_series_ValueError_when_len_y_less_than_window_size():
    """
    Test ValueError is raised when len(y) <= window_size.
    """
    y = pd.Series(np.arange(5), name='l1')
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        "Length of 'l1' must be greater than the maximum window size "
        "needed by the forecaster.\n"
        "    Length 'l1': 5.\n"
        "    Max window size: 5.\n"
        "    Lags window size: 5.\n"
        "    Window features window size: None."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y_single_series(y=y, ignore_exog=True)

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=6)
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=2, window_features=rolling)
    err_msg = re.escape(
        ("Length of 'l1' must be greater than the maximum window size "
         "needed by the forecaster.\n"
         "    Length 'l1': 5.\n"
         "    Max window size: 6.\n"
         "    Lags window size: 2.\n"
         "    Window features window size: 6.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y_single_series(y=y, ignore_exog=True)


@pytest.mark.parametrize("ignore_exog", 
                         [True, False], 
                         ids = lambda ie: f'ignore_exog: {ie}')
def test_create_train_X_y_single_series_output_when_series_and_exog_is_None(ignore_exog):
    """
    Test the output of _create_train_X_y_single_series when exog is None. 
    Check cases `ignore_exog` True and False.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.transformer_series_ = {'l1': StandardScaler()}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = ignore_exog,
                  exog        = None
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 'l1'],
                             [ 0. , -0.5, -1. , 'l1'],
                             [ 0.5,  0. , -0.5, 'l1'],
                             [ 1. ,  0.5,  0. , 'l1']]),
            index   = pd.RangeIndex(start=3, stop=7, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        None,
        pd.DataFrame(
            data    = np.nan,
            columns = ['_dummy_exog_col_to_keep_shape'],
            index   = pd.RangeIndex(start=3, stop=7, step=1)
        ),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5]),
            index = pd.RangeIndex(start=3, stop=7, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    if ignore_exog:
        assert isinstance(results[2], type(None))
    else:
        pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_and_exog():
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 'l1'],
                             [5., 4., 3., 2., 1., 'l1'],
                             [6., 5., 4., 3., 2., 'l1'],
                             [7., 6., 5., 4., 3., 'l1'],
                             [8., 7., 6., 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        None,
        pd.DataFrame(
            data  = np.array([105., 106., 107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column float and one column of cateogry.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame({'exog_1': np.arange(10, 20, dtype=float),
                         'exog_2': pd.Categorical(range(100, 110))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 'l1'],
                             [5., 4., 3., 2., 1., 'l1'],
                             [6., 5., 4., 3., 2., 'l1'],
                             [7., 6., 5., 4., 3., 'l1'],
                             [8., 7., 6., 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        None,
        pd.DataFrame(
            data  = np.array([15., 16., 17., 18., 19.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog_1']
        ).assign(exog_2 = pd.Categorical([105, 106, 107, 108, 109], categories=range(100, 110))),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_and_exog_is_DataFrame_datetime_index():
    """
    Test the output of _create_train_X_y_single_series when series and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')
    y.index = pd.date_range("1990-01-01", periods=7, freq='D')
    exog = pd.DataFrame({
               'exog_1': np.arange(100, 107, dtype=float),
               'exog_2': np.arange(1000, 1007, dtype=float)},
               index = pd.date_range("1990-01-01", periods=7, freq='D')
           )

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 'l1'],
                             [3.0, 2.0, 1.0, 'l1'],
                             [4.0, 3.0, 2.0, 'l1'],
                             [5.0, 4.0, 3.0, 'l1']]),
            index   = pd.date_range("1990-01-04", periods=4, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        None,
        pd.DataFrame(
            data = np.array([[103., 1003.],
                             [104., 1004.],
                             [105., 1005.],
                             [106., 1006.]]),
            index   = pd.date_range("1990-01-04", periods=4, freq='D'),
            columns = ['exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6.], dtype=float),
            index = pd.date_range("1990-01-04", periods=4, freq='D'),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_and_exog_is_DataFrame_with_NaNs():
    """
    Test the output of _create_train_X_y_single_series when series and 
    exog is a pandas dataframe with two columns and NaNs in between.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    y.iloc[6:8] = np.nan
    exog = pd.DataFrame({
               'exog_1': np.arange(100, 110, dtype=float),
               'exog_2': np.arange(1000, 1010, dtype=float)}
           )
    exog.iloc[2:7, 0] = np.nan

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 'l1'],
                             [5., 4., 3., 2., 1., 'l1'],
                             [np.nan, 5., 4., 3., 2., 'l1'],
                             [np.nan, np.nan, 5., 4., 3., 'l1'],
                             [8., np.nan, np.nan, 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        None,
        pd.DataFrame(
            data = np.array([[np.nan, 1005.],
                             [np.nan, 1006.],
                             [107., 1007.],
                             [108., 1008.],
                             [109., 1009.]]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([5., np.nan, np.nan, 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_transformer_and_fitted():
    """
    Test the output of _create_train_X_y_single_series when Forecaster as
    already been fitted, transformer is MinMaxScaler() and has a different 
    series as input.
    """
    y = pd.Series(np.arange(9, dtype=float), name='l1')
    transformer = MinMaxScaler()
    transformer.fit(y.values.reshape(-1, 1))

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': transformer}
    forecaster.is_fitted = True

    new_y = pd.Series(np.arange(10, 19, dtype=float), name='l1')
    results = forecaster._create_train_X_y_single_series(y=new_y, ignore_exog=True)
    
    expected = (
        pd.DataFrame(
            data = np.array([[1.5  , 1.375, 1.25 , 'l1'],
                             [1.625, 1.5  , 1.375, 'l1'],
                             [1.75 , 1.625, 1.5  , 'l1'],
                             [1.875, 1.75 , 1.625, 'l1'],
                             [2.   , 1.875, 1.75 , 'l1'],
                             [2.125, 2.   , 1.875, 'l1']]),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        None,
        None,
        pd.Series(
            data  = np.array([1.625, 1.75, 1.875, 2., 2.125, 2.25]),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    assert isinstance(results[2], type(None))
    pd.testing.assert_series_equal(results[3], expected[3])


@pytest.mark.parametrize("is_fitted", 
                         [True, False], 
                         ids = lambda is_fitted: f'is_fitted: {is_fitted}')
def test_create_train_X_y_single_series_output_when_series_and_exog_and_differentitation_1(is_fitted):
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column and differentiation=1.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series = None,
                                              differentiation    = 1)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.is_fitted = is_fitted

    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1']]),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        None,
        pd.DataFrame(
            data  = np.array([106., 107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([1., 1., 1., 1.], dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


@pytest.mark.parametrize("is_fitted", 
                         [True, False], 
                         ids = lambda is_fitted: f'is_fitted: {is_fitted}')
def test_create_train_X_y_single_series_output_when_series_and_exog_and_differentitation_2(is_fitted):
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column and differentiation=2.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                              transformer_series = None,
                                              differentiation    = 2)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.is_fitted = is_fitted

    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[0., 0., 0., 0., 0., 'l1'],
                             [0., 0., 0., 0., 0., 'l1'],
                             [0., 0., 0., 0., 0., 'l1']]),
            index   = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        None,
        pd.DataFrame(
            data  = np.array([107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([0., 0., 0.], dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_window_features_and_exog():
    """
    Test the output of _create_train_X_y_single_series when using window_features 
    and exog with datetime index.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    exog_datetime = pd.DataFrame(
        np.arange(100, 115, dtype=float), 
        index   = pd.date_range('2000-01-01', periods=15, freq='D'),
        columns = ['exog']
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, window_features=rolling
    )
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y_datetime,
                  ignore_exog = False,
                  exog        = exog_datetime
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 'l1'],
                             [6., 5., 4., 3., 2., 4., 4., 21., 'l1'],
                             [7., 6., 5., 4., 3., 5., 5., 27., 'l1'],
                             [8., 7., 6., 5., 4., 6., 6., 33., 'l1'],
                             [9., 8., 7., 6., 5., 7., 7., 39., 'l1'],
                             [10., 9., 8., 7., 6., 8., 8., 45., 'l1'],
                             [11., 10., 9., 8., 7., 9., 9., 51., 'l1'],
                             [12., 11., 10., 9., 8., 10., 10., 57., 'l1'],
                             [13., 12., 11., 10., 9., 11., 11., 63., 'l1']]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 
                       '_level_skforecast']
        ).astype(
            {'lag_1': float, 'lag_2': float, 'lag_3': float,
             'lag_4': float, 'lag_5': float, 'roll_mean_5': float,
             'roll_median_5': float, 'roll_sum_6': float}
        ),
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        pd.DataFrame(
            data  = np.arange(106, 115, dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_two_window_features_and_exog():
    """
    Test the output of _create_train_X_y_single_series when using 2 window_features 
    and exog with datetime index.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    exog_datetime = pd.DataFrame(
        np.arange(100, 115, dtype=float), 
        index   = pd.date_range('2000-01-01', periods=15, freq='D'),
        columns = ['exog']
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, window_features=[rolling, rolling_2]
    )
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y_datetime,
                  ignore_exog = False,
                  exog        = exog_datetime
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 'l1'],
                             [6., 5., 4., 3., 2., 4., 4., 21., 'l1'],
                             [7., 6., 5., 4., 3., 5., 5., 27., 'l1'],
                             [8., 7., 6., 5., 4., 6., 6., 33., 'l1'],
                             [9., 8., 7., 6., 5., 7., 7., 39., 'l1'],
                             [10., 9., 8., 7., 6., 8., 8., 45., 'l1'],
                             [11., 10., 9., 8., 7., 9., 9., 51., 'l1'],
                             [12., 11., 10., 9., 8., 10., 10., 57., 'l1'],
                             [13., 12., 11., 10., 9., 11., 11., 63., 'l1']]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 
                       '_level_skforecast']
        ).astype(
            {'lag_1': float, 'lag_2': float, 'lag_3': float,
             'lag_4': float, 'lag_5': float, 'roll_mean_5': float,
             'roll_median_5': float, 'roll_sum_6': float}
        ),
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        pd.DataFrame(
            data  = np.arange(106, 115, dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test__create_train_X_y_single_series_output_when_window_features_lags_None_and_exog():
    """
    Test the output of _create_train_X_y_single_series when using window_features 
    and exog with datetime index and lags=None.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    exog_datetime = pd.DataFrame(
        np.arange(100, 115, dtype=float), 
        index   = pd.date_range('2000-01-01', periods=15, freq='D'),
        columns = ['exog']
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=None, window_features=rolling
    )
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y_datetime,
                  ignore_exog = False,
                  exog        = exog_datetime
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[3., 3., 15., 'l1'],
                             [4., 4., 21., 'l1'],
                             [5., 5., 27., 'l1'],
                             [6., 6., 33., 'l1'],
                             [7., 7., 39., 'l1'],
                             [8., 8., 45., 'l1'],
                             [9., 9., 51., 'l1'],
                             [10., 10., 57., 'l1'],
                             [11., 11., 63., 'l1']]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 
                       '_level_skforecast']
        ).astype(
            {'roll_mean_5': float, 'roll_median_5': float, 'roll_sum_6': float}
        ),
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        pd.DataFrame(
            data  = np.arange(106, 115, dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_window_features_and_exog_transformers_diff():
    """
    Test the output of _create_train_X_y_single_series when using window_features, 
    exog, transformers and differentiation.
    """
    y_datetime = pd.Series(
        [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='l1', dtype=float
    )
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
               'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
               index = pd.date_range('2000-01-01', periods=10, freq='D')
           )

    transformer_series = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    # Exog is not transformed as it is done in _create_train_X_y
    forecaster = ForecasterRecursiveMultiSeries(
                     LinearRegression(), 
                     lags               = [1, 5], 
                     window_features    = rolling,
                     encoding           = 'onehot',
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                     differentiation    = 2
                 )
    forecaster.transformer_series_ = {'l1': transformer_series}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}

    results = forecaster._create_train_X_y_single_series(
                  y           = y_datetime,
                  ignore_exog = False,
                  exog        = exog
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[-1.56436158, -0.14173746, -0.89489489, -0.27035108, 'l1'],
                             [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472, 'l1'],
                             [-0.24672817, -0.49870587, -0.83943662,  0.75068358, 'l1']]),
            index   = pd.date_range('2000-01-08', periods=3, freq='D'),
            columns = ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4', 
                       '_level_skforecast']
        ).astype(
            {'lag_1': float, 'lag_5': float, 
             'roll_ratio_min_max_4': float, 'roll_median_4': float}
        ),
        ['roll_ratio_min_max_4', 'roll_median_4'],
        pd.DataFrame(
            data  = np.array([[47.4, 'b'],
                              [14.6, 'b'],
                              [73.5, 'b']]),
            index = pd.date_range('2000-01-08', periods=3, freq='D'),
            columns = ['col_1', 'col_2']
        ).astype({'col_1': float}
        ),
        pd.Series(
            data  = np.array([1.8635851, -0.24672817, -4.60909217]),
            index = pd.date_range('2000-01-08', periods=3, freq='D'),
            name  = 'y',
            dtype = float
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])
    pd.testing.assert_series_equal(results[3], expected[3])
