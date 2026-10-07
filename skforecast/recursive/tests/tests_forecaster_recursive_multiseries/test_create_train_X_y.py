# Unit test _create_train_X_y ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning, MissingExogWarning
from skforecast.exceptions import IgnoredArgumentWarning, DataTypeWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures, CalendarFeatures, reshape_series_wide_to_long
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_dict_unordered,
    exog_dict_unordered
)


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(4)),  
                           '2': pd.Series(np.arange(4))})
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.Series(['A', 'B', 'C', 'D'], name='exog', dtype='category')
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, categorical_features=None
    )

    err_msg = re.escape(
        "Categorical dtypes in exog must contain only integer values. "
        "See skforecast docs for more info about how to include "
        "categorical features https://skforecast.org/"
        "latest/user_guides/categorical-features.html"
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
    new_series.index = pd.date_range(start='2000-01-01', periods=len(new_series), freq='D')
    new_series = reshape_series_wide_to_long(new_series)

    err_msg = re.escape(
        "Once the Forecaster has been trained, `series` must contain "
        "the same series names as those used during training:\n"
        " Got      : ['l1', 'l4']\n"
        " Expected : ['l1', 'l2']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=new_series)


def test_create_train_X_y_TypeError_when_calendar_features_and_index_not_datetime():
    """
    Test TypeError is raised when calendar_features is not None and the index of 
    series is not a DatetimeIndex.
    """
    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })

    calendar = CalendarFeatures()
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, calendar_features=calendar
    )
    err_msg = re.escape(
        "When `calendar_features` is not `None`, the index of `series` "
        "must be a pandas DatetimeIndex."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_calendar_feature_name_duplicated_with_exog():
    """
    Test ValueError is raised when a calendar feature has the same name as an
    exogenous variable, producing duplicated feature names.
    """
    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10), index=pd.date_range('2000-01-01', periods=10, freq='D')),
        'l2': pd.Series(np.arange(10), index=pd.date_range('2000-01-01', periods=10, freq='D'))
    })
    exog = pd.Series(
        np.arange(100, 110, dtype=float),
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='day_of_week'
    )

    calendar = CalendarFeatures(features=['day_of_week'], encoding=None)
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=2, calendar_features=calendar
    )
    err_msg = re.escape(
        "Duplicated feature names detected in X_train: ['day_of_week']."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


@pytest.mark.parametrize(
    "exog_dtype",
    [float, int],
    ids=lambda dtype: f'exog_dtype: {dtype}'
)
def test_create_train_X_y_ValueError_when_exog_name_duplicated_with_lag(exog_dtype):
    """
    Test ValueError is raised when an exogenous variable has the same name as a
    lag, producing duplicated feature names. Both a float exog (written in the
    float block of X_train) and an int exog (inserted as its own column) are
    checked.
    """
    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    })
    exog = pd.Series(np.arange(100, 110, dtype=exog_dtype), name='lag_1')

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=2, encoding='ordinal'
    )
    err_msg = re.escape(
        "Duplicated feature names detected in X_train: ['lag_1']."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'ordinal_category', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
@pytest.mark.parametrize(
    "exog_dtype",
    [float, int],
    ids=lambda dtype: f'exog_dtype: {dtype}'
)
def test_create_train_X_y_ValueError_when_exog_name_duplicated_with_level_column(
    encoding, exog_dtype
):
    """
    Test ValueError is raised when an exogenous variable is named
    `_level_skforecast`, the column that identifies the series, producing
    duplicated feature names. Both a float exog (written in the float block of
    X_train) and an int exog (inserted as its own column) are checked with the
    encodings that create the column.
    """
    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    })
    exog = pd.Series(np.arange(100, 110, dtype=exog_dtype), name='_level_skforecast')

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=2, encoding=encoding
    )
    err_msg = re.escape(
        "Duplicated feature names detected in X_train: ['_level_skforecast']."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


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
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.Series(np.arange(10), name='exog')
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')

    err_msg = re.escape(
        "Once the Forecaster has been trained, `exog` must be `None` "
        "because no exogenous variables were added during training."
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
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    new_exog = pd.Series(np.arange(10), name='exog2')
    new_exog.index = pd.date_range(start='2000-01-01', periods=len(new_exog), freq='D')

    err_msg = re.escape(
        "Once the Forecaster has been trained, `exog` must contain "
        "the same exogenous variables as those used during training:\n" 
        " Got      : ['exog2']\n"
        " Expected : ['exog']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=new_exog)


@pytest.mark.parametrize(
    "forecaster_kwargs, exog, window_sizes",
    [
        ({'lags': 5}, None, (5, 5, None)),
        ({'lags': 2,
          'window_features': RollingFeatures(stats=['mean', 'median'], window_sizes=6)},
         None, (6, 2, 6)),
        ({'lags': 5},
         {'l1': pd.DataFrame({'exog': pd.Categorical([0, 1, 2, 0, 1])})},
         (5, 5, None)),
        ({'lags': 5, 'transformer_exog': StandardScaler()},
         {'l1': pd.DataFrame({'exog': np.arange(5, dtype=float)})},
         (5, 5, None)),
    ],
    ids=['lags', 'window_features', 'exog_categorical', 'exog_transformer']
)
def test_create_train_X_y_ValueError_when_len_series_less_than_window_size(
    forecaster_kwargs, exog, window_sizes
):
    """
    Test ValueError is raised when the length of a series is less than or
    equal to window_size. The length is checked before processing `exog`, so
    the error is the same when the categorical encoder or `transformer_exog`
    would have to be fitted without rows.
    """
    series = {'l1': pd.Series(np.arange(5, dtype=float), name='l1')}
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), **forecaster_kwargs)

    max_window_size, lags_window_size, window_features_window_size = window_sizes
    err_msg = re.escape(
        f"Length of 'l1' must be greater than the maximum window size "
        f"needed by the forecaster.\n"
        f"    Length 'l1': 5.\n"
        f"    Max window size: {max_window_size}.\n"
        f"    Lags window size: {lags_window_size}.\n"
        f"    Window features window size: {window_features_window_size}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        transformer_series=StandardScaler(),
        lags=3,
        encoding="onehot"
    )

    results = forecaster._create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 1, 0],
                             [ 0. , -0.5, -1. , 1, 0],
                             [ 0.5,  0. , -0.5, 1, 0],
                             [ 1. ,  0.5,  0. , 1, 0],
                             [-0.5, -1. , -1.5, 0, 1],
                             [ 0. , -0.5, -1. , 0, 1],
                             [ 0.5,  0. , -0.5, 0, 1],
                             [ 1. ,  0.5,  0. , 0, 1]]),
            index   = pd.DatetimeIndex([
                "2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07",
                "2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07"
            ]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index   = pd.DatetimeIndex([
                "2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07",
                "2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07"
            ]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.date_range(start='2000-01-01', periods=7, freq='D'),
         '2': pd.date_range(start='2000-01-01', periods=7, freq='D')},
        ['1', '2'],
        ['1', '2'],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {'1': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.date_range(start='2000-01-05', periods=3, freq='D'),
                  name  = '1',
                  dtype = float
              ),
         '2': pd.Series(
                  data  = np.array([4., 5., 6.]),
                  index = pd.date_range(start='2000-01-05', periods=3, freq='D'),
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("encoding, dtype", 
                         [('ordinal'         , float), 
                          ('ordinal_category', 'category'),
                          (None              , float)], 
                         ids = lambda dt: f'encoding, dtype: {dt}')
def test_create_train_X_y_output_when_series_and_exog_is_None_ordinal_encoding(encoding, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = {
        '1': pd.Series(np.arange(7, dtype=float)), 
        '2': pd.Series(np.arange(7, dtype=float))
    }
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
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas series of floats or ints.
    """
    series = {
        '1': pd.Series(np.arange(10, dtype=float)), 
        '2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None
    )
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
        ).astype({'exog': dtype}),
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype(dtype)},
        {'exog': np.dtype(dtype)},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


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
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')
    exog.index.name = "datetime"
    exog = [exog.assign(series_id=f"{i}") for i in range(1, 3)]
    exog = pd.concat(exog)
    exog = exog.set_index(["series_id", exog.index])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal_category', transformer_series=None
    )
    
    warn_msg = re.escape(
        "Series {'3'} are not present in `series`. No last window is stored for them."
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster._create_train_X_y(
            series=series, exog=exog, store_last_window=['3']
        )    

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
            index   = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast', 'exog_1', 'exog_2']
        ).astype(
            {'_level_skforecast': int, 'exog_1': dtype, 'exog_2': dtype}
        ).astype(
            {'_level_skforecast': 'category'}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.date_range(start='2000-01-01', periods=10, freq='D'),
         '2': pd.date_range(start='2000-01-01', periods=10, freq='D')},
        ['1', '2'],
        ['1', '2'],
        ['exog_1', 'exog_2'],
        [],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype(dtype), 'exog_2': np.dtype(dtype)},
        {'exog_1': np.dtype(dtype), 'exog_2': np.dtype(dtype)},
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


def test_create_train_X_y_output_when_MissingExogWarning_exog_not_for_any_level():
    """
    Test the output of _create_train_X_y when series has exog but it does not contain
    any series ID.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float),
                         'exog_2': np.arange(1000, 1010, dtype=float)})
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')
    exog.index.name = "datetime"
    # NOTE: Here series_id is different from series keys in `series` "series_{i}" != '1'
    exog = [exog.assign(series_id=f"series_{i}") for i in range(1, 3)]
    exog = pd.concat(exog)
    exog = exog.set_index(["series_id", exog.index])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal_category', transformer_series=None
    )
    
    warn_msg = re.escape(
        "No exogenous variables were found in `exog` that match the "
        "series IDs provided in `series`. As a result, no exogenous "
        "variables are included in the training matrices. Please "
        "review the series IDs in `exog` and ensure they match the "
        "following IDs: ['1', '2']. The forecaster will be "
        "trained without exogenous variables."
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        results = forecaster._create_train_X_y(
            series=series, exog=exog, store_last_window=['3']
        )    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 0.],
                             [5., 4., 3., 2., 1., 0.],
                             [6., 5., 4., 3., 2., 0.],
                             [7., 6., 5., 4., 3., 0.],
                             [8., 7., 6., 5., 4., 0.],
                             [4., 3., 2., 1., 0., 1.],
                             [5., 4., 3., 2., 1., 1.],
                             [6., 5., 4., 3., 2., 1.],
                             [7., 6., 5., 4., 3., 1.],
                             [8., 7., 6., 5., 4., 1.]]),
            index   = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', '_level_skforecast']
        ).astype(
            {'_level_skforecast': int}
        ).astype(
            {'_level_skforecast': 'category'}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.date_range(start='2000-01-01', periods=10, freq='D'),
         '2': pd.date_range(start='2000-01-01', periods=10, freq='D')},
        ['1', '2'],
        ['1', '2'],
        ['exog_1', 'exog_2'],
        None,
        None,
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
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    assert results[12] == expected[12]


def test_create_train_X_y_ValueError_when_transformer_exog_modifies_exog_index():
    """
    Test ValueError is raised when the transformer_exog modifies the index of exog
    and the index of series is not equal to the index of exog after transformation.
    """

    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })
    custom_exog = pd.Series(np.arange(10), name='exog1')

    class CustomTransformerExog:  # pragma: no cover
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            # This transformer modifies the index of the exog
            X.index = pd.date_range(start='2000-01-01', periods=len(X), freq='D')
            return X
        
        def fit_transform(self, X, y=None):
            return self.transform(X)
        
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, transformer_exog=CustomTransformerExog()
    )

    err_msg = re.escape(
        "Different index for `series` and `exog` after transformation. "
        "They must be equal to ensure the correct alignment of values."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=custom_exog)


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
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )

    if dtype is str:
        # 'auto' detects string (object) columns and OrdinalEncodes them
        exog_assign = {'exog': [0.0] * 10}
        exog_astype = {}
        expected_cat_names = ['exog']
        expected_dtypes_out = {'exog': np.dtype('float64')}
    else:
        # bool is not detected by 'auto'
        exog_assign = {'exog': exog_values * 5 + exog_values * 5}
        exog_astype = {'exog': dtype}
        expected_cat_names = []
        expected_dtypes_out = {'exog': np.dtype(dtype)}

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
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            l1   = [1.] * 5 + [0.] * 5, 
            l2   = [0.] * 5 + [1.] * 5,
            **exog_assign
        ).astype(exog_astype),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range(start='2000-01-01', periods=10, freq='D'),
         'l2': pd.date_range(start='2000-01-01', periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        expected_cat_names,
        None,
        None,
        ['exog'],
        {'exog': np.dtype(dtype)} if dtype is bool else {'exog': np.dtype('O')},
        expected_dtypes_out,
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
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    assert results[12] == expected[12]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal_category', transformer_series=None
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )    

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
            exog_1 = ([0.0] * 5 + [0.0] * 5) if dtype is str else (v_exog_1 * 5 + v_exog_1 * 5), 
            exog_2 = ([0.0] * 5 + [0.0] * 5) if dtype is str else (v_exog_2 * 5 + v_exog_2 * 5)
        ).astype({'_level_skforecast': int}
        ).astype(
            {'_level_skforecast': 'category', 
             'exog_1': dtype if dtype is bool else float,
             'exog_2': dtype if dtype is bool else float}
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
        ['exog_1', 'exog_2'] if dtype is str else [],
        None,
        None,
        ['exog_1', 'exog_2'],
        ({'exog_1': np.dtype(dtype), 'exog_2': np.dtype(dtype)} 
         if dtype is bool 
         else {'exog_1': np.dtype('O'), 'exog_2': np.dtype('O')}),
        ({'exog_1': np.dtype(dtype), 'exog_2': np.dtype(dtype)} 
         if dtype is bool 
         else {'exog_1': np.dtype('float64'), 'exog_2': np.dtype('float64')}),
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_category(categorical_features):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas series of category.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )   

    if categorical_features is None:
        exog_values = pd.Categorical([5, 6, 7, 8, 9] * 2, categories=range(10))
        expected_cat_names = None
        expected_dtypes_out = {'exog': exog.dtypes}
    else:
        exog_values = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
        expected_cat_names = ['exog']
        expected_dtypes_out = {'exog': np.dtype('float64')}

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
            exog = exog_values
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
        expected_cat_names,
        None,
        None,
        ['exog'],
        {'exog': pd.CategoricalDtype(categories=range(10))},
        expected_dtypes_out,
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_1', 'exog_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_category(categorical_features):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of category.
    """
    series = pd.DataFrame({
            'l1': np.arange(10, dtype=float), 
            'l2': np.arange(10, dtype=float)
        },
        index=pd.date_range(start='2000-01-01', periods=10, freq='D')
    )
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    exog.index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')
    exog.index.name = "datetime"
    exog = [exog.assign(series_id=f"l{i}") for i in range(1, 3)]
    exog = pd.concat(exog)
    exog = exog.set_index(["series_id", exog.index])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )   

    if categorical_features is None:
        exog_1_values = pd.Categorical([5, 6, 7, 8, 9] * 2, categories=range(10))
        exog_2_values = pd.Categorical([105, 106, 107, 108, 109] * 2, categories=range(100, 110))
        expected_cat_names = None
        expected_dtypes_out = {
            'exog_1': pd.CategoricalDtype(categories=range(10)),
            'exog_2': pd.CategoricalDtype(categories=range(100, 110))
        }
    else:
        exog_1_values = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
        exog_2_values = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
        expected_cat_names = ['exog_1', 'exog_2']
        expected_dtypes_out = {
            'exog_1': np.dtype('float64'),
            'exog_2': np.dtype('float64')
        }

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
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            l1     = [1.] * 5 + [0.] * 5, 
            l2     = [0.] * 5 + [1.] * 5,
            exog_1 = exog_1_values,
            exog_2 = exog_2_values
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.DatetimeIndex([
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
                "2000-01-06", "2000-01-07", "2000-01-08", "2000-01-09", "2000-01-10",
            ]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range(start='2000-01-01', periods=10, freq='D'),
         'l2': pd.date_range(start='2000-01-01', periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        expected_cat_names,
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': pd.CategoricalDtype(categories=range(10)), 
         'exog_2': pd.CategoricalDtype(categories=range(100, 110))},
        expected_dtypes_out,
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int_category(categorical_features):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of float, int, category.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )   

    if categorical_features is None:
        exog_3_values = pd.Categorical(
            [105, 106, 107, 108, 109] * 2, categories=range(100, 110)
        )
        expected_cat_names = None
        expected_dtypes_out = {
            'exog_1': np.dtype('float'),
            'exog_2': np.dtype('int'),
            'exog_3': pd.CategoricalDtype(categories=range(100, 110))
        }
    else:
        exog_3_values = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
        expected_cat_names = ['exog_3']
        expected_dtypes_out = {
            'exog_1': np.dtype('float'),
            'exog_2': np.dtype('int'),
            'exog_3': np.dtype('float64')
        }

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
            exog_3 = exog_3_values, 
        ).astype({'exog_1': float, 'exog_2': int}),
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
        expected_cat_names,
        None,
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        {'exog_1': np.dtype('float'), 
         'exog_2': np.dtype('int'), 
         'exog_3': pd.CategoricalDtype(categories=range(100, 110))
        },
        expected_dtypes_out,
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_series_of_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas series of
    string categories. OrdinalEncoder maps sorted unique training values
    ['f'..'j'] -> [0.0..4.0]. None is not parametrized because it raises
    TypeError (tested separately).
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.Series(
        pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
        name='exog'
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )

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
            exog = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
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
        ['exog'],
        None,
        None,
        ['exog'],
        {'exog': exog.dtypes},
        {'exog': np.dtype('float64')},
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['f', 'g', 'h', 'i', 'j'], dtype=object)
    )


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_1', 'exog_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas DataFrame with
    two string category columns. OrdinalEncoder maps each column alphabetically
    using only training window values.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.DataFrame({
        'exog_1': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
        'exog_2': pd.Categorical(['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'])
    })

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )

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
            exog_1 = [0.0, 1.0, 2.0, 3.0, 4.0] * 2,
            exog_2 = [0.0, 1.0, 2.0, 3.0, 4.0] * 2
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
        ['exog_1', 'exog_2'],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': np.dtype('float64'), 'exog_2': np.dtype('float64')},
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]
    assert len(forecaster.categorical_encoder.categories_) == 2
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['f', 'g', 'h', 'i', 'j'], dtype=object)
    )
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[1],
        np.array(['p', 'q', 'r', 's', 't'], dtype=object)
    )


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_float_int_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas DataFrame with
    float, int, and string category columns. Only the string category column
    is detected/encoded. auto correctly ignores float and int columns.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.DataFrame({
        'exog_1': pd.Series(np.arange(100, 110), dtype=float),
        'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
        'exog_3': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    })

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )

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
            exog_3 = [0.0, 1.0, 2.0, 3.0, 4.0] * 2,
        ).astype({'exog_1': float, 'exog_2': int}),
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
        ['exog_3'],
        None,
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes,
         'exog_2': exog['exog_2'].dtypes,
         'exog_3': exog['exog_3'].dtypes},
        {'exog_1': exog['exog_1'].dtypes,
         'exog_2': exog['exog_2'].dtypes,
         'exog_3': np.dtype('float64')},
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['f', 'g', 'h', 'i', 'j'], dtype=object)
    )


@pytest.mark.parametrize("encoding, dtype", 
                         [('ordinal'         , float), 
                          ('ordinal_category', 'category'),
                          (None              , float),], 
                         ids = lambda dt: f'encoding, dtype: {dt}')
def test_create_train_X_y_output_when_series_and_exog_is_dataframe_datetime_index(encoding, dtype):
    """
    Test the output of _create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    series = {
        '1': pd.Series(np.arange(7, dtype=float)), 
        '2': pd.Series(np.arange(7, dtype=float))
    }
    series['1'].index = pd.date_range("1990-01-01", periods=7, freq='D')
    series['2'].index = pd.date_range("1990-01-01", periods=7, freq='D')
    exog = pd.DataFrame({'exog_1': np.arange(100, 107, dtype=float),
                         'exog_2': np.arange(1000, 1007, dtype=float)},
                        index = pd.date_range("1990-01-01", periods=7, freq='D'))
                         
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, transformer_series=None
    )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=True
    )

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
        [],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of _create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(
                    estimator          = LinearRegression(),
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
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of _create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    series = {
        '1': pd.Series(np.arange(7, dtype=float)), 
        '2': pd.Series(np.arange(7, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
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
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    assert results[12] == expected[12]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog(transformer_series):
    """
    Test the output of _create_train_X_y when using transformer_series and 
    transformer_exog.
    """
    series = pd.DataFrame(
        {'1': np.arange(10, dtype=float), 
         '2': np.arange(10, dtype=float)},
        index = pd.date_range("1990-01-01", periods=10, freq='D')
    ).to_dict(orient='series')
    exog = pd.DataFrame({
        'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
        'exog_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
        index = pd.date_range("1990-01-01", periods=10, freq='D')
    )
    exog.index.name = "datetime"
    exog = [exog.assign(series_id=f"{i}") for i in range(1, 3)]
    exog = pd.concat(exog)
    exog = exog.set_index(["series_id", exog.index])

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
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
        [],
        None,
        None,
        ['exog_1', 'exog_2_a', 'exog_2_b'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('O')},
        {'exog_1': np.dtype('float'), 'exog_2_a': np.dtype('float'), 'exog_2_b': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_when_series_different_length_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when series has 2 columns with different 
    lengths and exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    series = reshape_series_wide_to_long(series)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog.index.name = "datetime"
    exog = [exog.assign(series_id=f"l{i}") for i in range(1, 3)]
    exog = pd.concat(exog)
    exog = exog.set_index(["series_id", exog.index])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', transformer_series=None
    )
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
        ).assign(exog_3 = [0.0, 1.0, 2.0, 3.0, 4.0, 
                          2.0, 3.0, 4.0]
        ).astype({'exog_1': float, 'exog_2': int}),
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
        ['exog_3'],
        None,
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        {'exog_1': np.dtype('float'), 
         'exog_2': np.dtype('int'), 
         'exog_3': pd.CategoricalDtype(categories=range(100, 110))
        },
        {'exog_1': np.dtype('float'), 
         'exog_2': np.dtype('int'), 
         'exog_3': np.dtype('float64')
        },
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), 'l3': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_with_different_series_lengths(transformer_series):
    """
    Test the output of _create_train_X_y when using transformer_series and 
    transformer_exog with series with different lengths.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.]), 
        'l3': pd.Series([np.nan, np.nan, np.nan, np.nan, 4., 5., 6., 7., 8., 9.])
    }
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l3'].index = pd.date_range("1990-01-01", periods=10, freq='D')
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
                     estimator          = LinearRegression(),
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
        [],
        None,
        None,
        ['exog_1', 'exog_2_a', 'exog_2_b'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('O')},
        {'exog_1': np.dtype('float'), 'exog_2_a': np.dtype('float'), 'exog_2_b': np.dtype('float')},
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_y_train():
    """
    Test the output of _create_train_X_y when series is a DataFrame and y_train
    has NaNs. Also test the MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[5] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', 
        transformer_series=None, dropna_from_series=False
    )
    
    warn_msg = re.escape(
        "NaNs detected in `y_train`. They have been dropped because the "
        "target variable cannot have NaN values. Same rows have been "
        "dropped from `X_train` to maintain alignment. This is caused by "
        "interspersed NaNs in `series`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_y_train_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and y_train
    has NaNs with datetime index. Also test the MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[5] = np.nan
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot',
        transformer_series=None, dropna_from_series=False
    )
    
    warn_msg = re.escape(
        "NaNs detected in `y_train`. They have been dropped because the "
        "target variable cannot have NaN values. Same rows have been "
        "dropped from `X_train` to maintain alignment. This is caused by "
        "interspersed NaNs in `series`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_True():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=True`. Also test the MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot',
        transformer_series=None, dropna_from_series=True
    )
    
    warn_msg = re.escape(
        "NaNs detected in `X_train`. They have been dropped. If "
        "you want to keep them, set `forecaster.dropna_from_series = False`. " 
        "Same rows have been removed from `y_train` to maintain alignment. "
        "This is caused by interspersed NaNs in `series` or `exog`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_True_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=True` with datetime index. Also test the 
    MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')

    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    multi_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(list(series.keys()), len(exog)), 
            np.tile(pd.date_range("1990-01-01", periods=10, freq='D'), len(series.keys()))
        ], 
        names=["series_id", "datetime"]
    )
    exog = pd.Series(
        np.tile(exog.to_numpy(), len(series.keys())), index=multi_index, name="exog"
    )
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot',
        transformer_series=None, dropna_from_series=True
    )
    
    warn_msg = re.escape(
        "NaNs detected in `X_train`. They have been dropped. If "
        "you want to keep them, set `forecaster.dropna_from_series = False`. " 
        "Same rows have been removed from `y_train` to maintain alignment. "
        "This is caused by interspersed NaNs in `series` or `exog`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_False():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=False`. Also test the MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', 
        transformer_series=None, dropna_from_series=False
    )
    
    warn_msg = re.escape(
        "NaNs detected in `X_train`. Some estimators do not allow "
        "NaN values during training. If you want to drop them, "
        "set `forecaster.dropna_from_series = True`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_DataFrame_and_NaNs_in_X_train_drop_nan_False_datetime():
    """
    Test the output of _create_train_X_y when series is a DataFrame and X_train
    has NaNs and `drop_nan=False` with datetime index. Also test the 
    MissingValuesWarning message.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].loc[3] = np.nan
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot',
        transformer_series=None, dropna_from_series=False
    )
    
    warn_msg = re.escape(
        "NaNs detected in `X_train`. Some estimators do not allow "
        "NaN values during training. If you want to drop them, "
        "set `forecaster.dropna_from_series = True`."
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
        [],
        None,
        None,
        ['exog'],
        {'exog': exog.dtypes},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_ValueError_create_train_X_series_DataFrame_exog_dict_and_empty_X_train_drop_nan_True():
    """
    Test ValueError is raised when series is a DataFrame and exog dict is used
    and all samples have been removed due to NaNs in exog.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)), 
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')

    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float),
                         'exog_2': np.arange(200, 210, dtype=float)})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog_dict = {
        'l1': exog['exog_1'].copy(),
        'l2': exog[['exog_2']].copy()
    }
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot', dropna_from_series=True
    )
    
    error_msg = re.escape(
        "All samples have been removed due to NaNs. Set "
        "`forecaster.dropna_from_series = False` or review `series` "
        "and `exog` values."
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

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='onehot',
        transformer_series=None, dropna_from_series=False
    )
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
                             [22., 21., 20., 0., 0., 1., 206., 0.0],
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
        ).astype({'exog_1': float, 'exog_2': float}
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
        ['exog_2'],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('O')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float64')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize(
    "encoding, dtype",
    [("ordinal", float), 
     ("ordinal_category", "category"),
     (None, float)],
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
        LinearRegression(), lags=3, encoding=encoding, 
        transformer_series=None, dropna_from_series=False
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
                    [22.0, 21.0, 20.0, 2, 206.0, 0.0],
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
                "exog_2": float,
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
        ["exog_2"],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('O')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float64')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


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
    series = {
        '1': pd.Series(np.arange(7, dtype=float)), 
        '2': pd.Series(np.arange(7, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding
    )
    _ = forecaster._create_train_X_y(series=series)
    
    assert forecaster.encoding_mapping_ == encoding_mapping_


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
@pytest.mark.parametrize("differentiation", 
                         [1, {'l1': 1, 'l2': 1, 'l3': 1, '_unknown_level': 1}], 
                         ids = lambda diff: f'differentiation: {diff}')
def test_create_train_X_y_output_when_series_and_differentiation_1_and_already_trained(fit_forecaster, differentiation):
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
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',
        transformer_series=StandardScaler(), differentiation=differentiation
    )
    
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
        ).astype({'_level_skforecast': float}
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
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
def test_create_train_X_y_output_when_series_and_already_trained_encoding_None(fit_forecaster):
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
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=None, transformer_series=StandardScaler()
    )
    
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
        ).astype({'_level_skforecast': float}
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
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
def test_create_train_X_y_output_when_series_and_differentiation_1_and_already_trained_encoding_None(fit_forecaster):
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
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=None,
        transformer_series=StandardScaler(), differentiation=1
    )
    
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
        ).astype({'_level_skforecast': float}
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
    assert results[9] == expected[9]
    assert results[10] == expected[10]
    assert results[11] == expected[11]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_series_dict_and_exog_dict_window_and_calendar_features():
    """
    Test the output of _create_train_X_y when series is a dict and exog is a
    dict with window features and calendar features.
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
    calendar = CalendarFeatures(
        features=['day_of_week', 'weekend'], encoding="cyclical"
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3,
        encoding='onehot',
        window_features=[rolling, rolling_2],
        calendar_features=calendar,
        transformer_series=None,
        dropna_from_series=False
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[np.nan, 2., 1., np.nan, np.nan, np.nan, 1., 0., 0., 104., np.nan, 0., -0.4338837391175581, -0.9009688679024191],
                             [4., np.nan, 2., np.nan, np.nan, np.nan, 1., 0., 0., 105., np.nan, 1., -0.9749279121818236, -0.2225209339563146],
                             [5., 4., np.nan, np.nan, np.nan, np.nan, 1., 0., 0., 106., np.nan, 1., -0.7818314824680299, 0.6234898018587334],
                             [6., 5., 4., 5.0, 5.0, np.nan, 1., 0., 0., 107., np.nan, 0., 0.0, 1.0],
                             [7., 6., 5., 6.0, 6.0, 22.0, 1., 0., 0., 108., np.nan, 0., 0.7818314824680298, 0.6234898018587336],
                             [8., 7., 6., 7.0, 7.0, 26.0, 1., 0., 0., 109., np.nan, 0., 0.9749279121818236, -0.22252093395631434],
                             [18., np.nan, 16., np.nan, np.nan, np.nan, 0., 1., 0., np.nan, np.nan, 0., 0.7818314824680298, 0.6234898018587336],
                             [23., 22., 21., 22.0, 22.0, 86.0, 0., 0., 1., 207., 0.0, 1., -0.7818314824680299, 0.6234898018587334]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                               '1990-01-09', '1990-01-10',
                               '1990-01-09', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'roll_median_3', 'roll_sum_4',
                       'l1', 'l2', 'l3', 'exog_1', 'exog_2',
                       'weekend', 'day_of_week_sin', 'day_of_week_cos']
        ).astype({'exog_1': float, 'exog_2': float}
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
        ['exog_2'],
        ['roll_mean_3', 'roll_median_3', 'roll_sum_4'],
        ['weekend', 'day_of_week_sin', 'day_of_week_cos'],
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('O')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float64')},
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
    expected[0].iloc[[0, 1, 2, 3, 4, 5, 6], expected[0].columns.get_loc('exog_2')] = np.nan

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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_output_when_series_and_exog_with_window_features_no_lags():
    """
    Test the output of _create_train_X_y when series and exog with window
    features but no lags.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, 20, dtype=float))
    }
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')
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
        ).astype({'_level_skforecast': float}),
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
        [],
        ['roll_mean_3', 'roll_median_5', 'roll_sum_4'],
        None,
        ['exog'],
        {'exog': np.dtype('float')},
        {'exog': np.dtype('float')},
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
    assert results[8] == expected[8]
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda is_fitted: f'fit_forecaster: {is_fitted}')
@pytest.mark.parametrize("differentiation", 
                         [{'l1': 1, 'l2': 2, 'l3': None, '_unknown_level': 1}, 
                          {'l1': 1, 'l2': 2, '_unknown_level': 1}], 
                         ids = lambda diff: f'differentiation: {diff}')
def test_create_train_X_y_output_when_series_and_exog_and_differentiation_dict_and_already_trained(fit_forecaster, differentiation):
    """
    Test the output of _create_train_X_y when differentiation=1,
    encoding 'ordinal' and already trained forecaster.
    """
    series = {
        "l1": pd.Series(np.array([14,  2, 85, 92, 77, 91, 63, 96, 11, 53], dtype=float)),
        "l2": pd.Series(np.array([16, 23, 98, 76, 75,  9, 23], dtype=float)),
        "l3": pd.Series(np.array([92,  2, 76, 94, 88, 10, 63], dtype=float)),
    }
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=7, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=7, freq="D")
    
    exog = {
        "l1": pd.Series(np.arange(100, 110), name="exog_1", dtype=float),
        "l3": pd.DataFrame(
            {"exog_1": np.arange(203, 210, dtype=float), 
             "exog_2": np.arange(303, 310, dtype=float)}
        ),
    }
    exog["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    exog["l3"].index = pd.date_range("1990-01-03", periods=7, freq="D")

    window_features = RollingFeatures(stats='mean', window_sizes=4)
    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, 
        encoding='ordinal', window_features=window_features,
        transformer_series=None, differentiation=differentiation
    )
    
    if fit_forecaster:
        forecaster.fit(series=series, exog=exog)

    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[ 14., -15.,   7.,  22.25, 0., 106.  , np.nan],
                             [-28.,  14., -15.,  -5.5 , 0., 107.  , np.nan],
                             [ 33., -28.,  14.,   1.  , 0., 108.  , np.nan],
                             [-85.,  33., -28., -16.5 , 0., 109.  , np.nan],
                             [-65.,  21., -97., -18.25, 1., np.nan, np.nan],
                             [ 10.,  88.,  94.,  67.  , 2., 209.  , 309.  ]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-11', 
                               '1990-01-09']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'roll_mean_4', 
                       '_level_skforecast', 'exog_1', 'exog_2']
        ).astype({'_level_skforecast': float}
        ),
        pd.Series(
            data  = np.array([-28., 33., -85., 42., 80., 63.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-11', 
                             '1990-01-09']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range("1990-01-01", periods=10, freq='D'),
         'l2': pd.date_range("1990-01-05", periods=7, freq='D'),
         'l3': pd.date_range("1990-01-03", periods=7, freq='D')},
        ['l1', 'l2', 'l3'],
        ['l1', 'l2', 'l3'],
        ['exog_1', 'exog_2'],
        [],
        ['roll_mean_4'],
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
        {'l1': pd.Series(
                   data  = np.array([77, 91, 63, 96, 11, 53]),
                   index = pd.date_range("1990-01-05", periods=6, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([23, 98, 76, 75,  9, 23]),
                   index = pd.date_range("1990-01-06", periods=6, freq='D'),
                   name  = 'l2',
                   dtype = float
               ),
         'l3': pd.Series(
                   data  = np.array([2, 76, 94, 88, 10, 63]),
                   index = pd.date_range("1990-01-04", periods=6, freq='D'),
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
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])


def test_create_train_X_y_ValueError_when_categorical_features_columns_not_in_exog():
    """
    Test ValueError is raised when explicit categorical_features list contains
    columns not present in exog after transformer_exog.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float)})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5,
        categorical_features=['exog_1', 'non_existent']
    )
    err_msg = re.escape(
        "The following columns specified in `categorical_features` "
        "are not present in `exog` after `transformer_exog`: "
        "{'non_existent'}."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['col_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_transformer_exog_is_make_column_transformer_and_categorical(categorical_features):
    """
    Test the output of _create_train_X_y when using make_column_transformer
    with StandardScaler only for numeric columns and a string categorical
    column passed through as remainder. With set_output(transform='pandas'),
    the category dtype is preserved and 'auto' correctly detects col_2.
    OrdinalEncoder maps ['a'..'c'] -> [0.0..2.0].
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    series['l2'].index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 30.1, 22.3],
               'col_2': pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c'])},
               index=pd.date_range("1990-01-01", periods=10, freq='D')
           )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), ['col_1']),
                           remainder='passthrough',
                           verbose_feature_names_out=False
                       ).set_output(transform='pandas')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     categorical_features = categorical_features
                 )
    results = forecaster._create_train_X_y(
        series=series, exog=exog, store_last_window=False
    )

    # OrdinalEncoder maps ['a','b','c'] -> [0.0, 1.0, 2.0]
    # col_1 is StandardScaled, col_2 is OrdinalEncoded
    expected = (
        pd.DataFrame(
            data = np.array(
                [[4., 3., 2., 1., 0., 0., -0.19009842, 2.],
                 [5., 4., 3., 2., 1., 0.,  1.84413235, 0.],
                 [6., 5., 4., 3., 2., 0.,  0.07639469, 1.],
                 [7., 6., 5., 4., 3., 0., -0.69199379, 2.],
                 [8., 7., 6., 5., 4., 0., -1.03843484, 2.],
                 [4., 3., 2., 1., 0., 1., -0.19009842, 2.],
                 [5., 4., 3., 2., 1., 1.,  1.84413235, 0.],
                 [6., 5., 4., 3., 2., 1.,  0.07639469, 1.],
                 [7., 6., 5., 4., 3., 1., -0.69199379, 2.],
                 [8., 7., 6., 5., 4., 1., -1.03843484, 2.]]
            ),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       '_level_skforecast', 'col_1', 'col_2']
        ).astype({'_level_skforecast': float}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9., 5., 6., 7., 8., 9.]),
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
        ['col_1', 'col_2'],
        ['col_2'],
        None,
        None,
        ['col_1', 'col_2'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': np.dtype('float64'), 'col_2': np.dtype('float64')},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0], atol=1e-08)
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
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    assert results[12] == expected[12]


@pytest.mark.parametrize("fit_forecaster",
                         [True, False],
                         ids=lambda fitted: f'fit_forecaster: {fitted}')
@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_cat']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_categorical_features_and_already_trained(
    fit_forecaster, categorical_features
):
    """
    Test that when is_fitted=True (after fit), _create_train_X_y uses
    transform (not fit_transform) on the categorical encoder and produces
    the same result as the first call.
    """
    series = {
        'l1': pd.Series(np.arange(10, dtype=float)),
        'l2': pd.Series(np.arange(10, dtype=float))
    }
    series['l1'].index = pd.date_range('2000-01-01', periods=10, freq='D')
    series['l2'].index = pd.date_range('2000-01-01', periods=10, freq='D')

    exog = pd.DataFrame({
        'exog_num': np.arange(100, 110, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    }, index=pd.date_range('2000-01-01', periods=10, freq='D'))

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal',
        transformer_series=None,
        categorical_features=categorical_features
    )

    if fit_forecaster:
        forecaster.fit(series=series, exog=exog)

    results = forecaster._create_train_X_y(series=series, exog=exog)

    # OrdinalEncoder maps ['f'..'j'] -> [0.0..4.0] (only training window values)
    expected = (
        pd.DataFrame(
            data = np.array(
                [[4., 3., 2., 1., 0., 0., 105., 0.],
                 [5., 4., 3., 2., 1., 0., 106., 1.],
                 [6., 5., 4., 3., 2., 0., 107., 2.],
                 [7., 6., 5., 4., 3., 0., 108., 3.],
                 [8., 7., 6., 5., 4., 0., 109., 4.],
                 [4., 3., 2., 1., 0., 1., 105., 0.],
                 [5., 4., 3., 2., 1., 1., 106., 1.],
                 [6., 5., 4., 3., 2., 1., 107., 2.],
                 [7., 6., 5., 4., 3., 1., 108., 3.],
                 [8., 7., 6., 5., 4., 1., 109., 4.]]
            ),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09', '2000-01-10',
                             '2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09', '2000-01-10']
                        )
                    ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       '_level_skforecast', 'exog_num', 'exog_cat']
        ).astype({'_level_skforecast': float}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9., 5., 6., 7., 8., 9.]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09', '2000-01-10',
                             '2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09', '2000-01-10']
                        )
                    ),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.date_range('2000-01-01', periods=10, freq='D'),
         'l2': pd.date_range('2000-01-01', periods=10, freq='D')},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_num', 'exog_cat'],
        ['exog_cat'],
        None,
        None,
        ['exog_num', 'exog_cat'],
        {'exog_num': exog['exog_num'].dtypes, 'exog_cat': exog['exog_cat'].dtypes},
        {'exog_num': np.dtype('float64'), 'exog_cat': np.dtype('float64')},
        {'l1': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range('2000-01-06', periods=5, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([5., 6., 7., 8., 9.]),
                   index = pd.date_range('2000-01-06', periods=5, freq='D'),
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
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['f', 'g', 'h', 'i', 'j'], dtype=object)
    )


@pytest.mark.parametrize(
    "encoding, expected_X_train_series_names_in_",
    [
        ('ordinal', ['a', 'b', 'c']),
        ('ordinal_category', ['a', 'b', 'c']),
        ('onehot', ['c', 'a', 'b']),
        (None, ['a', 'b', 'c'])
    ],
    ids=lambda value: f'encoding, X_train_series_names_in_: {value}'
)
def test_create_train_X_y_X_train_series_names_in_when_series_unordered_and_one_series_dropped(
    encoding, expected_X_train_series_names_in_
):
    """
    Test `X_train_series_names_in_` when the series are not in alphabetical order
    ('c', 'a', 'd', 'b') and all the rows of series 'd' are removed because it
    has no exog and `dropna_from_series=True`. With 'onehot' the names follow the
    order of the series and with the other encodings the alphabetical order of
    `encoding_mapping_`. The column of 'd' is kept in the one-hot block.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding=encoding, dropna_from_series=True
    )
    results = forecaster._create_train_X_y(
        series=series_dict_unordered, exog=exog_dict_unordered
    )

    expected_columns = (
        ['lag_1', 'lag_2', 'a', 'b', 'c', 'd', 'exog_1']
        if encoding == 'onehot'
        else ['lag_1', 'lag_2', '_level_skforecast', 'exog_1']
    )

    assert results[3] == ['c', 'a', 'd', 'b']
    assert results[4] == expected_X_train_series_names_in_
    assert results[0].columns.to_list() == expected_columns
    assert len(results[0]) == 17


def test_create_train_X_y_output_when_exog_dict_float_int_category_int_and_categorical_features_None():
    """
    Test the output of _create_train_X_y when exog is a dict of DataFrames with
    columns of dtypes float, int, category and int, and `categorical_features=None`.
    The category column keeps its dtype and every column keeps its position and
    dtype in X_train.
    """
    series = {
        'l1': pd.Series(np.arange(6, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 16, dtype=float), name='l2')
    }
    exog = {
        'l1': pd.DataFrame({
                  'exog_float': np.arange(100, 106, dtype=float),
                  'exog_int': np.arange(200, 206, dtype=int),
                  'exog_cat': pd.Categorical([0, 1, 2, 0, 1, 2], categories=[0, 1, 2]),
                  'exog_int_2': np.arange(300, 306, dtype=int)
              }),
        'l2': pd.DataFrame({
                  'exog_float': np.arange(110, 116, dtype=float),
                  'exog_int': np.arange(210, 216, dtype=int),
                  'exog_cat': pd.Categorical([2, 2, 1, 1, 0, 0], categories=[0, 1, 2]),
                  'exog_int_2': np.arange(310, 316, dtype=int)
              })
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding='ordinal', categorical_features=None
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected_dtypes = {
        'exog_float': np.dtype(float),
        'exog_int': np.dtype(int),
        'exog_cat': pd.CategoricalDtype(categories=[0, 1, 2]),
        'exog_int_2': np.dtype(int)
    }
    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 0., 102.],
                             [2., 1., 0., 103.],
                             [3., 2., 0., 104.],
                             [4., 3., 0., 105.],
                             [11., 10., 1., 112.],
                             [12., 11., 1., 113.],
                             [13., 12., 1., 114.],
                             [14., 13., 1., 115.]]),
            index   = pd.Index([2, 3, 4, 5, 2, 3, 4, 5]),
            columns = ['lag_1', 'lag_2', '_level_skforecast', 'exog_float']
        ).assign(
            exog_int   = np.array([202, 203, 204, 205, 212, 213, 214, 215], dtype=int),
            exog_cat   = pd.Categorical([2, 0, 1, 2, 1, 1, 0, 0], categories=[0, 1, 2]),
            exog_int_2 = np.array([302, 303, 304, 305, 312, 313, 314, 315], dtype=int)
        ),
        pd.Series(
            data  = np.array([2., 3., 4., 5., 12., 13., 14., 15.]),
            index = pd.Index([2, 3, 4, 5, 2, 3, 4, 5]),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[5] == ['exog_float', 'exog_int', 'exog_cat', 'exog_int_2']
    assert results[6] is None
    assert results[9] == ['exog_float', 'exog_int', 'exog_cat', 'exog_int_2']
    assert results[10] == expected_dtypes
    assert results[11] == expected_dtypes
    assert list(results[11]) == ['exog_float', 'exog_int', 'exog_cat', 'exog_int_2']


def test_create_train_X_y_output_when_dropna_from_series_and_nan_in_lags_and_category_exog():
    """
    Test the output of _create_train_X_y when `dropna_from_series=True`, series
    'l1' has an interspersed NaN (it appears in y and in the lags) and series
    'l2' has a NaN in a category exog kept as category (`categorical_features=None`).
    All the affected rows are dropped and the category column keeps its dtype.
    """
    series = {
        'l1': pd.Series([0., 1., 2., np.nan, 4., 5., 6., 7.], name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), name='l2')
    }
    exog = {
        'l1': pd.DataFrame({
                  'exog_float': np.arange(100, 108, dtype=float),
                  'exog_cat': pd.Categorical(
                                  [0, 1, 2, 0, 1, 2, 0, 1], categories=[0, 1, 2]
                              )
              }),
        'l2': pd.DataFrame({
                  'exog_float': np.arange(110, 118, dtype=float),
                  'exog_cat': pd.Categorical(
                                  [2, 2, 1, np.nan, 0, 0, 1, 1], categories=[0, 1, 2]
                              )
              })
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding='ordinal', categorical_features=None,
        dropna_from_series=True
    )

    warn_msg = re.escape(
        "NaNs detected in `X_train`. They have been dropped. If "
        "you want to keep them, set `forecaster.dropna_from_series = False`. "
        "Same rows have been removed from `y_train` to maintain alignment. "
        "This is caused by interspersed NaNs in `series` or `exog`."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 0., 102.],
                             [5., 4., 0., 106.],
                             [6., 5., 0., 107.],
                             [11., 10., 1., 112.],
                             [13., 12., 1., 114.],
                             [14., 13., 1., 115.],
                             [15., 14., 1., 116.],
                             [16., 15., 1., 117.]]),
            index   = pd.Index([2, 6, 7, 2, 4, 5, 6, 7]),
            columns = ['lag_1', 'lag_2', '_level_skforecast', 'exog_float']
        ).assign(
            exog_cat = pd.Categorical([2, 0, 1, 1, 0, 0, 1, 1], categories=[0, 1, 2])
        ),
        pd.Series(
            data  = np.array([2., 6., 7., 12., 14., 15., 16., 17.]),
            index = pd.Index([2, 6, 7, 2, 4, 5, 6, 7]),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[4] == ['l1', 'l2']


def test_create_train_X_y_output_when_exog_dict_object_column_with_different_values_by_series():
    """
    Test the output of _create_train_X_y when exog is a dict, the object column
    'exog_2' has different values in 'l1' and 'l2' and is missing in 'l3', and
    `categorical_features='auto'`. The column is encoded with the categories of
    all the series and is NaN for 'l3'.
    """
    series = {
        'l1': pd.Series(np.arange(5, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 15, dtype=float), name='l2'),
        'l3': pd.Series(np.arange(20, 25, dtype=float), name='l3')
    }
    exog = {
        'l1': pd.DataFrame({
                  'exog_1': np.arange(100, 105, dtype=float),
                  'exog_2': ['a', 'b', 'a', 'b', 'a']
              }),
        'l2': pd.DataFrame({
                  'exog_1': np.arange(110, 115, dtype=float),
                  'exog_2': ['c', 'd', 'c', 'd', 'c']
              }),
        'l3': pd.DataFrame({'exog_1': np.arange(120, 125, dtype=float)})
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding='ordinal', categorical_features='auto'
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 0., 102., 0.],
                             [2., 1., 0., 103., 1.],
                             [3., 2., 0., 104., 0.],
                             [11., 10., 1., 112., 2.],
                             [12., 11., 1., 113., 3.],
                             [13., 12., 1., 114., 2.],
                             [21., 20., 2., 122., np.nan],
                             [22., 21., 2., 123., np.nan],
                             [23., 22., 2., 124., np.nan]]),
            index   = pd.Index([2, 3, 4, 2, 3, 4, 2, 3, 4]),
            columns = ['lag_1', 'lag_2', '_level_skforecast', 'exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([2., 3., 4., 12., 13., 14., 22., 23., 24.]),
            index = pd.Index([2, 3, 4, 2, 3, 4, 2, 3, 4]),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[6] == ['exog_2']
    assert results[10] == {'exog_1': np.dtype(float), 'exog_2': np.dtype(object)}
    assert results[11] == {'exog_1': np.dtype(float), 'exog_2': np.dtype(float)}
    pd.testing.assert_index_equal(
        pd.Index(forecaster.categorical_encoder.categories_[0]),
        pd.Index(['a', 'b', 'c', 'd', np.nan], dtype=object)
    )


@pytest.mark.parametrize(
    "dropna_from_series",
    [False, True],
    ids=lambda dropna: f'dropna_from_series: {dropna}'
)
def test_create_train_X_y_output_when_exog_dict_float_column_with_NaNs(
    dropna_from_series
):
    """
    Test the output of _create_train_X_y when exog is a dict and a float column
    has NaNs between valid values in both series. The NaNs are kept in X_train,
    or the rows that have them are dropped when `dropna_from_series=True`.
    """
    series = {
        'l1': pd.Series(np.arange(6, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 16, dtype=float), name='l2')
    }
    exog = {
        'l1': pd.DataFrame({
                  'exog_1': [100., 101., np.nan, 103., np.nan, 105.],
                  'exog_2': np.arange(200, 206, dtype=float)
              }),
        'l2': pd.DataFrame({
                  'exog_1': [110., 111., 112., np.nan, 114., 115.],
                  'exog_2': np.arange(210, 216, dtype=float)
              })
    }
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 2,
                     encoding           = 'ordinal',
                     dropna_from_series = dropna_from_series
                 )

    warn_msg = re.escape("NaNs detected in `X_train`.")
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected_X = np.array([[1., 0., 0., np.nan, 202.],
                           [2., 1., 0., 103., 203.],
                           [3., 2., 0., np.nan, 204.],
                           [4., 3., 0., 105., 205.],
                           [11., 10., 1., 112., 212.],
                           [12., 11., 1., np.nan, 213.],
                           [13., 12., 1., 114., 214.],
                           [14., 13., 1., 115., 215.]])
    expected_y = np.array([2., 3., 4., 5., 12., 13., 14., 15.])
    expected_index = np.array([2, 3, 4, 5, 2, 3, 4, 5])
    if dropna_from_series:
        rows_kept = [1, 3, 4, 6, 7]
        expected_X = expected_X[rows_kept]
        expected_y = expected_y[rows_kept]
        expected_index = expected_index[rows_kept]

    expected = (
        pd.DataFrame(
            data    = expected_X,
            index   = pd.Index(expected_index),
            columns = ['lag_1', 'lag_2', '_level_skforecast', 'exog_1', 'exog_2']
        ),
        pd.Series(
            data  = expected_y,
            index = pd.Index(expected_index),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    assert results[4] == ['l1', 'l2']


@pytest.mark.parametrize(
    "encoding, use_exog, expected_no_copy, expected_order",
    [
        ('ordinal', False, True, 'F'),
        ('ordinal', True, True, 'F'),
        (None, True, True, 'F'),
        (None, False, False, 'C')
    ],
    ids=lambda value: f'encoding, exog, no_copy, order: {value}'
)
def test_create_train_X_y_X_train_layout_when_encoding_ordinal_or_None(
    encoding, use_exog, expected_no_copy, expected_order
):
    """
    Test the memory layout of X_train. With `encoding='ordinal'`, and with
    `encoding=None` and float exog, lags, window features, level and exog are
    stored in a single array with contiguous columns, so X_train is not copied
    when it is converted to numpy. With `encoding=None` and no exog, the level
    is stored apart (the conversion to numpy copies) and the array passed to
    the estimator, once the level column is dropped, is row-contiguous.
    """
    series = {
        'l1': pd.Series(np.arange(8, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), name='l2')
    }
    exog = None
    expected_columns = ['lag_1', 'lag_2', 'roll_mean_3', '_level_skforecast']
    if use_exog:
        exog = {
            'l1': pd.DataFrame({
                      'exog_1': np.arange(100, 108, dtype=float),
                      'exog_2': np.arange(200, 208, dtype=float)
                  }),
            'l2': pd.DataFrame({
                      'exog_1': np.arange(110, 118, dtype=float),
                      'exog_2': np.arange(210, 218, dtype=float)
                  })
        }
        expected_columns = expected_columns + ['exog_1', 'exog_2']

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator       = LinearRegression(),
                     lags            = 2,
                     window_features = RollingFeatures(stats='mean', window_sizes=3),
                     encoding        = encoding
                 )
    X_train = forecaster._create_train_X_y(series=series, exog=exog)[0]
    X_train_numpy = X_train.to_numpy()

    assert X_train.columns.to_list() == expected_columns
    assert (X_train.dtypes == np.dtype(float)).all()
    assert X_train['_level_skforecast'].to_numpy().strides == (8,)
    for col in expected_columns:
        col_values = X_train[col].to_numpy()
        assert np.shares_memory(X_train_numpy, col_values) == expected_no_copy
        if expected_no_copy:
            assert col_values.strides == (8,)

    X_train_estimator = (
        X_train if encoding is not None else X_train.drop(columns='_level_skforecast')
    )
    X_train_estimator = X_train_estimator.to_numpy()
    if expected_order == 'F':
        assert X_train_estimator.flags.f_contiguous
    else:
        assert X_train_estimator.flags.c_contiguous


@pytest.mark.parametrize(
    "use_exog", [False, True], ids=lambda value: f'use_exog: {value}'
)
def test_create_train_X_y_X_train_layout_when_encoding_onehot(use_exog):
    """
    Test the memory layout of X_train with `encoding='onehot'`. Lags, window
    features, the one-hot columns of the series and float exog are stored in
    a single float array with contiguous columns, so X_train is not copied
    when it is converted to numpy.
    """
    series = {
        'l1': pd.Series(np.arange(8, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), name='l2')
    }
    exog = None
    expected_columns = ['lag_1', 'lag_2', 'roll_mean_3', 'l1', 'l2']
    if use_exog:
        exog = {
            'l1': pd.DataFrame({'exog_1': np.arange(100, 108, dtype=float)}),
            'l2': pd.DataFrame({'exog_1': np.arange(110, 118, dtype=float)})
        }
        expected_columns = expected_columns + ['exog_1']

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator       = LinearRegression(),
                     lags            = 2,
                     window_features = RollingFeatures(stats='mean', window_sizes=3),
                     encoding        = 'onehot'
                 )
    X_train = forecaster._create_train_X_y(series=series, exog=exog)[0]
    X_train_numpy = X_train.to_numpy()

    assert X_train.columns.to_list() == expected_columns
    assert (X_train.dtypes == np.dtype(float)).all()
    assert X_train_numpy.flags.f_contiguous
    for col in expected_columns:
        col_values = X_train[col].to_numpy()
        assert np.shares_memory(X_train_numpy, col_values)
        assert col_values.strides == (8,)
    np.testing.assert_array_equal(
        X_train['l1'].to_numpy(), np.array([1.] * 5 + [0.] * 5)
    )
    np.testing.assert_array_equal(
        X_train['l2'].to_numpy(), np.array([0.] * 5 + [1.] * 5)
    )


@pytest.mark.parametrize(
    "calendar_encoding",
    ['cyclical', 'onehot'],
    ids=lambda value: f'calendar encoding: {value}'
)
@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'onehot', None],
    ids=lambda value: f'encoding: {value}'
)
def test_create_train_X_y_X_train_layout_when_calendar_features(
    encoding, calendar_encoding
):
    """
    Test the memory layout and the calendar columns of X_train when
    `calendar_features` is used, also with `encoding=None` and no exog. The
    calendar features are written as float, also the integer ones (`weekend`
    and the one-hot columns of `quarter`), at the end of the single float
    block, so X_train is not copied when it is converted to numpy.
    """
    index = pd.date_range(start='2020-01-01', periods=8, freq='D')
    series = {
        'l1': pd.Series(np.arange(8, dtype=float), index=index, name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), index=index, name='l2')
    }
    if calendar_encoding == 'cyclical':
        calendar = CalendarFeatures(
            features=['weekend', 'day_of_week'], encoding='cyclical'
        )
        expected_calendar = {
            'weekend': [1., 1., 0., 0., 0.],
            'day_of_week_sin': [-0.9749279121818236, -0.7818314824680299, 0.0,
                                0.7818314824680298, 0.9749279121818236],
            'day_of_week_cos': [-0.2225209339563146, 0.6234898018587334, 1.0,
                                0.6234898018587336, -0.22252093395631434]
        }
    else:
        calendar = CalendarFeatures(features=['weekend', 'quarter'], encoding='onehot')
        expected_calendar = {
            'weekend': [1., 1., 0., 0., 0.],
            'quarter_1': [1., 1., 1., 1., 1.],
            'quarter_2': [0., 0., 0., 0., 0.],
            'quarter_3': [0., 0., 0., 0., 0.],
            'quarter_4': [0., 0., 0., 0., 0.]
        }
    expected_calendar = pd.DataFrame(
        data  = {col: values * 2 for col, values in expected_calendar.items()},
        index = pd.DatetimeIndex(
                    ['2020-01-04', '2020-01-05', '2020-01-06',
                     '2020-01-07', '2020-01-08'] * 2
                )
    )
    level_columns = ['l1', 'l2'] if encoding == 'onehot' else ['_level_skforecast']
    expected_columns = (
        ['lag_1', 'lag_2', 'roll_mean_3'] + level_columns
        + expected_calendar.columns.to_list()
    )

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator         = LinearRegression(),
                     lags              = 2,
                     window_features   = RollingFeatures(stats='mean', window_sizes=3),
                     encoding          = encoding,
                     calendar_features = calendar
                 )
    X_train = forecaster._create_train_X_y(series=series)[0]
    X_train_numpy = X_train.to_numpy()

    assert X_train.columns.to_list() == expected_columns
    assert (X_train.dtypes == np.dtype(float)).all()
    assert X_train_numpy.flags.f_contiguous
    for col in expected_columns:
        col_values = X_train[col].to_numpy()
        assert np.shares_memory(X_train_numpy, col_values)
        assert col_values.strides == (8,)
    pd.testing.assert_frame_equal(
        X_train[expected_calendar.columns.to_list()], expected_calendar
    )

    X_train_estimator = (
        X_train if encoding is not None else X_train.drop(columns='_level_skforecast')
    )
    assert X_train_estimator.to_numpy().flags.f_contiguous


@pytest.mark.parametrize(
    "encoding, expected_level_dtype",
    [
        ('ordinal', np.dtype(float)),
        (
            'ordinal_category',
            pd.CategoricalDtype(categories=np.array([0, 1], dtype=int))
        )
    ],
    ids=lambda value: f'encoding, level dtype: {value}'
)
def test_create_train_X_y_X_train_layout_when_exog_has_non_float_columns(
    encoding, expected_level_dtype
):
    """
    Test the memory layout, column order and dtypes of X_train when exog has
    columns that are not float64 (int, category, float32 and object with
    datetime values) between float columns, and `categorical_features=None`.
    The float64 columns are stored one after another in a single array and the
    other columns, including the level when `encoding='ordinal_category'`, are
    inserted in their position with their dtype.
    """
    series = {
        'l1': pd.Series(np.arange(6, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 16, dtype=float), name='l2')
    }
    exog = {
        'l1': pd.DataFrame({
                  'exog_float': np.arange(100, 106, dtype=float),
                  'exog_int': np.arange(200, 206, dtype=int),
                  'exog_cat': pd.Categorical([0, 1, 2, 0, 1, 2], categories=[0, 1, 2]),
                  'exog_float_2': np.arange(300, 306, dtype=float),
                  'exog_float32': np.arange(400, 406, dtype='float32'),
                  'exog_object': pd.Series(
                                     [pd.Timestamp('2020-01-01')] * 6, dtype=object
                                 )
              }),
        'l2': pd.DataFrame({
                  'exog_float': np.arange(110, 116, dtype=float),
                  'exog_int': np.arange(210, 216, dtype=int),
                  'exog_cat': pd.Categorical([2, 2, 1, 1, 0, 0], categories=[0, 1, 2]),
                  'exog_float_2': np.arange(310, 316, dtype=float),
                  'exog_float32': np.arange(410, 416, dtype='float32'),
                  'exog_object': pd.Series(
                                     [pd.Timestamp('2020-01-02')] * 6, dtype=object
                                 )
              })
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, categorical_features=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DataTypeWarning)
        X_train = forecaster._create_train_X_y(series=series, exog=exog)[0]

    expected_dtypes = pd.Series({
        'lag_1': np.dtype(float),
        'lag_2': np.dtype(float),
        'lag_3': np.dtype(float),
        '_level_skforecast': expected_level_dtype,
        'exog_float': np.dtype(float),
        'exog_int': np.dtype(int),
        'exog_cat': pd.CategoricalDtype(categories=[0, 1, 2]),
        'exog_float_2': np.dtype(float),
        'exog_float32': np.dtype('float32'),
        'exog_object': np.dtype(object)
    })
    expected_float_columns = ['lag_1', 'lag_2', 'lag_3', 'exog_float', 'exog_float_2']
    if encoding == 'ordinal':
        expected_float_columns.insert(3, '_level_skforecast')
    expected_float_2 = np.array([303., 304., 305., 313., 314., 315.])
    expected_int = np.array([203, 204, 205, 213, 214, 215], dtype=int)
    expected_object = np.array(
        [pd.Timestamp('2020-01-01')] * 3 + [pd.Timestamp('2020-01-02')] * 3,
        dtype=object
    )

    pd.testing.assert_series_equal(X_train.dtypes, expected_dtypes)
    np.testing.assert_array_equal(X_train['exog_float_2'].to_numpy(), expected_float_2)
    np.testing.assert_array_equal(X_train['exog_int'].to_numpy(), expected_int)
    np.testing.assert_array_equal(X_train['exog_object'].to_numpy(), expected_object)

    # Each float64 column starts where the previous one ends.
    float_columns_values = [X_train[col].to_numpy() for col in expected_float_columns]
    float_columns_addresses = [values.ctypes.data for values in float_columns_values]
    assert all(values.strides == (8,) for values in float_columns_values)
    assert np.diff(float_columns_addresses).tolist() == (
        [len(X_train) * 8] * (len(expected_float_columns) - 1)
    )


@pytest.mark.parametrize(
    "encoding, n_float_exog, n_int_exog",
    [('ordinal', 0, 3), ('ordinal', 0, 4),
     ('ordinal', 98, 99), ('ordinal', 98, 100),
     ('ordinal_category', 0, 1), ('ordinal_category', 0, 2),
     ('ordinal_category', 98, 98), ('ordinal_category', 98, 99),
     ('onehot', 0, 4), ('onehot', 0, 5),
     ('onehot', 98, 99), ('onehot', 98, 100)],
    ids=lambda value: f'encoding, n_float_exog, n_int_exog: {value}'
)
def test_create_train_X_y_output_when_int_exog_columns_are_inserted_or_concatenated(
    encoding, n_float_exog, n_int_exog
):
    """
    Test the output of _create_train_X_y at both sides of the limits that
    decide how the columns that are not float are added to X_train. They are
    inserted one by one when they are no more than the float columns (2 lags,
    the float exog and the level: one column with `encoding='ordinal'` and one
    per series with `'onehot'`) and fewer than 100. Otherwise, X_train is
    assembled with `pd.concat`. With `encoding='ordinal_category'` the level is
    not float, so it counts as one more inserted column. The output is the same
    and pandas does not warn about a fragmented DataFrame.
    """
    series = {
        'l1': pd.Series(np.arange(5, dtype=float), name='l1'),
        'l2': pd.Series(np.arange(10, 15, dtype=float), name='l2')
    }
    float_cols = [f'exog_float_{i}' for i in range(n_float_exog)]
    int_cols = [f'exog_int_{i}' for i in range(n_int_exog)]
    exog_float = np.arange(5 * n_float_exog, dtype=float).reshape(5, n_float_exog)
    exog_int = np.arange(5 * n_int_exog, dtype=int).reshape(5, n_int_exog)
    exog = {
        'l1': pd.concat(
                  [pd.DataFrame(exog_float, columns=float_cols),
                   pd.DataFrame(exog_int, columns=int_cols)],
                  axis=1
              ),
        'l2': pd.concat(
                  [pd.DataFrame(exog_float + 1000, columns=float_cols),
                   pd.DataFrame(exog_int + 1000, columns=int_cols)],
                  axis=1
              )
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding=encoding
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=pd.errors.PerformanceWarning)
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected_index = pd.Index([2, 3, 4, 2, 3, 4])
    expected_lags = pd.DataFrame(
        data = np.array([[1., 0.],
                         [2., 1.],
                         [3., 2.],
                         [11., 10.],
                         [12., 11.],
                         [13., 12.]]),
        index   = expected_index,
        columns = ['lag_1', 'lag_2']
    )
    if encoding == 'onehot':
        expected_level = pd.DataFrame(
            data    = {'l1': [1., 1., 1., 0., 0., 0.],
                       'l2': [0., 0., 0., 1., 1., 1.]},
            index   = expected_index
        )
    else:
        expected_level = pd.DataFrame(
            data    = {'_level_skforecast': [0., 0., 0., 1., 1., 1.]},
            index   = expected_index
        )
        if encoding == 'ordinal_category':
            expected_level = expected_level.astype(
                {'_level_skforecast': int}
            ).astype(
                {'_level_skforecast': 'category'}
            )
    level_cols = expected_level.columns.to_list()
    expected_exog_float = pd.DataFrame(
        data    = np.vstack([exog_float[2:], exog_float[2:] + 1000]),
        index   = expected_index,
        columns = float_cols
    )
    expected_exog_int = pd.DataFrame(
        data    = np.vstack([exog_int[2:], exog_int[2:] + 1000]),
        index   = expected_index,
        columns = int_cols
    )
    n_autoreg_level_cols = 2 + len(level_cols)
    n_float_cols = n_autoreg_level_cols + n_float_exog

    assert results[0].columns.to_list() == (
        ['lag_1', 'lag_2'] + level_cols + float_cols + int_cols
    )
    pd.testing.assert_frame_equal(results[0].iloc[:, :2], expected_lags)
    pd.testing.assert_frame_equal(
        results[0].iloc[:, 2:n_autoreg_level_cols], expected_level
    )
    pd.testing.assert_frame_equal(
        results[0].iloc[:, n_autoreg_level_cols:n_float_cols], expected_exog_float
    )
    pd.testing.assert_frame_equal(results[0].iloc[:, n_float_cols:], expected_exog_int)
    assert results[9] == float_cols + int_cols


@pytest.mark.parametrize(
    "n_int_exog",
    [6, 7],
    ids=lambda value: f'n_int_exog: {value}'
)
def test_create_train_X_y_calendar_features_when_X_train_assembled_with_block_or_concat(
    n_int_exog
):
    """
    Test the calendar columns of X_train at both sides of the limit that
    decides how the int exog columns are added to X_train. With 6 int exog
    columns they are inserted next to the single float block (2 lags, the
    level and 3 calendar columns). With 7, X_train is assembled with
    `pd.concat`. The calendar columns are float and have the same values in
    both cases.
    """
    index = pd.date_range(start='2020-01-01', periods=8, freq='D')
    series = {
        'l1': pd.Series(np.arange(8, dtype=float), index=index, name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), index=index, name='l2')
    }
    int_cols = [f'exog_int_{i}' for i in range(n_int_exog)]
    exog_int = np.arange(8 * n_int_exog, dtype=int).reshape(8, n_int_exog)
    exog = {
        'l1': pd.DataFrame(exog_int, index=index, columns=int_cols),
        'l2': pd.DataFrame(exog_int + 1000, index=index, columns=int_cols)
    }
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator         = LinearRegression(),
                     lags              = 2,
                     encoding          = 'ordinal',
                     calendar_features = CalendarFeatures(
                                             features=['weekend', 'day_of_week'],
                                             encoding='cyclical'
                                         )
                 )
    X_train = forecaster._create_train_X_y(series=series, exog=exog)[0]

    expected_calendar = pd.DataFrame(
        data  = {
            'weekend': [0., 1., 1., 0., 0., 0.] * 2,
            'day_of_week_sin': [-0.433883739117558, -0.9749279121818236,
                                -0.7818314824680299, 0.0, 0.7818314824680298,
                                0.9749279121818236] * 2,
            'day_of_week_cos': [-0.9009688679024191, -0.2225209339563146,
                                0.6234898018587334, 1.0, 0.6234898018587336,
                                -0.22252093395631434] * 2
        },
        index = pd.DatetimeIndex(
                    ['2020-01-03', '2020-01-04', '2020-01-05',
                     '2020-01-06', '2020-01-07', '2020-01-08'] * 2
                )
    )
    assert X_train.columns.to_list() == (
        ['lag_1', 'lag_2', '_level_skforecast'] + int_cols
        + ['weekend', 'day_of_week_sin', 'day_of_week_cos']
    )
    pd.testing.assert_frame_equal(X_train.iloc[:, -3:], expected_calendar)


def test_create_train_X_y_calendar_features_same_dtype_as_create_predict_X():
    """
    Test that the calendar columns of the matrix returned by create_train_X_y
    have the same dtype as those of the matrix returned by create_predict_X,
    float, also for the integer ones (`weekend` and the one-hot columns of
    `quarter`).
    """
    index = pd.date_range(start='2020-01-01', periods=8, freq='D')
    series = {
        'l1': pd.Series(np.arange(8, dtype=float), index=index, name='l1'),
        'l2': pd.Series(np.arange(10, 18, dtype=float), index=index, name='l2')
    }
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator         = LinearRegression(),
                     lags              = 2,
                     calendar_features = CalendarFeatures(
                                             features=['weekend', 'quarter'],
                                             encoding='onehot'
                                         )
                 )
    forecaster.fit(series=series)
    X_train, _ = forecaster.create_train_X_y(series=series)
    X_predict = forecaster.create_predict_X(steps=2, suppress_warnings=True)

    calendar_columns = ['weekend', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
    expected_dtypes = pd.Series(
        data  = [np.dtype(float)] * len(calendar_columns),
        index = calendar_columns
    )

    pd.testing.assert_series_equal(X_train.dtypes[calendar_columns], expected_dtypes)
    pd.testing.assert_series_equal(X_predict.dtypes[calendar_columns], expected_dtypes)


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'ordinal_category', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
@pytest.mark.parametrize(
    "series_index_name, exog_index_name",
    [
        ('date', None),
        ('date', 'other'),
        ('date', 'date'),
        (None, 'date')
    ],
    ids=lambda value: f'index names: {value}'
)
@pytest.mark.parametrize(
    "exog_dtype, n_exog_cols",
    [(float, 1), (int, 5)],
    ids=['1 float exog (single block)', '5 int exog (pd.concat)']
)
def test_create_train_X_y_index_names_when_series_and_exog_index_have_names(
    encoding, series_index_name, exog_index_name, exog_dtype, n_exog_cols
):
    """
    Test the name of the index of X_train and y_train when the indexes of
    series and exog have names. Both have the name of the index of series,
    whatever the name of the index of exog. If the index of series has no
    name, they have no name. With one float exog X_train is built as a single
    block; with 5 int exog (more inserted columns than block columns) it is
    built with `pd.concat`, which drops the index name when series and exog
    do not share it.
    """
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    series_index = index.rename(series_index_name)
    exog_index = index.rename(exog_index_name)
    series = {
        'l1': pd.Series(np.arange(6, dtype=float), index=series_index, name='l1'),
        'l2': pd.Series(np.arange(10, 16, dtype=float), index=series_index, name='l2')
    }
    exog = {
        'l1': pd.DataFrame(
                  {f'exog_{i}': np.arange(100, 106, dtype=exog_dtype)
                   for i in range(n_exog_cols)},
                  index=exog_index
              ),
        'l2': pd.DataFrame(
                  {f'exog_{i}': np.arange(110, 116, dtype=exog_dtype)
                   for i in range(n_exog_cols)},
                  index=exog_index
              )
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding=encoding
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    assert results[0].index.name == series_index_name
    assert results[1].index.name == series_index_name


def test_create_train_X_y_output_when_exog_dict_and_one_series_without_exog():
    """
    Test the output of _create_train_X_y when exog is a dict and one of the
    series has no exogenous variables (`None`). The rows of that series have NaN
    in the exogenous columns and keep their own training index, and the column
    `_dummy_exog_col_to_keep_shape`, used internally to keep the shape of the
    exogenous matrix, is not in the output.
    """
    index_l1 = pd.date_range('2000-01-01', periods=10, freq='D')
    index_l2 = pd.date_range('2000-01-02', periods=8, freq='D')
    series = {
        'l1': pd.Series(np.arange(10, dtype=float), index=index_l1, name='l1'),
        'l2': pd.Series(np.arange(20, 28, dtype=float), index=index_l2, name='l2')
    }
    exog = {
        'l1': pd.DataFrame(
                  {'exog_1': np.arange(100, 110, dtype=float),
                   'exog_2': np.arange(200, 210, dtype=float)},
                  index = index_l1
              ),
        'l2': None
    }
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)

    warn_msg = re.escape(
        "NaNs detected in `X_train`. Some estimators do not allow "
        "NaN values during training. If you want to drop them, "
        "set `forecaster.dropna_from_series = True`."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        results = forecaster._create_train_X_y(series=series, exog=exog)

    expected_index = pd.DatetimeIndex(
        ['2000-01-04', '2000-01-05', '2000-01-06', '2000-01-07', '2000-01-08',
         '2000-01-09', '2000-01-10',
         '2000-01-05', '2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09']
    )
    expected = (
        pd.DataFrame(
            data = np.array([[ 2.,  1.,  0., 0., 103., 203.],
                             [ 3.,  2.,  1., 0., 104., 204.],
                             [ 4.,  3.,  2., 0., 105., 205.],
                             [ 5.,  4.,  3., 0., 106., 206.],
                             [ 6.,  5.,  4., 0., 107., 207.],
                             [ 7.,  6.,  5., 0., 108., 208.],
                             [ 8.,  7.,  6., 0., 109., 209.],
                             [22., 21., 20., 1., np.nan, np.nan],
                             [23., 22., 21., 1., np.nan, np.nan],
                             [24., 23., 22., 1., np.nan, np.nan],
                             [25., 24., 23., 1., np.nan, np.nan],
                             [26., 25., 24., 1., np.nan, np.nan]]),
            index   = expected_index,
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast',
                       'exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 7., 8., 9., 23., 24., 25., 26., 27.]),
            index = expected_index,
            name  = 'y',
            dtype = float
        ),
        {'l1': index_l1, 'l2': index_l2},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        [],
        None,
        None,
        ['exog_1', 'exog_2'],
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
        {'exog_1': np.dtype('float'), 'exog_2': np.dtype('float')},
        {'l1': pd.Series(
                   data  = np.array([7., 8., 9.]),
                   index = pd.date_range('2000-01-08', periods=3, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([25., 26., 27.]),
                   index = pd.date_range('2000-01-07', periods=3, freq='D'),
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
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])
    assert '_dummy_exog_col_to_keep_shape' not in results[0].columns


def test_create_train_X_y_output_when_exog_dict_window_features_transformers_and_differentiation_2():
    """
    Test the output of _create_train_X_y when exog is a dict and the forecaster
    has window features, transformer_series, transformer_exog and
    differentiation=2.
    """
    index = pd.date_range('2000-01-01', periods=10, freq='D')
    series = {
        'l1': pd.Series(
                  [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
                  index=index, name='l1', dtype=float
              ),
        'l2': pd.Series(
                  [10.2, 15.1, 12.3, 18.9, 11.4, 22.8, 19.5, 25.1, 31.2, 28.2],
                  index=index, name='l2', dtype=float
              )
    }
    exog = {
        'l1': pd.DataFrame({
                  'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
                  'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
                  index = index
              ),
        'l2': pd.DataFrame({
                  'col_1': [12.1, 33.5, 48.2, 29.9, 51.3, 60.8, 41.7, 55.4, 38.6, 22.9],
                  'col_2': ['b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a']},
                  index = index
              )
    }
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    rolling = RollingFeatures(stats=['ratio_min_max', 'median'], window_sizes=4)
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = [1, 5],
                     window_features    = rolling,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                     differentiation    = 2
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected_index = pd.DatetimeIndex(
        ['2000-01-08', '2000-01-09', '2000-01-10',
         '2000-01-08', '2000-01-09', '2000-01-10']
    )
    expected = (
        pd.DataFrame(
            data = np.array([
                       [-1.56436158, -0.14173746, -0.89489489, -0.27035108,
                         0.,  0.27075471, 0., 1.],
                       [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472,
                         0., -1.39438677, 0., 1.],
                       [-0.24672817, -0.49870587, -0.83943662,  0.75068358,
                         0.,  1.59576059, 0., 1.],
                       [-2.12512748, -1.11316201, -0.77777778, -0.33973126,
                         1.,  0.67688678, 1., 0.],
                       [ 1.2866418 ,  1.35892505, -0.77777778, -0.37587289,
                         1., -0.17599056, 0., 1.],
                       [ 0.07228325, -2.03838758, -0.77777778,  0.67946253,
                         1., -0.97302475, 1., 0.]]),
            index   = expected_index,
            columns = ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4',
                       '_level_skforecast', 'col_1', 'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([ 1.8635851 , -0.24672817, -4.60909217,
                               1.2866418 ,  0.07228325, -1.3155551 ]),
            index = expected_index,
            name  = 'y',
            dtype = float
        ),
        {'l1': index, 'l2': index},
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['col_1', 'col_2'],
        [],
        ['roll_ratio_min_max_4', 'roll_median_4'],
        None,
        ['col_1', 'col_2_a', 'col_2_b'],
        {'col_1': np.dtype('float'), 'col_2': np.dtype('O')},
        {'col_1': np.dtype('float'), 'col_2_a': np.dtype('float'),
         'col_2_b': np.dtype('float')},
        {'l1': pd.Series(
                   data  = np.array([24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3]),
                   index = pd.date_range('2000-01-04', periods=7, freq='D'),
                   name  = 'l1',
                   dtype = float
               ),
         'l2': pd.Series(
                   data  = np.array([18.9, 11.4, 22.8, 19.5, 25.1, 31.2, 28.2]),
                   index = pd.date_range('2000-01-04', periods=7, freq='D'),
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
    assert results[9] == expected[9]
    for k in results[10].keys():
        assert results[10][k] == expected[10][k]
    for k in results[11].keys():
        assert results[11][k] == expected[11][k]
    for k in results[12].keys():
        pd.testing.assert_series_equal(results[12][k], expected[12][k])
