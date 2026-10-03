# Unit test _create_train_X_y_single_series ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_create_train_X_y_single_series_output_when_transformer_series():
    """
    Test the output of _create_train_X_y_single_series when the series is
    transformed with StandardScaler.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.transformer_series_ = {'l1': StandardScaler()}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[-0.5, -1. , -1.5],
                  [ 0. , -0.5, -1. ],
                  [ 0.5,  0. , -0.5],
                  [ 1. ,  0.5,  0. ]]),
        'l1',
        None,
        np.array([0., 0.5, 1., 1.5])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_10():
    """
    Test the output of _create_train_X_y_single_series when the series has
    10 values and no transformer.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                                transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[4., 3., 2., 1., 0.],
                  [5., 4., 3., 2., 1.],
                  [6., 5., 4., 3., 2.],
                  [7., 6., 5., 4., 3.],
                  [8., 7., 6., 5., 4.]]),
        'l1',
        None,
        np.array([5., 6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_datetime_index():
    """
    Test the output of _create_train_X_y_single_series when the series has
    a datetime index.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')
    y.index = pd.date_range("1990-01-01", periods=7, freq='D')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                                transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[2.0, 1.0, 0.0],
                  [3.0, 2.0, 1.0],
                  [4.0, 3.0, 2.0],
                  [5.0, 4.0, 3.0]]),
        'l1',
        None,
        np.array([3., 4., 5., 6.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_series_with_NaNs():
    """
    Test the output of _create_train_X_y_single_series when the series has
    NaNs in between.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    y.iloc[6:8] = np.nan

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                                transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[4., 3., 2., 1., 0.],
                  [5., 4., 3., 2., 1.],
                  [np.nan, 5., 4., 3., 2.],
                  [np.nan, np.nan, 5., 4., 3.],
                  [8., np.nan, np.nan, 5., 4.]]),
        'l1',
        None,
        np.array([5., np.nan, np.nan, 8., 9.])
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_equal(results[3], expected[3])


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
                                                transformer_series=MinMaxScaler())
    forecaster.transformer_series_ = {'l1': transformer}
    forecaster.differentiator_ = {'l1': None}
    forecaster.is_fitted = True

    new_y = pd.Series(np.arange(10, 19, dtype=float), name='l1')
    results = forecaster._create_train_X_y_single_series(y=new_y)

    expected = (
        np.array([[1.5  , 1.375, 1.25 ],
                  [1.625, 1.5  , 1.375],
                  [1.75 , 1.625, 1.5  ],
                  [1.875, 1.75 , 1.625],
                  [2.   , 1.875, 1.75 ],
                  [2.125, 2.   , 1.875]]),
        'l1',
        None,
        np.array([1.625, 1.75, 1.875, 2., 2.125, 2.25])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


@pytest.mark.parametrize("is_fitted",
                         [True, False],
                         ids = lambda is_fitted: f'is_fitted: {is_fitted}')
def test_create_train_X_y_single_series_output_when_differentiation_1(is_fitted):
    """
    Test the output of _create_train_X_y_single_series when differentiation=1.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                                transformer_series = None,
                                                differentiation    = 1)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.is_fitted = is_fitted

    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.]]),
        'l1',
        None,
        np.array([1., 1., 1., 1.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


@pytest.mark.parametrize("is_fitted",
                         [True, False],
                         ids = lambda is_fitted: f'is_fitted: {is_fitted}')
def test_create_train_X_y_single_series_output_when_differentiation_2(is_fitted):
    """
    Test the output of _create_train_X_y_single_series when differentiation=2.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5,
                                                transformer_series = None,
                                                differentiation    = 2)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.is_fitted = is_fitted

    results = forecaster._create_train_X_y_single_series(y=y)

    expected = (
        np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]),
        'l1',
        None,
        np.array([0., 0., 0.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] is None
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_window_features():
    """
    Test the output of _create_train_X_y_single_series when using window_features
    and a datetime index.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, window_features=rolling
    )
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y_datetime)

    expected = (
        np.array([[5., 4., 3., 2., 1., 3., 3., 15.],
                  [6., 5., 4., 3., 2., 4., 4., 21.],
                  [7., 6., 5., 4., 3., 5., 5., 27.],
                  [8., 7., 6., 5., 4., 6., 6., 33.],
                  [9., 8., 7., 6., 5., 7., 7., 39.],
                  [10., 9., 8., 7., 6., 8., 8., 45.],
                  [11., 10., 9., 8., 7., 9., 9., 51.],
                  [12., 11., 10., 9., 8., 10., 10., 57.],
                  [13., 12., 11., 10., 9., 11., 11., 63.]]),
        'l1',
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.]),
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_two_window_features():
    """
    Test the output of _create_train_X_y_single_series when using 2 window_features
    and a datetime index.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, window_features=[rolling, rolling_2]
    )
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y_datetime)

    expected = (
        np.array([[5., 4., 3., 2., 1., 3., 3., 15.],
                  [6., 5., 4., 3., 2., 4., 4., 21.],
                  [7., 6., 5., 4., 3., 5., 5., 27.],
                  [8., 7., 6., 5., 4., 6., 6., 33.],
                  [9., 8., 7., 6., 5., 7., 7., 39.],
                  [10., 9., 8., 7., 6., 8., 8., 45.],
                  [11., 10., 9., 8., 7., 9., 9., 51.],
                  [12., 11., 10., 9., 8., 10., 10., 57.],
                  [13., 12., 11., 10., 9., 11., 11., 63.]]),
        'l1',
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.]),
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_window_features_and_lags_None():
    """
    Test the output of _create_train_X_y_single_series when using window_features
    with a datetime index and lags=None.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='l1', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=None, window_features=rolling
    )
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(y=y_datetime)

    expected = (
        np.array([[3., 3., 15.],
                  [4., 4., 21.],
                  [5., 5., 27.],
                  [6., 6., 33.],
                  [7., 7., 39.],
                  [8., 8., 45.],
                  [9., 9., 51.],
                  [10., 10., 57.],
                  [11., 11., 63.]]),
        'l1',
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6'],
        np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.]),
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    np.testing.assert_array_almost_equal(results[3], expected[3])


def test_create_train_X_y_single_series_output_when_window_features_transformer_series_and_differentiation():
    """
    Test the output of _create_train_X_y_single_series when using window_features,
    transformer_series and differentiation.
    """
    y_datetime = pd.Series(
        [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='l1', dtype=float
    )

    transformer_series = StandardScaler()
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     LinearRegression(),
                     lags               = [1, 5],
                     window_features    = rolling,
                     encoding           = 'onehot',
                     transformer_series = transformer_series,
                     differentiation    = 2
                 )
    forecaster.transformer_series_ = {'l1': transformer_series}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}

    results = forecaster._create_train_X_y_single_series(y=y_datetime)

    expected = (
        np.array([[-1.56436158, -0.14173746, -0.89489489, -0.27035108],
                  [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472],
                  [-0.24672817, -0.49870587, -0.83943662,  0.75068358]]),
        'l1',
        ['roll_ratio_min_max_4', 'roll_median_4'],
        np.array([1.8635851, -0.24672817, -4.60909217]),
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    np.testing.assert_array_almost_equal(results[3], expected[3])
