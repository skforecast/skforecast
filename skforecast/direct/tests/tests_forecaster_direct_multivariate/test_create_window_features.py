# Unit test _create_window_features ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate


class WindowFeatureNoPandas:
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self, y):
        return y

    def transform(self):
        pass


class WindowFeatureNoCorrectLength:
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self, y):
        y = pd.DataFrame(y).iloc[-1:, :]
        return y

    def transform(self):
        pass


class WindowFeatureNoCorrectIndex:
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self, y):
        y = pd.DataFrame(y)
        y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
        return y

    def transform(self):
        pass


def test_create_window_features_TypeError_when_transform_batch_not_pandas():
    """
    Test TypeError is raised when `transform_batch` does not return 
    a pandas DataFrame.
    """
    wf = WindowFeatureNoPandas(window_sizes=5, features_names='feature_1')
    y = pd.Series(np.arange(10), name='l1')
    train_index = pd.RangeIndex(start=6, stop=10, step=1)

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=5, steps=2, window_features=wf
    )
    err_msg = re.escape(
        ("The method `transform_batch` of WindowFeatureNoPandas "
         "must return a pandas DataFrame.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_window_features(y=y, train_index=train_index)


def test_create_window_features_ValueError_when_transform_batch_not_correct_length():
    """
    Test ValueError is raised when `transform_batch` does not return
    a DataFrame with the correct length.
    """
    wf = WindowFeatureNoCorrectLength(window_sizes=5, features_names='feature_1')
    y = pd.Series(np.arange(10), name='l1')
    train_index = pd.RangeIndex(start=6, stop=10, step=1)

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=5, steps=2, window_features=wf
    )
    err_msg = re.escape(
        ("The method `transform_batch` of WindowFeatureNoCorrectLength "
         "must return a DataFrame with the same number of rows as "
         "the input time series - (`window_size` + (`steps` - 1)): 4.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_window_features(y=y, train_index=train_index)


def test_create_window_features_output():
    """
    Test window features are created properly.
    """
    y = pd.Series(np.arange(10), name='l1')
    train_index = pd.RangeIndex(start=6, stop=10, step=1)
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=5, steps=2, window_features=rolling
    )
    results = forecaster._create_window_features(
        y=y, train_index=train_index
    )
    expected = (
        [
            np.array([[3., 3., 15.],
                      [4., 4., 21.],
                      [5., 5., 27.],
                      [6., 6., 33.]])
        ],
        ['l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6']
    )

    for result, exp in zip(results[0], expected[0]):
        np.testing.assert_array_almost_equal(result, exp)
    assert results[1] == expected[1]


def test_create_window_features_output_as_pandas():
    """
    Test window features are created properly as pandas.
    """
    y_datetime = pd.Series(
        np.arange(10), index=pd.date_range(start='2020-01-01', periods=10), name='l1'
    )
    train_index = pd.date_range(start='2020-01-08', periods=3)
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=3, window_features=rolling
    )
    results = forecaster._create_window_features(
        y=y_datetime, train_index=train_index, X_as_pandas=True
    )
    expected = (
        [
            pd.DataFrame(
                data = np.array(
                           [[4., 4., 21.],
                            [5., 5., 27.],
                            [6., 6., 33.]]),
                index = train_index,
                columns = ['l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6']
            )
        ],
        ['l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6']
    )

    for result, exp in zip(results[0], expected[0]):
        np.testing.assert_array_almost_equal(result, exp)
    assert results[1] == expected[1]


def test_create_window_features_output_list():
    """
    Test window features are created properly when `window_features` is a list.
    """
    y = pd.Series(np.arange(10), name='l2')
    train_index = pd.RangeIndex(start=7, stop=10, step=1)
    rolling_1 = RollingFeatures(
        stats=['mean', 'median'], window_sizes=[5, 5]
    )
    rolling_2 = RollingFeatures(
        stats='sum', window_sizes=6, features_names=['feature_2']
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=2, window_features=[rolling_1, rolling_2]
    )
    results = forecaster._create_window_features(y=y, train_index=train_index)
    expected = (
        [
            np.array([[4., 4.],
                      [5., 5.],
                      [6., 6.]]),
            np.array([[21.],
                      [27.],
                      [33.]])
        ],
        ['l2_roll_mean_5', 'l2_roll_median_5', 'l2_feature_2']
    )

    for result, exp in zip(results[0], expected[0]):
        np.testing.assert_array_almost_equal(result, exp)
    assert results[1] == expected[1]
