
# Unit test _create_lags ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

rolling = RollingFeatures(
    stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
)


def test_create_lags_output():
    """
    Test matrix of lags is created properly when lags=3 and y=np.arange(10).
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]]),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_interspersed_lags():
    """
    Test matrix of lags if list with interspersed lags.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=[2, 3])
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[1., 0.],
                  [2., 1.],
                  [3., 2.],
                  [4., 3.],
                  [5., 4.],
                  [6., 5.],
                  [7., 6.]]),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_pandas():
    """
    Test matrix of lags is created properly when X_as_pandas=True.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    results = forecaster._create_lags(
        y=np.arange(10), X_as_pandas=True, 
        train_index=pd.date_range('2020-01-03', periods=7, freq='D')
    )
    expected = (
        pd.DataFrame(
            data = np.array([
                       [2., 1., 0.],
                       [3., 2., 1.],
                       [4., 3., 2.],
                       [5., 4., 3.],
                       [6., 5., 4.],
                       [7., 6., 5.],
                       [8., 7., 6.]]
                   ),
            columns = ["lag_1", "lag_2", "lag_3"],
            index = pd.date_range('2020-01-03', periods=7, freq='D')
        ),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_window_size_window_features_greater_than_max_lag():
    """
    Test matrix of lags created properly when window_size of 
    window_features is greater than max lag.
    """
    forecaster = ForecasterRecursive(
        LinearRegression(), lags=3, window_features=rolling
    )
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]]),
        np.array([6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_lags_None():
    """
    Test matrix of lags when lags=None.
    """
    forecaster = ForecasterRecursive(
        LinearRegression(), lags=None, window_features=rolling
    )
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        None,
        np.array([6., 7., 8., 9.])
    )

    assert results[0] == expected[0]
    np.testing.assert_array_almost_equal(results[1], expected[1])
