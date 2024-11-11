# Unit test _create_lags ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect

rolling = RollingFeatures(
    stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
)

  
def test_create_lags_when_lags_is_3_steps_1_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 1 and y is
    np.arange(10).
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]]),
        np.array(
            [[3.], 
             [4.], 
             [5.], 
             [6.], 
             [7.], 
             [8.], 
             [9.]]
        )
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_pandas():
    """
    Test matrix of lags is created properly when X_as_pandas=True.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)
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
        np.array(
            [[3.], 
             [4.], 
             [5.], 
             [6.], 
             [7.], 
             [8.], 
             [9.]]
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_lags_is_list_interspersed_lags_steps_1_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is a list with interspersed 
    lags, steps is 1 and y is np.arange(10).
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=[1, 5], steps=1)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[4., 0.],
                  [5., 1.],
                  [6., 2.],
                  [7., 3.],
                  [8., 4.]]),
        np.array(
            [[5.], 
             [6.], 
             [7.], 
             [8.], 
             [9.]]
        )
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_lags_is_3_steps_2_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 2 and y is
    np.arange(10).
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.]]),
        np.array([
            [3., 4.],
            [4., 5.],
            [5., 6.],
            [6., 7.],
            [7., 8.],
            [8., 9.]]
        )
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_lags_is_3_steps_5_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 5 and y is
    np.arange(10).
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.]]),
        np.array([[3., 4., 5., 6., 7.],
                  [4., 5., 6., 7., 8.],
                  [5., 6., 7., 8., 9.]])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_lags_None():
    """
    Test matrix of lags when lags=None.
    """
    forecaster = ForecasterDirect(
        LinearRegression(), lags=None, steps=2, window_features=rolling
    )
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        None,
        np.array([[6., 7.],
                  [7., 8.],
                  [8., 9.]])
    )

    assert results[0] == expected[0]
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_window_size_window_features_greater_than_max_lag():
    """
    Test matrix of lags created properly when lags is 3, steps is 2, y is
    np.arange(10) and window_size of window_features is greater than max lag.
    """
    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=2, window_features=rolling
    )
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.]]),
        np.array([
            [6., 7.],
            [7., 8.],
            [8., 9.]]
        )
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
