
# Unit test _create_lags ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

rolling = RollingFeaturesClassification(
    stats=['proportion', 'mode', 'entropy'], window_sizes=[5, 5, 6]
)


def test_create_lags_output():
    """
    Test matrix of lags is created properly when lags=3 and y=np.arange(10).
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
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
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=[2, 3])
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


def test_create_lags_output_pandas_use_native_categoricals_False():
    """
    Test matrix of lags is created properly when X_as_pandas=True but not 
    using native pandas categoricals.
    """
    y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=int)
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    results = forecaster._create_lags(
        y=y, X_as_pandas=True, train_index=pd.date_range('2020-01-03', periods=7, freq='D')
    )

    expected = (
        pd.DataFrame(
            data = np.array([
                       [3, 2, 1],
                       [1, 3, 2],
                       [2, 1, 3],
                       [3, 2, 1],
                       [1, 3, 2],
                       [2, 1, 3],
                       [3, 2, 1]]
                   ),
            columns = ["lag_1", "lag_2", "lag_3"],
            index = pd.date_range('2020-01-03', periods=7, freq='D')
        ),
        np.array([1, 2, 3, 1, 2, 3, 1], dtype=int)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_pandas_use_native_categoricals_True():
    """
    Test matrix of lags is created properly when X_as_pandas=True  
    using native pandas categoricals.
    """
    y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=int)
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=3)
    results = forecaster._create_lags(
        y=y, X_as_pandas=True, train_index=pd.date_range('2020-01-03', periods=7, freq='D')
    )

    expected = (
        pd.DataFrame(
            data = np.array([
                       [3, 2, 1],
                       [1, 3, 2],
                       [2, 1, 3],
                       [3, 2, 1],
                       [1, 3, 2],
                       [2, 1, 3],
                       [3, 2, 1]]
                   ),
            columns = ["lag_1", "lag_2", "lag_3"],
            index = pd.date_range('2020-01-03', periods=7, freq='D')
        ),
        np.array([1, 2, 3, 1, 2, 3, 1], dtype=int)
    )

    for col in expected[0].columns:
        expected[0][col] = pd.Categorical(
                               values     = expected[0][col],
                               categories = [1, 2, 3],
                               ordered    = False
                           )

    pd.testing.assert_frame_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_when_window_size_window_features_greater_than_max_lag():
    """
    Test matrix of lags created properly when window_size of 
    window_features is greater than max lag.
    """
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=3, window_features=rolling
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
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=None, window_features=rolling
    )
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        None,
        np.array([6., 7., 8., 9.])
    )

    assert results[0] == expected[0]
    np.testing.assert_array_almost_equal(results[1], expected[1])
