# Unit test create_train_X_y ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from skforecast.recursive import ForecasterRecursiveClassifier


def test_create_train_X_y_output_when_encoded_False():
    """
    Test the output of create_train_X_y when encoded is False.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=False)

    expected = (
        pd.DataFrame(
            data = np.array([['b', 'a', 'c', 'b', 'a'],
                             ['c', 'b', 'a', 'c', 'b'],
                             ['a', 'c', 'b', 'a', 'c'],
                             ['b', 'a', 'c', 'b', 'a'],
                             ['c', 'b', 'a', 'c', 'b']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array(['c', 'a', 'b', 'c', 'a']),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_encoded_True():
    """
    Test the output of create_train_X_y when encoded is True.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.],
                             [0., 2., 1., 0., 2.],
                             [1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_encoded_True_as_categorical():
    """
    Test the output of create_train_X_y when encoded is True and lags are categorical.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.],
                             [0., 2., 1., 0., 2.],
                             [1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = int
        )
    )

    for col in expected[0].columns:
        expected[0][col] = pd.Categorical(
                               values     = expected[0][col],
                               categories = [0, 1, 2],
                               ordered    = False
                           )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])

