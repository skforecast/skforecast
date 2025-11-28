# Unit test _train_test_split_one_step_ahead ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from skforecast.recursive import ForecasterRecursiveClassifier


def test_train_test_split_one_step_ahead_when_y_is_series_15_and_exog_is_dataframe_of_float_int():
    """
    Test the output of _train_test_split_one_step_ahead when 
    y=pd.Series(np.arange(10)) and exog is a pandas dataframe with 
    2 columns of floats or ints.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']), 
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)

    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [1., 2., 0., 1., 2.],
            "lag_2": [0., 1., 2., 0., 1.],
            "lag_3": [2., 0., 1., 2., 0.],
            "lag_4": [1., 2., 0., 1., 2.],
            "lag_5": [0., 1., 2., 0., 1.],
            "exog_1": [105., 106., 107., 108., 109.],
            "exog_2": [1005., 1006., 1007., 1008., 1009.],
        },
        index=pd.DatetimeIndex(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"],
            freq='D'
        ),
    )
    expected_y_train = pd.Series(
        [2., 0., 1., 2., 0.],
        index=pd.DatetimeIndex(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"],
            freq='D'
        ),
        name = 'y'
    )

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [0., 1., 2., 0., 1.],
            "lag_2": [2., 0., 1., 2., 0.],
            "lag_3": [1., 2., 0., 1., 2.],
            "lag_4": [0., 1., 2., 0., 1.],
            "lag_5": [2., 0., 1., 2., 0.],
            "exog_1": [110., 111., 112., 113., 114.],
            "exog_2": [1010., 1011., 1012., 1013., 1014.],
        },
        index=pd.DatetimeIndex(
            ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
            freq='D'
        ),
    )
    expected_y_test = pd.Series(
        [1., 2., 0., 1., 2.],
        index=pd.DatetimeIndex(
            ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
            freq='D'
        ),
        name = 'y'
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
