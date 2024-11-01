# Unit test _train_test_split_one_step_ahead ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import LinearRegression


def test_train_test_split_one_step_ahead_when_y_is_series_15_and_exog_is_dataframe_of_float_int():
    """
    Test the output of _train_test_split_one_step_ahead when
    y=pd.Series(np.arange(10)) and exog is a pandas dataframe with
    2 columns of floats or ints.
    """
    y = pd.Series(
        np.arange(15), index=pd.date_range("2020-01-01", periods=15), dtype=float
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=5, steps=1)

    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [4.0, 5.0, 6.0, 7.0, 8.0],
            "lag_2": [3.0, 4.0, 5.0, 6.0, 7.0],
            "lag_3": [2.0, 3.0, 4.0, 5.0, 6.0],
            "lag_4": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lag_5": [0.0, 1.0, 2.0, 3.0, 4.0],
            "exog_1_step_1": [105.0, 106.0, 107.0, 108.0, 109.0],
            "exog_2_step_1": [1005., 1006., 1007., 1008., 1009.],
        },
        index=pd.DatetimeIndex(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"],
            freq="D",
        ),
    )

    expected_y_train = {
        1: pd.Series(
            [5.0, 6.0, 7.0, 8.0, 9.0],
            index=pd.DatetimeIndex(
                ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"],
                freq="D",
            ),
            name="y_step_1",
        )
    }

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [9.0, 10.0, 11.0, 12.0, 13.0],
            "lag_2": [8.0, 9.0, 10.0, 11.0, 12.0],
            "lag_3": [7.0, 8.0, 9.0, 10.0, 11.0],
            "lag_4": [6.0, 7.0, 8.0, 9.0, 10.0],
            "lag_5": [5.0, 6.0, 7.0, 8.0, 9.0],
            "exog_1_step_1": [110.0, 111.0, 112.0, 113.0, 114.0],
            "exog_2_step_1": [1010., 1011., 1012., 1013., 1014.],
        },
        index=pd.DatetimeIndex(
            ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
            freq="D",
        ),
    )

    expected_y_test = {
        1: pd.Series(
            [10.0, 11.0, 12.0, 13.0, 14.0],
            index=pd.DatetimeIndex(
                ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
                freq="D",
            ),
            name="y_step_1",
        )
    }

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train[1], expected_y_train[1])
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test[1], expected_y_test[1])
