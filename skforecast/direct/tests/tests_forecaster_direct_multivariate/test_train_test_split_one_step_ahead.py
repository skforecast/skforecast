# Unit test _train_test_split_one_step_ahead ForecasterDirectMultiVariate
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirectMultiVariate


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    pandas dataframes.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(15),
            "series_2": np.arange(50, 65),
        },
        index=pd.date_range("2020-01-01", periods=15),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), lags=5, level="series_1", steps=1, transformer_series=None
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = pd.DataFrame(
        {
            "series_1_lag_1": [4.0, 5.0, 6.0, 7.0, 8.0],
            "series_1_lag_2": [3.0, 4.0, 5.0, 6.0, 7.0],
            "series_1_lag_3": [2.0, 3.0, 4.0, 5.0, 6.0],
            "series_1_lag_4": [1.0, 2.0, 3.0, 4.0, 5.0],
            "series_1_lag_5": [0.0, 1.0, 2.0, 3.0, 4.0],
            "series_2_lag_1": [54.0, 55.0, 56.0, 57.0, 58.0],
            "series_2_lag_2": [53.0, 54.0, 55.0, 56.0, 57.0],
            "series_2_lag_3": [52.0, 53.0, 54.0, 55.0, 56.0],
            "series_2_lag_4": [51.0, 52.0, 53.0, 54.0, 55.0],
            "series_2_lag_5": [50.0, 51.0, 52.0, 53.0, 54.0],
            "exog_1_step_1": [105.0, 106.0, 107.0, 108.0, 109.0],
            "exog_2_step_1": [1005.0, 1006.0, 1007.0, 1008.0, 1009.0],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ],
            freq="D",
        ),
    )

    expected_y_train = {
        1: pd.Series(
            [5.0, 6.0, 7.0, 8.0, 9.0],
            index=pd.DatetimeIndex(
                [
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                    "2020-01-10",
                ],
                freq="D",
            ),
            name="series_1_step_1",
        )
    }

    expected_X_test = pd.DataFrame(
        {
            "series_1_lag_1": [9.0, 10.0, 11.0, 12.0, 13.0],
            "series_1_lag_2": [8.0, 9.0, 10.0, 11.0, 12.0],
            "series_1_lag_3": [7.0, 8.0, 9.0, 10.0, 11.0],
            "series_1_lag_4": [6.0, 7.0, 8.0, 9.0, 10.0],
            "series_1_lag_5": [5.0, 6.0, 7.0, 8.0, 9.0],
            "series_2_lag_1": [59.0, 60.0, 61.0, 62.0, 63.0],
            "series_2_lag_2": [58.0, 59.0, 60.0, 61.0, 62.0],
            "series_2_lag_3": [57.0, 58.0, 59.0, 60.0, 61.0],
            "series_2_lag_4": [56.0, 57.0, 58.0, 59.0, 60.0],
            "series_2_lag_5": [55.0, 56.0, 57.0, 58.0, 59.0],
            "exog_1_step_1": [110.0, 111.0, 112.0, 113.0, 114.0],
            "exog_2_step_1": [1010.0, 1011.0, 1012.0, 1013.0, 1014.0],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ],
            freq="D",
        ),
    )

    expected_y_test = {
        1: pd.Series(
            [10.0, 11.0, 12.0, 13.0, 14.0],
            index=pd.DatetimeIndex(
                [
                    "2020-01-11",
                    "2020-01-12",
                    "2020-01-13",
                    "2020-01-14",
                    "2020-01-15",
                ],
                freq="D",
            ),
            name="series_1_step_1",
        )
    }

    expected_X_train_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ],
            freq="D",
        ),
    )

    expected_X_test_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
        ],
        index=pd.DatetimeIndex(
            ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
            freq="D",
        ),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train[1], expected_y_train[1])
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test[1], expected_y_test[1])
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)
