# Unit test _train_test_split_one_step_ahead ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
from ....recursive import ForecasterRecursiveMultiSeries
from sklearn.linear_model import LinearRegression


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe_encoding_ordinal():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    pandas dataframes, and encoding is 'ordinal'.
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
            "exog_2": np.arange(1000, 1015, dtype=int),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
        encoding='ordinal'
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [4.0, 5.0, 6.0, 7.0, 8.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            "lag_2": [3.0, 4.0, 5.0, 6.0, 7.0, 53.0, 54.0, 55.0, 56.0, 57.0],
            "lag_3": [2.0, 3.0, 4.0, 5.0, 6.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            "lag_4": [1.0, 2.0, 3.0, 4.0, 5.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "lag_5": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0, 52.0, 53.0, 54.0],
            "_level_skforecast": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "exog_2": [1005, 1006, 1007, 1008, 1009, 1005, 1006, 1007, 1008, 1009],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    ).astype({"_level_skforecast": int, "exog_2": int})

    expected_y_train = pd.Series(
        [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
        name="y",
    )

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [9.0, 10.0, 11.0, 12.0, 13.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            "lag_2": [8.0, 9.0, 10.0, 11.0, 12.0, 58.0, 59.0, 60.0, 61.0, 62.0],
            "lag_3": [7.0, 8.0, 9.0, 10.0, 11.0, 57.0, 58.0, 59.0, 60.0, 61.0],
            "lag_4": [6.0, 7.0, 8.0, 9.0, 10.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            "lag_5": [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            "_level_skforecast": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
            ],
            "exog_2": [1010, 1011, 1012, 1013, 1014, 1010, 1011, 1012, 1013, 1014],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    ).astype({"_level_skforecast": int, "exog_2": int})

    expected_y_test = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
        name="y",
    )

    expected_X_train_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    )

    expected_X_test_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe_encoding_onehot():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    pandas dataframes, and encoding is 'onehot'.
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
            "exog_2": np.arange(1000, 1015, dtype=int),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
        encoding='onehot',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [4.0, 5.0, 6.0, 7.0, 8.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            "lag_2": [3.0, 4.0, 5.0, 6.0, 7.0, 53.0, 54.0, 55.0, 56.0, 57.0],
            "lag_3": [2.0, 3.0, 4.0, 5.0, 6.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            "lag_4": [1.0, 2.0, 3.0, 4.0, 5.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "lag_5": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0, 52.0, 53.0, 54.0],
            'series_1': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'series_2': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "exog_2": [1005, 1006, 1007, 1008, 1009, 1005, 1006, 1007, 1008, 1009],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    ).astype({"series_1": int, "series_2": int, "exog_2": int})

    expected_y_train = pd.Series(
        [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
        name="y",
    )

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [9.0, 10.0, 11.0, 12.0, 13.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            "lag_2": [8.0, 9.0, 10.0, 11.0, 12.0, 58.0, 59.0, 60.0, 61.0, 62.0],
            "lag_3": [7.0, 8.0, 9.0, 10.0, 11.0, 57.0, 58.0, 59.0, 60.0, 61.0],
            "lag_4": [6.0, 7.0, 8.0, 9.0, 10.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            "lag_5": [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            'series_1': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'series_2': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
            ],
            "exog_2": [1010, 1011, 1012, 1013, 1014, 1010, 1011, 1012, 1013, 1014],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    ).astype({"series_1": int, "series_2": int, "exog_2": int})

    expected_y_test = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
        name="y",
    )

    expected_X_train_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    )

    expected_X_test_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe_encoding_none():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    pandas dataframes, and encoding is None.
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
            "exog_2": np.arange(1000, 1015, dtype=int),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
        encoding=None,
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [4.0, 5.0, 6.0, 7.0, 8.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            "lag_2": [3.0, 4.0, 5.0, 6.0, 7.0, 53.0, 54.0, 55.0, 56.0, 57.0],
            "lag_3": [2.0, 3.0, 4.0, 5.0, 6.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            "lag_4": [1.0, 2.0, 3.0, 4.0, 5.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "lag_5": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0, 52.0, 53.0, 54.0],
            "exog_1": [
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "exog_2": [1005, 1006, 1007, 1008, 1009, 1005, 1006, 1007, 1008, 1009],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    ).astype({"exog_2": int})

    expected_y_train = pd.Series(
        [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
        name="y",
    )

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [9.0, 10.0, 11.0, 12.0, 13.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            "lag_2": [8.0, 9.0, 10.0, 11.0, 12.0, 58.0, 59.0, 60.0, 61.0, 62.0],
            "lag_3": [7.0, 8.0, 9.0, 10.0, 11.0, 57.0, 58.0, 59.0, 60.0, 61.0],
            "lag_4": [6.0, 7.0, 8.0, 9.0, 10.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            "lag_5": [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            "exog_1": [
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
            ],
            "exog_2": [1010, 1011, 1012, 1013, 1014, 1010, 1011, 1012, 1013, 1014],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    ).astype({"exog_2": int})

    expected_y_test = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
        name="y",
    )

    expected_X_train_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    )

    expected_X_test_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)

    
def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dict():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    dictionaries.
    """
    series = {
            "series_1": pd.Series(np.arange(15, dtype=float), index=pd.date_range("2020-01-01", periods=15)),
            "series_2": pd.Series(np.arange(50, 65, dtype=float), index=pd.date_range("2020-01-01", periods=15))
        }
    exog = {
            "series_1": pd.DataFrame(
                {
                    "exog_1": np.arange(100, 115, dtype=float),
                    "exog_2": np.arange(1000, 1015, dtype=int),
                },
                index=pd.date_range("2020-01-01", periods=15),
            ),
            "series_2": pd.DataFrame(
                {
                    "exog_1": np.arange(100, 115, dtype=float),
                    "exog_2": np.arange(1000, 1015, dtype=int),
                },
                index=pd.date_range("2020-01-01", periods=15),
            ),
        }

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = pd.DataFrame(
        {
            "lag_1": [4.0, 5.0, 6.0, 7.0, 8.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            "lag_2": [3.0, 4.0, 5.0, 6.0, 7.0, 53.0, 54.0, 55.0, 56.0, 57.0],
            "lag_3": [2.0, 3.0, 4.0, 5.0, 6.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            "lag_4": [1.0, 2.0, 3.0, 4.0, 5.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "lag_5": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0, 52.0, 53.0, 54.0],
            "_level_skforecast": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "exog_2": [1005, 1006, 1007, 1008, 1009, 1005, 1006, 1007, 1008, 1009],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    ).astype({"_level_skforecast": int, "exog_2": int})

    expected_y_train = pd.Series(
        [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
        name="y",
    )

    expected_X_test = pd.DataFrame(
        {
            "lag_1": [9.0, 10.0, 11.0, 12.0, 13.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            "lag_2": [8.0, 9.0, 10.0, 11.0, 12.0, 58.0, 59.0, 60.0, 61.0, 62.0],
            "lag_3": [7.0, 8.0, 9.0, 10.0, 11.0, 57.0, 58.0, 59.0, 60.0, 61.0],
            "lag_4": [6.0, 7.0, 8.0, 9.0, 10.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            "lag_5": [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            "_level_skforecast": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "exog_1": [
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
            ],
            "exog_2": [1010, 1011, 1012, 1013, 1014, 1010, 1011, 1012, 1013, 1014],
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    ).astype({"_level_skforecast": int, "exog_2": int})

    expected_y_test = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
        name="y",
    )

    expected_X_train_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        ),
    )

    expected_X_test_encoding = pd.Series(
        [
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_1",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
            "series_2",
        ],
        index=pd.DatetimeIndex(
            [
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
            ]
        ),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)


def test_train_test_split_one_step_ahead_raise_error_when_series_is_dict_with_different_freq():
    """
    Test _train_test_split_one_step_ahead method raises error when series is dict
    and has different frequencies.
    """
    series = {
            "series_1": pd.Series(np.arange(15, dtype=float), index=pd.date_range("2020-01-01", periods=15, freq="D")),
            "series_2": pd.Series(np.arange(50, 65, dtype=float), index=pd.date_range("2020-01-01", periods=15, freq="H"))
        }

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
    )

    with np.testing.assert_raises(ValueError):
        _ = forecaster._train_test_split_one_step_ahead(series=series, initial_train_size=10)


def test_train_test_split_one_step_ahead_raise_error_when_series_is_dict_without_frequency():
    """
    Test _train_test_split_one_step_ahead method raises error when series is dict
    and has no frequencies.
    """
    series = {
            "series_1": pd.Series(np.arange(15, dtype=float), index=pd.date_range("2020-01-01", periods=15)),
            "series_2": pd.Series(np.arange(50, 65, dtype=float), index=pd.date_range("2020-01-01", periods=15))
        }
    series['series_1'].index.freq = None
    series['series_2'].index.freq = None

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(),
        lags=5,
    )

    with np.testing.assert_raises(ValueError):
        _ = forecaster._train_test_split_one_step_ahead(series=series, initial_train_size=10)