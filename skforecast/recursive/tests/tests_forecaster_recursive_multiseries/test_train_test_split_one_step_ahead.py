# Unit test _train_test_split_one_step_ahead ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
series = pd.DataFrame(
    {
        "series_1": np.arange(15),
        "series_2": np.arange(50, 65),
    },
    index=pd.date_range("2020-01-01", periods=15),
    dtype=float,
).to_dict(orient='series')
exog = pd.DataFrame(
    {
        "exog_1": np.arange(100, 115, dtype=float),
        "exog_2": np.arange(1000, 1015, dtype=int),
    },
    index=pd.date_range("2020-01-01", periods=15),
)
exog = {
    "series_1": exog,
    "series_2": exog
}


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe_encoding_ordinal():
    """
    Test the output of _train_test_split_one_step_ahead when series is dict and exog is
    pandas dataframe, and encoding is 'ordinal'.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal'
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
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
            "exog_1": [105., 106., 107., 108., 109., 105., 106., 107., 108., 109.],
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
    ).astype({"_level_skforecast": float, "exog_2": int})

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
    ).astype({"_level_skforecast": float, "exog_2": int})

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
    Test the output of _train_test_split_one_step_ahead when series is dict, exog is
    pandas dataframe, and encoding is 'onehot'.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='onehot',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
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


def test_train_test_split_one_step_ahead_when_y_is_series_and_exog_are_dataframe_encoding_none():
    """
    Test the output of _train_test_split_one_step_ahead when series is dict, exog is
    pandas dataframe, and encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding=None,
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
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

    
def test_train_test_split_one_step_ahead_when_series_and_exog_are_dict_RangeIndex():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    dictionaries with RangeIndex.
    """
    series_2 = {
        "series_1": pd.Series(np.arange(15, dtype=float)),
        "series_2": pd.Series(np.arange(50, 65, dtype=float)),
    }
    exog_2 = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=int),
        },
    )
    exog_2 = {"series_1": exog_2, "series_2": exog_2}

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_2, exog=exog_2, initial_train_size=10
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
        index=pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
    ).astype({"_level_skforecast": float, "exog_2": int})

    expected_y_train = pd.Series(
        [5.0, 6.0, 7.0, 8.0, 9.0, 55.0, 56.0, 57.0, 58.0, 59.0],
        pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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
        index=pd.Index([10, 11, 12, 13, 14, 10, 11, 12, 13, 14]),
    ).astype({"_level_skforecast": float, "exog_2": int})

    expected_y_test = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        index=pd.Index([10, 11, 12, 13, 14, 10, 11, 12, 13, 14]),
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
        index=pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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
        index=pd.Index([10, 11, 12, 13, 14, 10, 11, 12, 13, 14]),
    )

    pd.testing.assert_frame_equal(X_train, expected_X_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(X_test, expected_X_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)
    pd.testing.assert_series_equal(X_train_encoding, expected_X_train_encoding)
    pd.testing.assert_series_equal(X_test_encoding, expected_X_test_encoding)


def test_train_test_split_one_step_ahead_when_no_exog():
    """
    Test _train_test_split_one_step_ahead when exog is None. Verify that
    X_train and X_test contain only lag and encoding columns.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal'
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    # 3 lags + 1 encoding column = 4 columns
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4
    assert '_level_skforecast' in X_train.columns
    assert 'exog_1' not in X_train.columns
    # 2 series × 7 train obs each = 14 rows (lags=3, so 10-3=7 per series)
    assert X_train.shape[0] == 14
    # 2 series × 5 test obs each = 10 rows
    assert X_test.shape[0] == 10
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_weight_func():
    """
    Test _train_test_split_one_step_ahead precomputes sample_weight when
    weight_func is provided. Verifies weights are correctly mapped per
    observation for all series.
    """
    def custom_weights(index):
        return np.where(index >= pd.Timestamp("2020-01-07"), 2.0, 1.0)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',
        weight_func=custom_weights
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # 7 obs per series: 2020-01-04..2020-01-10
    # weights per series: 04,05,06 -> 1.0; 07,08,09,10 -> 2.0
    # 2 series concatenated
    expected_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                                 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_when_series_weights():
    """
    Test _train_test_split_one_step_ahead precomputes sample_weight when
    series_weights is provided. Verifies weights are correctly applied per
    series.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',
        series_weights={'series_1': 1.0, 'series_2': 3.0}
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # 7 obs per series: series_1 weight=1.0, series_2 weight=3.0
    expected_weights = np.array([1.0] * 7 + [3.0] * 7)
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_when_window_features_and_transformers_and_differentiation():
    """
    Test _train_test_split_one_step_ahead with window_features,
    transformer_series, transformer_exog, and differentiation. Verify
    shapes and return types.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    rolling = RollingFeatures(stats=['mean'], window_sizes=3)

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=3,
        window_features=rolling,
        encoding='ordinal',
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(X_train_encoding, pd.Series)
    assert isinstance(X_test_encoding, pd.Series)
    # 3 lags + 1 window_feature (rolling mean) + 1 encoding + 1 exog = 6 columns
    assert X_train.shape[1] == 6
    assert X_test.shape[1] == 6
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_lightgbm_categorical():
    """
    Test _train_test_split_one_step_ahead with LGBMRegressor and categorical
    features. Verify fit_kwargs contains 'categorical_feature' with correct
    indices.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        encoding='ordinal',
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    # Features: lag_1, lag_2, lag_3, _level_skforecast, exog_num, exog_cat
    assert 'categorical_feature' in fit_kwargs
    cat_idx = fit_kwargs['categorical_feature']
    # Only exog_cat is categorical (ordinal encoding is float, not category)
    exog_cat_pos = X_train.columns.get_loc('exog_cat')
    assert cat_idx == [exog_cat_pos]
    # The original forecaster.fit_kwargs should not be mutated
    assert 'categorical_feature' not in forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_xgboost_categorical():
    """
    Test _train_test_split_one_step_ahead with XGBRegressor and categorical
    features. Verify estimator is configured with feature_types and
    enable_categorical via set_params.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    estimator = XGBRegressor(random_state=123, enable_categorical=False)
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator,
        lags=3,
        encoding='ordinal',
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    # XGBoost uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # Estimator should be configured
    params = forecaster.estimator.get_params()
    assert params['enable_categorical'] is True
    # exog_cat should map to 'c' in feature_types
    exog_cat_pos = X_train.columns.get_loc('exog_cat')
    assert params['feature_types'][exog_cat_pos] == 'c'


def test_train_test_split_one_step_ahead_when_histgradientboosting_categorical():
    """
    Test _train_test_split_one_step_ahead with HistGradientBoostingRegressor
    and categorical features. Verify estimator is configured with
    categorical_features via set_params.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=HistGradientBoostingRegressor(random_state=123),
        lags=3,
        encoding='ordinal',
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    # HistGradientBoosting uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # Estimator should be configured with cat indices
    params = forecaster.estimator.get_params()
    exog_cat_pos = X_train.columns.get_loc('exog_cat')
    assert exog_cat_pos in params['categorical_features']


def test_train_test_split_one_step_ahead_when_catboost_categorical():
    """
    Test _train_test_split_one_step_ahead with CatBoostRegressor and
    categorical features. Verify fit_kwargs contains 'cat_features' with
    correct indices. Note: CatBoost int-cast is handled by the consumer
    (_predict_and_calculate_metrics_one_step_ahead_multiseries), not here.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=CatBoostRegressor(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        lags=3,
        encoding='ordinal',
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    # CatBoost uses cat_features in fit_kwargs
    assert 'cat_features' in fit_kwargs
    exog_cat_pos = X_train.columns.get_loc('exog_cat')
    assert exog_cat_pos in fit_kwargs['cat_features']
    # X_train/X_test should remain as DataFrames (no numpy conversion)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)


def test_train_test_split_one_step_ahead_when_fit_kwargs_no_categorical():
    """
    Test _train_test_split_one_step_ahead passes user-defined fit_kwargs
    through the else branch (categorical_features=None). Verify that
    fit_kwargs is a copy of forecaster.fit_kwargs and the original is
    not mutated.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        encoding='ordinal',
        categorical_features=None,
        fit_kwargs={'categorical_feature': 'auto'},
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    # User fit_kwargs passed through the else branch (no categorical config)
    assert fit_kwargs == {'categorical_feature': 'auto'}
    # fit_kwargs is a copy, not the same object
    assert fit_kwargs is not forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_non_contiguous_lags():
    """
    Test _train_test_split_one_step_ahead with non-contiguous lags (gaps).
    Verify that window_size is driven by the maximum lag and that shapes
    are correct.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=[1, 7], encoding='ordinal'
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    # window_size = max_lag = 7
    # train rows per series: 15 - 7 = 8, total = 16
    # test rows per series: 20 - 15 + 15 - 7 - (20 - 15) = 5, total = 10
    # columns: lag_1, lag_7, _level_skforecast, exog_1 = 4
    assert X_train.shape == (16, 4)
    assert X_test.shape == (10, 4)
    assert len(y_train) == 16
    assert len(y_test) == 10
    assert sample_weight is None
    assert fit_kwargs == {}


@pytest.mark.parametrize(
    'estimator, categorical_features, expected_fit_kwarg',
    [
        (
            LGBMRegressor(verbose=-1, random_state=123, n_estimators=10),
            'auto',
            'categorical_feature',
        ),
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            'auto',
            'cat_features',
        ),
        (
            XGBRegressor(random_state=123, n_estimators=10),
            'auto',
            None,
        ),
        (
            HistGradientBoostingRegressor(random_state=123, max_iter=10),
            'auto',
            None,
        ),
    ],
    ids=['LightGBM', 'CatBoost', 'XGBoost', 'HistGradientBoosting']
)
def test_train_test_split_one_step_ahead_fit_predict_with_categorical(
    estimator, categorical_features, expected_fit_kwarg
):
    """
    Integration test: verify that the output of _train_test_split_one_step_ahead
    can be used directly to fit and predict with each estimator that supports
    native categorical features. This catches mismatches between categorical
    indices and actual data layout.
    """
    series_50 = pd.DataFrame(
        {
            "series_1": np.arange(50),
            "series_2": np.arange(50, 100),
        },
        index=pd.date_range("2020-01-01", periods=50),
        dtype=float,
    ).to_dict(orient='series')
    exog_50 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 150, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 16 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=50),
    )
    exog_50 = {"series_1": exog_50, "series_2": exog_50}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator,
        lags=3,
        encoding='ordinal',
        categorical_features=categorical_features,
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_50, exog=exog_50, initial_train_size=40
        )
    )

    # Handle CatBoost int-cast (same as consumer does)
    if (
        'cat_features' in fit_kwargs
        and type(forecaster.estimator).__name__ == 'CatBoostRegressor'
    ):
        X_train_fit = X_train.copy()
        X_test_fit = X_test.copy()
        cat_cols = [X_train_fit.columns[i] for i in fit_kwargs['cat_features']]
        for df in (X_train_fit, X_test_fit):
            for col in cat_cols:
                if hasattr(df[col].dtype, 'categories'):
                    df[col] = df[col].cat.codes.astype(int)
                else:
                    df[col] = df[col].fillna(-1).astype(int)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    # Fit must succeed with the provided data and kwargs
    forecaster.estimator.fit(X_train_fit, y_train, **fit_kwargs)

    # Predict must succeed and return correct shape
    predictions = forecaster.estimator.predict(X_test_fit)
    assert predictions.shape[0] == y_test.shape[0]

    # Verify categorical config was applied
    if expected_fit_kwarg is not None:
        assert expected_fit_kwarg in fit_kwargs
    else:
        params = forecaster.estimator.get_params()
        est_name = type(forecaster.estimator).__name__
        if 'XGB' in est_name:
            assert params['enable_categorical'] is True
            assert 'c' in params['feature_types']
        elif 'HistGradient' in est_name:
            exog_cat_pos = X_train.columns.get_loc('exog_cat')
            assert exog_cat_pos in params['categorical_features']


@pytest.mark.parametrize(
    'initial_is_fitted',
    [True, False],
    ids=lambda v: f'is_fitted={v}'
)
def test_train_test_split_one_step_ahead_restores_is_fitted(initial_is_fitted):
    """
    Test _train_test_split_one_step_ahead preserves the original is_fitted
    state of the forecaster after execution.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal'
    )
    forecaster.is_fitted = initial_is_fitted

    forecaster._train_test_split_one_step_ahead(
        series=series, initial_train_size=10
    )

    assert forecaster.is_fitted == initial_is_fitted


def test_train_test_split_one_step_ahead_sample_weight_and_fit_kwargs_ordinal():
    """
    Test that sample_weight is None and fit_kwargs is empty dict when no
    weight_func, no series_weights, and no categorical_features.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, encoding='ordinal',
        categorical_features=None
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_encoding_ordinal_category():
    """
    Test _train_test_split_one_step_ahead with encoding='ordinal_category'.
    Verify that _level_skforecast column has Categorical dtype and that
    X_train_encoding correctly maps back to series names.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal_category'
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    assert isinstance(X_train, pd.DataFrame)
    assert '_level_skforecast' in X_train.columns
    # ordinal_category creates a Categorical dtype column
    assert hasattr(X_train['_level_skforecast'].dtype, 'categories')
    assert hasattr(X_test['_level_skforecast'].dtype, 'categories')
    # Encoding maps back to series names
    assert set(X_train_encoding.unique()) == {'series_1', 'series_2'}
    assert set(X_test_encoding.unique()) == {'series_1', 'series_2'}
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_ordinal_category_and_lightgbm_categorical():
    """
    Test _train_test_split_one_step_ahead with encoding='ordinal_category'
    and LGBMRegressor with categorical_features='auto'. Verify that
    _level_skforecast is included as categorical alongside exog_cat,
    resulting in 2 categorical indices.
    """
    series_20 = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    ).to_dict(orient='series')
    exog_20 = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog_20 = {"series_1": exog_20, "series_2": exog_20}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        encoding='ordinal_category',
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series_20, exog=exog_20, initial_train_size=15
        )
    )

    assert 'categorical_feature' in fit_kwargs
    cat_idx = fit_kwargs['categorical_feature']
    # Both _level_skforecast (Categorical dtype) and exog_cat should be categorical
    level_pos = X_train.columns.get_loc('_level_skforecast')
    exog_cat_pos = X_train.columns.get_loc('exog_cat')
    assert level_pos in cat_idx
    assert exog_cat_pos in cat_idx
    assert len(cat_idx) == 2


def test_train_test_split_one_step_ahead_when_weight_func_and_series_weights():
    """
    Test _train_test_split_one_step_ahead when both weight_func and
    series_weights are provided. Verify that the resulting sample_weight
    is the element-wise product of both weight sources.
    """
    def custom_weights(index):
        return np.where(index >= pd.Timestamp("2020-01-07"), 2.0, 1.0)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',
        weight_func=custom_weights,
        series_weights={'series_1': 1.0, 'series_2': 3.0}
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # 7 obs per series: 2020-01-04..2020-01-10
    # weight_func: 04,05,06 -> 1.0; 07,08,09,10 -> 2.0
    # series_weights: series_1=1.0, series_2=3.0
    # Result = series_weights * weight_func
    expected_weights = np.array(
        [1.0*1.0, 1.0*1.0, 1.0*1.0, 1.0*2.0, 1.0*2.0, 1.0*2.0, 1.0*2.0,
         3.0*1.0, 3.0*1.0, 3.0*1.0, 3.0*2.0, 3.0*2.0, 3.0*2.0, 3.0*2.0]
    )
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_when_weight_func_is_dict():
    """
    Test _train_test_split_one_step_ahead when weight_func is a dict
    mapping different weight functions per series. Verify that each
    series gets its own weight function applied.
    """
    def weights_series_1(index):
        return np.where(index >= pd.Timestamp("2020-01-08"), 5.0, 1.0)

    def weights_series_2(index):
        return np.full(len(index), 2.0)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',
        weight_func={
            'series_1': weights_series_1,
            'series_2': weights_series_2
        }
    )

    X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # 7 obs per series: 2020-01-04..2020-01-10
    # series_1: 04,05,06,07 -> 1.0; 08,09,10 -> 5.0
    # series_2: all -> 2.0
    expected_weights = np.array(
        [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0,
         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    )
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_encoding_onehot_with_3_series():
    """
    Test _train_test_split_one_step_ahead with onehot encoding and 3 series.
    Verify that X_train_encoding and X_test_encoding correctly recover the
    series name for every row, including the third series.
    """
    series_3 = pd.DataFrame(
        {
            'series_1': np.arange(20, dtype=float),
            'series_2': np.arange(50, 70, dtype=float),
            'series_3': np.arange(100, 120, dtype=float),
        },
        index=pd.date_range('2020-01-01', periods=20),
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='onehot'
    )

    (
        X_train, y_train, X_test, y_test,
        X_train_encoding, X_test_encoding,
        sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series=series_3, initial_train_size=15
        )

    # Verify all 3 series are present in encoding
    assert set(X_train_encoding.unique()) == {'series_1', 'series_2', 'series_3'}
    assert set(X_test_encoding.unique()) == {'series_1', 'series_2', 'series_3'}

    # Verify encoding length matches X_train/X_test rows
    assert len(X_train_encoding) == X_train.shape[0]
    assert len(X_test_encoding) == X_test.shape[0]

    # Verify one-hot columns are consistent with encoding:
    # for each row, the active one-hot column name must match the encoding value
    encoding_keys = ['series_1', 'series_2', 'series_3']
    onehot_train = X_train[encoding_keys].to_numpy()
    for i in range(len(X_train)):
        active_idx = np.flatnonzero(onehot_train[i] == 1)
        assert len(active_idx) == 1
        assert X_train_encoding.iloc[i] == encoding_keys[active_idx[0]]

    onehot_test = X_test[encoding_keys].to_numpy()
    for i in range(len(X_test)):
        active_idx = np.flatnonzero(onehot_test[i] == 1)
        assert len(active_idx) == 1
        assert X_test_encoding.iloc[i] == encoding_keys[active_idx[0]]


def test_train_test_split_one_step_ahead_encoding_onehot_with_5_series_and_exog():
    """
    Test _train_test_split_one_step_ahead with onehot encoding, 5 series,
    and exogenous variables. Verify that encoding extraction is correct
    when one-hot columns are interleaved with exog columns.
    """
    n_obs = 30
    series_5 = pd.DataFrame(
        {f'series_{i}': np.arange(i * 100, i * 100 + n_obs, dtype=float)
         for i in range(1, 6)},
        index=pd.date_range('2020-01-01', periods=n_obs),
    ).to_dict(orient='series')
    exog_5 = pd.DataFrame(
        {'exog_1': np.arange(n_obs, dtype=float)},
        index=pd.date_range('2020-01-01', periods=n_obs),
    )
    exog_5 = {f'series_{i}': exog_5 for i in range(1, 6)}

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='onehot'
    )

    (
        X_train, y_train, X_test, y_test,
        X_train_encoding, X_test_encoding,
        sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series=series_5, exog=exog_5, initial_train_size=20
        )

    expected_series = {f'series_{i}' for i in range(1, 6)}
    assert set(X_train_encoding.unique()) == expected_series
    assert set(X_test_encoding.unique()) == expected_series

    # Each series block should have the same number of rows in train
    train_counts = X_train_encoding.value_counts()
    assert train_counts.nunique() == 1  # all series have same count

    # Verify consistency: encoding label matches the onehot column that is 1
    encoding_keys = [f'series_{i}' for i in range(1, 6)]
    for i in range(len(X_train)):
        row_onehot = X_train.iloc[i][encoding_keys].to_numpy()
        active_idx = np.flatnonzero(row_onehot == 1)
        assert len(active_idx) == 1
        assert X_train_encoding.iloc[i] == encoding_keys[active_idx[0]]


def test_train_test_split_one_step_ahead_encoding_onehot_with_window_features():
    """
    Test _train_test_split_one_step_ahead with onehot encoding and
    window_features. Verify that encoding extraction is not confused by
    additional window feature columns.
    """
    series_20 = pd.DataFrame(
        {
            'series_1': np.arange(20, dtype=float),
            'series_2': np.arange(50, 70, dtype=float),
            'series_3': np.arange(100, 120, dtype=float),
        },
        index=pd.date_range('2020-01-01', periods=20),
    ).to_dict(orient='series')

    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=3)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, window_features=rolling, encoding='onehot'
    )

    (
        X_train, y_train, X_test, y_test,
        X_train_encoding, X_test_encoding,
        sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series=series_20, initial_train_size=15
        )

    assert set(X_train_encoding.unique()) == {'series_1', 'series_2', 'series_3'}
    assert set(X_test_encoding.unique()) == {'series_1', 'series_2', 'series_3'}

    # Columns should include lags, window features, and one-hot columns
    assert 'roll_mean_3' in X_train.columns
    assert 'roll_std_3' in X_train.columns
    assert 'series_1' in X_train.columns
    assert 'series_2' in X_train.columns
    assert 'series_3' in X_train.columns

    # Verify encoding matches one-hot columns
    encoding_keys = ['series_1', 'series_2', 'series_3']
    onehot_arr = X_train[encoding_keys].to_numpy()
    for i in range(len(X_train)):
        active_idx = np.flatnonzero(onehot_arr[i] == 1)
        assert len(active_idx) == 1
        assert X_train_encoding.iloc[i] == encoding_keys[active_idx[0]]
