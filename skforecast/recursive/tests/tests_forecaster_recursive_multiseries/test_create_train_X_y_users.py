# Unit test create_train_X_y ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
from ....recursive import ForecasterRecursiveMultiSeries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterRecursiveMultiSeries(
        regressor = LinearRegression(),
        lags      = 3,
        transformer_series=StandardScaler(),
        encoding  = "onehot"
    )

    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 1., 0.],
                             [ 0. , -0.5, -1. , 1., 0.],
                             [ 0.5,  0. , -0.5, 1., 0.],
                             [ 1. ,  0.5,  0. , 1., 0.],
                             [-0.5, -1. , -1.5, 0., 1.],
                             [ 0. , -0.5, -1. , 0., 1.],
                             [ 0.5,  0. , -0.5, 0., 1.],
                             [ 1. ,  0.5,  0. , 0., 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ).astype({'1': int, '2': int}),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_series_and_exog_and_encoding_None():
    """
    Test the output of create_train_X_y when encoding None 
    """
    series = {
        "l1": pd.Series(np.arange(10, dtype=float)),
        "l2": pd.Series(np.arange(15, 20, dtype=float)),
        "l3": pd.Series(np.arange(20, 25, dtype=float)),
    }
    series["l1"].index = pd.date_range("1990-01-01", periods=10, freq="D")
    series["l2"].index = pd.date_range("1990-01-05", periods=5, freq="D")
    series["l3"].index = pd.date_range("1990-01-03", periods=5, freq="D")
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding           = None,
                                              transformer_series = StandardScaler())
    
    forecaster.fit(series=series)

    results = forecaster._create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[-1.24514561, -1.36966017, -1.49417474, 0.],
                             [-1.12063105, -1.24514561, -1.36966017, 0.],
                             [-0.99611649, -1.12063105, -1.24514561, 0.],
                             [-0.87160193, -0.99611649, -1.12063105, 0.],
                             [-0.74708737, -0.87160193, -0.99611649, 0.],
                             [-0.62257281, -0.74708737, -0.87160193, 0.],
                             [-0.49805825, -0.62257281, -0.74708737, 0.],
                             [ 0.62257281,  0.49805825,  0.37354368, 1.],
                             [ 0.74708737,  0.62257281,  0.49805825, 1.],
                             [ 1.24514561,  1.12063105,  0.99611649, 2.],
                             [ 1.36966017,  1.24514561,  1.12063105, 2.]]),
            index   = pd.Index(
                          pd.DatetimeIndex(
                              ['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                               '1990-01-08', '1990-01-09', '1990-01-10',
                               '1990-01-08', '1990-01-09', 
                               '1990-01-06', '1990-01-07']
                          )
                      ),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'_level_skforecast': int}
        ),
        pd.Series(
            data  = np.array([
                        -1.12063105, -0.99611649, -0.87160193, -0.74708737, -0.62257281,
                        -0.49805825, -0.37354368,  0.74708737,  0.87160193,  1.36966017,
                        1.49417474]),
            index = pd.Index(
                        pd.DatetimeIndex(
                            ['1990-01-04', '1990-01-05', '1990-01-06',
                             '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                             '1990-01-08', '1990-01-09', 
                             '1990-01-06', '1990-01-07']
                        )
                    ),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
