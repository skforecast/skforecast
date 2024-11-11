# Unit test create_train_X_y ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    y = pd.Series(np.arange(10), index=np.arange(3, 13), dtype=float)
    exog = pd.Series(np.arange(100, 110), index=np.arange(3, 13), 
                     name='exog', dtype=float)
    forecaster = ForecasterRecursive(LinearRegression(), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
