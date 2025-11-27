# Unit test create_train_X_y ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirect


def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_float():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of floats.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_step_1']
        ).astype({'exog_step_1': float}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
