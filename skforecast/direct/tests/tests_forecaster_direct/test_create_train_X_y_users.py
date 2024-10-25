# Unit test create_train_X_y ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirect


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    y = pd.Series(np.arange(10), index=np.arange(6, 16), dtype=float)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)
    
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 103.],
                             [3., 2., 1., 104.],
                             [4., 3., 2., 105.],
                             [5., 4., 3., 106.],
                             [6., 5., 4., 107.],
                             [7., 6., 5., 108.],
                             [8., 7., 6., 109.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
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
