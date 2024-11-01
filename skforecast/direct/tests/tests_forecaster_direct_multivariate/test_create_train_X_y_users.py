# Unit test create_train_X_y ForecasterDirectMultiVariate
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', [-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828, 1.21854359,  1.5666989 ]), 
                          ('l2', [-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828, 1.21854359,  1.5666989 ])])
def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags=3, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.87038828, -1.21854359, -1.5666989 , -0.87038828, -1.21854359, -1.5666989 ],
                             [-0.52223297, -0.87038828, -1.21854359, -0.52223297, -0.87038828, -1.21854359],
                             [-0.17407766, -0.52223297, -0.87038828, -0.17407766, -0.52223297, -0.87038828],
                             [ 0.17407766, -0.17407766, -0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [ 0.52223297,  0.17407766, -0.17407766,  0.52223297,  0.17407766, -0.17407766],
                             [ 0.87038828,  0.52223297,  0.17407766,  0.87038828,  0.52223297, 0.17407766],
                             [ 1.21854359,  0.87038828,  0.52223297,  1.21854359,  0.87038828, 0.52223297]], 
                            dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values, dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = f'{level}_step_1'
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key])