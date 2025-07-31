# Unit test _create_lags ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
import pytest
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model


# parametrize tests
@pytest.mark.parametrize(
    "lags, steps, expected",
    [
        # test_create_lags_when_lags_is_3_steps_1_and_y_is_numpy_arange_10
        (
            [1, 2, 3],
            1,
            (
                np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 3.0, 4.0],
                        [3.0, 4.0, 5.0],
                        [4.0, 5.0, 6.0],
                        [5.0, 6.0, 7.0],
                        [6.0, 7.0, 8.0],
                    ]
                ),
                np.array([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]),
            ),
        ),
        # test_create_lags_when_lags_is_list_interspersed_lags_steps_1_and_y_is_numpy_arange_10
        (
            [1, 5],
            1,
            (
                np.array([[0.0, 4.0], [1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]),
                np.array([[5.0, 6.0, 7.0, 8.0, 9.0]]),
            ),
        ),
        # test_create_lags_when_lags_is_3_steps_2_and_y_is_numpy_arange_10
        (
            3,
            2,
            (
                np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 3.0, 4.0],
                        [3.0, 4.0, 5.0],
                        [4.0, 5.0, 6.0],
                        [5.0, 6.0, 7.0],
                    ]
                ),
                np.array(
                    [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
                ),
            ),
        ),
        # test_create_lags_when_lags_is_3_steps_5_and_y_is_numpy_arange_10
        (
            3,
            5,
            (
                np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                np.array(
                    [
                        [3.0, 4.0, 5.0],
                        [4.0, 5.0, 6.0],
                        [5.0, 6.0, 7.0],
                        [6.0, 7.0, 8.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
            ),
        ),
    ],
)
def test_create_lags_several_configurations(lags, steps, expected):
    """
    Test matrix of lags created with different configurations.
    """
    series = pd.DataFrame(np.arange(10), columns=["l1"])
    model = create_and_compile_model(
                series=series, 
                levels="l1",    
                lags=lags,           
                steps=steps,              
                recurrent_layer="LSTM",
                recurrent_units=128,
                dense_units=64,
            )
    forecaster = ForecasterRnn(regressor=model, levels='l1', lags=lags)
    results = forecaster._create_lags(y=np.arange(10))

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], np.transpose(expected[1]))
