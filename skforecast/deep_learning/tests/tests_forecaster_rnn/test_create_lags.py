# Unit test _create_lags ForecasterRnn
# ==============================================================================
import re

import keras
import numpy as np
import pandas as pd
import pytest

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

lags = 6
steps = 3
levels = "l1"
activation = "relu"
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss = keras.losses.MeanSquaredError()
recurrent_units = 100
dense_units = [128, 64]


def test_check_create_lags_exception_when_n_splits_less_than_0():
    """
    Check exception is raised when n_splits in _create_lags is less than 0.
    """
    series = pd.DataFrame(np.arange(10), columns=["l1"])
    y_array = np.arange(10)
    lags_20 = 20
    model = create_and_compile_model(
        series=series,
        lags=lags_20,
        steps=steps,
        levels=levels,
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
    )
    forecaster = ForecasterRnn(model, levels, lags=lags_20)

    err_msg = re.escape(
        f"The maximum lag ({forecaster.max_lag}) must be less than the length "
        f"of the series minus the maximum of steps ({len(series)-forecaster.max_step})."
    )

    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_lags(y=y_array)


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
        lags=lags,
        steps=steps,
        levels=levels,
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
    )
    forecaster = ForecasterRnn(regressor=model, levels=levels, lags=lags, steps=steps)

    results = forecaster._create_lags(y=np.arange(10))

    np.testing.assert_array_equal(results[0], expected[0])
    np.testing.assert_array_equal(results[1], np.transpose(expected[1]))
