import os

import numpy as np
import pandas as pd

from skforecast.deep_learning._forecaster_rnn import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

os.environ["KERAS_BACKEND"] = "torch"
import keras
import pytest
from keras.losses import MeanSquaredError
from keras.optimizers import Adam


# test levels, steps, lags
@pytest.mark.parametrize(
    "levels, steps, lags",
    [
        ("t+1", 5, 10),
        (["t+1"], [24, 48], 10),
        (["t+1", "t+2"], 5, [2, 4, 6]),
    ],
)
def test_init(levels, steps, lags):
    """
    Test case for the initialization of the ForecasterRnn class
    """
    print(f"levels: {levels}")
    # Generate dummy data for testing
    series = pd.DataFrame(np.random.randn(100, 3))
    series.columns = [f"t+{step}" for step in range(1, series.shape[1] + 1)]
    recurrent_units = 100
    dense_units = 64
    activation = "relu"
    optimizer = Adam(learning_rate=0.01)
    loss = MeanSquaredError()

    # Call the function to create and compile the model
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

    forecaster = ForecasterRnn(
        regressor=model,
        levels=levels,
        steps=steps,
        lags=lags,
    )

    # Assert that the forecaster is an instance of ForecasterRecursive
    assert isinstance(forecaster, ForecasterRnn)

    # Assert that the forecaster is not fitted
    assert forecaster.is_fitted is False

    # Assert that the model is compiled with the correct optimizer and loss
    assert forecaster.regressor.optimizer == optimizer
    assert forecaster.regressor.loss == loss

    # Assert that the model has the correct number of layers
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]
    if isinstance(levels, str):
        levels = [levels]

    assert (
        len(forecaster.regressor.layers)
        == 5 + len(dense_units) - 1 + len(recurrent_units) - 1
    )

    # Assert that the forecaster has the correct number of levels
    assert forecaster.levels == levels

    # Assert that the forecaster has the correct lags
    if isinstance(lags, int):
        np.testing.assert_equal(forecaster.lags, np.array(range(lags)) + 1)
    else:
        np.testing.assert_equal(forecaster.lags, np.array(lags))

    # Assert that the forecaster has the correct steps
    if isinstance(steps, int):
        np.testing.assert_equal(forecaster.steps, np.array(range(steps)) + 1)
    else:
        np.testing.assert_equal(forecaster.steps, np.array(steps))
