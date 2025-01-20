# Unit test predict method
# ==============================================================================
import os

import numpy as np
import pandas as pd
import pytest

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

os.environ["KERAS_BACKEND"] = "torch"
import keras

series = pd.DataFrame(
    {
        "1": pd.Series(np.arange(50)),
        "2": pd.Series(np.arange(50)),
        "3": pd.Series(np.arange(50)),
    }
)
lags = 3
steps = 4
levels = ["1", "2"]
activation = "relu"
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss = keras.losses.MeanSquaredError()
recurrent_units = 100
dense_units = [128, 64]


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

def test_predict_bootstrapping_output_size_with_steps_by_default():
    """
    Test output sizes for predicting steps defined by default with bootstrapping
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    boot_preds = forecaster.predict_bootstrapping()

    # Check the shape and values of the predictions
    assert boot_preds.shape == (steps * len(levels), 251)


def test_predict_bootstrapping_output_size_3_steps_ahead():
    """
    Test output sizes for predicting 3 steps ahead with bootstrapping
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    boot_preds = forecaster.predict_bootstrapping(steps=3)

    # Check the shape and values of the predictions
    assert boot_preds.shape == (3 * len(levels), 251)


def test_predict_2_steps_ahead_specific_levels():
    """
    Test output sizes for predicting 2 steps ahead with bootstrapping and specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    boot_preds = forecaster.predict_bootstrapping(steps=2, levels="1")

    # Check the shape and values of the predictions
    assert boot_preds.shape == (2 * 1, 251)
