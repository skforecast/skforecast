# Unit test predict method
# ==============================================================================
import keras
import numpy as np
import pandas as pd

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

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


# Test case for predicting 3 steps ahead
def test_predict_3_steps_ahead():
    """
    Test case for predicting 3 steps ahead
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    predictions = forecaster.predict(steps=3)

    # Check the shape and values of the predictions
    assert predictions.shape == (3, 2)


# Test case for predicting 2 steps ahead with specific levels
def test_predict_2_steps_ahead_specific_levels():
    """
    Test case for predicting 2 steps ahead with specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    predictions = forecaster.predict(steps=3, levels="1")

    # Check the shape and values of the predictions
    assert predictions.shape == (3, 1)
