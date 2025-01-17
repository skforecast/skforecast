# Unit test predict bootstrapping method
# ==============================================================================
import os
import keras
import numpy as np
import pandas as pd
import scipy

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

os.environ["KERAS_BACKEND"] = "torch"

series = pd.DataFrame(
    {
        "1": pd.Series(np.arange(50)),
        "2": pd.Series(np.arange(50)),
        "3": pd.Series(np.arange(50)),
    }
)
lags = 3
steps = 4
distribution = scipy.stats.norm
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

# Adjust after including steps by default
"""
def test_predict_dist_output_size_with_steps_by_default():
    
    #Test output sizes for predicting distribution with steps defined by default
    
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    dist_preds = forecaster.predict_dist(distribution = distribution)

    # Check the shape and values of the predictions
    assert dist_preds.shape == (steps * len(levels), 3)
    """


def test_predict_dist_output_size_3_steps_ahead():
    """
    Test output sizes for predicting distribution 3 steps ahead
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    dist_preds = forecaster.predict_dist(steps=3, distribution=distribution)

    # Check the shape and values of the predictions
    assert dist_preds.shape == (3 * len(levels), 3)


def test_predict_dist_output_size_2_steps_ahead_specific_levels():
    """
    Test output sizes for predicting distribution 2 steps ahead and specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    dist_preds = forecaster.predict_dist(steps=2, distribution=distribution, levels="1")

    # Check the shape and values of the predictions
    assert dist_preds.shape == (2 * 1, 3)

