# Unit test predict method using TensorFlow backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

series = pd.DataFrame(
    {
        "1": pd.Series(np.arange(50)),
        "2": pd.Series(np.arange(50)),
        "3": pd.Series(np.arange(50)),
    }
)

model = create_and_compile_model(
            series=series, 
            levels=["1", "2"],    
            lags=3,           
            steps=4,              
            recurrent_layer="LSTM",
            recurrent_units=100,
            recurrent_layers_kwargs={"activation": "relu"},
            dense_units=[128, 64],
            dense_layers_kwargs={"activation": "relu"},
            output_dense_layer_kwargs={"activation": "linear"},
            compile_kwargs={"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
        )


def test_predict_3_steps_ahead():
    """
    Test case for predicting 3 steps ahead
    """
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series)
    predictions = forecaster.predict(steps=3)

    assert predictions.shape == (6, 2)


def test_predict_2_steps_ahead_specific_levels():
    """
    Test case for predicting 2 steps ahead with specific levels
    """
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series)
    predictions = forecaster.predict(steps=3, levels=["1"])

    assert predictions.shape == (3, 2)