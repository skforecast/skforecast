# Unit test predict interval method with PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

os.environ["KERAS_BACKEND"] = "torch"
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

def test_predict_interval_output_size_with_steps_by_default():
    
    #Test output sizes for predicting steps defined by default with intervals
    
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval()

    # Check the shape and values of the predictions
    assert int_preds.shape == (4 * len(["1", "2"]), 4)


def test_predict_interval_output_size_3_steps_ahead():
    """
    Test output sizes for predicting 3 steps ahead with intervals
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval(steps=3)

    # Check the shape and values of the predictions
    assert int_preds.shape == (3 * len(["1", "2"]), 4)


def test_predict_interval_output_size_2_steps_ahead_specific_levels():
    """
    Test output sizes for predicting 2 steps ahead with intervals and specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval(steps=2, levels="1")

    # Check the shape and values of the predictions
    assert int_preds.shape == (2 * 1, 4)

