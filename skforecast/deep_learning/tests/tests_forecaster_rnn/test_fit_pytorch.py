# Unit test fit method with PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model
os.environ["KERAS_BACKEND"] = "torch"

series = pd.DataFrame({"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))})
lags = 3
steps = 1
levels = "1"

model = create_and_compile_model(
    series=series, 
    levels=levels,    
    lags=lags,           
    steps=steps,              
    recurrent_layer="LSTM",
    recurrent_units=100,
    recurrent_layers_kwargs={"activation": "relu"},
    dense_units=[128, 64],
    dense_layers_kwargs={"activation": "relu"},
    output_dense_layer_kwargs={"activation": "linear"},
    compile_kwargs={"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
)


# Test case for fitting the forecaster without validation data
def test_fit_without_validation_data():
    """
    Test case for fitting the forecaster without validation data
    """

    forecaster = ForecasterRnn(model, levels, lags=lags)
    assert forecaster.is_fitted is False
    forecaster.fit(series)
    assert forecaster.is_fitted is True
    assert all(forecaster.training_range_ == (0, 4))
    last_window = pd.DataFrame({"1": [2, 3, 4], "2": [2, 3, 4]})
    np.testing.assert_array_almost_equal(forecaster.last_window_, last_window)


# Test case for fitting the forecaster with validation data
def test_fit_with_validation_data():
    """
    Test case for fitting the forecaster with validation data
    """

    series_val = pd.DataFrame(
        {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
    )
    forecaster = ForecasterRnn(
        regressor=model,
        levels=levels,
        fit_kwargs={
            "epochs": 10,  # Number of epochs to train the model.
            "batch_size": 32,  # Batch size to train the model.
            "series_val": series_val,  # Validation data for model training.
        },
        lags=lags
    )
    assert forecaster.is_fitted is False
    forecaster.fit(series)
    assert forecaster.is_fitted is True
    assert forecaster.history_ is not None
