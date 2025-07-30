# Unit test plot history method with PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
from skforecast.deep_learning import ForecasterRnn

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.optimizers import Adam, MeanSquaredError
import matplotlib.pyplot as plt

from skforecast.deep_learning.utils import create_and_compile_model

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
forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
forecaster.fit(series)


def test_plot_history_with_val_loss():
    """
    Test case for the plot_history method
    """
    # Call the plot_history method
    fig, ax = plt.subplots()
    forecaster.plot_history(ax=ax)

    # Assert that the figure is of type matplotlib.figure.Figure
    assert isinstance(fig, plt.Figure)

    # Assert that the plot contains the training loss curve
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "Training Loss"

    # Assert that the plot contains the validation loss curve
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "Training Loss"
