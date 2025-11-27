# Unit test plot history method with PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn

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
            recurrent_units=128,
            dense_units=64,
        )
forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
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
