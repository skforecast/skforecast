# Unit test predict quantiles method
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

# Adjust after including steps by default
"""
def test_predict_quantiles_output_size_with_steps_by_default():
    
    #Test output sizes for predicting steps defined by default with quantiles
    
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    quant_preds = forecaster.predict_quantiles()

    # Check the shape and values of the predictions
    assert quant_preds.shape == (steps * len(levels), 4)
    """


def test_predict_quantiles_output_size_3_steps_ahead():
    """
    Test output sizes for predicting 3 steps ahead with quantiles by default
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    quant_preds = forecaster.predict_quantiles(steps=3)

    # Check the shape and values of the predictions
    assert quant_preds.shape == (3 * len(levels), 4)

def test_predict_deciles_output_size_3_steps_ahead():
    """
    Test output sizes for predicting 3 steps ahead with deciles
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    quant_preds = forecaster.predict_quantiles(steps=3,
                                               quantiles = [0.1 * i for i in range(1, 10)])

    # Check the shape and values of the predictions
    assert quant_preds.shape == (3 * len(levels), 10)


def test_predict_quantiles_output_size_2_steps_ahead_specific_levels():
    """
    Test output sizes for predicting 2 steps ahead with quantiles by default and specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels, lags=lags)
    forecaster.fit(series)

    # Call the predict method
    quant_preds = forecaster.predict_quantiles(steps=2, levels="1")

    # Check the shape and values of the predictions
    assert quant_preds.shape == (2 * 1, 4)

