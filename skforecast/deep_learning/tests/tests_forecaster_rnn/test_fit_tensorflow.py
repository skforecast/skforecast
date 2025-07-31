# Unit test fit method with TensorFlow backend
# ==============================================================================
import sys
import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 13, 0),
    reason="TensorFlow does not support Python 3.13+",
)

import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

series = pd.DataFrame(
    {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
)
exog = pd.DataFrame(
    {"exog1": pd.Series(np.arange(5)), "exog2": pd.Series(np.arange(5))}
)
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

model_exog = create_and_compile_model(
    series=series, 
    exog=exog,
    levels=levels,    
    lags=lags,           
    steps=steps,              
    recurrent_layer="LSTM",
    recurrent_units=128,
    dense_units=64,
)


# Test case for fitting the forecaster without validation data
def test_fit_without_validation_data():
    """
    Test case for fitting the forecaster without validation data
    """
    # Call the function to create and compile the model

    forecaster = ForecasterRnn(model, levels, lags=lags)

    # Assert that the forecaster is fitted
    assert forecaster.is_fitted is False

    # Fit the forecaster
    forecaster.fit(series)

    # Assert that the forecaster is fitted
    assert forecaster.is_fitted is True
    print("GITHUB_ACTIONS =", os.environ.get("GITHUB_ACTIONS"))
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert forecaster.keras_backend_ == "tensorflow"

    assert forecaster.series_names_in_ == ["1", "2"]
    assert forecaster.X_train_series_names_in_ == ["1", "2"]

    # Assert that the training range is set correctly
    assert all(forecaster.training_range_ == (0, 4))

    # Assert that the last window is set correctly
    last_window = pd.DataFrame({"1": [2, 3, 4], "2": [2, 3, 4]})

    np.testing.assert_array_almost_equal(forecaster.last_window_, last_window)


# Test case for fitting the forecaster with validation data
def test_fit_with_validation_data():
    """
    Test case for fitting the forecaster with validation data
    """

    # Create a validation series
    series_val = pd.DataFrame(
        {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
    )

    # Create an instance of ForecasterRnn
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

    # Assert that the forecaster is not fitted
    assert forecaster.is_fitted is False

    # Fit the forecaster
    forecaster.fit(series)

    # Assert that the forecaster is fitted
    assert forecaster.is_fitted is True
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert forecaster.keras_backend_ == "tensorflow"

    # # Assert that the history is not None
    assert forecaster.history_ is not None


def test_fit_with_exog_and_validation_data():
    """
    Test case for fitting the forecaster with validation data
    """

    series_val = pd.DataFrame(
        {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
    )
    exog_val = pd.DataFrame(
        {"exog1": pd.Series(np.arange(5)), "exog2": pd.Series(np.arange(5))}
    )
    forecaster = ForecasterRnn(
        regressor=model_exog,
        levels=levels,
        fit_kwargs={
            "epochs": 10,  # Number of epochs to train the model.
            "batch_size": 32,  # Batch size to train the model.
            "series_val": series_val,  # Validation data for model training.
            "exog_val": exog_val,  # Validation exogenous data for model training.
        },
        lags=lags
    )
    assert forecaster.is_fitted is False
    forecaster.fit(series, exog=exog)
    assert forecaster.is_fitted is True
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert forecaster.keras_backend_ == "tensorflow"
    assert forecaster.history_ is not None
    assert forecaster.exog_names_in_ == ["exog1", "exog2"]
