import os

import numpy as np
import pandas as pd
import pytest

os.environ["KERAS_BACKEND"] = "torch"
import keras

from skforecast.deep_learning.utils import create_and_compile_model

# test with several parameters for dense_units and recurrent_units


@pytest.mark.parametrize(
    "dense_units, recurrent_layer, recurrent_units",
    [
        (64, "LSTM", 100),
        ([64], "LSTM", 100),
        ([64, 32], "LSTM", 100),
        (64, "RNN", [100]),
        (64, "LSTM", [100, 50]),
        ([64], "LSTM", [100]),
        ([64], "LSTM", [100, 50]),
        ([64, 32], "LSTM", [100]),
        ([64, 32], "LSTM", [100, 50]),
        (None, "RNN", [100, 50]),
    ],
)
def test_units(dense_units, recurrent_layer, recurrent_units):
    """
    Test case for testing the create_and_compile_model function with different dense_units and recurrent_units
    """
    print(f"dense_units: {dense_units}")
    print(f"recurrent_units: {recurrent_units}")
    # Generate dummy data for testing
    series = pd.DataFrame(np.random.randn(100, 3))
    lags = 10
    steps = 5
    levels = 1
    activation = "relu"
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss = keras.losses.MeanSquaredError()

    # Call the function to create and compile the model
    model = create_and_compile_model(
        series=series,
        lags=lags,
        steps=steps,
        levels=levels,
        recurrent_layer=recurrent_layer,
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
    )
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]

    # Assert that the model is an instance of tf.keras.models.Model
    assert isinstance(model, keras.models.Model)

    # Assert that the model has the correct number of layers
    if dense_units is None:
        assert len(model.layers) == 3 + len(recurrent_units)
    else:
        assert len(model.layers) == 3 + len(dense_units) + len(recurrent_units)

    # Assert that the model is compiled with the correct optimizer and loss
    assert model.optimizer == optimizer
    assert model.loss == loss


# Mock data for testing
series_data = pd.DataFrame(np.random.randn(100, 2), columns=["feature1", "feature2"])
lags_data = 5
steps_data = 3
levels_data = "feature1"


def test_correct_input_type():
    # Test if the function works with the correct input type
    model = create_and_compile_model(series_data, lags_data, steps_data, levels_data)
    assert isinstance(model, keras.models.Model)


def test_incorrect_series_type():
    # Test if the function raises an error for incorrect series type
    with pytest.raises(TypeError, match="`series` must be a pandas DataFrame. Got .*"):
        create_and_compile_model(
            np.random.randn(100, 2), lags_data, steps_data, levels_data
        )


def test_incorrect_dense_units_type():
    # Test if the function raises an error for incorrect dense_units type
    with pytest.raises(
        TypeError, match="`dense_units` argument must be a list or int. Got .*"
    ):
        create_and_compile_model(
            series_data, lags_data, steps_data, levels_data, dense_units="invalid"
        )


def test_incorrect_recurrent_units_type():
    # Test if the function raises an error for incorrect recurrent_units type
    with pytest.raises(
        TypeError, match="`recurrent_units` argument must be a list or int. Got .*"
    ):
        create_and_compile_model(
            series_data, lags_data, steps_data, levels_data, recurrent_units="invalid"
        )


def test_incorrect_lags_type():
    # Test if the function raises an error for incorrect lags type
    with pytest.raises(
        TypeError, match="`lags` argument must be a list or int. Got .*"
    ):
        create_and_compile_model(series_data, "invalid", steps_data, levels_data)


def test_incorrect_steps_type():
    # Test if the function raises an error for incorrect steps type
    with pytest.raises(
        TypeError, match="`steps` argument must be a list or int. Got .*"
    ):
        create_and_compile_model(series_data, lags_data, "invalid", levels_data)


def test_incorrect_levels_type():
    # Test if the function raises an error for incorrect levels type
    with pytest.raises(
        TypeError, match="`levels` argument must be a string, list or int. Got .*"
    ):
        create_and_compile_model(
            series_data, lags_data, steps_data, np.array([1, 2, 3])
        )
