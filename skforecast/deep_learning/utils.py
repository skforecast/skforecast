################################################################################
#                      skforecast.ForecasterRnn.utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations
import pandas as pd
from ..utils import check_optional_dependency

try:
    import keras
    from keras.layers import (
        LSTM,
        Concatenate,
        Dense,
        Input,
        RepeatVector,
        SimpleRNN,
        TimeDistributed,
    )
    from keras.losses import MeanSquaredError
    from keras.models import Model
    from keras.optimizers import Adam
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def create_and_compile_model(
    series: pd.DataFrame,
    lags: int | list[int],
    steps: int | list[int],
    levels: str | list[str] | None = None,
    exog: pd.DataFrame | None = None,
    recurrent_layer: str = "LSTM",
    recurrent_units: int | list[int] = 100,
    dense_units: int | list[int] = 64,
    activation: str | dict[str, str | list[str]] = "relu",
    optimizer: object = Adam(learning_rate=0.01),
    loss: object = MeanSquaredError(),
    compile_kwargs: dict[str, object] = {},
) -> keras.models.Model:
    """
    Creates a neural network model for time series prediction with flexible 
    recurrent layers and optional exogenous variables.

    Parameters
    ----------
    series : pandas DataFrame
        Input time series.
    lags : int, list
        Number of lagged time steps to consider in the input, or a list of specific lag indices.
    steps : int, list
        Number of steps to predict into the future, or a list of specific step indices.
    levels : str, int, list, default None
        Number of output levels (features) to predict, or a list of specific level indices. If None, defaults to the number of input series.
    exog : pandas DataFrame, default None
        Exogenous variables to be included as input, should have the same number of rows as `series`.
    recurrent_layer : str, default 'LSTM'
        Type of recurrent layer to be used ('LSTM' or 'RNN').
    recurrent_units : int, list, default 100
        Number of units in the recurrent layer(s). Can be an integer or a list of integers for multiple layers.
    dense_units : int, list, default 64
        List of integers representing the number of units in each dense layer.
    activation : str, dict, default 'relu'
        Activation function for the recurrent and dense layers. Can be a single
        string for all layers or a dictionary specifying different activations
        for 'recurrent_units' and 'dense_units'.
    optimizer : object, default Adam(learning_rate=0.01)
        Optimization algorithm and learning rate.
    loss : object, default MeanSquaredError()
        Loss function for model training.
    compile_kwargs : dict, default {}
        Additional arguments for model compilation.

    Returns
    -------
    model : keras.models.Model
        Compiled neural network model.
    
    """

    if not isinstance(series, pd.DataFrame):
        raise TypeError("`series` must be a pandas DataFrame.")
    if exog is not None and not isinstance(exog, pd.DataFrame):
        raise TypeError("`exog` must be a pandas DataFrame or None.")

    n_series = series.shape[1]
    n_exog = exog.shape[1] if exog is not None else 0

    if isinstance(lags, list):
        lags = len(lags)
    if isinstance(steps, list):
        steps = len(steps)
    if isinstance(levels, list):
        levels = len(levels)
    elif levels is None:
        levels = n_series

    # Input
    series_input = Input(shape=(lags, n_series), name="series_input")
    inputs = [series_input]

    if exog is not None:
        exog_input = Input(shape=(steps, n_exog), name="exog_input")
        inputs.append(exog_input)

    x = series_input
    if not isinstance(recurrent_units, list):
        recurrent_units = [recurrent_units]

    for i, units in enumerate(recurrent_units):
        return_sequences = i < len(recurrent_units) - 1
        recurrent_activation = (
            activation if isinstance(activation, str)
            else activation.get("recurrent_units", "relu")
        )

        if recurrent_layer == "LSTM":
            x = LSTM(units, activation=recurrent_activation,
                     return_sequences=return_sequences)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(units, activation=recurrent_activation,
                          return_sequences=return_sequences)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")

    x = RepeatVector(steps)(x)  # (batch, steps, features)

    if exog is not None:
        x = Concatenate(axis=-1)([x, exog_input])  # (batch, steps, features + n_exog)

    for units in dense_units if isinstance(dense_units, list) else [dense_units]:
        dense_activation = (
            activation if isinstance(activation, str)
            else activation.get("dense_units", "relu")
        )
        x = TimeDistributed(Dense(units, activation=dense_activation))(x)

    output = TimeDistributed(Dense(levels, activation="linear"))(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)
    
    model.exog = True if exog is not None else False

    return model
