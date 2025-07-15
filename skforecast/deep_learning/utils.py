################################################################################
#                      skforecast.ForecasterRnn.utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import (
    initialize_lags,
    check_optional_dependency
)

try:
    import keras
    from keras.layers import (
        Input,
        LSTM,
        GRU,
        SimpleRNN,
        RepeatVector,
        Concatenate,
        Dense,
        TimeDistributed,
    )
    from keras.optimizers import Adam
    from keras.losses import MeanSquaredError
    from keras.models import Model
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def create_and_compile_model(
    series: pd.DataFrame,
    lags: int | list[int] | np.ndarray[int] | range[int],
    steps: int,
    levels: str | list[str] | tuple[str] | None = None,
    exog: pd.DataFrame | None = None,
    recurrent_layer: str = "LSTM",
    recurrent_units: int | list[int] | tuple[int] = 100,
    recurrent_layers_kwargs: dict[str, str | list[str]] | None = {"activation": "relu"},
    dense_units: int | list[int] | tuple[int] = 64,
    dense_layers_kwargs: dict[str, str | list[str]] | None = {"activation": "relu"},
    # activation: str | dict[str, str | list[str]] = "relu",
    output_layer_kwargs: dict[str, str | list[str]] | None = {"activation": "linear"},
    # output_activation: str = "linear",
    # optimizer: object = Adam(learning_rate=0.01),
    # loss: object = MeanSquaredError(),
    compile_kwargs: dict[str, object] = {"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
    model_name: str | None = None
) -> keras.models.Model:
    """
    Build and compile a RNN-based Keras model for time series prediction, 
    supporting exogenous variables.

    Parameters
    ----------
    series : pandas DataFrame
        Input time series with shape (n_obs, n_series). Each column is a time series.
    lags : int, list
        Number of lagged time steps to consider in the input, index starts at 1, 
        so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
    steps : int
        Number of steps to predict.
    levels : str, list, default None
       Output levels (features) to predict, or a list of specific level(s). If None, 
       defaults to the number of input series.
    exog : pandas DataFrame, default None
        Exogenous variables to be included as input, should have the same number of rows as `series`.
    recurrent_layer : str, default 'LSTM'
        Type of recurrent layer to be used ('LSTM' or 'RNN').
    recurrent_units : int, list, default 100
        Number of units in the recurrent layer(s). Can be an integer or a list of integers for multiple layers.
    dense_units : int, list, tuple, default 64
        List of integers representing the number of units in each dense layer.
    activation : str, dict, default 'relu'
        Activation function for the recurrent and dense layers. Can be a single
        string for all layers or a dictionary specifying different activations
        for 'recurrent_units' and 'dense_units'.
    optimizer : object, default Adam(learning_rate=0.01)
        Optimization algorithm and learning rate.
    loss : object, default MeanSquaredError()
        Loss function for model training.
    output_activation : str, default 'linear'
        Activation function for the output layer.
    compile_kwargs : dict, default {}
        Additional arguments for model compilation.

    Returns
    -------
    model : keras.models.Model
        Compiled neural network model.
    
    """

    if not isinstance(series, pd.DataFrame):
        raise TypeError(
            f"`series` must be a pandas DataFrame. Got {type(series)}."
        )
    if exog is not None and not isinstance(exog, pd.DataFrame):
        raise TypeError(
            f"`exog` must be a pandas DataFrame or None. Got {type(exog)}."
        )

    n_series = series.shape[1]
    n_exog = exog.shape[1] if exog is not None else 0

    lags, _, _ = initialize_lags('ForecasterRNN', lags)
    n_lags = len(lags)

    if not isinstance(steps, int):
        raise TypeError(
            f"`steps` argument must be an int greater than or equal to 1. "
            f"Got {type(steps)}."
        )

    if steps < 1:
        raise ValueError(
            f"`steps` argument must be greater than or equal to 1. Got {steps}."
        )

    if levels is None:
        n_levels = n_series
    elif isinstance(levels, (list, tuple)):
        n_levels = len(levels)
    elif isinstance(levels, str):
        n_levels = 1
    else:
        raise ValueError(f"Invalid type for `levels`: {type(levels)}.")
    
    # TODO: levels must be one of the series columns names? Yes
    series_names_in = series.columns.tolist()
    missing_levels = [level for level in levels if level not in series_names_in]
    if missing_levels:
        raise ValueError(
            f"Levels {missing_levels} not found in series columns: {series_names_in}."
        )

    series_input = Input(shape=(n_lags, n_series), name="series_input")
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

        layer_kwargs = {
            "units": units,
            "activation": recurrent_activation,
            "return_sequences": return_sequences,
            "name": f"{recurrent_layer.lower()}_{i + 1}"
        }

        if dropout_rate is not None:
            layer_kwargs["dropout"] = dropout_rate
            layer_kwargs["recurrent_dropout"] = dropout_rate
        
        if recurrent_layer == "LSTM":
            x = LSTM(**layer_kwargs)(x)
        elif recurrent_layer == "GRU":
            x = GRU(**layer_kwargs)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(**layer_kwargs)(x)
        else:
            valid_layers = ["LSTM", "GRU", "RNN"]
            raise ValueError(
                f"`recurrent_layer` must be one of {valid_layers}. Got '{recurrent_layer}'."
            )

    # NOTE: Shape (batch, steps, features)
    x = RepeatVector(steps, name="repeat_vector")(x)

    if exog is not None:
        # NOTE: Shape (batch, steps, features + n_exog)
        x = Concatenate(axis=-1, name="concat_exog")([x, exog_input])  

    dense_units = dense_units if isinstance(dense_units, (list, tuple)) else [dense_units]
    for i, units in enumerate(dense_units):
        dense_activation = (
            activation if isinstance(activation, str)
            else activation.get("dense_units", "relu")
        )
        x = TimeDistributed(
            Dense(units, activation=dense_activation), name=f"dense_td_{i+1}"
        )(x)

    output = TimeDistributed(
        Dense(n_levels, activation=output_activation), name="output_layer"
    )(x)

    # NOTE: decide name
    model = Model(inputs=inputs, outputs=output, name=model_name)
    model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)

    return model
