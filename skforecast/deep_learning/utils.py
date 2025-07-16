################################################################################
#                      skforecast.ForecasterRnn.utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
from copy import deepcopy
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
    recurrent_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    dense_units: int | list[int] | tuple[int] = 64,
    dense_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = {"activation": "relu"},
    output_dense_layer_kwargs: dict[str, Any] | None = {"activation": "linear"},
    compile_kwargs: dict[str, Any] = {"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
    model_name: str | None = None
) -> keras.models.Model:
    """
    Build and compile a RNN-based Keras model for time series prediction, 
    supporting exogenous variables.

    Parameters
    ----------
    series : pandas DataFrame
        Input time series with shape (n_obs, n_series). Each column is a time series.
    lags : int, list, numpy ndarray, range
        Number of lagged time steps to consider in the input, index starts at 1, 
        so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
    steps : int
        Number of steps to predict.
    levels : str, list, default None
       Output level(s) (features) to predict. If None, defaults to the names of 
       input series.
    exog : pandas DataFrame, default None
        Exogenous variables to be included as input, should have the same number 
        of rows as `series`.
    recurrent_layer : str, default 'LSTM'
        Type of recurrent layer to be used, 'LSTM' [1]_, 'GRU' [2]_, or 'RNN' [3]_.
    recurrent_units : int, list, default 100
        Number of units in the recurrent layer(s). Can be an integer for single 
        recurrent layer, or a list of integers for multiple recurrent layers.
    recurrent_layers_kwargs : dict, list, default None
        Additional keyword arguments for the recurrent layers [1]_, [2]_, [3]_. 
        Can be a single dictionary for all layers or a list of dictionaries 
        specifying different parameters for each recurrent layer.
    dense_units : int, list, tuple, default 64
        Number of units in the dense layer(s) [4]_. Can be an integer for single
        dense layer, or a list of integers for multiple dense layers.
    dense_layers_kwargs : dict, list, default {'activation': 'relu'}
        Additional keyword arguments for the dense layers [4]_. Can be a single
        dictionary for all layers or a list of dictionaries specifying different
        parameters for each dense layer.
    output_dense_layer_kwargs : dict, default {'activation': 'linear'}
        Additional keyword arguments for the output dense layer.
    compile_kwargs : dict, default {'optimizer': Adam(learning_rate=0.01), 'loss': MeanSquaredError()}
        Additional keyword arguments for the model compilation, such as optimizer 
        and loss function.
    model_name : str, default None
        Name of the model.

    Returns
    -------
    model : keras.models.Model
        Compiled Keras model ready for training.

    References
    ----------
    .. [1] LSTM layer Keras documentation.
           https://keras.io/api/layers/recurrent_layers/lstm/

    .. [2] GRU layer Keras documentation.
           https://keras.io/api/layers/recurrent_layers/gru/

    .. [3] SimpleRNN layer Keras documentation.
           https://keras.io/api/layers/recurrent_layers/simple_rnn/

    .. [4] Dense layer Keras documentation.
           https://keras.io/api/layers/core_layers/dense/

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
    else:
        if isinstance(levels, str):
            levels = [levels]
        elif not isinstance(levels, (list, tuple)):
            raise TypeError(f"Invalid type for `levels`: {type(levels)}.")
        
        series_names_in = series.columns.tolist()
        missing_levels = [level for level in levels if level not in series_names_in]
        if missing_levels:
            raise ValueError(
                f"Levels {missing_levels} not found in series columns: {series_names_in}."
            )
        
        n_levels = len(levels)

    series_input = Input(shape=(n_lags, n_series), name="series_input")
    inputs = [series_input]
    if exog is not None:
        exog_input = Input(shape=(steps, n_exog), name="exog_input")
        inputs.append(exog_input)

    x = series_input
    if not isinstance(recurrent_units, (list, tuple)):
        recurrent_units = [recurrent_units]

    if isinstance(recurrent_layers_kwargs, dict):
        recurrent_layers_kwargs = [recurrent_layers_kwargs] * len(recurrent_units)
    elif isinstance(recurrent_layers_kwargs, (list, tuple)):
        if len(recurrent_layers_kwargs) != len(recurrent_units):
            raise ValueError(
                "If `recurrent_layers_kwargs` is a list, it must have the same "
                "length as `recurrent_units`. One dict of kwargs per recurrent layer."
            )
    elif recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = [{}] * len(recurrent_units)
    else:
        raise TypeError(
            f"`recurrent_layers_kwargs` must be a dict, a list of dicts or None. "
            f"Got {type(recurrent_layers_kwargs)}."
        )

    for i, units in enumerate(recurrent_units):

        return_sequences = i < len(recurrent_units) - 1

        layer_kwargs = deepcopy(recurrent_layers_kwargs[i])
        layer_kwargs.update({
            "units": units,
            "return_sequences": return_sequences,
        })
        if "name" not in layer_kwargs:
            layer_kwargs["name"] = f"{recurrent_layer.lower()}_{i + 1}"
        
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

    if not isinstance(dense_units, (list, tuple)):
        dense_units = [dense_units]

    if isinstance(dense_layers_kwargs, dict):
        dense_layers_kwargs = [dense_layers_kwargs] * len(dense_units)
    elif isinstance(dense_layers_kwargs, (list, tuple)):
        if len(dense_layers_kwargs) != len(dense_units):
            raise ValueError(
                "If `dense_layers_kwargs` is a list, it must have the same "
                "length as `dense_units`. One dict of kwargs per dense layer."
            )
    elif dense_layers_kwargs is None:
        dense_layers_kwargs = [{}] * len(dense_units)
    else:
        raise TypeError(
            f"`dense_layers_kwargs` must be a dict, a list of dicts or None. "
            f"Got {type(dense_layers_kwargs)}."
        )
    
    for i, units in enumerate(dense_units):
        
        layer_kwargs = deepcopy(dense_layers_kwargs[i])
        layer_kwargs.update({
            "units": units,
        })
        if "name" in layer_kwargs:
            layer_name = layer_kwargs.pop("name")
        else:
            layer_name = f"dense_td_{i + 1}"

        x = TimeDistributed(Dense(**layer_kwargs), name=layer_name)(x)

    if output_dense_layer_kwargs is None:
        output_layer_kwargs = {}
    else:
        output_layer_kwargs = deepcopy(output_dense_layer_kwargs)
    
    output_layer_kwargs.update({
        "units": n_levels,
    })
    if "name" in output_layer_kwargs:
        layer_name = output_layer_kwargs.pop("name")
    else:
        layer_name = "output_dense_td_layer"

    output = TimeDistributed(Dense(**output_layer_kwargs), name=layer_name)(x)

    model = Model(inputs=inputs, outputs=output, name=model_name)
    model.compile(**compile_kwargs)

    return model
