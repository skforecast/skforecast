################################################################################
#                      skforecast.ForecasterRnn.utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import warnings
from typing import Union, Any, Optional
import pandas as pd 
from ..utils import check_optional_dependency

try:
    import keras
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, SimpleRNN
    from keras.optimizers import Adam
    from keras.losses import MeanSquaredError
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def create_and_compile_model(
    series: pd.DataFrame,
    lags: Union[int, list],
    steps: Union[int, list],
    levels: Optional[Union[str, int, list]]=None,
    recurrent_layer: str="LSTM",
    recurrent_units: Union[int, list]=100,
    dense_units: Union[int, list]=64,
    activation: str="relu",
    optimizer: object=Adam(learning_rate=0.01),
    loss: object=MeanSquaredError(),
    compile_kwargs: dict={},
) -> keras.models.Model:
    """
    Creates a neural network model for time series prediction with flexible recurrent layers.

    Parameters
    ----------
    series : pandas DataFrame
        Input time series.
    lags : int, list
        Number of lagged time steps to consider in the input, or a list of 
        specific lag indices.
    steps : int, list
        Number of steps to predict into the future, or a list of specific step 
        indices.
    levels : str, int, list, default `None`
        Number of output levels (features) to predict, or a list of specific 
        level indices. If None, defaults to the number of input series.
    recurrent_layer : str, default `'LSTM'`
        Type of recurrent layer to be used ('LSTM' or 'RNN').
    recurrent_units : int, list, default `100`
        Number of units in the recurrent layer(s). Can be an integer or a 
        list of integers for multiple layers.
    dense_units : int, list, default `64`
        List of integers representing the number of units in each dense layer.
    activation : str, default `'relu'`
        Activation function for the recurrent and dense layers.
    optimizer : object, default `Adam(learning_rate=0.01)`
        Optimization algorithm and learning rate.
    loss : object, default `MeanSquaredError()`
        Loss function for model training.
    compile_kwargs : dict, default `{}` 
        Additional arguments for model compilation.

    Returns
    -------
    model : keras.models.Model
        Compiled neural network model.
    
    """
    
    if keras.__version__ > "3":
        print(f"keras version: {keras.__version__}")
        print(f"Using backend: {keras.backend.backend()}")
        if keras.backend.backend() == "tensorflow":
            import tensorflow
            print(f"tensorflow version: {tensorflow.__version__}")
        elif keras.backend.backend() == "torch":
            import torch
            print(f"torch version: {torch.__version__}")
        elif keras.backend.backend() == "jax":
            import jax
            print(f"jax version: {jax.__version__}")
        else:
            print("Backend not recognized")
            
    err_msg = f"`series` must be a pandas DataFrame. Got {type(series)}."

    if not isinstance(series, pd.DataFrame):
        raise TypeError(err_msg)

    n_series = series.shape[1]

    # Dense units must be a list, None or int
    if not isinstance(dense_units, (list, int, type(None))):
        raise TypeError(
            f"`dense_units` argument must be a list or int. Got {type(dense_units)}."
        )
    if isinstance(dense_units, int):
        dense_units = [dense_units]

    # Recurrent units must be a list or int
    if not isinstance(recurrent_units, (list, int)):
        raise TypeError(
            f"`recurrent_units` argument must be a list or int. Got {type(recurrent_units)}."
        )
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]

    # Lags, steps and levels must be int or list
    if not isinstance(lags, (int, list)):
        raise TypeError(f"`lags` argument must be a list or int. Got {type(lags)}.")
    if not isinstance(steps, (int, list)):
        raise TypeError(f"`steps` argument must be a list or int. Got {type(steps)}.")
    if not isinstance(levels, (str, int, list, type(None))):
        raise TypeError(
            f"`levels` argument must be a string, list or int. Got {type(levels)}."
        )

    if isinstance(lags, list):
        lags = len(lags)
    if isinstance(steps, list):
        steps = len(steps)
    if isinstance(levels, list):
        levels = len(levels)
    elif isinstance(levels, (str)):
        levels = 1
    elif isinstance(levels, type(None)):
        levels = series.shape[1]
    elif isinstance(levels, int):
        pass
    else:
        raise TypeError(
            f"`levels` argument must be a string, list or int. Got {type(levels)}."
        )

    input_layer = Input(shape=(lags, n_series))
    x = input_layer

    # Dynamically create multiple recurrent layers if recurrent_units is a list
    if isinstance(recurrent_units, list):
        for units in recurrent_units[:-1]:  # All layers except the last one
            if recurrent_layer == "LSTM":
                x = LSTM(units, activation=activation, return_sequences=True)(x)
            elif recurrent_layer == "RNN":
                x = SimpleRNN(units, activation=activation, return_sequences=True)(x)
            else:
                raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")
        # Last layer without return_sequences
        if recurrent_layer == "LSTM":
            x = LSTM(recurrent_units[-1], activation=activation)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(recurrent_units[-1], activation=activation)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")
    else:
        # Single recurrent layer
        if recurrent_layer == "LSTM":
            x = LSTM(recurrent_units, activation=activation)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(recurrent_units, activation=activation)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")

    # Dense layers
    if dense_units is not None:
        for nn in dense_units:
            x = Dense(nn, activation=activation)(x)

    # Output layer
    x = Dense(levels * steps, activation="linear")(x)
    # model = Model(inputs=input_layer, outputs=x)
    output_layer = keras.layers.Reshape((steps, levels))(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model if optimizer, loss or compile_kwargs are passed
    if optimizer is not None or loss is not None or compile_kwargs:
        # give more priority to the parameters passed in the function check if the 
        # parameters passes in compile_kwargs include optimizer and loss if so, 
        # delete them from compile_kwargs and raise a warning
        if "optimizer" in compile_kwargs.keys():
            compile_kwargs.pop("optimizer")
            warnings.warn("`optimizer` passed in `compile_kwargs`. Ignoring it.")
        if "loss" in compile_kwargs.keys():
            compile_kwargs.pop("loss")
            warnings.warn("`loss` passed in `compile_kwargs`. Ignoring it.")

        model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)

    return model