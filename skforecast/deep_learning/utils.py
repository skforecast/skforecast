################################################################################
#                      skforecast.ForecasterRnn.utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import warnings
from typing import Union, Any, Optional, Callable
import pandas as pd 
import pytest
import tensorflow as tf
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
    levels: Optional[Union[str, int, list]] = None,
    recurrent_layer: str = "LSTM",
    recurrent_units: Union[int, list] = 100,
    dense_units: Union[int, list] = 64,
    activation: Union[str, dict] = "relu",
    optimizer: object = Adam(learning_rate=0.01),
    loss: Union[str, Callable, object] = MeanSquaredError(),
    compile_kwargs: dict = {},
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
    activation : str, dict, default `'relu'`
        Activation function for the recurrent and dense layers. Can be a single 
        string for all layers or a dictionary specifying different activations 
        for 'recurrent_units' and 'dense_units'.
    optimizer : object, default `Adam(learning_rate=0.01)`
        Optimization algorithm and learning rate.
    loss : str, callable, or keras.losses.Loss, default `MeanSquaredError()`
        Loss function for model training. Can be:
        - A string identifier for predefined losses (e.g., 'mse', 'mae')
        - A callable custom loss function that accepts y_true and y_pred arguments
        - A Keras Loss instance
    compile_kwargs : dict, default `{}`
        Additional arguments for model compilation.

    Returns
    -------
    model : keras.models.Model
        Compiled neural network model.

    Examples
    --------
    >>> # Using a predefined loss
    >>> model1 = create_and_compile_model(
    ...     series=data,
    ...     lags=10,
    ...     steps=1,
    ...     loss='mse'
    ... )
    
    >>> # Using a custom loss function
    >>> def custom_loss_function(y_true, y_pred):
    ...     return tf.reduce_mean(tf.square(y_true - y_pred))
    >>> 
    >>> model2 = create_and_compile_model(
    ...     series=data,
    ...     lags=10,
    ...     steps=1,
    ...     loss=custom_loss_function
    ... )
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
            
    # Input validation
    err_msg = f"`series` must be a pandas DataFrame. Got {type(series)}."
    if not isinstance(series, pd.DataFrame):
        raise TypeError(err_msg)

    n_series = series.shape[1]

    # Validate loss argument
    if not (isinstance(loss, (str, object)) or callable(loss)):
        raise TypeError(
            "`loss` must be a string (predefined loss), a callable (custom loss function), "
            f"or a Keras Loss instance. Got {type(loss)}."
        )

    # If loss is a callable but not a Keras Loss instance, validate its signature
    if callable(loss) and not isinstance(loss, keras.losses.Loss):
        import inspect
        sig = inspect.signature(loss)
        if len(sig.parameters) != 2:
            raise ValueError(
                "Custom loss function must accept exactly two arguments (y_true, y_pred). "
                f"Got {len(sig.parameters)} arguments."
            )

    # Dense units validation
    if not isinstance(dense_units, (list, int, type(None))):
        raise TypeError(
            f"`dense_units` argument must be a list or int. Got {type(dense_units)}."
        )
    if isinstance(dense_units, int):
        dense_units = [dense_units]

    # Recurrent units validation
    if not isinstance(recurrent_units, (list, int)):
        raise TypeError(
            f"`recurrent_units` argument must be a list or int. Got {type(recurrent_units)}."
        )
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]

    # Lags, steps and levels validation
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
    elif isinstance(levels, str):
        levels = 1
    elif levels is None:
        levels = series.shape[1]
    elif isinstance(levels, int):
        pass
    else:
        raise TypeError(
            f"`levels` argument must be a string, list or int. Got {type(levels)}."
        )

    # Process activation
    if isinstance(activation, str):
        if dense_units is not None:
            activation = {
                "recurrent_units": [activation]*len(recurrent_units), 
                "dense_units": [activation]*len(dense_units)  
            }
        else:
            activation = {
                "recurrent_units": [activation]*len(recurrent_units)
            }
    elif isinstance(activation, dict):
        # Validate activation dictionary
        if "recurrent_units" not in activation:
            raise ValueError("The activation dictionary must have a 'recurrent_units' key.")
        if dense_units is not None and "dense_units" not in activation:
            raise ValueError("The activation dictionary must have a 'dense_units' key if dense_units is not None.")
        
        # Check if values are lists
        if not isinstance(activation["recurrent_units"], list):
            raise TypeError("The 'recurrent_units' value in the activation dictionary must be a list.")
        if dense_units is not None and not isinstance(activation["dense_units"], list):
            raise TypeError("The 'dense_units' value in the activation dictionary must be a list if dense_units is not None.")
        
        # Check list lengths
        if len(activation["recurrent_units"]) != len(recurrent_units):
            raise ValueError("The 'recurrent_units' list in the activation dictionary must have the same length as the recurrent_units parameter.")
        if dense_units is not None and len(activation["dense_units"]) != len(dense_units):
            raise ValueError("The 'dense_units' list in the activation dictionary must have the same length as the dense_units parameter.")
    else:
        raise TypeError(f"`activation` argument must be a string or dict. Got {type(activation)}.")

    # Build model architecture
    input_layer = Input(shape=(lags, n_series))
    x = input_layer

    # Recurrent layers
    for i, units in enumerate(recurrent_units[:-1]):
        if recurrent_layer == "LSTM":
            x = LSTM(units, activation=activation["recurrent_units"][i], return_sequences=True)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(units, activation=activation["recurrent_units"][i], return_sequences=True)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")
    
    # Last recurrent layer
    if recurrent_layer == "LSTM":
        x = LSTM(recurrent_units[-1], activation=activation["recurrent_units"][-1])(x)
    elif recurrent_layer == "RNN":
        x = SimpleRNN(recurrent_units[-1], activation=activation["recurrent_units"][-1])(x)
    else:
        raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")

    # Dense layers
    if dense_units is not None:
        for i, nn in enumerate(dense_units):
            x = Dense(nn, activation=activation["dense_units"][i])(x)

    # Output layer
    x = Dense(levels * steps, activation="linear")(x)
    output_layer = keras.layers.Reshape((steps, levels))(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    # Handle compilation arguments
    if optimizer is not None or loss is not None or compile_kwargs:
        if "optimizer" in compile_kwargs:
            compile_kwargs.pop("optimizer")
            warnings.warn("`optimizer` passed in `compile_kwargs`. Using the optimizer parameter instead.")
        
        if "loss" in compile_kwargs:
            compile_kwargs.pop("loss")
            warnings.warn("`loss` passed in `compile_kwargs`. Using the loss parameter instead.")

        model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)

    return model


# Unit tests
def test_predefined_loss_string():
    """Test model creation with predefined loss string"""
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    model = create_and_compile_model(
        series=data,
        lags=2,
        steps=1,
        loss='mse'
    )
    assert model.loss == 'mse'

def test_predefined_loss_object():
    """Test model creation with predefined loss object"""
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    model = create_and_compile_model(
        series=data,
        lags=2,
        steps=1,
        loss=MeanSquaredError()
    )
    assert isinstance(model.loss, MeanSquaredError)

def test_custom_loss_function():
    """Test model creation with custom loss function"""
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    model = create_and_compile_model(
        series=data,
        lags=2,
        steps=1,
        loss=custom_loss
    )
    assert model.loss == custom_loss

def test_invalid_custom_loss():
    """Test that invalid custom loss raises error"""
    def invalid_loss(y_true, y_pred, extra_param):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    with pytest.raises(ValueError) as exc_info:
        create_and_compile_model(
            series=data,
            lags=2,
            steps=1,
            loss=invalid_loss
        )
    assert "Custom loss function must accept exactly two arguments" in str(exc_info.value)

def test_compile_kwargs_with_custom_loss():
    """Test that custom loss takes priority over compile_kwargs"""
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    with pytest.warns(UserWarning):
        model = create_and_compile_model(
            series=data,
            lags=2,
            steps=1,
            loss=custom_loss,
            compile_kwargs={'loss': 'mse'}
        )
    assert model.loss == custom_loss

def test_loss_and_optimizer_precedence():
    """Test that main parameters take precedence over compile_kwargs"""
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    with pytest.warns(UserWarning, match="Using the optimizer parameter instead"):
        with pytest.warns(UserWarning, match="Using the loss parameter instead"):
            model = create_and_compile_model(
                series=data,
                lags=2,
                steps=1,
                optimizer="adam",
                loss="mse",
                compile_kwargs={'optimizer': 'sgd', 'loss': 'mae'}
            )
    assert model.loss == "mse"