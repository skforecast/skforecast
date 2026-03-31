################################################################################
#                      skforecast.deep_learning.utils                          #
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
    input_to_frame,
    check_optional_dependency
)

try:
    import keras
    from keras.layers import (
        Input,
        LSTM,
        GRU,
        SimpleRNN,
        Flatten,
        Concatenate,
        Dense,
        Reshape,
    )
    from keras.optimizers import Adam
    from keras.losses import MeanSquaredError
    from keras.models import Model
except ImportError as e:
    import sys
    if sys.version_info >= (3, 13):
        raise ImportError(
            "Python 3.13+ is not supported by TensorFlow, which is the default "
            "backend used by Keras. To use Keras with Python 3.13+, the KERAS_BACKEND "
            "environment variable needs to be set to 'torch', `os.environ['KERAS_BACKEND'] = 'torch'`."
            "Make sure you have PyTorch installed to use Keras with the torch backend. "
            "For installation instructions, visit https://pytorch.org/get-started/locally/"
        )
    else:
        package_name = str(e).split(" ")[-1].replace("'", "")
        check_optional_dependency(package_name=package_name)


def create_and_compile_model(
    series: pd.DataFrame,
    lags: int | list[int] | np.ndarray[int] | range[int],
    steps: int,
    levels: str | list[str] | tuple[str] | None = None,
    exog: pd.Series | pd.DataFrame | None = None,
    recurrent_layer: str = 'LSTM',
    recurrent_units: int | list[int] | tuple[int] = 100,
    recurrent_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    dense_units: int | list[int] | tuple[int] | None = 64,
    dense_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    output_dense_layer_kwargs: dict[str, Any] | None = None,
    compile_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None
) -> keras.models.Model:
    """
    Build and compile a RNN-based Keras model for time series prediction, 
    supporting exogenous variables.

    The model uses a unified Dense + Reshape output architecture for both
    cases (with and without exogenous variables). When exogenous variables
    are provided, they are flattened and concatenated with the RNN output
    before the dense layers, giving the model independent weights per
    prediction step.

    Parameters
    ----------
    series : pandas DataFrame
        Input time series with shape (n_obs, n_series). Each column is a 
        time series.
    lags : int, list, numpy ndarray, range
        Number of lagged time steps to consider in the input, index starts 
        at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present 
        in `lags`, all elements must be int.
    steps : int
        Number of steps to predict.
    levels : str, list, default None
        Output level(s) (features) to predict. If None, defaults to the 
        names of input series.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variables to be included as input, should have the same 
        number of rows as `series`.
    recurrent_layer : str, default 'LSTM'
        Type of recurrent layer to be used, 'LSTM' [1]_, 'GRU' [2]_, or 
        'RNN' [3]_.
    recurrent_units : int, list, default 100
        Number of units in the recurrent layer(s). Can be an integer for 
        single recurrent layer, or a list of integers for multiple 
        recurrent layers.
    recurrent_layers_kwargs : dict, list, default None
        Additional keyword arguments for the recurrent layers [1]_, [2]_, 
        [3]_. Can be a single dictionary for all layers or a list of 
        dictionaries specifying different parameters for each recurrent 
        layer. If None, defaults to ``{'activation': 'tanh'}``.
    dense_units : int, list, tuple, None, default 64
        Number of units in the dense layer(s) [4]_. Can be an integer for 
        single dense layer, or a list of integers for multiple dense layers.
    dense_layers_kwargs : dict, list, default None
        Additional keyword arguments for the dense layers [4]_. Can be a 
        single dictionary for all layers or a list of dictionaries 
        specifying different parameters for each dense layer. If None, 
        defaults to ``{'activation': 'relu'}``.
    output_dense_layer_kwargs : dict, default None
        Additional keyword arguments for the output dense layer. If None, 
        defaults to ``{'activation': 'linear'}``.
    compile_kwargs : dict, default None
        Additional keyword arguments for the model compilation, such as 
        optimizer and loss function [5]_. If None, defaults to 
        ``{'optimizer': Adam(), 'loss': MeanSquaredError()}``.
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

    .. [5] Model training APIs: compile method.
           https://keras.io/api/models/model_training_apis/
    
    """

    # Immutable defaults
    if recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = {'activation': 'tanh'}
    if dense_layers_kwargs is None:
        dense_layers_kwargs = {'activation': 'relu'}
    if output_dense_layer_kwargs is None:
        output_dense_layer_kwargs = {'activation': 'linear'}
    if compile_kwargs is None:
        compile_kwargs = {'optimizer': Adam(), 'loss': MeanSquaredError()}

    keras_backend = keras.backend.backend()

    print(f'keras version: {keras.__version__}')
    print(f'Using backend: {keras_backend}')
    if keras_backend == 'tensorflow':
        import tensorflow
        print(f'tensorflow version: {tensorflow.__version__}')
    elif keras_backend == 'torch':
        import torch
        print(f'torch version: {torch.__version__}')
    elif keras_backend == 'jax':
        import jax
        print(f'jax version: {jax.__version__}')
    else:
        print('Backend not recognized')
    print('')

    # Validate series
    if not isinstance(series, pd.DataFrame):
        raise TypeError(
            f'`series` must be a pandas DataFrame. Got {type(series)}.'
        )
    n_series = series.shape[1]

    # Validate exog
    if exog is not None:
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f'`exog` must be a pandas Series, DataFrame or None. '
                f'Got {type(exog)}.'
            )
        exog = input_to_frame(data=exog, input_name='exog')
        n_exog = exog.shape[1]

    # Validate lags
    lags, _, _ = initialize_lags('ForecasterRnn', lags)
    n_lags = len(lags)

    # Validate steps
    if not isinstance(steps, int):
        raise TypeError(
            f'`steps` argument must be an int greater than or equal to 1. '
            f'Got {type(steps)}.'
        )

    if steps < 1:
        raise ValueError(
            f'`steps` argument must be greater than or equal to 1. '
            f'Got {steps}.'
        )

    # Validate levels
    if levels is None:
        n_levels = n_series
    else:
        if isinstance(levels, str):
            levels = [levels]
        elif not isinstance(levels, (list, tuple)):
            raise TypeError(f'Invalid type for `levels`: {type(levels)}.')

        series_names_in = series.columns.tolist()
        missing_levels = [
            level for level in levels if level not in series_names_in
        ]
        if missing_levels:
            raise ValueError(
                f'Levels {missing_levels} not found in series columns: '
                f'{series_names_in}.'
            )

        n_levels = len(levels)

    # === INPUTS ===
    series_input = Input(shape=(n_lags, n_series), name='series_input')
    inputs = [series_input]

    if exog is not None:
        exog_input = Input(shape=(steps, n_exog), name='exog_input')
        inputs.append(exog_input)

    x = series_input

    # === RECURRENT LAYERS ===
    if not isinstance(recurrent_units, (list, tuple, int)):
        raise TypeError(
            f'At least one recurrent layer (LSTM, GRU, or SimpleRNN) is '
            f'required. `recurrent_units` argument must be an int or a list '
            f'of ints. Got {type(recurrent_units)}.'
        )
    if not isinstance(recurrent_units, (list, tuple)):
        recurrent_units = [recurrent_units]

    if isinstance(recurrent_layers_kwargs, dict):
        recurrent_layers_kwargs = [recurrent_layers_kwargs] * len(recurrent_units)
    elif isinstance(recurrent_layers_kwargs, (list, tuple)):
        if len(recurrent_layers_kwargs) != len(recurrent_units):
            raise ValueError(
                'If `recurrent_layers_kwargs` is a list, it must have the '
                'same length as `recurrent_units`. One dict of kwargs per '
                'recurrent layer.'
            )
    elif recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = [{}] * len(recurrent_units)
    else:
        raise TypeError(
            f'`recurrent_layers_kwargs` must be a dict, a list of dicts or '
            f'None. Got {type(recurrent_layers_kwargs)}.'
        )

    for i, units in enumerate(recurrent_units):

        return_sequences = i < len(recurrent_units) - 1

        layer_kwargs = deepcopy(recurrent_layers_kwargs[i])
        layer_kwargs.update({
            'units': units,
            'return_sequences': return_sequences,
        })
        if 'name' not in layer_kwargs:
            layer_kwargs['name'] = f'{recurrent_layer.lower()}_{i + 1}'

        if recurrent_layer == 'LSTM':
            x = LSTM(**layer_kwargs)(x)
        elif recurrent_layer == 'GRU':
            x = GRU(**layer_kwargs)(x)
        elif recurrent_layer == 'RNN':
            x = SimpleRNN(**layer_kwargs)(x)
        else:
            valid_layers = ['LSTM', 'GRU', 'RNN']
            raise ValueError(
                f'`recurrent_layer` must be one of {valid_layers}. '
                f"Got '{recurrent_layer}'."
            )

    # === EXOG CONCATENATION ===
    if exog is not None:
        exog_flat = Flatten(name='exog_flatten')(exog_input)
        x = Concatenate(axis=-1, name='concat_exog')([x, exog_flat])

    # === DENSE LAYERS ===
    if dense_units is not None:
        if not isinstance(dense_units, (list, tuple, int)):
            raise TypeError(
                f'`dense_units` argument must be an int, a list of ints or '
                f'None. Got {type(dense_units)}.'
            )
        if not isinstance(dense_units, (list, tuple)):
            dense_units = [dense_units]

        if isinstance(dense_layers_kwargs, dict):
            dense_layers_kwargs = [dense_layers_kwargs] * len(dense_units)
        elif isinstance(dense_layers_kwargs, (list, tuple)):
            if len(dense_layers_kwargs) != len(dense_units):
                raise ValueError(
                    'If `dense_layers_kwargs` is a list, it must have the '
                    'same length as `dense_units`. One dict of kwargs per '
                    'dense layer.'
                )
        elif dense_layers_kwargs is None:
            dense_layers_kwargs = [{}] * len(dense_units)
        else:
            raise TypeError(
                f'`dense_layers_kwargs` must be a dict, a list of dicts or '
                f'None. Got {type(dense_layers_kwargs)}.'
            )

        for i, units in enumerate(dense_units):

            layer_kwargs = deepcopy(dense_layers_kwargs[i])
            layer_kwargs.update({
                'units': units,
            })
            if 'name' not in layer_kwargs:
                layer_kwargs['name'] = f'dense_{i + 1}'

            x = Dense(**layer_kwargs)(x)

    # === OUTPUT LAYER (Dense + Reshape) ===
    output_layer_kwargs = deepcopy(output_dense_layer_kwargs)
    output_layer_kwargs.update({
        'units': n_levels * steps,
    })
    if 'name' not in output_layer_kwargs:
        output_layer_kwargs['name'] = 'output_dense_layer'

    x = Dense(**output_layer_kwargs)(x)
    output = Reshape((steps, n_levels), name='reshape')(x)

    model = Model(inputs=inputs, outputs=output, name=model_name)
    model.compile(**compile_kwargs)

    return model
