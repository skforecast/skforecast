
# Unit test _create_and_compile_model_no_exog
# ==============================================================================
import sys
import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 13, 0),
    reason="TensorFlow does not support Python 3.13+",
)

import re
import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning.utils import _create_and_compile_model_no_exog


def test__create_and_compile_model_no_exog_raise_TypeError_if_series_is_not_dataframe():
    """
    Raise TypeError if series is not a pandas DataFrame.
    """
    series = np.arange(10)
    err_msg = re.escape(
        f"`series` must be a pandas DataFrame. Got {type(series)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1)


def test__create_and_compile_model_no_exog_raise_TypeError_if_steps_is_not_int():
    """
    Raise TypeError if steps is not int.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    err_msg = re.escape(
        "`steps` argument must be an int greater than or equal to 1. "
        f"Got {type('1')}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps="1")


def test__create_and_compile_model_no_exog_raise_ValueError_if_steps_less_than_1():
    """
    Raise ValueError if steps < 1.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    err_msg = re.escape(
        "`steps` argument must be greater than or equal to 1. Got 0."
    )
    with pytest.raises(ValueError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=0)


def test__create_and_compile_model_no_exog_raise_TypeError_if_levels_is_wrong_type():
    """
    Raise TypeError if levels is not str, list, tuple or None.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    err_msg = re.escape(
        f"Invalid type for `levels`: {type(5.0)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, levels=5.0)


def test__create_and_compile_model_no_exog_raise_ValueError_if_levels_not_in_series():
    """
    Raise ValueError if levels contains names not present in series columns.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    err_msg = re.escape(
        f"Levels {['b']} not found in series columns: {series.columns.tolist()}."
    )
    with pytest.raises(ValueError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, levels="b")


def test__create_and_compile_model_no_exog_raise_TypeError_if_recurrent_units_wrong_type():
    """
    Raise TypeError if recurrent_units is not int or list of ints.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    recurrent_units = "100"
    err_msg = re.escape(
        f"At least one recurrent layer (LSTM, GRU, or SimpleRNN) is required."
        f"`recurrent_units` argument must be an int or a list of ints. "
        f"Got {type(recurrent_units)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(
            series=series, lags=2, steps=1, recurrent_units=recurrent_units
        )


def test__create_and_compile_model_no_exog_raise_ValueError_if_recurrent_layers_kwargs_list_length_mismatch():
    """
    Raise ValueError if recurrent_layers_kwargs is a list and length does not match recurrent_units.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    rkwargs = [{"activation": "tanh"}, {"activation": "relu"}]
    err_msg = re.escape(
        "If `recurrent_layers_kwargs` is a list, it must have the same "
        "length as `recurrent_units`. One dict of kwargs per recurrent layer."
    )
    with pytest.raises(ValueError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, recurrent_units=[10], recurrent_layers_kwargs=rkwargs)


def test__create_and_compile_model_no_exog_raise_TypeError_if_recurrent_layers_kwargs_is_wrong_type():
    """
    Raise TypeError if recurrent_layers_kwargs is not dict, list of dicts, or None.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    rkwargs = 5
    err_msg = re.escape(
        f"`recurrent_layers_kwargs` must be a dict, a list of dicts or None. Got {type(rkwargs)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(
            series=series, lags=2, steps=1, recurrent_layers_kwargs=rkwargs
        )


def test__create_and_compile_model_exog_raise_TypeError_if_dense_units_is_wrong_type():
    """
    Raise TypeError if dense_units is not int or list of ints.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    dense_units = "100"
    err_msg = re.escape(
        f"`dense_units` argument must be an int, a list of ints or None. "
        f"Got {type(dense_units)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(
            series=series, lags=2, steps=1, dense_units=dense_units
        )


def test__create_and_compile_model_no_exog_raise_ValueError_if_dense_layers_kwargs_list_length_mismatch():
    """
    Raise ValueError if dense_layers_kwargs is a list and length does not match dense_units.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    dkwargs = [{"activation": "relu"}, {"activation": "sigmoid"}]
    err_msg = re.escape(
        "If `dense_layers_kwargs` is a list, it must have the same "
        "length as `dense_units`. One dict of kwargs per dense layer."
    )
    with pytest.raises(ValueError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, dense_units=[32], dense_layers_kwargs=dkwargs)


def test__create_and_compile_model_no_exog_raise_TypeError_if_dense_layers_kwargs_is_wrong_type():
    """
    Raise TypeError if dense_layers_kwargs is not dict, list of dicts, or None.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    dkwargs = 5
    err_msg = re.escape(
        f"`dense_layers_kwargs` must be a dict, a list of dicts or None. Got {type(dkwargs)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, dense_layers_kwargs=dkwargs)


def test__create_and_compile_model_no_exog_raise_ValueError_if_invalid_recurrent_layer():
    """
    Raise ValueError if recurrent_layer is not valid.
    """
    series = pd.DataFrame({"a": np.arange(10)})
    valid_layers = ["LSTM", "GRU", "RNN"]
    err_msg = re.escape(
        f"`recurrent_layer` must be one of {valid_layers}. Got 'FOO'."
    )
    with pytest.raises(ValueError, match=err_msg):
        _create_and_compile_model_no_exog(series=series, lags=2, steps=1, recurrent_layer="FOO")


def test__create_and_compile_model_no_exog_model_output_shape_single_series():
    """
    Output shape for a single series, default params.
    """
    series = pd.DataFrame({"a": np.arange(10, dtype=float)})
    model = _create_and_compile_model_no_exog(series=series, lags=2, steps=3)
    
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 3, 1)
    assert len(model.inputs) == 1
    assert isinstance(model.name, str)


def test__create_and_compile_model_no_exog_model_output_shape_multi_series():
    """
    Output shape for multi-series, default params.
    """
    series = pd.DataFrame({
        "a": np.arange(10, dtype=float),
        "b": np.arange(10, 20, dtype=float),
    })
    model = _create_and_compile_model_no_exog(
        series=series, lags=[1, 2], steps=2, recurrent_layer='RNN',
        recurrent_layers_kwargs=None, dense_layers_kwargs=None, output_dense_layer_kwargs=None
    )
    
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 2, 2)
    assert len(model.inputs) == 1


def test__create_and_compile_model_no_exog_model_with_levels_as_str():
    """
    Output shape when levels is str (only one target series).
    """
    series = pd.DataFrame({
        "a": np.arange(10, dtype=float),
        "b": np.arange(10, 20, dtype=float),
    })
    model = _create_and_compile_model_no_exog(series=series, lags=2, steps=2, levels="a")
    assert model.output_shape == (None, 2, 1)


def test__create_and_compile_model_no_exog_model_with_levels_as_list():
    """
    Output shape when levels is a list (only those columns as targets).
    """
    series = pd.DataFrame({
        "a": np.arange(10, dtype=float),
        "b": np.arange(10, 20, dtype=float),
        "c": np.arange(10, 30, 2, dtype=float),
    })
    model = _create_and_compile_model_no_exog(series=series, lags=2, steps=2, levels=["a", "c"])
    assert model.output_shape == (None, 2, 2)


def test__create_and_compile_model_no_exog_model_with_custom_compile_kwargs():
    """
    Custom compile_kwargs are respected.
    """
    series = pd.DataFrame({"a": np.arange(10, dtype=float)})
    compile_kwargs = {
        "optimizer": Adam(),
        "loss": MeanSquaredError()
    }
    model = _create_and_compile_model_no_exog(
        series=series, lags=2, steps=1, compile_kwargs=compile_kwargs
    )
    
    assert model.loss.name == "mean_squared_error"
    assert model.optimizer.name == "adam"


@pytest.mark.parametrize("dense_units", 
                         [None, [16, 8]], 
                         ids=lambda units: f'dense_units: {units}')
def test__create_and_compile_model_no_exog_model_with_multiple_dense_and_recurrent_layers(dense_units):
    """
    Multiple recurrent and dense layers with lists/kwargs.
    """
    series = pd.DataFrame({"a": np.arange(10, dtype=float)})
    model = _create_and_compile_model_no_exog(
        series=series,
        lags=2,
        steps=1,
        recurrent_layer="GRU",
        recurrent_units=[16, 8],
        recurrent_layers_kwargs=[{"activation": "relu"}, {"activation": "tanh"}],
        dense_units=dense_units,
        dense_layers_kwargs=[{"activation": "relu"}, {"activation": "linear"}]
    )
    
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 1, 1)


def test__create_and_compile_model_no_exog_model_name_custom():
    """
    Check custom model name.
    """
    series = pd.DataFrame({"a": np.arange(10, dtype=float)})
    model = create_and_compile_model(
        series=series, lags=2, steps=1, model_name="my_model"
    )
    
    assert model.name == "my_model"
