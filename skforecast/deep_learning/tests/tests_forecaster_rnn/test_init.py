
# Unit test __init__ ForecasterRnn using PyTorch backend
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model


def test_ForecasterRnn_init_basic_and_attr_types():
    """
    Basic initialization of ForecasterRnn with attributes types.
    """
    series = pd.DataFrame({'A': np.arange(10, dtype=float)})
    model = create_and_compile_model(
        series=series, lags=3, steps=2, model_name="modelo"
    )
    forecaster = ForecasterRnn(
        estimator=model,
        levels='A',
        lags=3,
        transformer_series=None,
        transformer_exog=None,
        fit_kwargs={},
        forecaster_id="f-id"
    )

    assert isinstance(forecaster.estimator, keras.Model)
    assert forecaster.levels == ['A']
    assert forecaster.transformer_series is None
    assert forecaster.transformer_exog is None
    assert forecaster.is_fitted is False
    assert isinstance(forecaster.creation_date, str)
    assert forecaster.fit_date is None
    assert forecaster.keras_backend_ is None
    assert isinstance(forecaster.skforecast_version, str)
    assert isinstance(forecaster.python_version, str)
    assert forecaster.forecaster_id == "f-id"
    assert forecaster._probabilistic_mode == "no_binned"
    assert forecaster.dropna_from_series is False
    assert forecaster.encoding is None
    assert forecaster.differentiation is None
    assert forecaster.layers_names == [
        'series_input', 'lstm_1', 'dense_1', 'output_dense_td_layer', 'reshape'
    ]
    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.window_size == forecaster.max_lag
    np.testing.assert_array_almost_equal(forecaster.steps, np.array([1, 2]))
    assert forecaster.max_step == 2
    assert forecaster.n_series_in == 1
    assert forecaster.n_levels_out == 1
    assert forecaster.exog_in_ is False
    assert forecaster.n_exog_in is None
    assert forecaster.series_val is None
    assert forecaster.exog_val is None
    assert isinstance(forecaster.fit_kwargs, dict)


def test_ForecasterRnn_init_levels_list_and_lags_list():
    """
    Supports levels and lags as lists.
    """
    series = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20)})
    model = create_and_compile_model(series=series, lags=[1, 2], steps=2)
    forecaster = ForecasterRnn(
        estimator=model,
        levels=['A', 'B'],
        lags=[1, 2]
    )

    assert forecaster.levels == ['A', 'B']
    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2]))
    assert forecaster.n_series_in == 2
    assert forecaster.n_levels_out == 2


def test_ForecasterRnn_init_raises_ValueError_if_number_of_lags_not_matching_estimator():
    """
    Raise ValueError if number of lags does not match the model.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    model = create_and_compile_model(series=series, lags=2, steps=2)
    err_msg = re.escape(
        "Number of lags (3) does not match the number of "
        "lags expected by the estimator architecture (2)."
    )
    with pytest.raises(ValueError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels='A',
            lags=3
        )


def test_ForecasterRnn_init_raises_TypeError_if_levels_not_str_or_list():
    """
    Raise TypeError if levels is not str or list.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2)
    err_msg = re.escape(
        "`levels` argument must be a string or list. Got <class 'int'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels=7,
            lags=3
        )


def test_ForecasterRnn_init_raises_ValueError_if_levels_list_length_not_matching_estimator():
    """
    Raise ValueError if levels list length does not match the model.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, levels=['A'])
    err_msg = re.escape(
        "Number of levels (3) does not match the number of "
        "levels expected by the estimator architecture (1)."
    )
    with pytest.raises(ValueError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels=['A', 'B', 'C'],
            lags=3
        )


def test_ForecasterRnn_init_with_exog_and_check_exog_attrs():
    """
    Initialization with model that has exogenous variables: related attributes.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    exog = pd.DataFrame({'exog1': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, exog=exog)
    forecaster = ForecasterRnn(
        estimator=model,
        levels='A',
        lags=3
    )
    assert forecaster.exog_in_ is True
    assert forecaster.n_exog_in == 1
    assert "exog_input" in forecaster.layers_names
    assert forecaster.n_levels_out == 1


def test_ForecasterRnn_init_with_series_val_and_exog_val():
    """
    Initialization with fit_kwargs: series_val and exog_val.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    exog = pd.DataFrame({'exog1': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, exog=exog)
    series_val = pd.DataFrame({'A': np.arange(3)})
    exog_val = pd.DataFrame({'exog1': np.arange(3)})
    fit_kwargs = {
        "series_val": series_val,
        "exog_val": exog_val
    }
    forecaster = ForecasterRnn(
        estimator=model,
        levels='A',
        lags=3,
        fit_kwargs=fit_kwargs
    )
    assert isinstance(forecaster.series_val, pd.DataFrame)
    assert isinstance(forecaster.exog_val, pd.DataFrame)
    assert "series_val" not in forecaster.fit_kwargs and "exog_val" not in forecaster.fit_kwargs


def test_ForecasterRnn_init_raises_TypeError_if_series_val_not_dataframe():
    """
    Raise TypeError if series_val is not DataFrame.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    exog = pd.DataFrame({'exog1': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, exog=exog)
    fit_kwargs = {
        "series_val": [1, 2, 3],
        "exog_val": pd.DataFrame({'exog1': np.arange(3)})
    }
    err_msg = re.escape(
        "`series_val` must be a pandas DataFrame. Got <class 'list'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels='A',
            lags=3,
            fit_kwargs=fit_kwargs
        )


def test_ForecasterRnn_init_raises_ValueError_if_exog_in_but_missing_exog_val():
    """
    Raise ValueError if there is series_val but missing exog_val and the model uses exogenous variables.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    exog = pd.DataFrame({'exog1': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, exog=exog)
    fit_kwargs = {
        "series_val": pd.DataFrame({'A': np.arange(3)})
        # No exog_val
    }
    err_msg = re.escape(
        "If `series_val` is provided, `exog_val` must also be "
        "provided using the `fit_kwargs` argument when the "
        "estimator has exogenous variables."
    )
    with pytest.raises(ValueError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels='A',
            lags=3,
            fit_kwargs=fit_kwargs
        )


def test_ForecasterRnn_init_raises_TypeError_if_exog_val_wrong_type():
    """
    Raise TypeError if exog_val is not DataFrame or Series.
    """
    series = pd.DataFrame({'A': np.arange(10)})
    exog = pd.DataFrame({'exog1': np.arange(10)})
    model = create_and_compile_model(series=series, lags=3, steps=2, exog=exog)
    fit_kwargs = {
        "series_val": pd.DataFrame({'A': np.arange(3)}),
        "exog_val": 42
    }
    err_msg = re.escape(
        "`exog_val` must be a pandas Series or DataFrame. Got <class 'int'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterRnn(
            estimator=model,
            levels='A',
            lags=3,
            fit_kwargs=fit_kwargs
        )
