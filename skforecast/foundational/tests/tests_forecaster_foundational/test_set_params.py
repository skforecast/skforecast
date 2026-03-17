# Unit test set_params ForecasterFoundational
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational import ForecasterFoundational

# Fixtures
from .fixtures_forecaster_foundational import make_forecaster, y


# Tests set_params
# ==============================================================================

def test_set_params_ValueError_when_invalid_key():
    """
    Raise ValueError when an invalid parameter key is provided.
    """
    forecaster = make_forecaster()

    allowed = {'context_length', 'predict_kwargs', 'device_map', 'torch_dtype', 'cross_learning'}
    err_msg = re.escape(
        f"Invalid parameter(s): {{'bad_param'}}. "
        f"Allowed parameters are: {allowed}."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.set_params({"bad_param": 42})


def test_set_params_updates_context_length():
    """
    set_params updates adapter.context_length correctly.
    """
    forecaster = make_forecaster()
    forecaster.set_params({"context_length": 256})
    assert forecaster.estimator.adapter.context_length == 256


def test_set_params_updates_window_size_when_context_length_changes():
    """
    window_size is synchronised with context_length after set_params.
    """
    forecaster = make_forecaster()
    forecaster.set_params({"context_length": 64})
    assert forecaster.window_size == 64


def test_set_params_window_size_becomes_1_when_context_length_set_to_None():
    """
    window_size falls back to 1 when context_length is set to None.
    """
    forecaster = make_forecaster(context_length=128)
    assert forecaster.window_size == 128

    forecaster.set_params({"context_length": None})
    assert forecaster.window_size == 1


def test_set_params_updates_cross_learning():
    """
    set_params updates adapter.cross_learning correctly.
    """
    forecaster = make_forecaster()
    assert forecaster.estimator.adapter.cross_learning is False
    forecaster.set_params({"cross_learning": True})
    assert forecaster.estimator.adapter.cross_learning is True


def test_set_params_updates_predict_kwargs():
    """
    set_params updates adapter.predict_kwargs correctly.
    """
    forecaster = make_forecaster()
    forecaster.set_params({"predict_kwargs": {"num_samples": 20}})
    assert forecaster.estimator.adapter.predict_kwargs == {"num_samples": 20}


def test_set_params_invalidates_fit_state():
    """
    After set_params, is_fitted is reset to False.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.is_fitted is True

    forecaster.set_params({"context_length": 32})
    assert forecaster.is_fitted is False


def test_set_params_clears_training_metadata():
    """
    After set_params, all training-related metadata attributes are reset to None/False.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)

    forecaster.set_params({"context_length": 32})

    assert forecaster.fit_date is None
    assert forecaster.training_range_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.last_window_ is None
    assert forecaster.extended_index_ is None
    assert forecaster.series_name_in_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster._is_multiseries is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None


def test_set_params_device_map_resets_pipeline():
    """
    Changing device_map resets the cached _pipeline to None.
    """
    from .fixtures_forecaster_foundational import FakePipeline

    forecaster = make_forecaster()
    # Ensure a pipeline is loaded.
    assert forecaster.estimator.adapter._pipeline is not None

    forecaster.set_params({"device_map": "cpu"})
    assert forecaster.estimator.adapter._pipeline is None


def test_set_params_multiple_params_at_once():
    """
    Multiple parameters can be updated in a single set_params call.
    """
    forecaster = make_forecaster()
    forecaster.set_params({"context_length": 512, "cross_learning": True})
    assert forecaster.estimator.adapter.context_length == 512
    assert forecaster.estimator.adapter.cross_learning is True
    assert forecaster.window_size == 512
