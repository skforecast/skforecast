# Unit test set_params ForecasterFoundation
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation import ForecasterFoundation

# Fixtures
from .fixtures_forecaster_foundation import make_forecaster, FakePipeline, y


# Tests set_params — errors
# ==============================================================================

def test_set_params_ValueError_when_invalid_key():
    """
    Raise ValueError when an invalid parameter key is provided.
    """
    forecaster = make_forecaster()
    with pytest.raises(ValueError, match="Invalid parameter"):
        forecaster.set_params({"bad_param": 42})


# Tests set_params — updates
# ==============================================================================

def test_set_params_updates_context_length_and_window_size():
    """
    set_params updates adapter.context_length, forecaster.context_length,
    and synchronises window_size.
    """
    forecaster = make_forecaster()
    forecaster.set_params({"context_length": 256})

    assert forecaster.estimator.adapter.context_length == 256
    assert forecaster.context_length == 256
    assert forecaster.window_size == 256


def test_set_params_updates_adapter_params():
    """
    set_params updates cross_learning, predict_kwargs individually and
    multiple params at once.
    """
    forecaster = make_forecaster()

    # cross_learning
    assert forecaster.estimator.adapter.cross_learning is False
    forecaster.set_params({"cross_learning": True})
    assert forecaster.estimator.adapter.cross_learning is True

    # predict_kwargs
    forecaster.set_params({"predict_kwargs": {"num_samples": 20}})
    assert forecaster.estimator.adapter.predict_kwargs == {"num_samples": 20}

    # Multiple params at once
    forecaster.set_params({"context_length": 512, "cross_learning": False})
    assert forecaster.estimator.adapter.context_length == 512
    assert forecaster.estimator.adapter.cross_learning is False
    assert forecaster.window_size == 512


def test_set_params_resets_fitted_state():
    """
    After set_params, is_fitted is reset to False and all training-related
    metadata attributes are cleared.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.is_fitted is True

    forecaster.set_params({"context_length": 32})

    assert forecaster.is_fitted is False
    assert forecaster.fit_date is None
    assert forecaster.context_range_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.context_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster.is_multiple_series_ is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None


def test_set_params_device_map_resets_pipeline():
    """
    Changing device_map resets the cached _pipeline to None.
    """
    forecaster = make_forecaster()
    assert forecaster.estimator.adapter._pipeline is not None

    forecaster.set_params({"device_map": "cpu"})
    assert forecaster.estimator.adapter._pipeline is None
