# Unit test __init__ ForecasterFoundational
# ==============================================================================
import re
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.foundational import ForecasterFoundational, FoundationalModel
from skforecast.foundational._foundational_model import Chronos2Adapter

# Fixtures
from .fixtures_forecaster_foundational import make_forecaster, FakePipeline


# Tests __init__
# ==============================================================================

def test_init_TypeError_when_estimator_is_not_FoundationalModel():
    """
    Raise TypeError if `estimator` is not a FoundationalModel instance.
    """
    err_msg = re.escape(
        f"`estimator` must be a `FoundationalModel` instance. "
        f"Got {type(LinearRegression())}."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterFoundational(estimator=LinearRegression())


def test_init_TypeError_when_estimator_is_plain_Chronos2Adapter():
    """
    Raise TypeError if `estimator` is a raw Chronos2Adapter (not FoundationalModel).
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    err_msg = re.escape(
        f"`estimator` must be a `FoundationalModel` instance. "
        f"Got {type(adapter)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterFoundational(estimator=adapter)


def test_init_stores_estimator():
    """
    The estimator attribute must be the exact FoundationalModel instance passed in.
    """
    estimator = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    forecaster = ForecasterFoundational(estimator=estimator)
    assert forecaster.estimator is estimator


def test_init_window_size_uses_context_length_when_set():
    """
    window_size is set to adapter.context_length when it is not None.
    """
    estimator = FoundationalModel(
        "autogluon/chronos-2-small", context_length=128, pipeline=FakePipeline()
    )
    forecaster = ForecasterFoundational(estimator=estimator)
    assert forecaster.window_size == 128


def test_init_window_size_defaults_to_1_when_context_length_is_None():
    """
    window_size falls back to 1 when adapter.context_length is None.
    """
    estimator = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    forecaster = ForecasterFoundational(estimator=estimator)
    assert forecaster.window_size == 1


def test_init_default_attributes_before_fit():
    """
    All fit-time attributes are initialised to their 'unfitted' defaults.
    """
    forecaster = make_forecaster()

    assert forecaster.last_window_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.training_range_ is None
    assert forecaster.series_name_in_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster._is_multiseries is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None
    assert forecaster.is_fitted is False
    assert forecaster.fit_date is None


def test_init_forecaster_id_default_is_None():
    """
    forecaster_id defaults to None when not provided.
    """
    forecaster = make_forecaster()
    assert forecaster.forecaster_id is None


def test_init_forecaster_id_stored_correctly():
    """
    forecaster_id is stored exactly as provided.
    """
    forecaster = ForecasterFoundational(
        estimator=FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline()),
        forecaster_id="my_forecaster",
    )
    assert forecaster.forecaster_id == "my_forecaster"


def test_init_metadata_attributes():
    """
    Metadata attributes (skforecast_version, python_version, creation_date) are set.
    """
    from skforecast import __version__ as sfv

    forecaster = make_forecaster()

    assert forecaster.skforecast_version == sfv
    assert forecaster.python_version == sys.version.split(" ")[0]
    assert forecaster.creation_date is not None


def test_init_skforecast_tags():
    """
    __skforecast_tags__ contains the expected keys and values.
    """
    forecaster = make_forecaster()
    tags = forecaster.__skforecast_tags__

    assert tags["forecaster_name"] == "ForecasterFoundational"
    assert tags["supports_exog"] is True
    assert tags["supports_lags"] is False
    assert tags["supports_probabilistic"] is True
    assert "quantile_native" in tags["probabilistic_methods"]
