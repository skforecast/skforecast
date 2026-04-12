# Unit test __init__ ForecasterFoundation
# ==============================================================================
import re
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.foundation import ForecasterFoundation, FoundationModel
from skforecast.foundation._adapters import Chronos2Adapter

# Fixtures
from .fixtures_forecaster_foundation import make_forecaster, FakePipeline


# Tests __init__
# ==============================================================================

@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), Chronos2Adapter(model_id="autogluon/chronos-2-small")],
    ids=lambda e: f"estimator: {type(e).__name__}",
)
def test_init_TypeError_when_estimator_is_not_FoundationModel(estimator):
    """
    Raise TypeError if `estimator` is not a FoundationModel instance.
    """
    err_msg = re.escape(
        f"`estimator` must be a `FoundationModel` instance. "
        f"Got {type(estimator)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterFoundation(estimator=estimator)


def test_init_estimator_derived_attributes_correctly_stored():
    """
    estimator, context_length, model_id, and window_size are correctly
    derived from the FoundationModel instance.
    """
    estimator = FoundationModel(
        "autogluon/chronos-2-small", context_length=128, pipeline=FakePipeline()
    )
    forecaster = ForecasterFoundation(estimator=estimator)

    assert forecaster.estimator is estimator
    assert forecaster.context_length == 128
    assert forecaster.model_id == "autogluon/chronos-2-small"
    assert forecaster.model_id == estimator.model_id
    assert forecaster.window_size == 128


def test_init_default_attributes_before_fit():
    """
    All fit-time attributes are initialised to their 'unfitted' defaults.
    """
    forecaster = make_forecaster()

    assert forecaster.last_window_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.training_range_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster.is_multiple_series_ is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None
    assert forecaster.is_fitted is False
    assert forecaster.fit_date is None


@pytest.mark.parametrize(
    "forecaster_id, expected",
    [(None, None), ("my_forecaster", "my_forecaster")],
    ids=lambda v: f"forecaster_id={v}",
)
def test_init_forecaster_id_correctly_stored(forecaster_id, expected):
    """
    forecaster_id is stored exactly as provided (or defaults to None).
    """
    estimator = FoundationModel(
        "autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    forecaster = ForecasterFoundation(
        estimator=estimator, forecaster_id=forecaster_id
    )
    assert forecaster.forecaster_id == expected


def test_init_metadata_and_tags_correctly_stored():
    """
    Metadata attributes (skforecast_version, python_version, creation_date)
    and __skforecast_tags__ are set with expected keys and values.
    """
    from skforecast import __version__ as sfv

    forecaster = make_forecaster()

    # Metadata
    assert forecaster.skforecast_version == sfv
    assert forecaster.python_version == sys.version.split(" ")[0]
    assert forecaster.creation_date is not None

    # Tags
    tags = forecaster.__skforecast_tags__
    assert tags["forecaster_name"] == "ForecasterFoundation"
    assert tags["supports_exog"] is True
    assert tags["supports_lags"] is False
    assert tags["supports_probabilistic"] is True
    assert "quantile_native" in tags["probabilistic_methods"]
