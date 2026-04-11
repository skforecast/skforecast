# Unit test __init__ FoundationModel
# ==============================================================================
import pytest
import pandas as pd
from skforecast.foundation._adapters import Chronos2Adapter
from skforecast.foundation._foundation_model import FoundationModel


# Tests FoundationModel.__init__
# ==============================================================================
def test_init_output_when_default_params():
    """
    Test that FoundationModel creates the correct adapter with default
    attribute values: adapter type, model_id, context_length, allow_exogenous,
    is_fitted, and metadata attributes.
    """
    m = FoundationModel("autogluon/chronos-2-small")

    assert isinstance(m.adapter, Chronos2Adapter)
    assert m.model_id == "autogluon/chronos-2-small"
    assert m.model_id == m.adapter.model_id
    assert m.context_length == 2048
    assert m.context_length == m.adapter.context_length
    assert m.allow_exogenous is True
    assert m.allow_exogenous is m.adapter.allow_exogenous
    assert m.is_fitted is False
    assert m.is_multiple_series_ is False
    assert m.index_type_ is None
    assert m.index_freq_ is None
    assert m.training_range_ is None
    assert m.series_names_in_ is None
    assert m.exog_in_ is False
    assert m.exog_names_in_ is None
    assert m.exog_names_in_per_series_ is None
    assert m.fit_date is None
    assert m.creation_date is not None


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"context_length": 128, "device_map": "cpu"},
            {"context_length": 128, "device_map": "cpu", "cross_learning": False},
        ),
        (
            {"context_length": 512, "cross_learning": True},
            {"context_length": 512, "cross_learning": True},
        ),
    ],
    ids=["context_length+device_map", "context_length+cross_learning"],
)
def test_init_output_when_custom_params(kwargs, expected):
    """
    Test that keyword arguments are correctly forwarded to the adapter.
    """
    m = FoundationModel("autogluon/chronos-2-small", **kwargs)

    for attr, value in expected.items():
        assert getattr(m.adapter, attr) == value
    assert m.context_length == m.adapter.context_length


def test_init_ValueError_when_unknown_model():
    """
    Test that FoundationModel raises ValueError when the model ID does
    not match any registered adapter prefix.
    """
    with pytest.raises(ValueError, match="No adapter found"):
        FoundationModel("unknown/unsupported-model")
