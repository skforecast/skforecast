# Unit test _resolve_adapter FoundationModel
# ==============================================================================
import re
import pytest
from skforecast.foundation._adapters import (
    ChronosAdapter,
    TimesFMAdapter,
    MoiraiAdapter,
    TabICLAdapter,
    TabPFNAdapter,
    NoriAdapter,
    T0Adapter,
    _resolve_adapter,
    _ADAPTER_REGISTRY,
)


# Tests _resolve_adapter
# ==============================================================================
@pytest.mark.parametrize(
    "model_id, expected_cls",
    [
        ("autogluon/chronos-2-small", ChronosAdapter),
        ("autogluon/chronos-2-large", ChronosAdapter),
        ("google/timesfm-2.5-200m-pytorch", TimesFMAdapter),
        ("google/timesfm-2.5-200m-flax", TimesFMAdapter),
        ("Salesforce/moirai-2-base", MoiraiAdapter),
        ("soda-inria/tabicl", TabICLAdapter),
        ("priorlabs/tabpfn-ts", TabPFNAdapter),
        ("Synthefy/Nori", NoriAdapter),
        ("theforecastingcompany/t0-alpha", T0Adapter),
    ],
    ids=lambda x: str(x),
)
def test_resolve_adapter_returns_correct_class(model_id, expected_cls):
    """
    Test that _resolve_adapter returns the correct adapter class for each
    registered model prefix.
    """
    assert _resolve_adapter(model_id) is expected_cls


def test_resolve_adapter_ValueError_when_unknown_prefix():
    """
    Test that _resolve_adapter raises ValueError with a clear message
    including the registered prefixes when no prefix matches.
    """
    err_msg = re.escape("No adapter found for model 'unknown/my-model'.")
    with pytest.raises(ValueError, match=err_msg) as exc_info:
        _resolve_adapter("unknown/my-model")
    assert "Registered prefixes" in str(exc_info.value)


# Tests _ADAPTER_REGISTRY
# ==============================================================================
def test_ADAPTER_REGISTRY_contains_all_expected_entries():
    """
    Test that _ADAPTER_REGISTRY maps each expected prefix to its adapter.
    """
    expected = {
        "autogluon/chronos": ChronosAdapter,
        "google/timesfm": TimesFMAdapter,
        "Salesforce/moirai": MoiraiAdapter,
        "soda-inria/tabicl": TabICLAdapter,
        "priorlabs/tabpfn": TabPFNAdapter,
        "Synthefy/Nori": NoriAdapter,
        "theforecastingcompany/t0": T0Adapter,
    }
    for prefix, cls in expected.items():
        assert prefix in _ADAPTER_REGISTRY
        assert _ADAPTER_REGISTRY[prefix] is cls
