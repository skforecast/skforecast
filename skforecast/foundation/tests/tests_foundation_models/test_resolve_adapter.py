# Unit test _resolve_adapter FoundationModel
# ==============================================================================
import re
import pytest
from skforecast.foundation._adapters import (
    ChronosAdapter,
    TimesFM25Adapter,
    TimesFM3Adapter,
    MoiraiAdapter,
    TabICLAdapter,
    TabPFNAdapter,
    T0Adapter,
    NoriAdapter,
    TSICLAdapter,
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
        ("google/timesfm-2.5-200m-pytorch", TimesFM25Adapter),
        ("google/timesfm-2.5-200m-flax", TimesFM25Adapter),
        ("google/timesfm-3.0-pytorch", TimesFM3Adapter),
        ("Salesforce/moirai-2-base", MoiraiAdapter),
        ("soda-inria/tabicl", TabICLAdapter),
        ("priorlabs/tabpfn-ts", TabPFNAdapter),
        ("theforecastingcompany/t0-alpha", T0Adapter),
        ("Synthefy/Nori", NoriAdapter),
        ("taharnbl/TS-ICL", TSICLAdapter),
    ],
    ids=lambda x: str(x),
)
def test_resolve_adapter_returns_correct_class(model_id, expected_cls):
    """
    Test that _resolve_adapter returns the correct adapter class for each
    registered model prefix.
    """
    assert _resolve_adapter(model_id) is expected_cls


@pytest.mark.parametrize(
    "model_id",
    [
        "unknown/my-model",
        "google/timesfm-1.0-200m-pytorch",
        "google/timesfm-2.0-500m-pytorch",
        "google/timesfm",
    ],
    ids=lambda x: str(x),
)
def test_resolve_adapter_ValueError_when_unknown_prefix(model_id):
    """
    Test that _resolve_adapter raises ValueError with a clear message
    including the registered prefixes when no prefix matches. TimesFM ids
    that are not 2.5 or 3.0 are not served by any adapter.
    """
    err_msg = re.escape(f"No adapter found for model '{model_id}'.")
    with pytest.raises(ValueError, match=err_msg) as exc_info:
        _resolve_adapter(model_id)
    assert "Registered prefixes" in str(exc_info.value)
    assert "'google/timesfm-2.5'" in str(exc_info.value)
    assert "'google/timesfm-3.0'" in str(exc_info.value)


# Tests _ADAPTER_REGISTRY
# ==============================================================================
def test_ADAPTER_REGISTRY_contains_all_expected_entries():
    """
    Test that _ADAPTER_REGISTRY maps each expected prefix to its adapter.
    """
    expected = {
        "autogluon/chronos": ChronosAdapter,
        "google/timesfm-2.5": TimesFM25Adapter,
        "google/timesfm-3.0": TimesFM3Adapter,
        "Salesforce/moirai": MoiraiAdapter,
        "soda-inria/tabicl": TabICLAdapter,
        "priorlabs/tabpfn": TabPFNAdapter,
        "theforecastingcompany/t0": T0Adapter,
        "Synthefy/Nori": NoriAdapter,
        "taharnbl/TS-ICL": TSICLAdapter,
    }
    for prefix, cls in expected.items():
        assert prefix in _ADAPTER_REGISTRY
        assert _ADAPTER_REGISTRY[prefix] is cls

    # The prefix each TimesFM adapter validates must be its registry key.
    assert _ADAPTER_REGISTRY[TimesFM25Adapter._MODEL_ID_PREFIX] is TimesFM25Adapter
    assert _ADAPTER_REGISTRY[TimesFM3Adapter._MODEL_ID_PREFIX] is TimesFM3Adapter


# Tests TimesFM adapters parameter surfaces
# ==============================================================================
def test_TimesFM_adapters_get_params_share_only_common_keys():
    """
    Test that the two TimesFM adapters only share the parameters common to
    every adapter (model_id and context_length), so a search space built for
    one version cannot silently tune a no-op parameter on the other.
    """
    keys_25 = set(TimesFM25Adapter("google/timesfm-2.5-200m-pytorch").get_params())
    keys_3 = set(TimesFM3Adapter("google/timesfm-3.0-pytorch").get_params())
    assert keys_25 & keys_3 == {"model_id", "context_length"}
