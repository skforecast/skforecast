# Unit test list_adapters
# ==============================================================================
import inspect
import dataclasses
import pytest
import pandas as pd
from skforecast.foundation import FoundationModelInfo, get_model_info, list_adapters
from skforecast.foundation._adapters import _ADAPTER_REGISTRY


def test_list_adapters_output():
    """
    Test that list_adapters returns one FoundationModelInfo per adapter, in
    registration order, each described by its default model ID.
    """
    adapters = list_adapters()

    expected = [
        ("ChronosAdapter", "autogluon/chronos-2-small"),
        ("TimesFM25Adapter", "google/timesfm-2.5-200m-pytorch"),
        ("TimesFM3Adapter", "google/timesfm-3.0-pytorch"),
        ("MoiraiAdapter", "Salesforce/moirai-2.0-R-small"),
        ("TabICLAdapter", "soda-inria/tabicl"),
        ("TabPFNAdapter", "priorlabs/tabpfn-ts"),
        ("T0Adapter", "theforecastingcompany/t0-alpha"),
        ("NoriAdapter", "Synthefy/Nori"),
        ("TSICLAdapter", "taharnbl/TS-ICL"),
    ]

    assert all(isinstance(info, FoundationModelInfo) for info in adapters)
    assert [(info.adapter, info.model_id) for info in adapters] == expected
    assert all(info.model_id == info.default_model_id for info in adapters)


def test_list_adapters_output_as_frame():
    """
    Test that list_adapters with `as_frame=True` returns a DataFrame with one
    row per adapter, indexed by the adapter name, and the same values as the
    list of FoundationModelInfo.
    """
    adapters = list_adapters()
    results = list_adapters(as_frame=True)

    expected_columns = [
        field.name for field in dataclasses.fields(FoundationModelInfo)
        if field.name != "adapter"
    ]

    assert isinstance(results, pd.DataFrame)
    assert results.index.name == "adapter"
    assert results.index.tolist() == [info.adapter for info in adapters]
    assert results.columns.tolist() == expected_columns
    for info in adapters:
        expected = dataclasses.asdict(info)
        expected.pop("adapter")
        assert results.loc[info.adapter].to_dict() == expected


def test_list_adapters_output_equals_get_model_info_of_default_model_id():
    """
    Test that every entry of list_adapters is the same object get_model_info
    returns for the adapter default model ID.
    """
    for info in list_adapters():
        assert info == get_model_info(model_id=info.default_model_id)


def test_list_adapters_model_id_prefixes_cover_the_whole_registry():
    """
    Test that the prefixes of all the adapters are exactly the registered
    prefixes, so every model ID FoundationModel accepts is described.
    """
    prefixes = [prefix for info in list_adapters() for prefix in info.model_id_prefixes]

    assert sorted(prefixes) == sorted(_ADAPTER_REGISTRY)


@pytest.mark.parametrize(
    "adapter_cls",
    list(dict.fromkeys(_ADAPTER_REGISTRY.values())),
    ids=lambda cls: cls.__name__,
)
def test_list_adapters_every_adapter_declares_its_capabilities(adapter_cls):
    """
    Contract test: every registered adapter declares, as class attributes,
    all the capabilities exposed by FoundationModelInfo with the right type,
    and its default model ID resolves to itself. A new adapter that misses
    any of them fails here instead of breaking get_model_info.
    """
    bool_attributes = [
        "allow_exog",
        "supports_past_only_covariates",
        "supports_categorical_covariates",
        "supports_heterogeneous_covariates",
        "supports_nan_in_series",
        "requires_hf_auth",
    ]
    for name in bool_attributes:
        assert isinstance(vars(adapter_cls).get(name), bool), name

    assert isinstance(vars(adapter_cls).get("backend_package"), str)
    assert isinstance(vars(adapter_cls).get("default_model_id"), str)
    assert "SUPPORTED_QUANTILES" in vars(adapter_cls)
    quantiles = adapter_cls.SUPPORTED_QUANTILES
    assert quantiles is None or all(0 < q < 1 for q in quantiles)

    context_length = inspect.signature(adapter_cls).parameters["context_length"]
    assert isinstance(context_length.default, int)

    assert get_model_info(adapter_cls.default_model_id).adapter == adapter_cls.__name__
