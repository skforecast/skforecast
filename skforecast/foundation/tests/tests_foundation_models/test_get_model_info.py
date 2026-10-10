# Unit test get_model_info
# ==============================================================================
import re
import sys
import subprocess
import dataclasses
import pytest
from skforecast.foundation import FoundationModelInfo, get_model_info
from skforecast.foundation._utils import _MODEL_LICENSES


def test_get_model_info_TypeError_when_model_id_is_not_str():
    """
    Test that get_model_info raises a TypeError when `model_id` is not a
    string.
    """
    err_msg = re.escape("`model_id` must be a string. Got int.")
    with pytest.raises(TypeError, match=err_msg):
        get_model_info(model_id=123)


def test_get_model_info_ValueError_when_no_adapter_matches_model_id():
    """
    Test that get_model_info raises the same ValueError as FoundationModel
    when no registered prefix matches `model_id`.
    """
    err_msg = re.escape("No adapter found for model 'unknown/my-model'.")
    with pytest.raises(ValueError, match=err_msg):
        get_model_info(model_id="unknown/my-model")


def test_get_model_info_ValueError_when_no_license_is_registered(monkeypatch):
    """
    Test that get_model_info raises a ValueError when the adapter is
    registered but the license registry has no entry for `model_id`.
    """
    monkeypatch.delitem(_MODEL_LICENSES, "Synthefy/Nori")

    err_msg = re.escape(
        "No license is registered for model 'Synthefy/Nori'. Add its prefix "
        "to `_MODEL_LICENSES`."
    )
    with pytest.raises(ValueError, match=err_msg):
        get_model_info(model_id="Synthefy/Nori")


def test_get_model_info_output_Chronos():
    """
    Test the full output of get_model_info for a Chronos-2 checkpoint that is
    not the default one of its adapter.
    """
    info = get_model_info(model_id="autogluon/chronos-2-synth")

    expected = FoundationModelInfo(
        model_id                          = "autogluon/chronos-2-synth",
        adapter                           = "ChronosAdapter",
        model_id_prefixes                 = ("amazon/chronos-2", "autogluon/chronos-2"),
        default_model_id                  = "autogluon/chronos-2-small",
        default_context_length            = 8192,
        backend_package                   = "chronos-forecasting",
        allow_exog                        = True,
        supports_past_only_covariates     = True,
        supports_categorical_covariates   = True,
        supports_heterogeneous_covariates = False,
        supports_nan_in_series            = True,
        supported_quantiles               = None,
        requires_hf_auth                  = False,
        requires_provider_auth            = False,
        weights_repo_id                   = "autogluon/chronos-2-synth",
        weights_in_hf_cache               = True,
        license                           = "Apache-2.0",
        license_url                       = (
            "https://huggingface.co/autogluon/chronos-2-synth"
        ),
        commercial_use_restricted         = False,
    )

    assert info == expected


def test_get_model_info_output_TimesFM3():
    """
    Test the full output of get_model_info for TimesFM 3.0, whose weights
    carry a registered non-commercial license and which only supports a fixed
    set of quantiles.
    """
    info = get_model_info(model_id="google/timesfm-3.0-pytorch")

    expected = FoundationModelInfo(
        model_id                          = "google/timesfm-3.0-pytorch",
        adapter                           = "TimesFM3Adapter",
        model_id_prefixes                 = ("google/timesfm-3.0",),
        default_model_id                  = "google/timesfm-3.0-pytorch",
        default_context_length            = 2048,
        backend_package                   = "timesfm[torch]",
        allow_exog                        = True,
        supports_past_only_covariates     = True,
        supports_categorical_covariates   = False,
        supports_heterogeneous_covariates = False,
        supports_nan_in_series            = True,
        supported_quantiles               = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        requires_hf_auth                  = False,
        requires_provider_auth            = False,
        weights_repo_id                   = "google/timesfm-3.0-pytorch",
        weights_in_hf_cache               = True,
        license                           = "timesfm-non-commercial-license-v1.0",
        license_url                       = (
            "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"
        ),
        commercial_use_restricted         = True,
    )

    assert info == expected


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("autogluon/chronos-2-small",
         ("ChronosAdapter", 8192, "chronos-forecasting", True, True, False)),
        ("google/timesfm-2.5-200m-pytorch",
         ("TimesFM25Adapter", 512, "timesfm[torch]", False, False, False)),
        ("google/timesfm-3.0-pytorch",
         ("TimesFM3Adapter", 2048, "timesfm[torch]", True, False, False)),
        ("NX-AI/TiRex-2",
         ("TiRex2Adapter", 2048, "tirex-2", True, False, False)),
        ("Salesforce/moirai-2.0-R-small",
         ("MoiraiAdapter", 2048, "uni2ts", False, False, False)),
        ("soda-inria/tabicl",
         ("TabICLAdapter", 4096, "tabicl[forecast]", True, False, False)),
        ("priorlabs/tabpfn-ts",
         ("TabPFNAdapter", 32768, "tabpfn-time-series>=1.3", True, False, False)),
        ("theforecastingcompany/t0-alpha",
         ("T0Adapter", 8192, "tfc-t0", True, False, False)),
        ("Synthefy/Nori",
         ("NoriAdapter", 4096, "synthefy-nori", True, False, False)),
        ("taharnbl/TS-ICL",
         ("TSICLAdapter", 4096, "tsicl", True, False, False)),
    ],
    ids=lambda x: str(x),
)
def test_get_model_info_output_for_each_adapter(model_id, expected):
    """
    Test the adapter, default context length, backend package, exog support,
    categorical support and Hugging Face gating returned for every adapter.
    """
    info = get_model_info(model_id=model_id)

    assert (
        info.adapter,
        info.default_context_length,
        info.backend_package,
        info.allow_exog,
        info.supports_categorical_covariates,
        info.requires_hf_auth,
    ) == expected


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("autogluon/chronos-2-small", None),
        ("soda-inria/tabicl", None),
        ("google/timesfm-2.5-200m-pytorch", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        ("Salesforce/moirai-2.0-R-small", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        ("NX-AI/TiRex-2", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        ("taharnbl/TS-ICL", tuple(round(0.01 * i, 2) for i in range(1, 100))),
    ],
    ids=lambda x: str(x)[:40],
)
def test_get_model_info_supported_quantiles(model_id, expected):
    """
    Test that `supported_quantiles` is `None` for adapters that accept any
    level and the tuple of accepted levels otherwise (a 0.01 grid for TS-ICL).
    """
    info = get_model_info(model_id=model_id)

    assert info.supported_quantiles == expected
    if expected is not None:
        assert isinstance(info.supported_quantiles, tuple)


@pytest.mark.parametrize(
    "model_id, prefix, restricted",
    [
        ("google/timesfm-3.0-pytorch", "google/timesfm-3.0", True),
        ("Salesforce/moirai-2.0-R-small", "Salesforce/moirai", True),
        ("priorlabs/tabpfn-ts", "priorlabs/tabpfn", True),
        ("taharnbl/TS-ICL", "taharnbl/TS-ICL", True),
        ("autogluon/chronos-2-small", "autogluon/chronos-2", False),
        ("amazon/chronos-2", "amazon/chronos-2", False),
        ("google/timesfm-2.5-200m-pytorch", "google/timesfm-2.5", False),
        ("soda-inria/tabicl", "soda-inria/tabicl", False),
        ("theforecastingcompany/t0-alpha", "theforecastingcompany/t0", False),
        ("Synthefy/Nori", "Synthefy/Nori", False),
        ("NX-AI/TiRex-2", "NX-AI/TiRex-2", False),
    ],
    ids=lambda x: str(x),
)
def test_get_model_info_license_fields_match_LicenseWarning_registry(
    model_id, prefix, restricted
):
    """
    Test that the license fields come from the same registry used by
    `LicenseWarning`: `license` and `license_url` are always informed, and
    `commercial_use_restricted` is `True` only for the licenses that restrict
    commercial use.
    """
    info = get_model_info(model_id=model_id)
    license_name, license_url, _, _ = _MODEL_LICENSES[prefix]
    if license_url is None:
        license_url = f"https://huggingface.co/{info.weights_repo_id}"

    assert info.license == license_name
    assert info.license_url == license_url
    assert info.commercial_use_restricted is restricted


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("amazon/chronos-2", "https://huggingface.co/amazon/chronos-2"),
        ("Synthefy/Nori-30M", "https://huggingface.co/Synthefy/Nori-30M"),
        ("soda-inria/tabicl", "https://huggingface.co/jingang/TabICL"),
        ("priorlabs/tabpfn-ts",
         "https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE"),
        ("taharnbl/TS-ICL", "https://huggingface.co/taharnbl/TS-ICL/blob/main/LICENSE"),
    ],
    ids=lambda x: str(x),
)
def test_get_model_info_license_url(model_id, expected):
    """
    Test that `license_url` is the license file registered for licenses that
    are not standard, and the model card of the repository the weights are
    downloaded from otherwise, which follows `model_id` unless the backend
    uses a fixed repository (TabICL).
    """
    info = get_model_info(model_id=model_id)

    assert info.license_url == expected


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("autogluon/chronos-2-small", ("autogluon/chronos-2-small", True, False)),
        ("amazon/chronos-2", ("amazon/chronos-2", True, False)),
        ("google/timesfm-2.5-200m-pytorch",
         ("google/timesfm-2.5-200m-pytorch", True, False)),
        ("google/timesfm-3.0-pytorch", ("google/timesfm-3.0-pytorch", True, False)),
        ("Salesforce/moirai-2.0-R-base",
         ("Salesforce/moirai-2.0-R-base", True, False)),
        ("soda-inria/tabicl", ("jingang/TabICL", True, False)),
        ("priorlabs/tabpfn-ts", ("Prior-Labs/tabpfn_3_5", False, True)),
        ("theforecastingcompany/t0-alpha",
         ("theforecastingcompany/t0-alpha", True, False)),
        ("Synthefy/Nori", ("Synthefy/Nori", True, False)),
        ("taharnbl/TS-ICL", ("taharnbl/TS-ICL", True, False)),
        ("NX-AI/TiRex-2", ("NX-AI/TiRex-2", True, False)),
    ],
    ids=lambda x: str(x),
)
def test_get_model_info_weights_location_for_each_adapter(model_id, expected):
    """
    Test that `weights_repo_id` is the Hugging Face repository the backend
    downloads the weights from (`model_id` itself, or the fixed repository of
    backends that ignore it, as TabICL and TabPFN), and the values of
    `weights_in_hf_cache` and `requires_provider_auth` for every adapter.
    """
    info = get_model_info(model_id=model_id)

    assert (
        info.weights_repo_id,
        info.weights_in_hf_cache,
        info.requires_provider_auth,
    ) == expected


def test_get_model_info_output_is_frozen_and_convertible_to_dict():
    """
    Test that the returned FoundationModelInfo cannot be modified and can be
    converted to a dict with `dataclasses.asdict`.
    """
    info = get_model_info(model_id="autogluon/chronos-2-small")

    with pytest.raises(dataclasses.FrozenInstanceError):
        info.allow_exog = False

    info_dict = dataclasses.asdict(info)
    assert list(info_dict) == [field.name for field in dataclasses.fields(info)]
    assert info_dict["adapter"] == "ChronosAdapter"


def test_get_model_info_does_not_import_backend_libraries():
    """
    Test that get_model_info and list_adapters do not import any backend
    library, so they work without the backends installed. Run in a clean
    interpreter because other tests may have imported them already.
    """
    backends = [
        "chronos", "timesfm", "uni2ts", "tabicl", "tabpfn_time_series",
        "t0", "tsicl", "synthefy_nori", "tirex2",
    ]
    code = (
        "import sys\n"
        "from skforecast.foundation import get_model_info, list_adapters\n"
        "get_model_info('autogluon/chronos-2-small')\n"
        "list_adapters()\n"
        f"print(sorted(m for m in {backends!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
