# Unit test get_model_info
# ==============================================================================
import re
import sys
import subprocess
import dataclasses
import pytest
from skforecast.foundation import FoundationModelInfo, get_model_info
from skforecast.foundation._utils import _NON_COMMERCIAL_LICENSES


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
        license_restriction               = None,
        license_url                       = None,
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
        license_restriction               = "TimesFM Non-Commercial License v1.0",
        license_url                       = (
            "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"
        ),
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
        ("Salesforce/moirai-2.0-R-small",
         ("MoiraiAdapter", 2048, "uni2ts", False, False, False)),
        ("soda-inria/tabicl",
         ("TabICLAdapter", 4096, "tabicl[forecast]", True, False, False)),
        ("priorlabs/tabpfn-ts",
         ("TabPFNAdapter", 32768, "tabpfn-time-series", True, False, False)),
        ("theforecastingcompany/t0-alpha",
         ("T0Adapter", 8192, "tfc-t0", True, False, True)),
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
    "model_id, prefix",
    [
        ("google/timesfm-3.0-pytorch", "google/timesfm-3.0"),
        ("Salesforce/moirai-2.0-R-small", "Salesforce/moirai"),
        ("priorlabs/tabpfn-ts", "priorlabs/tabpfn"),
        ("taharnbl/TS-ICL", "taharnbl/TS-ICL"),
        ("autogluon/chronos-2-small", None),
        ("google/timesfm-2.5-200m-pytorch", None),
        ("theforecastingcompany/t0-alpha", None),
    ],
    ids=lambda x: str(x),
)
def test_get_model_info_license_fields_match_LicenseWarning_registry(model_id, prefix):
    """
    Test that the license fields come from the same registry used by
    `LicenseWarning`, and are `None` when no restriction is registered.
    """
    info = get_model_info(model_id=model_id)

    if prefix is None:
        assert info.license_restriction is None
        assert info.license_url is None
    else:
        license_name, license_url = _NON_COMMERCIAL_LICENSES[prefix]
        assert info.license_restriction == license_name
        assert info.license_url == license_url


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
        "t0", "tsicl", "synthefy_nori",
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
