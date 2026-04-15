# Unit test _resolve_torch_device and adapter device handling
# ==============================================================================
import re
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import (
    _resolve_torch_device,
    Chronos2Adapter,
    MoiraiAdapter,
)
from .fixtures_adapters import (
    y, y_dict,
    FakePipeline, FakeMoirai2Forecast,
    prepare_fit_args, prepare_predict_args
)


# ==============================================================================
# Tests _resolve_torch_device
# ==============================================================================
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
    ids=lambda d: f"device={d}"
)
def test_resolve_torch_device_explicit_values_returned_as_is(device):
    """
    Test that explicit device strings (anything other than "auto") are
    returned unchanged without importing torch.
    """
    result = _resolve_torch_device(device)
    assert result == device


def test_resolve_torch_device_auto_returns_cuda_when_available():
    """
    Test that "auto" resolves to "cuda" when CUDA is available.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _resolve_torch_device("auto")

    assert result == "cuda"


def test_resolve_torch_device_auto_returns_mps_when_cuda_unavailable():
    """
    Test that "auto" resolves to "mps" when CUDA is unavailable but MPS
    (Apple Silicon) is available.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _resolve_torch_device("auto")

    assert result == "mps"


def test_resolve_torch_device_auto_returns_cpu_when_no_accelerator():
    """
    Test that "auto" resolves to "cpu" when neither CUDA nor MPS is
    available.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _resolve_torch_device("auto")

    assert result == "cpu"


def test_resolve_torch_device_auto_priority_cuda_over_mps():
    """
    Test that CUDA takes priority over MPS when both are available.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _resolve_torch_device("auto")

    assert result == "cuda"


# ==============================================================================
# Tests Chronos2Adapter device_map handling
# ==============================================================================
def test_Chronos2Adapter_device_map_default_is_auto():
    """
    Test that Chronos2Adapter default device_map is "auto".
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    assert adapter.device_map == "auto"


@pytest.mark.parametrize(
    "device_map",
    ["auto", "cpu", "cuda", "mps"],
    ids=lambda d: f"device_map={d}"
)
def test_Chronos2Adapter_device_map_stored_in_get_params(device_map):
    """
    Test that custom device_map values are stored and returned by
    get_params.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", device_map=device_map
    )
    assert adapter.device_map == device_map
    assert adapter.get_params()["device_map"] == device_map


def test_Chronos2Adapter_set_params_device_map_resets_pipeline():
    """
    Test that changing device_map via set_params resets the cached
    _pipeline to None.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    assert adapter._pipeline is not None
    adapter.set_params(device_map="cpu")
    assert adapter._pipeline is None
    assert adapter.device_map == "cpu"


def test_Chronos2Adapter_load_pipeline_passes_device_map_to_from_pretrained():
    """
    Test that _load_pipeline passes device_map directly to
    BaseChronosPipeline.from_pretrained without resolving it
    (Chronos handles "auto" internally).
    """
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = FakePipeline()

    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", device_map="auto"
    )

    with patch.dict(
        "sys.modules",
        {"chronos": MagicMock(BaseChronosPipeline=mock_pipeline_cls)}
    ):
        adapter._load_pipeline()

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        "autogluon/chronos-2-small", device_map="auto"
    )


def test_Chronos2Adapter_load_pipeline_passes_explicit_device_map():
    """
    Test that an explicit device_map like "cuda" is forwarded as-is to
    from_pretrained.
    """
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = FakePipeline()

    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", device_map="cuda"
    )

    with patch.dict(
        "sys.modules",
        {"chronos": MagicMock(BaseChronosPipeline=mock_pipeline_cls)}
    ):
        adapter._load_pipeline()

    call_kwargs = mock_pipeline_cls.from_pretrained.call_args
    assert call_kwargs[1]["device_map"] == "cuda"


def test_Chronos2Adapter_predict_full_pipeline_with_device_map():
    """
    Test that the full fit → predict pipeline works with an explicit
    device_map. Uses FakePipeline so no actual GPU is needed.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small",
        pipeline=FakePipeline(),
        device_map="cpu",
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.5, 0.9]
    )

    assert raw["sales"].shape == (5, 3)
    for i, q in enumerate([0.1, 0.5, 0.9]):
        np.testing.assert_array_almost_equal(raw["sales"][:, i], np.full(5, q))


# ==============================================================================
# Tests MoiraiAdapter device handling
# ==============================================================================
def test_MoiraiAdapter_device_default_is_auto():
    """
    Test that MoiraiAdapter default device is "auto".
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small")
    assert adapter.device == "auto"


@pytest.mark.parametrize(
    "device",
    ["auto", "cpu", "cuda", "mps"],
    ids=lambda d: f"device={d}"
)
def test_MoiraiAdapter_device_stored_in_get_params(device):
    """
    Test that custom device values are stored and returned by get_params.
    """
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small", device=device
    )
    assert adapter.device == device
    assert adapter.get_params()["device"] == device


def test_MoiraiAdapter_set_params_device_resets_forecast_obj():
    """
    Test that changing device via set_params resets _module and
    _forecast_obj to None.
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small")
    adapter._module = MagicMock()
    adapter._forecast_obj = FakeMoirai2Forecast()
    adapter.set_params(device="cpu")
    assert adapter._module is None
    assert adapter._forecast_obj is None
    assert adapter.device == "cpu"


def test_MoiraiAdapter_ensure_forecast_obj_calls_to_with_resolved_device():
    """
    Test that _ensure_forecast_obj moves the model to the resolved device
    via .to(). Mocks uni2ts imports and _resolve_torch_device.
    """
    mock_module = MagicMock()
    mock_forecast_cls = MagicMock()
    mock_forecast_instance = MagicMock()
    mock_forecast_instance.eval.return_value = mock_forecast_instance
    mock_forecast_cls.return_value = mock_forecast_instance

    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small",
        module=mock_module,
        device="cuda",
    )

    with patch(
        "skforecast.foundation._adapters._resolve_torch_device",
        return_value="cuda"
    ) as mock_resolve:
        with patch.dict(
            "sys.modules",
            {"uni2ts": MagicMock(), "uni2ts.model": MagicMock(),
             "uni2ts.model.moirai2": MagicMock(Moirai2Forecast=mock_forecast_cls)}
        ):
            adapter._ensure_forecast_obj()

    mock_resolve.assert_called_once_with("cuda")
    mock_forecast_instance.to.assert_called_once_with("cuda")


def test_MoiraiAdapter_ensure_forecast_obj_auto_resolves_to_mps():
    """
    Test that _ensure_forecast_obj with device="auto" calls
    _resolve_torch_device("auto") and moves the model to the resolved
    device (mps in this mock scenario).
    """
    mock_module = MagicMock()
    mock_forecast_cls = MagicMock()
    mock_forecast_instance = MagicMock()
    mock_forecast_instance.eval.return_value = mock_forecast_instance
    mock_forecast_cls.return_value = mock_forecast_instance

    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small",
        module=mock_module,
        device="auto",
    )

    with patch(
        "skforecast.foundation._adapters._resolve_torch_device",
        return_value="mps"
    ):
        with patch.dict(
            "sys.modules",
            {"uni2ts": MagicMock(), "uni2ts.model": MagicMock(),
             "uni2ts.model.moirai2": MagicMock(Moirai2Forecast=mock_forecast_cls)}
        ):
            adapter._ensure_forecast_obj()

    mock_forecast_instance.to.assert_called_once_with("mps")


def test_MoiraiAdapter_predict_full_pipeline_with_device():
    """
    Test that the full fit → predict pipeline works with an explicit
    device. Uses FakeMoirai2Forecast so no actual GPU is needed.
    """
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small", device="cpu"
    )
    adapter._forecast_obj = FakeMoirai2Forecast()

    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.5, 0.9]
    )

    assert raw["sales"].shape == (5, 3)
    for i, q in enumerate([0.1, 0.5, 0.9]):
        np.testing.assert_array_almost_equal(raw["sales"][:, i], np.full(5, q))


def test_MoiraiAdapter_predict_full_pipeline_multiseries_with_device():
    """
    Test that the full fit → predict pipeline works with multi-series input
    and an explicit device. Uses FakeMoirai2Forecast.
    """
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small", device="cpu"
    )
    adapter._forecast_obj = FakeMoirai2Forecast()

    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert set(raw.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw[name].shape == (4, 1)
        np.testing.assert_array_almost_equal(raw[name][:, 0], np.full(4, 0.5))
