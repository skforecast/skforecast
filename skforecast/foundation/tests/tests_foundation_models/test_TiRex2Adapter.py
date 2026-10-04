# Unit tests for the TiRex-2 adapter.
# ==============================================================================
import builtins
import re
import sys
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TiRex2Adapter
from .fixtures_adapters import (
    FakeTirex2Model,
    fake_tirex2_modules,
)


# Fixtures
# ==============================================================================
@pytest.fixture
def index():
    return pd.date_range("2024-01-01", periods=4, freq="D")


@pytest.fixture
def context(index):
    return {"series": pd.Series([1, 2, 3, 4], index=index, dtype=float)}


@pytest.fixture
def context_exog(index):
    return {
        "series": pd.DataFrame(
            {"past": [10, 11, 12, 13], "future": [0, 1, 2, 3]}, index=index
        )
    }


@pytest.fixture
def exog():
    return {
        "series": pd.DataFrame(
            {"future": [4, 5]},
            index=pd.date_range("2024-01-05", periods=2, freq="D"),
        )
    }


# Tests TiRex2Adapter.__init__
# ==============================================================================
def test_TiRex2Adapter_init_default_params():
    """
    Test that the constructor stores the default parameters and starts
    unfitted, and that the capability class attributes have the values the
    TiRex-2 backend supports.
    """
    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2")

    assert adapter.model_id == "NX-AI/TiRex-2"
    assert adapter.context_length == 2048
    assert adapter.device == "auto"
    assert adapter.predict_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TiRex2Adapter.allow_exog is True
    assert TiRex2Adapter.supports_past_only_covariates is True
    assert TiRex2Adapter.supports_categorical_covariates is False
    assert TiRex2Adapter.supports_heterogeneous_covariates is True
    assert TiRex2Adapter.supports_nan_in_series is True
    assert TiRex2Adapter.requires_hf_auth is False
    assert TiRex2Adapter.requires_provider_auth is False
    assert TiRex2Adapter.weights_repo_id is None
    assert TiRex2Adapter.weights_in_hf_cache is True
    assert TiRex2Adapter.backend_package == "tirex-2"
    assert TiRex2Adapter.default_model_id == "NX-AI/TiRex-2"
    assert TiRex2Adapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]


@pytest.mark.parametrize(
    "model_id",
    ["google/timesfm-3.0-pytorch", "NX-AI/TiRex", "tirex-2"],
    ids=lambda m: f"model_id={m}",
)
def test_TiRex2Adapter_init_ValueError_when_model_id_prefix_not_supported(model_id):
    """
    Test that __init__ raises ValueError when `model_id` is not served by this
    adapter.
    """
    err_msg = re.escape(
        "`model_id` must start with 'NX-AI/TiRex-2' for TiRex2Adapter."
    )
    with pytest.raises(ValueError, match=err_msg):
        TiRex2Adapter(model_id=model_id)


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}",
)
def test_TiRex2Adapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    err_msg = re.escape("`context_length` must be a positive integer")
    with pytest.raises(ValueError, match=err_msg):
        TiRex2Adapter(model_id="NX-AI/TiRex-2", context_length=context_length)


def test_TiRex2Adapter_init_custom_params_stored():
    """
    Test that custom constructor parameters are stored correctly.
    """
    adapter = TiRex2Adapter(
        model_id="NX-AI/TiRex-2",
        context_length=512,
        device="cpu",
        predict_kwargs={"tta_sign_flip": True},
    )

    assert adapter.context_length == 512
    assert adapter.device == "cpu"
    assert adapter.predict_kwargs == {"tta_sign_flip": True}


@pytest.mark.parametrize(
    "reserved",
    ["timeseries", "prediction_length", "output_type", "yield_per_batch",
     "return_inference_time"],
    ids=lambda k: f"predict_kwargs={k}",
)
def test_TiRex2Adapter_init_ValueError_when_predict_kwargs_managed(reserved):
    """
    Test that predict_kwargs owned by the adapter are rejected at construction
    time, since forwarding them would override the arguments built from
    `steps` and the pre-processed inputs.
    """
    with pytest.raises(ValueError, match="adapter-managed"):
        TiRex2Adapter(
            model_id="NX-AI/TiRex-2", predict_kwargs={reserved: 1}
        )


# Tests TiRex2Adapter.get_params / set_params
# ==============================================================================
def test_TiRex2Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction, and `None` (not an empty dict) when predict_kwargs is empty.
    """
    adapter = TiRex2Adapter(
        model_id="NX-AI/TiRex-2",
        context_length=512,
        device="cpu",
    )
    params = adapter.get_params()

    assert set(params.keys()) == {
        "model_id", "context_length", "device", "predict_kwargs"
    }
    assert params["model_id"] == "NX-AI/TiRex-2"
    assert params["context_length"] == 512
    assert params["device"] == "cpu"
    assert params["predict_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"model_id": "unknown/model"},
         "`model_id` must start with 'NX-AI/TiRex-2'"),
        ({"predict_kwargs": {"output_type": "numpy"}}, "adapter-managed"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=0", "context_length=-1", "model_id", "predict_kwargs",
         "unknown_param"],
)
def test_TiRex2Adapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values, for a model_id
    served by another adapter, for adapter-managed predict_kwargs and for
    unknown parameter names.
    """
    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value, resets_model",
    [
        ("model_id", "NX-AI/TiRex-2", False),
        ("device", "cpu", True),
        ("context_length", 1024, False),
        ("predict_kwargs", {"tta_sign_flip": True}, False),
    ],
    ids=["model_id", "device", "context_length", "predict_kwargs"],
)
def test_TiRex2Adapter_set_params_updates_and_resets_model(
    param, value, resets_model
):
    """
    Test that set_params updates the given parameter and resets `_model` only
    when `model_id` or `device` changes, since those control which checkpoint
    is loaded and where it runs. Returns self.
    """
    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2", model=FakeTirex2Model())
    assert adapter._model is not None

    result = adapter.set_params(**{param: value})

    assert result is adapter
    if resets_model:
        assert adapter._model is None
    else:
        assert adapter._model is not None


def test_TiRex2Adapter_set_params_no_reset_when_value_unchanged():
    """
    Test that set_params does not reset `_model` when the reset keys are set to
    their current values (no actual change).
    """
    adapter = TiRex2Adapter(
        model_id="NX-AI/TiRex-2", model=FakeTirex2Model(), device="auto"
    )

    adapter.set_params(model_id="NX-AI/TiRex-2", device="auto")

    assert adapter._model is not None


# Tests TiRex2Adapter.fit
# ==============================================================================
def test_TiRex2Adapter_fit_stores_context_and_exog(context, context_exog):
    """
    Test that fit stores the series and the historical exog, marks the adapter
    as fitted, and returns self. No training occurs.
    """
    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2")

    result = adapter.fit(context=context, context_exog=context_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    assert adapter.context_ is context
    assert adapter.context_exog_ is context_exog


# Tests TiRex2Adapter.predict — inputs and outputs
# ==============================================================================
def test_TiRex2Adapter_predict_maps_covariates_and_native_quantiles(
    monkeypatch, context, context_exog, exog
):
    """
    Test that the adapter builds one `TimeseriesType` per series with the target
    and the past and future covariates as float32 tensors, and that the native
    `(1, 9, steps)` output is mapped to `(steps, n_quantiles)` with the rows of
    the requested levels.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(2, context, context_exog, exog, [0.1, 0.9])

    ts = model.calls[0][0][0]
    np.testing.assert_array_equal(ts.target, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(ts.past_covariates, [[10, 11, 12, 13]])
    np.testing.assert_array_equal(ts.future_covariates, [[0, 1, 2, 3, 4, 5]])
    assert ts.target.dtype == np.float32
    # Quantile row 1 is 0.1 and row 9 is 0.9.
    np.testing.assert_array_equal(result["series"], [[1, 9], [1, 9]])


def test_TiRex2Adapter_predict_point_forecast_uses_median(monkeypatch, context):
    """
    Test that a point forecast (quantiles=None) returns the native median
    quantile (row 5) as a single-column array.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(2, context, None, None, None)

    np.testing.assert_array_equal(result["series"], [[5], [5]])


def test_TiRex2Adapter_predict_forwards_predict_kwargs(monkeypatch, context):
    """
    Test that predict_kwargs are forwarded to the backend call alongside the
    arguments the adapter owns, which are never overridden.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter(
        "NX-AI/TiRex-2", model=model, predict_kwargs={"tta_sign_flip": True}
    )

    adapter.predict(3, context, None, None, None)

    _, prediction_length, output_type, kwargs = model.calls[0]
    assert prediction_length == 3
    assert output_type == "numpy"
    assert kwargs == {"tta_sign_flip": True}


def test_TiRex2Adapter_predict_multi_series_with_and_without_covariates(monkeypatch):
    """
    Test that every series of a multi-series call produces its own
    `TimeseriesType` and its own forecast, and that a series without exog keys
    is forwarded without covariates instead of raising KeyError.
    """
    model = fake_tirex2_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    context = {
        "with_covariates": pd.Series([1, 2, 3], index=index, dtype=float),
        "without_covariates": pd.Series([4, 5, 6], index=index, dtype=float),
    }
    context_exog = {
        "with_covariates": pd.DataFrame({"past": [0.0, 1.0, 2.0]}, index=index)
    }
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(2, context, context_exog, None, None)

    timeseries = model.calls[0][0]
    assert len(timeseries) == 2
    np.testing.assert_array_equal(timeseries[0].past_covariates, [[0.0, 1.0, 2.0]])
    assert timeseries[1].past_covariates is None
    assert timeseries[1].future_covariates is None
    assert set(result) == {"with_covariates", "without_covariates"}
    assert result["with_covariates"].shape == (2, 1)
    assert result["without_covariates"].shape == (2, 1)


def test_TiRex2Adapter_predict_heterogeneous_covariates_in_single_call(monkeypatch):
    """
    Test that series with different covariate columns are accepted in a single
    backend call, which is what `supports_heterogeneous_covariates = True`
    declares to FoundationModel.
    """
    model = fake_tirex2_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    context = {
        "s1": pd.Series([1, 2, 3], index=index, dtype=float),
        "s2": pd.Series([4, 5, 6], index=index, dtype=float),
    }
    context_exog = {
        "s1": pd.DataFrame({"a": [0.0, 1.0, 2.0]}, index=index),
        "s2": pd.DataFrame(
            {"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]}, index=index
        ),
    }
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    adapter.predict(2, context, context_exog, None, None)

    timeseries = model.calls[0][0]
    assert len(model.calls) == 1
    assert timeseries[0].past_covariates.shape == (1, 3)
    assert timeseries[1].past_covariates.shape == (2, 3)


def test_TiRex2Adapter_predict_accepts_nan_in_series(monkeypatch):
    """
    Test that a series containing NaN is forwarded to the backend unchanged
    instead of being rejected, matching `supports_nan_in_series = True`.
    """
    model = fake_tirex2_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    context = {"s": pd.Series([1.0, np.nan, 3.0], index=index)}
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    adapter.predict(2, context, None, None, None)

    target = model.calls[0][0][0].target
    assert np.isnan(target[0, 1])
    assert target[0, 0] == 1.0


def test_TiRex2Adapter_predict_ValueError_when_steps_above_model_limit(
    monkeypatch, context
):
    """
    Test that a horizon longer than the model `future_len` is rejected before
    the backend is called, since the backend would truncate it with only a log
    warning and return a shorter forecast than requested.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    err_msg = re.escape(
        "`steps` must be <= 320, the maximum forecast horizon of NX-AI/TiRex-2"
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(321, context, None, None, None)

    assert model.calls == []


def test_TiRex2Adapter_predict_steps_at_model_limit_is_accepted(monkeypatch, context):
    """
    Test that exactly `_MODEL_MAX_PREDICTION_LENGTH` steps is accepted, so the
    limit is inclusive.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(
        TiRex2Adapter._MODEL_MAX_PREDICTION_LENGTH, context, None, None, None
    )

    assert result["series"].shape == (TiRex2Adapter._MODEL_MAX_PREDICTION_LENGTH, 1)


def test_TiRex2Adapter_predict_ValueError_when_backend_returns_wrong_count(
    monkeypatch, context
):
    """
    Test that a backend returning fewer forecasts than series raises a clear
    ValueError instead of silently misaligning the series names.
    """
    class _ShortModel(FakeTirex2Model):
        def forecast(self, timeseries, prediction_length, output_type, **kwargs):
            return super().forecast(
                timeseries[:-1], prediction_length, output_type, **kwargs
            )

    model = fake_tirex2_modules(monkeypatch, model=_ShortModel())
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)
    context = {
        "s1": pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2)),
        "s2": pd.Series([3.0, 4.0], index=pd.date_range("2024-01-01", periods=2)),
    }

    err_msg = re.escape("TiRex-2 returned an unexpected number of forecasts")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(2, context, None, None, None)


def test_TiRex2Adapter_predict_ValueError_when_backend_returns_wrong_shape(
    monkeypatch, context
):
    """
    Test that a forecast whose shape is not `(1, 9, steps)` raises a clear
    ValueError naming both the received and the expected shape.
    """
    class _WrongShapeModel(FakeTirex2Model):
        def forecast(self, timeseries, prediction_length, output_type, **kwargs):
            return [np.zeros((1, 9, prediction_length - 1)) for _ in timeseries]

    model = fake_tirex2_modules(monkeypatch, model=_WrongShapeModel())
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    err_msg = re.escape("TiRex-2 returned an unexpected forecast shape")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(2, context, None, None, None)


@pytest.mark.parametrize(
    "quantiles, expected",
    [
        ([0.5], [[5], [5]]),
        ([0.1000000005], [[1], [1]]),
        ([0.1, 0.5, 0.9], [[1, 5, 9], [1, 5, 9]]),
    ],
    ids=["median", "tolerant_level", "three_levels"],
)
def test_TiRex2Adapter_predict_selects_native_quantile_rows(
    monkeypatch, context, quantiles, expected
):
    """
    Test that the requested levels are matched tolerantly against the native
    nine-quantile grid and mapped to the corresponding rows, in the requested
    order.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(2, context, None, None, quantiles)

    np.testing.assert_array_equal(result["series"], expected)


def test_TiRex2Adapter_predict_ValueError_when_quantile_not_on_native_grid(
    monkeypatch, context
):
    """
    Test that a quantile level outside the native grid raises ValueError, since
    TiRex-2 does not interpolate quantiles.
    """
    model = fake_tirex2_modules(monkeypatch)
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    err_msg = re.escape("TiRex-2 only supports quantile levels")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(2, context, None, None, [0.25])


# Tests TiRex2Adapter covariate conversion
# ==============================================================================
def test_TiRex2Adapter_to_covariate_array_non_pandas_inputs():
    """
    Test that lists and numpy arrays are converted to float32 arrays, and that
    boolean columns are accepted as numeric.
    """
    assert TiRex2Adapter._to_covariate_array([1, 2, 3]).dtype == np.float32
    assert TiRex2Adapter._to_covariate_array(np.array([1, 2, 3])).dtype == np.float32
    assert (
        TiRex2Adapter._to_covariate_array(
            pd.Series([True, False])
        ).dtype == np.float32
    )


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["a", "b"], name="cat"),
        np.array(["a", "b"]),
        [1, "b"],
    ],
    ids=["series_object", "array_object", "mixed_list"],
)
def test_TiRex2Adapter_to_covariate_array_ValueError_when_not_numeric(values):
    """
    Test that non-numeric covariates raise a ValueError naming the adapter,
    since TiRex-2 only accepts numeric covariates.
    """
    err_msg = re.escape("TiRex2Adapter supports only numeric covariates")
    with pytest.raises(ValueError, match=err_msg):
        TiRex2Adapter._to_covariate_array(values)


def test_TiRex2Adapter_predict_ValueError_when_covariate_not_numeric(monkeypatch):
    """
    Test that a non-numeric exog column is rejected during predict, before the
    backend is called.
    """
    model = fake_tirex2_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    context = {"s": pd.Series([1.0, 2.0, 3.0], index=index)}
    context_exog = {"s": pd.DataFrame({"cat": ["a", "b", "c"]}, index=index)}
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    err_msg = re.escape("TiRex2Adapter supports only numeric covariates")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(2, context, context_exog, None, None)

    assert model.calls == []


# Tests TiRex2Adapter._load_model
# ==============================================================================
def test_TiRex2Adapter_load_model_ImportError_when_backend_missing(monkeypatch):
    """
    Test that _load_model raises an ImportError with the install instruction
    when tirex-2 is not installed, instead of a bare ModuleNotFoundError.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("tirex2"):
            raise ModuleNotFoundError("No module named 'tirex2'")
        return real_import(name, *args, **kwargs)

    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2")
    monkeypatch.setattr(builtins, "__import__", mock_import)

    err_msg = re.escape(
        "tirex-2 is required for TiRex2Adapter. Install it with "
        "`pip install tirex-2`."
    )
    with pytest.raises(ImportError, match=err_msg):
        adapter._load_model()


def test_TiRex2Adapter_predict_ImportError_when_backend_missing(monkeypatch, context):
    """
    Test that predict surfaces the same ImportError as _load_model when the
    backend is not installed.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("tirex2"):
            raise ModuleNotFoundError("No module named 'tirex2'")
        return real_import(name, *args, **kwargs)

    adapter = TiRex2Adapter(model_id="NX-AI/TiRex-2")
    monkeypatch.setattr(builtins, "__import__", mock_import)

    err_msg = re.escape("tirex-2 is required for TiRex2Adapter")
    with pytest.raises(ImportError, match=err_msg):
        adapter.predict(2, context, None, None, None)


def test_TiRex2Adapter_load_model_uses_model_id_and_device(monkeypatch):
    """
    Test that _load_model resolves the checkpoint from `model_id` (TiRex-2 has
    no fixed weights repository) and passes the resolved device, and that a
    second call reuses the loaded model.
    """
    calls = []

    def fake_load_model(model_id, device=None, **kwargs):
        calls.append((model_id, device))
        return FakeTirex2Model()

    fake_module = type(sys)("tirex2")
    fake_module.TimeseriesType = object
    fake_module.load_model = fake_load_model
    monkeypatch.setitem(sys.modules, "tirex2", fake_module)

    adapter = TiRex2Adapter("NX-AI/TiRex-2", device="cpu")
    adapter._load_model()
    loaded = adapter._model
    adapter._load_model()

    assert calls == [("NX-AI/TiRex-2", "cpu")]
    assert adapter._model is loaded


def test_TiRex2Adapter_load_model_is_noop_when_model_injected():
    """
    Test that an injected model is used as-is and the backend is never
    imported, which is what makes the adapter testable without tirex-2.
    """
    model = FakeTirex2Model()
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    adapter._load_model()

    assert adapter._model is model
