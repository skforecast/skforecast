# Unit test TimesFMAdapter
# ==============================================================================
import re
import sys
import types
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TimesFMAdapter
from skforecast.exceptions import LicenseWarning
from .fixtures_adapters import (
    y, y_wide, y_dict,
    FakeTimesFM25Model,
    FakeTimesFM3Forecaster,
    prepare_fit_args, prepare_predict_args
)


# Helpers
# ==============================================================================
def make_adapter(**kwargs) -> TimesFMAdapter:
    """
    Return a TimesFMAdapter pre-loaded with FakeTimesFM25Model.
    """
    defaults = dict(
        model_id="google/timesfm-2.5-200m-pytorch",
        model=FakeTimesFM25Model()
    )
    defaults.update(kwargs)
    return TimesFMAdapter(**defaults)


def make_v3_adapter(**kwargs) -> TimesFMAdapter:
    """
    Return a TimesFMAdapter (v3.0 backend) pre-loaded with
    FakeTimesFM3Forecaster.
    """
    defaults = dict(
        model_id="google/timesfm-3.0-pytorch",
        model=FakeTimesFM3Forecaster()
    )
    defaults.update(kwargs)
    return TimesFMAdapter(**defaults)


# ==============================================================================
# Tests TimesFMAdapter.__init__
# ==============================================================================
def test_TimesFMAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TimesFMAdapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter.context_length == 512
    assert adapter.max_horizon == 512
    assert adapter.forecast_config_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.is_fitted is False
    assert TimesFMAdapter.allow_exog is False
    assert TimesFMAdapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]


@pytest.mark.parametrize(
    "param, value",
    [
        ("context_length", 0),
        ("context_length", -1),
        ("max_horizon", 0),
        ("max_horizon", -1),
        ("max_horizon", None),
    ],
    ids=lambda x: str(x)
)
def test_TimesFMAdapter_init_ValueError_when_invalid_params(param, value):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length or max_horizon. `context_length=None` is not included
    since it is a valid sentinel that resolves to a backend-specific default.
    """
    with pytest.raises(ValueError, match=re.escape(f"`{param}` must be a positive integer")):
        TimesFMAdapter(
            model_id="google/timesfm-2.5-200m-pytorch", **{param: value}
        )


def test_TimesFMAdapter_init_context_length_none_resolves_to_backend_default():
    """
    Test that `context_length=None` resolves to 512 for the v2.5 backend and
    2048 for the v3.0 backend.
    """
    adapter_v25 = TimesFMAdapter(
        model_id="google/timesfm-2.5-200m-pytorch", context_length=None
    )
    adapter_v3 = TimesFMAdapter(
        model_id="google/timesfm-3.0-pytorch", context_length=None
    )
    assert adapter_v25.context_length == 512
    assert adapter_v3.context_length == 2048


@pytest.mark.parametrize(
    "model_id, expected_backend, expected_allow_exog",
    [
        ("google/timesfm-2.5-200m-pytorch", "v25", False),
        ("google/timesfm-2.5-200m-flax", "v25", False),
        ("google/timesfm-3.0-pytorch", "v3", True),
    ],
    ids=["v2.5-pytorch", "v2.5-flax", "v3.0"]
)
def test_TimesFMAdapter_init_backend_and_allow_exog_per_model_id(
    model_id, expected_backend, expected_allow_exog
):
    """
    Test that __init__ detects the correct backend and sets allow_exog
    accordingly: True for the v3.0 backend, False for the v2.5 backend.
    """
    adapter = TimesFMAdapter(model_id=model_id)
    assert adapter._backend == expected_backend
    assert adapter.allow_exog is expected_allow_exog


def test_TimesFMAdapter_init_ValueError_for_unrecognized_model_id():
    """
    Test that __init__ raises ValueError for a model_id that does not match
    either the v2.5 or v3.0 pattern.
    """
    err_msg = re.escape(
        "Could not determine the TimesFM backend for model_id "
        "'google/timesfm-1.0-pytorch'."
    )
    with pytest.raises(ValueError, match=err_msg):
        TimesFMAdapter(model_id="google/timesfm-1.0-pytorch")


@pytest.mark.parametrize(
    "reserved_key",
    [
        "contexts", "horizon", "return_quantiles", "past_only_covariates",
        "past_future_covariates", "padding_mode", "ts_ids",
    ],
)
def test_TimesFMAdapter_init_ValueError_when_predict_kwargs_has_reserved_key(reserved_key):
    """
    Test that __init__ raises ValueError when predict_kwargs includes a key
    that the v3.0 backend manages internally.
    """
    with pytest.raises(ValueError, match=re.escape("`predict_kwargs` cannot include")):
        TimesFMAdapter(
            model_id="google/timesfm-3.0-pytorch",
            predict_kwargs={reserved_key: "value"},
        )


def test_TimesFMAdapter_init_forecast_config_kwargs_stored_by_reference():
    """
    Test that forecast_config_kwargs is stored by reference (not copied), so
    the same object is returned by get_params and the adapter stays compatible
    with sklearn.base.clone.
    """
    original = {"normalize_inputs": True}
    adapter = TimesFMAdapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config_kwargs=original
    )
    assert adapter.forecast_config_kwargs is original


# ==============================================================================
# Tests TimesFMAdapter.get_params / set_params
# ==============================================================================
def test_TimesFMAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with correct values, and
    that forecast_config_kwargs and predict_kwargs are None when empty.
    """
    adapter = TimesFMAdapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        max_horizon=128,
        forecast_config_kwargs={"normalize_inputs": True},
        device="cpu",
        predict_kwargs={"use_znorm": True},
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "max_horizon", "forecast_config_kwargs",
        "device", "predict_kwargs",
    }
    assert params["model_id"] == "google/timesfm-2.5-200m-pytorch"
    assert params["context_length"] == 256
    assert params["max_horizon"] == 128
    assert params["forecast_config_kwargs"] == {"normalize_inputs": True}
    assert params["device"] == "cpu"
    assert params["predict_kwargs"] == {"use_znorm": True}

    # Empty kwargs → None
    adapter2 = TimesFMAdapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter2.get_params()["forecast_config_kwargs"] is None
    assert adapter2.get_params()["predict_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"max_horizon": 0}, "`max_horizon` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=-1", "max_horizon=0", "unknown_param"]
)
def test_TimesFMAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values or unknown
    parameter names.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value",
    [
        ("model_id", "google/timesfm-2.5-200m-pytorch-v2"),
        ("context_length", 256),
        ("max_horizon", 128),
        ("forecast_config_kwargs", {"normalize_inputs": True}),
    ],
    ids=lambda x: str(x)
)
def test_TimesFMAdapter_set_params_updates_and_resets_model(param, value):
    """
    Test that set_params updates the given parameter, resets _model (since
    all TimesFM params affect compilation), and returns self.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert adapter._model is None


def test_TimesFMAdapter_set_params_no_reset_when_value_unchanged():
    """
    Test that set_params does not reset _model when reset keys are set to
    their current values (no actual change).
    """
    adapter = make_adapter()
    assert adapter._model is not None

    adapter.set_params(
        model_id="google/timesfm-2.5-200m-pytorch",
        context_length=512,
        max_horizon=512,
    )

    assert adapter._model is not None  # not reset because values unchanged


def test_TimesFMAdapter_set_params_model_id_change_updates_backend_and_allow_exog():
    """
    Test that changing model_id via set_params re-detects the backend and
    allow_exog, in addition to resetting the cached model.
    """
    adapter = make_adapter()
    assert adapter._backend == "v25"
    assert adapter.allow_exog is False

    adapter.set_params(model_id="google/timesfm-3.0-pytorch")

    assert adapter._backend == "v3"
    assert adapter.allow_exog is True
    assert adapter._model is None


def test_TimesFMAdapter_set_params_ValueError_when_predict_kwargs_has_reserved_key():
    """
    Test that set_params raises ValueError when predict_kwargs includes a
    key that the v3.0 backend manages internally.
    """
    adapter = make_v3_adapter()
    with pytest.raises(ValueError, match=re.escape("`predict_kwargs` cannot include")):
        adapter.set_params(predict_kwargs={"padding_mode": "edge"})


def test_TimesFMAdapter_set_params_predict_kwargs_does_not_reset_model():
    """
    Test that changing predict_kwargs via set_params does not reset the
    cached model, unlike model_id/context_length/max_horizon/
    forecast_config_kwargs/device.
    """
    adapter = make_v3_adapter()
    assert adapter._model is not None

    adapter.set_params(predict_kwargs={"use_znorm": True})

    assert adapter._model is not None
    assert adapter.predict_kwargs == {"use_znorm": True}


# ==============================================================================
# Tests TimesFMAdapter.fit
# ==============================================================================
def test_TimesFMAdapter_fit_error_handling():
    """
    Test fit raises TypeError for unsupported series types.
    """
    with pytest.raises(TypeError):
        prepare_fit_args(np.arange(50))


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (20, 20), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TimesFMAdapter_fit_output_single_series(context_length, expected_len):
    """
    Test fit on a single series: returns self, sets is_fitted=True,
    stores history trimmed to context_length,
    and does not modify the input series.
    """
    adapter = make_adapter(context_length=context_length)
    y_copy = y.copy()
    ctx, ctx_exog = prepare_fit_args(y, context_length=context_length)
    result = adapter.fit(
        context=ctx, context_exog=ctx_exog
    )

    assert result is adapter
    assert adapter.is_fitted is True
    hist = next(iter(adapter.context_.values()))
    assert len(hist) == expected_len
    pd.testing.assert_series_equal(hist, y.iloc[-expected_len:])
    pd.testing.assert_series_equal(y, y_copy)


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TimesFMAdapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input: sets is_fitted=True,
    stores a dict of Series keyed by series names,
    each trimmed to context_length.
    """
    context_length = 10
    adapter = make_adapter(context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(series_input, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == context_length


def test_TimesFMAdapter_fit_exog_ignored_silently():
    """
    Test that passing exog to fit completes successfully (exog handling
    is done upstream by FoundationModel).
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df)
    adapter.fit(context=ctx, context_exog=ctx_exog)
    assert adapter.is_fitted is True


def test_TimesFMAdapter_fit_stores_context_exog():
    """
    Test that fit stores context_exog_ (used by the v3.0 backend at predict
    time), regardless of backend.
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_v3_adapter()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    assert adapter.context_exog_ is not None
    pd.testing.assert_frame_equal(
        adapter.context_exog_["sales"], ctx_exog["sales"]
    )


# ==============================================================================
# Tests TimesFMAdapter.predict — error handling
# ==============================================================================
@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q={x}"
)
def test_TimesFMAdapter_predict_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in
    SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    with pytest.raises(ValueError, match=re.escape("TimesFM only supports quantile levels")):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=[0.5, bad_quantile],
            
        )


def test_TimesFMAdapter_predict_ValueError_when_steps_exceed_max_horizon():
    """
    Test predict raises ValueError when steps > max_horizon.
    """
    adapter = make_adapter(max_horizon=10)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=15)
    err_msg = re.escape("`steps` (15) exceeds `max_horizon` (10).")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(
            steps=15, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=None
        )


# ==============================================================================
# Tests TimesFMAdapter.predict — single series
# ==============================================================================
def test_TimesFMAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns dict
    with one key, shape (steps, 1), values = 0.0 (FakeTimesFM25Model zeros).
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=12)
    raw = adapter.predict(
        steps=12, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert list(raw.keys()) == ["sales"]
    arr = raw["sales"]
    assert arr.shape == (12, 1)
    np.testing.assert_array_equal(arr[:, 0], np.zeros(12))


def test_TimesFMAdapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: returns dict with correct
    shape and values matching FakeTimesFM25Model output (q_level at each
    quantile index).
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    arr = raw["sales"]
    assert arr.shape == (5, 3)
    for i, q in enumerate(quantiles):
        np.testing.assert_array_almost_equal(arr[:, i], np.full(5, q))


def test_TimesFMAdapter_predict_all_supported_quantiles():
    """
    Test that all 9 supported quantile levels are accepted without error.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    raw = adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p,
        quantiles=TimesFMAdapter.SUPPORTED_QUANTILES,
        
    )
    assert raw["sales"].shape == (3, 9)


# ==============================================================================
# Tests TimesFMAdapter.predict — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TimesFMAdapter_predict_point_forecast_multi_series(series_input):
    """
    Test point forecast on multi-series: returns dict with one array per
    series, each of shape (steps, 1) with value 0.0.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(series_input)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert set(raw.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw[name].shape == (5, 1)
        np.testing.assert_array_equal(raw[name][:, 0], np.zeros(5))


def test_TimesFMAdapter_predict_quantile_forecast_multi_series():
    """
    Test quantile forecast on multi-series: returns dict with one array per
    series, each of shape (steps, n_quantiles) with correct values.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    for name in ["s1", "s2"]:
        assert raw[name].shape == (4, 3)
        for i, q in enumerate(quantiles):
            np.testing.assert_array_almost_equal(raw[name][:, i], np.full(4, q))


# ==============================================================================
# Tests TimesFMAdapter.predict — pipeline receives correct args
# ==============================================================================
def test_TimesFMAdapter_predict_model_receives_correct_args():
    """
    Test that the model's forecast receives the correct horizon and number
    of input arrays.
    """
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=7)
    adapter.predict(
        steps=7, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert fake_model.last_horizon == 7
    assert len(fake_model.last_inputs) == 1


def test_TimesFMAdapter_predict_context_length_trims_history():
    """
    Test that the history passed to the model is trimmed to context_length.
    """
    context_length = 10
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model, context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(y, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert len(fake_model.last_inputs[0]) == context_length


# ==============================================================================
# Tests TimesFMAdapter._ensure_compiled
# ==============================================================================
def test_TimesFMAdapter_ensure_compiled_calls_compile_with_actual_steps():
    """
    Test that _ensure_compiled compiles the model with max_horizon equal to
    the requested steps, not to the adapter's max_horizon ceiling. This is
    key for backtesting performance: TimesFM always runs max_horizon
    autoregressive decode iterations internally.
    """

    class _TrackingModel(FakeTimesFM25Model):
        def __init__(self):
            super().__init__()
            self.compile_calls = []
            self.forecast_config = None

        def compile(self, forecast_config):
            self.compile_calls.append(forecast_config)
            self.forecast_config = forecast_config

    class _MockForecastConfig:
        def __init__(self, **kwargs):
            self.max_horizon = kwargs.get("max_horizon")

    tracking_model = _TrackingModel()
    adapter = TimesFMAdapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        model=tracking_model,
        context_length=128,
        max_horizon=512
    )

    mock_timesfm = types.ModuleType("timesfm")
    mock_timesfm.ForecastConfig = _MockForecastConfig
    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter._ensure_compiled(steps=12)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original

    assert len(tracking_model.compile_calls) == 1
    assert tracking_model.compile_calls[0].max_horizon == 12


def test_TimesFMAdapter_ensure_compiled_noop_when_already_compiled():
    """
    Test that _ensure_compiled is a no-op when the model is already compiled
    for a horizon >= steps.
    """

    class _TrackingModel(FakeTimesFM25Model):
        def __init__(self):
            super().__init__()
            self.compile_calls = 0

        def compile(self, forecast_config):
            self.compile_calls += 1
            self.forecast_config = forecast_config

    tracking_model = _TrackingModel()
    tracking_model.forecast_config = type("_FC", (), {"max_horizon": 100})()

    adapter = TimesFMAdapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        model=tracking_model
    )

    adapter._ensure_compiled(steps=12)
    adapter._ensure_compiled(steps=50)
    adapter._ensure_compiled(steps=100)

    assert tracking_model.compile_calls == 0


# ==============================================================================
# Tests TimesFMAdapter.predict — v3.0 backend
# ==============================================================================
def test_TimesFMAdapter_v3_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series using the v3.0
    backend: shape (steps, 1), values = 0.0 (FakeTimesFM3Forecaster zeros),
    no covariates passed so padding_mode is 'none'.
    """
    fake_model = FakeTimesFM3Forecaster()
    adapter = make_v3_adapter(model=fake_model)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=12)
    raw = adapter.predict(
        steps=12, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert list(raw.keys()) == ["sales"]
    arr = raw["sales"]
    assert arr.shape == (12, 1)
    np.testing.assert_array_equal(arr[:, 0], np.zeros(12))
    assert fake_model.last_padding_mode == "none"
    assert fake_model.last_past_only_covariates is None
    assert fake_model.last_past_future_covariates is None


def test_TimesFMAdapter_v3_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series using the v3.0 backend:
    shape (steps, n_quantiles), values matching FakeTimesFM3Forecaster
    output (q_level at each quantile index).
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_v3_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    arr = raw["sales"]
    assert arr.shape == (5, 3)
    for i, q in enumerate(quantiles):
        np.testing.assert_array_almost_equal(arr[:, i], np.full(5, q))


def test_TimesFMAdapter_v3_predict_point_and_quantile_multi_series():
    """
    Test point and quantile forecasts on multi-series input using the v3.0
    backend: one array per series with the correct shape and values.
    """
    adapter = make_v3_adapter()
    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw_point = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    raw_quantile = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.5, 0.9]
    )

    assert set(raw_point.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw_point[name].shape == (4, 1)
        np.testing.assert_array_equal(raw_point[name][:, 0], np.zeros(4))
        assert raw_quantile[name].shape == (4, 3)


def test_TimesFMAdapter_v3_predict_ignores_max_horizon_ceiling():
    """
    Test that the v3.0 backend has no max_horizon ceiling: predict succeeds
    even when steps far exceeds a small max_horizon, unlike the v2.5
    backend.
    """
    adapter = make_v3_adapter(max_horizon=5)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=20)
    raw = adapter.predict(
        steps=20, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert raw["sales"].shape == (20, 1)


# ==============================================================================
# Tests TimesFMAdapter.predict — v3.0 covariate wiring
# ==============================================================================
def test_TimesFMAdapter_v3_predict_covariates_builds_past_only_and_past_future():
    """
    Test that predict builds past_only_covariates (columns only in
    context_exog) and past_future_covariates (columns in exog, concatenated
    with the matching historical column), with the correct shapes, and
    switches padding_mode to 'edge'.
    """
    context_length = 20
    steps = 5
    idx = y.index[-context_length:]
    context_exog_df = pd.DataFrame(
        {
            "known": np.arange(context_length, dtype=float),
            "past_only": np.arange(context_length, dtype=float) * 2,
        },
        index=idx,
    )
    future_idx = pd.date_range(
        idx[-1] + pd.DateOffset(months=1), periods=steps, freq="ME"
    )
    future_exog_df = pd.DataFrame(
        {"known": np.arange(steps, dtype=float) + 100}, index=future_idx
    )

    fake_model = FakeTimesFM3Forecaster()
    adapter = make_v3_adapter(model=fake_model, context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(
        y, exog=context_exog_df, context_length=context_length
    )
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(
        adapter, steps=steps, exog=future_exog_df
    )
    adapter.predict(
        steps=steps, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert fake_model.last_padding_mode == "edge"
    past_only = fake_model.last_past_only_covariates[0]
    past_future = fake_model.last_past_future_covariates[0]
    assert past_only.shape == (1, context_length)
    assert past_future.shape == (1, context_length + steps)
    np.testing.assert_array_almost_equal(
        past_only[0], context_exog_df["past_only"].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        past_future[0],
        np.concatenate(
            [context_exog_df["known"].to_numpy(), future_exog_df["known"].to_numpy()]
        ),
    )


def test_TimesFMAdapter_v3_predict_non_numeric_covariate_raises_ValueError():
    """
    Test that predict raises ValueError naming the offending column when a
    covariate is not numeric or boolean.
    """
    context_length = 10
    idx = y.index[-context_length:]
    context_exog_df = pd.DataFrame(
        {"category": pd.Categorical(["a", "b"] * (context_length // 2))},
        index=idx,
    )

    adapter = make_v3_adapter(context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(
        y, exog=context_exog_df, context_length=context_length
    )
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    err_msg = re.escape(
        "TimesFMAdapter supports only numeric covariates for the v3.0 backend. "
        "Column 'category'"
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=None
        )


# ==============================================================================
# Tests TimesFMAdapter.predict — v3.0 quantile index mapping
# ==============================================================================
def test_TimesFMAdapter_v3_predict_quantile_index_mapping_uses_model_quantile_grid():
    """
    Test that quantile columns are selected by matching against the model's
    own `config.quantiles` grid rather than a fixed formula: a model whose
    quantile grid is reversed still returns the columns matching the
    requested levels.
    """
    reversed_grid = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    fake_model = FakeTimesFM3Forecaster(quantiles=reversed_grid)
    adapter = make_v3_adapter(model=fake_model)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    raw = adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.9]
    )

    # FakeTimesFM3Forecaster fills column i with reversed_grid[i], so
    # matching by value (not by a fixed *10 formula) must select column 8
    # for 0.1 and column 0 for 0.9.
    np.testing.assert_array_almost_equal(raw["sales"][:, 0], np.full(3, 0.1))
    np.testing.assert_array_almost_equal(raw["sales"][:, 1], np.full(3, 0.9))


def test_TimesFMAdapter_v3_predict_ValueError_when_quantile_missing_from_model_grid():
    """
    Test that predict raises ValueError when a requested quantile (valid
    against SUPPORTED_QUANTILES) has no match in the model's own quantile
    grid within tolerance.
    """
    incomplete_grid = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # missing 0.5
    fake_model = FakeTimesFM3Forecaster(quantiles=incomplete_grid)
    adapter = make_v3_adapter(model=fake_model)
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    err_msg = re.escape("Quantile 0.5 not found in the model's quantile grid")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=[0.5]
        )


# ==============================================================================
# Tests TimesFMAdapter._load_model — version dispatch and LicenseWarning
# ==============================================================================
def test_TimesFMAdapter_load_model_v3_LicenseWarning_and_no_warning_for_v25():
    """
    Test that _load_model issues a LicenseWarning when loading the v3.0
    backend (non-commercial license) but not when loading the v2.5 backend.
    The real `timesfm` module is mocked so no network call happens.
    """

    class _FakeV25Base:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls._from_pretrained(**kwargs)

        @classmethod
        def _from_pretrained(cls, **kwargs):
            return cls()

    mock_timesfm = types.ModuleType("timesfm")
    mock_timesfm.TimesFM_2p5_200M_torch = _FakeV25Base
    mock_timesfm.TimesFM3Forecaster = FakeTimesFM3Forecaster

    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter_v25 = TimesFMAdapter(model_id="google/timesfm-2.5-200m-pytorch")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter_v25._load_model()
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
        assert adapter_v25._model is not None

        adapter_v3 = TimesFMAdapter(model_id="google/timesfm-3.0-pytorch")
        with pytest.warns(LicenseWarning, match="TimesFM Non-Commercial License"):
            adapter_v3._load_model()
        assert isinstance(adapter_v3._model, FakeTimesFM3Forecaster)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original


def test_TimesFMAdapter_load_model_v3_ImportError_when_TimesFM3Forecaster_missing():
    """
    Test that _load_model_v3 raises a clear ImportError with an upgrade hint
    when the installed `timesfm` package predates 3.0 (no
    TimesFM3Forecaster attribute).
    """
    mock_timesfm = types.ModuleType("timesfm")

    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter = TimesFMAdapter(model_id="google/timesfm-3.0-pytorch")
        err_msg = re.escape("TimesFM 3.0 requires `timesfm>=3.0`")
        with pytest.raises(ImportError, match=err_msg):
            adapter._load_model()
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original
