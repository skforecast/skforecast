# Unit test TimesFM25Adapter
# ==============================================================================
import re
import sys
import types
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TimesFM25Adapter
from .fixtures_adapters import (
    y, y_wide, y_dict,
    FakeTimesFM25Model,
    prepare_fit_args, prepare_predict_args
)


# Helpers
# ==============================================================================
def make_adapter(**kwargs) -> TimesFM25Adapter:
    """
    Return a TimesFM25Adapter pre-loaded with FakeTimesFM25Model.
    """
    defaults = dict(
        model_id="google/timesfm-2.5-200m-pytorch",
        model=FakeTimesFM25Model()
    )
    defaults.update(kwargs)
    return TimesFM25Adapter(**defaults)


# ==============================================================================
# Tests TimesFM25Adapter.__init__
# ==============================================================================
def test_TimesFM25Adapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter.context_length == 512
    assert adapter.max_horizon == 512
    assert adapter.forecast_config_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.is_fitted is False
    assert TimesFM25Adapter.allow_exog is False
    assert TimesFM25Adapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]


@pytest.mark.parametrize(
    "param, value",
    [
        ("context_length", 0),
        ("context_length", -1),
        ("context_length", None),
        ("max_horizon", 0),
        ("max_horizon", -1),
        ("max_horizon", None),
    ],
    ids=lambda x: str(x)
)
def test_TimesFM25Adapter_init_ValueError_when_invalid_params(param, value):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length or max_horizon.
    """
    with pytest.raises(ValueError, match=re.escape(f"`{param}` must be a positive integer")):
        TimesFM25Adapter(
            model_id="google/timesfm-2.5-200m-pytorch", **{param: value}
        )


def test_TimesFM25Adapter_init_forecast_config_kwargs_is_independent_copy():
    """
    Test that forecast_config_kwargs is stored as an independent copy.
    """
    original = {"normalize_inputs": True}
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config_kwargs=original
    )
    original["extra"] = "should_not_appear"
    assert "extra" not in adapter.forecast_config_kwargs


# ==============================================================================
# Tests TimesFM25Adapter.get_params / set_params
# ==============================================================================
def test_TimesFM25Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with correct values, and
    that forecast_config_kwargs is None when empty.
    """
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        max_horizon=128,
        forecast_config_kwargs={"normalize_inputs": True}
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "max_horizon", "forecast_config_kwargs",
    }
    assert params["model_id"] == "google/timesfm-2.5-200m-pytorch"
    assert params["context_length"] == 256
    assert params["max_horizon"] == 128
    assert params["forecast_config_kwargs"] == {"normalize_inputs": True}

    # Empty kwargs → None
    adapter2 = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter2.get_params()["forecast_config_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"max_horizon": 0}, "`max_horizon` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=-1", "max_horizon=0", "unknown_param"]
)
def test_TimesFM25Adapter_set_params_ValueError_when_invalid(params, match):
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
def test_TimesFM25Adapter_set_params_updates_and_resets_model(param, value):
    """
    Test that set_params updates the given parameter, resets _model (since
    all TimesFM params affect compilation), and returns self.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert adapter._model is None


# ==============================================================================
# Tests TimesFM25Adapter.fit
# ==============================================================================
def test_TimesFM25Adapter_fit_error_handling():
    """
    Test fit raises TypeError for unsupported series types.
    """
    adapter = make_adapter()
    with pytest.raises(TypeError):
        prepare_fit_args(np.arange(50))


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (20, 20), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TimesFM25Adapter_fit_output_single_series(context_length, expected_len):
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
def test_TimesFM25Adapter_fit_output_multi_series(series_input):
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


def test_TimesFM25Adapter_fit_exog_ignored_silently():
    """
    Test that passing exog to fit completes successfully (exog handling
    is done upstream by FoundationModel).
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df)
    adapter.fit(context=ctx, context_exog=ctx_exog)
    assert adapter.is_fitted is True


# ==============================================================================
# Tests TimesFM25Adapter.predict — error handling
# ==============================================================================
@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q={x}"
)
def test_TimesFM25Adapter_predict_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in
    SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    with pytest.raises(ValueError, match=re.escape("TimesFM 2.5 only supports quantile levels")):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=[0.5, bad_quantile],
            
        )


def test_TimesFM25Adapter_predict_ValueError_when_steps_exceed_max_horizon():
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
# Tests TimesFM25Adapter.predict — single series
# ==============================================================================
def test_TimesFM25Adapter_predict_point_forecast_single_series():
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


def test_TimesFM25Adapter_predict_quantile_forecast_single_series():
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


def test_TimesFM25Adapter_predict_all_supported_quantiles():
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
        quantiles=TimesFM25Adapter.SUPPORTED_QUANTILES,
        
    )
    assert raw["sales"].shape == (3, 9)


# ==============================================================================
# Tests TimesFM25Adapter.predict — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TimesFM25Adapter_predict_point_forecast_multi_series(series_input):
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


def test_TimesFM25Adapter_predict_quantile_forecast_multi_series():
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
# Tests TimesFM25Adapter.predict — pipeline receives correct args
# ==============================================================================
def test_TimesFM25Adapter_predict_model_receives_correct_args():
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


def test_TimesFM25Adapter_predict_context_length_trims_history():
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
# Tests TimesFM25Adapter._ensure_compiled
# ==============================================================================
def test_TimesFM25Adapter_ensure_compiled_calls_compile_with_actual_steps():
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
    adapter = TimesFM25Adapter(
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


def test_TimesFM25Adapter_ensure_compiled_noop_when_already_compiled():
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

    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        model=tracking_model
    )

    adapter._ensure_compiled(steps=12)
    adapter._ensure_compiled(steps=50)
    adapter._ensure_compiled(steps=100)

    assert tracking_model.compile_calls == 0
