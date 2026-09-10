# Unit test TimesFM25Adapter
# ==============================================================================
import re
import sys
import types
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TimesFM25Adapter
from skforecast.exceptions import LicenseWarning
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
    Test that default parameter values are set correctly, that the
    capability flags are class-level attributes never shadowed on the
    instance, and that get_params exposes only TimesFM 2.5 parameters.
    """
    adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter.context_length == 512
    assert adapter.max_horizon == 512
    assert adapter.forecast_config_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TimesFM25Adapter.allow_exog is False
    assert TimesFM25Adapter.supports_past_only_covariates is False
    assert TimesFM25Adapter.supports_heterogeneous_covariates is True
    assert TimesFM25Adapter.supports_nan_in_series is True
    assert "allow_exog" not in vars(adapter)
    assert TimesFM25Adapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]
    assert "device" not in adapter.get_params()
    assert "predict_kwargs" not in adapter.get_params()


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
    context_length or max_horizon. There is no None sentinel for
    context_length: the default is 512.
    """
    with pytest.raises(ValueError, match=re.escape(f"`{param}` must be a positive integer")):
        TimesFM25Adapter(
            model_id="google/timesfm-2.5-200m-pytorch", **{param: value}
        )


@pytest.mark.parametrize(
    "model_id",
    [
        "google/timesfm-3.0-pytorch",
        "google/timesfm-1.0-200m-pytorch",
        "autogluon/chronos-2-small",
    ],
    ids=lambda x: str(x)
)
def test_TimesFM25Adapter_init_ValueError_when_model_id_not_timesfm_2_5(model_id):
    """
    Test that __init__ raises ValueError when model_id does not start with
    the TimesFM 2.5 prefix served by this adapter.
    """
    err_msg = re.escape(
        f"`model_id` must start with 'google/timesfm-2.5' for TimesFM25Adapter. "
        f"Got {model_id!r}."
    )
    with pytest.raises(ValueError, match=err_msg):
        TimesFM25Adapter(model_id=model_id)


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"predict_kwargs": {}}],
    ids=lambda x: str(x)
)
def test_TimesFM25Adapter_init_TypeError_when_timesfm_3_only_kwargs(kwargs):
    """
    Test that the TimesFM 3.0 constructor parameters are not accepted by
    TimesFM25Adapter, whose parameter surface is disjoint from it.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch", **kwargs)


def test_TimesFM25Adapter_init_forecast_config_kwargs_stored_by_reference():
    """
    Test that forecast_config_kwargs is stored by reference (not copied), so
    the same object is returned by get_params and the adapter stays compatible
    with sklearn.base.clone.
    """
    original = {"normalize_inputs": True}
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config_kwargs=original
    )
    assert adapter.forecast_config_kwargs is original


# ==============================================================================
# Tests TimesFM25Adapter.get_params / set_params
# ==============================================================================
def test_TimesFM25Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns exactly the TimesFM 2.5 parameters with the
    correct values, and that forecast_config_kwargs is None when empty.
    """
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        max_horizon=128,
        forecast_config_kwargs={"normalize_inputs": True},
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "max_horizon", "forecast_config_kwargs"
    }
    assert params["model_id"] == "google/timesfm-2.5-200m-pytorch"
    assert params["context_length"] == 256
    assert params["max_horizon"] == 128
    assert params["forecast_config_kwargs"] == {"normalize_inputs": True}

    # Empty kwargs -> None
    adapter2 = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter2.get_params()["forecast_config_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"max_horizon": 0}, "`max_horizon` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter(s) for TimesFM25Adapter"),
        ({"device": "cpu"}, "Invalid parameter(s) for TimesFM25Adapter"),
        ({"predict_kwargs": {}}, "Invalid parameter(s) for TimesFM25Adapter"),
    ],
    ids=["context_length=-1", "max_horizon=0", "unknown_param", "device", "predict_kwargs"]
)
def test_TimesFM25Adapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values, unknown
    parameter names, and the TimesFM 3.0-only parameters.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


def test_TimesFM25Adapter_set_params_ValueError_when_model_id_not_timesfm_2_5():
    """
    Test that set_params rejects a model_id of another TimesFM version and
    leaves the adapter untouched (validation runs before anything is
    applied).
    """
    adapter = make_adapter()
    err_msg = re.escape(
        "`model_id` must start with 'google/timesfm-2.5' for TimesFM25Adapter. "
        "Got 'google/timesfm-3.0-pytorch'."
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.set_params(model_id="google/timesfm-3.0-pytorch")

    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter._model is not None


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
    all TimesFM 2.5 params affect loading or compilation), and returns self.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert getattr(adapter, param) == value
    assert adapter._model is None


def test_TimesFM25Adapter_set_params_no_reset_when_value_unchanged():
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


# ==============================================================================
# Tests TimesFM25Adapter.fit
# ==============================================================================
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


def test_TimesFM25Adapter_fit_stores_context_exog():
    """
    Test that fit stores context_exog_ for API consistency even though
    TimesFM 2.5 never uses it.
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    assert adapter.context_exog_ is not None
    pd.testing.assert_frame_equal(
        adapter.context_exog_["sales"], ctx_exog["sales"]
    )


# ==============================================================================
# Tests TimesFM25Adapter.predict: error handling
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
    with pytest.raises(ValueError, match=re.escape("TimesFM only supports quantile levels")):
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
# Tests TimesFM25Adapter.predict: single series
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
# Tests TimesFM25Adapter.predict: multi-series
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
# Tests TimesFM25Adapter.predict: model receives correct args
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


# ==============================================================================
# Tests TimesFM25Adapter._load_model
# ==============================================================================
def test_TimesFM25Adapter_load_model_no_LicenseWarning_and_noop_when_model_set():
    """
    Test that _load_model loads the model without a LicenseWarning (TimesFM
    2.5 weights are not under a non-commercial license) and is a no-op once
    the model is loaded. The real `timesfm` module is mocked so no network
    call happens.
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

    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter._load_model()
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
        assert isinstance(adapter._model, _FakeV25Base)

        loaded = adapter._model
        adapter._load_model()
        assert adapter._model is loaded
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original


def test_TimesFM25Adapter_load_model_ImportError_when_timesfm_not_installed():
    """
    Test that _load_model raises ImportError naming the adapter when
    `timesfm` is not installed.
    """
    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = None  # forces `import timesfm` to raise ImportError
    try:
        adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
        err_msg = re.escape("timesfm is required for TimesFM25Adapter")
        with pytest.raises(ImportError, match=err_msg):
            adapter._load_model()
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original
