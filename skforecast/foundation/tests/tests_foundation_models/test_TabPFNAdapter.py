# Unit test TabPFNAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TabPFNAdapter
from .fixtures_adapters import (
    y, exog, y_wide, y_dict, exog_shared,
    FakeTabPFNTSPipeline,
    prepare_fit_args, prepare_predict_args,
)


# ==============================================================================
# Tests TabPFNAdapter.__init__
# ==============================================================================
def test_TabPFNAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts")
    assert adapter.model_id == "priorlabs/tabpfn-ts"
    assert adapter.context_length == 32768
    assert adapter.mode == "local"
    assert adapter.point_estimate == "median"
    assert adapter.tabpfn_model_config == {}
    assert adapter.temporal_features is None
    assert adapter.show_progress is False
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TabPFNAdapter.allow_exog is True


def test_TabPFNAdapter_init_custom_params_stored():
    """
    Test that custom constructor parameters are stored correctly.
    """
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        context_length=512,
        mode="client",
        point_estimate="mean",
        tabpfn_model_config={"model_path": "my.ckpt"},
        temporal_features=[],
        show_progress=True,
    )
    assert adapter.context_length == 512
    assert adapter.mode == "client"
    assert adapter.point_estimate == "mean"
    assert adapter.tabpfn_model_config == {"model_path": "my.ckpt"}
    assert adapter.temporal_features == []
    assert adapter.show_progress is True


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_TabPFNAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        TabPFNAdapter(model_id="priorlabs/tabpfn-ts", context_length=context_length)


@pytest.mark.parametrize(
    "mode",
    ["cloud", "LOCAL", None, 123],
    ids=lambda m: f"mode={m}"
)
def test_TabPFNAdapter_init_ValueError_when_mode_invalid(mode):
    """
    Test that __init__ raises ValueError for unsupported mode values.
    """
    with pytest.raises(ValueError, match=re.escape("`mode` must be 'local' or 'client'")):
        TabPFNAdapter(model_id="priorlabs/tabpfn-ts", mode=mode)


@pytest.mark.parametrize(
    "point_estimate",
    ["sum", "average", None, 123],
    ids=lambda pe: f"point_estimate={pe}"
)
def test_TabPFNAdapter_init_ValueError_when_point_estimate_invalid(point_estimate):
    """
    Test that __init__ raises ValueError for unsupported point_estimate values.
    """
    with pytest.raises(ValueError, match=re.escape("`point_estimate` must be 'mean', 'median' or 'mode'")):
        TabPFNAdapter(model_id="priorlabs/tabpfn-ts", point_estimate=point_estimate)


# ==============================================================================
# Tests TabPFNAdapter.get_params / set_params
# ==============================================================================
def test_TabPFNAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction, and that tabpfn_model_config returns None when empty.
    """
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        context_length=512,
        point_estimate="mean",
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "mode", "point_estimate",
        "tabpfn_model_config", "temporal_features", "show_progress",
    }
    assert params["model_id"] == "priorlabs/tabpfn-ts"
    assert params["context_length"] == 512
    assert params["mode"] == "local"
    assert params["point_estimate"] == "mean"
    assert params["tabpfn_model_config"] is None   # empty dict → None
    assert params["temporal_features"] is None
    assert params["show_progress"] is False


def test_TabPFNAdapter_get_params_tabpfn_model_config_non_empty():
    """
    Test that tabpfn_model_config is returned as-is when non-empty.
    """
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        tabpfn_model_config={"device": "cpu"},
    )
    params = adapter.get_params()
    assert params["tabpfn_model_config"] == {"device": "cpu"}


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"context_length": -5}, "`context_length` must be a positive integer"),
        ({"mode": "cloud"}, "`mode` must be 'local' or 'client'"),
        ({"point_estimate": "sum"}, "`point_estimate` must be 'mean', 'median' or 'mode'"),
        ({"show_progress": "yes"}, "`show_progress` must be a bool"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=[
        "context_length=0", "context_length=-5", "mode=cloud",
        "point_estimate=sum", "show_progress=yes", "unknown_param",
    ]
)
def test_TabPFNAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values or unknown keys.
    """
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value",
    [
        ("model_id", "priorlabs/tabpfn-ts-v2"),
        ("context_length", 128),
        ("mode", "client"),
        ("point_estimate", "mean"),
        ("tabpfn_model_config", {"device": "cpu"}),
        ("temporal_features", []),
    ],
    ids=lambda x: str(x)
)
def test_TabPFNAdapter_set_params_resets_model_when_param_changes(
    param, value
):
    """
    Test that set_params resets the internal _model when any parameter
    actually changes value, and returns self.
    """
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", model=fake)
    assert adapter._model is not None

    result = adapter.set_params(**{param: value})

    assert result is adapter
    assert adapter._model is None


def test_TabPFNAdapter_set_params_no_reset_when_value_unchanged():
    """
    Test that set_params does not reset _model when the same value is
    passed (no actual change).
    """
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=fake,
        context_length=32768,
        point_estimate="median",
    )
    assert adapter._model is not None

    adapter.set_params(context_length=32768, point_estimate="median")

    assert adapter._model is not None  # not reset because values unchanged


def test_TabPFNAdapter_set_params_show_progress_does_not_reset_model():
    """
    Test that set_params does not reset _model when only show_progress
    changes, since it does not affect the underlying TabPFNTSPipeline.
    """
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts", model=fake, show_progress=False
    )
    assert adapter._model is not None

    result = adapter.set_params(show_progress=True)

    assert result is adapter
    assert adapter.show_progress is True
    assert adapter._model is not None  # not reset, show_progress is not a model param


def test_TabPFNAdapter_set_params_tabpfn_model_config_none_normalises_to_empty_dict():
    """
    Test that passing tabpfn_model_config=None via set_params normalises
    the internal value to an empty dict.
    """
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        tabpfn_model_config={"device": "cpu"},
    )
    adapter.set_params(tabpfn_model_config=None)
    assert adapter.tabpfn_model_config == {}


# ==============================================================================
# Tests TabPFNAdapter.fit
# ==============================================================================
@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (25, 25), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TabPFNAdapter_fit_single_series_stores_context(
    context_length, expected_len
):
    """
    Test fit on a single series: returns self, sets is_fitted=True,
    stores history trimmed to context_length, and does not modify the
    input series.
    """
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts", context_length=context_length
    )
    y_copy = y.copy()
    context, context_exog = prepare_fit_args(y, context_length=context_length)

    result = adapter.fit(context=context, context_exog=context_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"sales"}
    hist = adapter.context_["sales"]
    assert len(hist) == expected_len
    pd.testing.assert_series_equal(hist, y.iloc[-expected_len:])
    pd.testing.assert_series_equal(y, y_copy)


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TabPFNAdapter_fit_multi_series_stores_context(series_input):
    """
    Test fit on multi-series (wide DataFrame or dict): stores all series
    in context_ keyed by name.
    """
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", context_length=15)
    context, context_exog = prepare_fit_args(series_input, context_length=15)

    adapter.fit(context=context, context_exog=context_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == 15


def test_TabPFNAdapter_fit_stores_exog_and_handles_none_exog():
    """
    Test that exog is stored in context_exog_ and that None exog
    results in context_exog_ = None.
    """
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", context_length=15)

    # Fit with exog
    context, context_exog = prepare_fit_args(y, exog=exog, context_length=15)
    adapter.fit(context=context, context_exog=context_exog)
    hist_exog = adapter.context_exog_["sales"]
    assert len(hist_exog) == 15

    # Fit without exog → context_exog_ is None
    ctx2, ctx_exog2 = prepare_fit_args(y)
    adapter.fit(context=ctx2, context_exog=ctx_exog2)
    assert adapter.context_exog_ is None


def test_TabPFNAdapter_fit_multiseries_exog_broadcast_and_per_series():
    """
    Test that a shared exog DataFrame is broadcast to all series, and that
    a per-series exog dict stores correctly (present key → stored, absent
    key → None). Context_length trims exog per series.
    """
    context_length = 8
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts", context_length=context_length
    )

    # Shared exog broadcast
    context, context_exog = prepare_fit_args(
        y_dict, exog=exog_shared, context_length=context_length
    )
    adapter.fit(context=context, context_exog=context_exog)
    for name in y_dict:
        assert adapter.context_exog_[name] is not None
        assert len(adapter.context_exog_[name]) == context_length

    # Per-series exog dict: s1 has exog, s2 does not
    ctx2, ctx_exog2 = prepare_fit_args(
        y_dict, exog={"s1": exog_shared.copy()}
    )
    adapter.fit(context=ctx2, context_exog=ctx_exog2)
    assert adapter.context_exog_["s1"] is not None
    assert adapter.context_exog_["s2"] is None


# ==============================================================================
# Tests TabPFNAdapter.predict — single series
# ==============================================================================
def test_TabPFNAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns a dict
    with one entry, shape (steps, 1), and numeric values.
    """
    steps = 5
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=FakeTabPFNTSPipeline(),
    )
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=None,
    )

    assert set(predictions.keys()) == {"sales"}
    arr = predictions["sales"]
    assert arr.shape == (steps, 1)
    assert np.issubdtype(arr.dtype, np.floating)


def test_TabPFNAdapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: returns (steps, n_quantiles)
    shape, and quantile values match those defined in FakeTabPFNTSPipeline
    (each column value equals the quantile level).
    """
    steps = 6
    quantiles = [0.1, 0.5, 0.9]
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=FakeTabPFNTSPipeline(),
    )
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=quantiles,
    )

    assert set(predictions.keys()) == {"sales"}
    arr = predictions["sales"]
    assert arr.shape == (steps, len(quantiles))
    # FakeTabPFNTSPipeline returns q value itself for every step
    for col_idx, q in enumerate(quantiles):
        np.testing.assert_allclose(arr[:, col_idx], q)


# ==============================================================================
# Tests TabPFNAdapter.predict — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TabPFNAdapter_predict_point_forecast_multi_series(series_input):
    """
    Test point forecast on multi-series: returns a dict with one array per
    series, each of shape (steps, 1) with value 0.0 (FakeTabPFNTSPipeline
    target).
    """
    steps = 4
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=FakeTabPFNTSPipeline(),
    )
    context, context_exog = prepare_fit_args(series_input)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=None,
    )

    assert set(predictions.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert predictions[name].shape == (steps, 1)
        np.testing.assert_array_almost_equal(
            predictions[name][:, 0], np.full(steps, 0.0)
        )


def test_TabPFNAdapter_predict_quantile_forecast_multi_series():
    """
    Test quantile forecast on multi-series: returns a dict with one array
    per series, each of shape (steps, n_quantiles) with correct values.
    """
    steps = 4
    quantiles = [0.1, 0.5, 0.9]
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=FakeTabPFNTSPipeline(),
    )
    context, context_exog = prepare_fit_args(y_dict)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=quantiles,
    )

    assert set(predictions.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert predictions[name].shape == (steps, len(quantiles))
        for col_idx, q in enumerate(quantiles):
            np.testing.assert_allclose(predictions[name][:, col_idx], q)


# ==============================================================================
# Tests TabPFNAdapter.predict — exog forwarding
# ==============================================================================
def test_TabPFNAdapter_predict_exog_forwarded_to_context_df_and_future_df():
    """
    Test that context exog columns appear in context_df and future exog
    columns appear in future_df passed to the underlying pipeline.
    """
    steps = 3
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts", model=fake
    )
    context, context_exog = prepare_fit_args(y, exog=exog, context_length=20)
    adapter.fit(context=context, context_exog=context_exog)

    # Prepare future exog (same columns, next `steps` rows)
    future_exog_df = exog.iloc[-steps:].copy()
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps, exog=future_exog_df
    )

    adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=[0.5],
    )

    assert fake.last_context_df is not None
    assert fake.last_future_df is not None
    # context_df must contain the exog columns
    for col in exog.columns:
        assert col in fake.last_context_df.columns
    # future_df must contain the exog columns
    for col in exog.columns:
        assert col in fake.last_future_df.columns


# ==============================================================================
# Tests TabPFNAdapter.predict — model receives correct args
# ==============================================================================
def test_TabPFNAdapter_predict_model_receives_correct_quantiles_and_structure():
    """
    Test that the underlying FakeTabPFNTSPipeline receives the correct
    quantiles, and that context_df and future_df have the expected
    structure (item_id, timestamp, target columns).
    """
    steps = 5
    quantiles = [0.2, 0.8]
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=quantiles,
    )

    assert fake.last_quantiles == quantiles
    # context_df structure
    assert "item_id" in fake.last_context_df.columns
    assert "timestamp" in fake.last_context_df.columns
    assert "target" in fake.last_context_df.columns
    # future_df structure
    assert "item_id" in fake.last_future_df.columns
    assert "timestamp" in fake.last_future_df.columns
    # future_df has exactly `steps` rows per series
    future_counts = fake.last_future_df.groupby("item_id").size()
    assert (future_counts == steps).all()


def test_TabPFNAdapter_predict_point_forecast_passes_default_quantiles():
    """
    Test that when quantiles=None (point forecast), the default quantile
    set [0.1, ..., 0.9] is passed to predict_df.
    """
    steps = 3
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=None,
    )

    expected_default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert fake.last_quantiles == expected_default


# ==============================================================================
# Tests TabPFNAdapter.predict — context_length trims history
# ==============================================================================
def test_TabPFNAdapter_predict_context_length_trims_history():
    """
    Test that the context_df passed to the underlying model contains at
    most context_length observations per series.
    """
    context_length = 10
    steps = 3
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(
        model_id="priorlabs/tabpfn-ts",
        model=fake,
        context_length=context_length,
    )
    context, context_exog = prepare_fit_args(y, context_length=context_length)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=None,
    )

    context_counts = fake.last_context_df.groupby("item_id").size()
    assert (context_counts == context_length).all()


# ==============================================================================
# Tests TabPFNAdapter.predict — RangeIndex series
# ==============================================================================
def test_TabPFNAdapter_predict_range_index_issues_warning_and_returns_arrays():
    """
    Test that predicting on a RangeIndex series issues a UserWarning about
    the synthetic DatetimeIndex and still returns arrays with correct shape.
    """
    steps = 4
    y_range = pd.Series(
        data=np.arange(30, dtype=float),
        index=pd.RangeIndex(30),
        name="ts",
    )
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", model=fake)
    context, context_exog = prepare_fit_args(y_range)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    with pytest.warns(UserWarning, match="non-DatetimeIndex"):
        predictions = adapter.predict(
            steps=steps,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=[0.5],
        )

    assert "ts" in predictions
    assert predictions["ts"].shape == (steps, 1)


def test_TabPFNAdapter_predict_range_index_timestamps_are_datetime():
    """
    Test that the context_df and future_df built from a RangeIndex series
    contain valid datetime timestamps.
    """
    steps = 3
    y_range = pd.Series(
        data=np.arange(20, dtype=float),
        index=pd.RangeIndex(20),
        name="ts",
    )
    fake = FakeTabPFNTSPipeline()
    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts", model=fake)
    context, context_exog = prepare_fit_args(y_range)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    with pytest.warns(UserWarning):
        adapter.predict(
            steps=steps,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=None,
        )

    assert pd.api.types.is_datetime64_any_dtype(fake.last_context_df["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(fake.last_future_df["timestamp"])
    # future timestamps must be strictly after context timestamps for same series
    ctx_max = fake.last_context_df["timestamp"].max()
    fut_min = fake.last_future_df["timestamp"].min()
    assert fut_min > ctx_max


# ==============================================================================
# Tests TabPFNAdapter.predict — ImportError
# ==============================================================================
def test_TabPFNAdapter_predict_ImportError_when_tabpfn_time_series_not_installed(monkeypatch):
    """
    Test that predict raises ImportError with a clear install message when
    the tabpfn_time_series package is not available and no _model is
    injected.
    """
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("tabpfn_time_series"):
            raise ImportError("No module named 'tabpfn_time_series'")
        return real_import(name, *args, **kwargs)

    adapter = TabPFNAdapter(model_id="priorlabs/tabpfn-ts")
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=3)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="tabpfn-time-series"):
        adapter.predict(
            steps=3,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=None,
        )


# ==============================================================================
# Tests _ADAPTER_REGISTRY and _resolve_adapter
# ==============================================================================
def test_TabPFNAdapter_registered_in_registry():
    """
    Test that 'priorlabs/tabpfn-ts' resolves to TabPFNAdapter via
    _resolve_adapter.
    """
    from skforecast.foundation._adapters import _resolve_adapter
    cls = _resolve_adapter("priorlabs/tabpfn-ts")
    assert cls is TabPFNAdapter


def test_TabPFNAdapter_used_by_FoundationModel():
    """
    Test that FoundationModel instantiates a TabPFNAdapter when given
    the priorlabs/tabpfn-ts model_id.
    """
    from skforecast.foundation import FoundationModel
    fm = FoundationModel("priorlabs/tabpfn-ts")
    assert isinstance(fm.adapter, TabPFNAdapter)
