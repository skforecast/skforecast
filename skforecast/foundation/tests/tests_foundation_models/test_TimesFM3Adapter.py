# Unit test TimesFM3Adapter
# ==============================================================================
import re
import sys
import types
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TimesFM3Adapter
from skforecast.exceptions import LicenseWarning
from .fixtures_adapters import (
    y, y_wide, y_dict,
    FakeTimesFM3Forecaster,
    prepare_fit_args, prepare_predict_args
)


# Helpers
# ==============================================================================
def make_adapter(**kwargs) -> TimesFM3Adapter:
    """
    Return a TimesFM3Adapter pre-loaded with FakeTimesFM3Forecaster.
    """
    defaults = dict(
        model_id="google/timesfm-3.0-pytorch",
        model=FakeTimesFM3Forecaster()
    )
    defaults.update(kwargs)
    return TimesFM3Adapter(**defaults)


# ==============================================================================
# Tests TimesFM3Adapter.__init__
# ==============================================================================
def test_TimesFM3Adapter_init_default_params():
    """
    Test that default parameter values are set correctly and that the
    capability flags are class-level attributes, never shadowed on the
    instance.
    """
    adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
    assert adapter.model_id == "google/timesfm-3.0-pytorch"
    assert adapter.context_length == 2048
    assert adapter.device == "auto"
    assert adapter.predict_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TimesFM3Adapter.allow_exog is True
    assert TimesFM3Adapter.supports_past_only_covariates is True
    assert TimesFM3Adapter.supports_heterogeneous_covariates is False
    assert TimesFM3Adapter.supports_nan_in_series is True
    assert "allow_exog" not in vars(adapter)
    assert TimesFM3Adapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]


@pytest.mark.parametrize(
    "value",
    [0, -1, None, "a"],
    ids=lambda x: str(x)
)
def test_TimesFM3Adapter_init_ValueError_when_invalid_context_length(value):
    """
    Test that __init__ raises ValueError for a non-positive-integer
    context_length. There is no None sentinel: the default is 2048.
    """
    err_msg = re.escape("`context_length` must be a positive integer")
    with pytest.raises(ValueError, match=err_msg):
        TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch", context_length=value)


@pytest.mark.parametrize(
    "model_id",
    [
        "google/timesfm-2.5-200m-pytorch",
        "google/timesfm-1.0-200m-pytorch",
        "autogluon/chronos-2-small",
    ],
    ids=lambda x: str(x)
)
def test_TimesFM3Adapter_init_ValueError_when_model_id_not_timesfm_3(model_id):
    """
    Test that __init__ raises ValueError when model_id does not start with
    the TimesFM 3.0 prefix served by this adapter.
    """
    err_msg = re.escape(
        f"`model_id` must start with 'google/timesfm-3.0' for TimesFM3Adapter. "
        f"Got {model_id!r}."
    )
    with pytest.raises(ValueError, match=err_msg):
        TimesFM3Adapter(model_id=model_id)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_horizon": 5}, {"forecast_config_kwargs": {}}],
    ids=lambda x: str(x)
)
def test_TimesFM3Adapter_init_TypeError_when_timesfm_2_5_only_kwargs(kwargs):
    """
    Test that the TimesFM 2.5 constructor parameters are not accepted by
    TimesFM3Adapter, whose parameter surface is disjoint from it.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch", **kwargs)


@pytest.mark.parametrize(
    "reserved_key",
    [
        "contexts", "horizon", "return_quantiles", "past_only_covariates",
        "past_future_covariates", "padding_mode", "ts_ids",
    ],
)
def test_TimesFM3Adapter_init_ValueError_when_predict_kwargs_has_reserved_key(reserved_key):
    """
    Test that __init__ raises ValueError when predict_kwargs includes a key
    that the adapter manages internally.
    """
    err_msg = re.escape("`predict_kwargs` cannot include")
    with pytest.raises(ValueError, match=err_msg):
        TimesFM3Adapter(
            model_id="google/timesfm-3.0-pytorch",
            predict_kwargs={reserved_key: "value"},
        )


def test_TimesFM3Adapter_init_predict_kwargs_stored_by_reference():
    """
    Test that predict_kwargs is stored by reference (not copied), so the
    same object is returned by get_params and the adapter stays compatible
    with sklearn.base.clone.
    """
    original = {"use_znorm": True}
    adapter = TimesFM3Adapter(
        model_id="google/timesfm-3.0-pytorch",
        predict_kwargs=original
    )
    assert adapter.predict_kwargs is original


# ==============================================================================
# Tests TimesFM3Adapter.get_params / set_params
# ==============================================================================
def test_TimesFM3Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns exactly the TimesFM 3.0 parameters with the
    correct values, and that predict_kwargs is None when empty.
    """
    adapter = TimesFM3Adapter(
        model_id="google/timesfm-3.0-pytorch",
        context_length=256,
        device="cpu",
        predict_kwargs={"use_znorm": True},
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "device", "predict_kwargs"
    }
    assert params["model_id"] == "google/timesfm-3.0-pytorch"
    assert params["context_length"] == 256
    assert params["device"] == "cpu"
    assert params["predict_kwargs"] == {"use_znorm": True}

    # Empty kwargs -> None
    adapter2 = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
    assert adapter2.get_params()["predict_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter(s) for TimesFM3Adapter"),
        ({"max_horizon": 1}, "Invalid parameter(s) for TimesFM3Adapter"),
        ({"forecast_config_kwargs": {}}, "Invalid parameter(s) for TimesFM3Adapter"),
    ],
    ids=["context_length=-1", "unknown_param", "max_horizon", "forecast_config_kwargs"]
)
def test_TimesFM3Adapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values, unknown
    parameter names, and the TimesFM 2.5-only parameters.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


def test_TimesFM3Adapter_set_params_ValueError_when_model_id_not_timesfm_3():
    """
    Test that set_params rejects a model_id of another TimesFM version and
    leaves the adapter untouched (validation runs before anything is
    applied).
    """
    adapter = make_adapter()
    err_msg = re.escape(
        "`model_id` must start with 'google/timesfm-3.0' for TimesFM3Adapter. "
        "Got 'google/timesfm-2.5-200m-pytorch'."
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.set_params(model_id="google/timesfm-2.5-200m-pytorch")

    assert adapter.model_id == "google/timesfm-3.0-pytorch"
    assert adapter._model is not None


def test_TimesFM3Adapter_set_params_ValueError_when_predict_kwargs_has_reserved_key():
    """
    Test that set_params raises ValueError when predict_kwargs includes a
    key that the adapter manages internally.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape("`predict_kwargs` cannot include")):
        adapter.set_params(predict_kwargs={"padding_mode": "edge"})


@pytest.mark.parametrize(
    "param, value",
    [
        ("model_id", "google/timesfm-3.0-pytorch-v2"),
        ("device", "cuda"),
    ],
    ids=lambda x: str(x)
)
def test_TimesFM3Adapter_set_params_updates_and_resets_model(param, value):
    """
    Test that set_params updates model_id or device, resets _model (both are
    forwarded to TimesFM3Forecaster.from_pretrained), returns self, and does
    not reset _model when the value is unchanged.
    """
    adapter = make_adapter(device="cpu")
    assert adapter._model is not None

    adapter.set_params(**{param: getattr(adapter, param)})
    assert adapter._model is not None  # not reset because value unchanged

    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert getattr(adapter, param) == value
    assert adapter._model is None


@pytest.mark.parametrize(
    "param, value",
    [
        ("context_length", 4096),
        ("predict_kwargs", {"use_znorm": True}),
    ],
    ids=lambda x: str(x)
)
def test_TimesFM3Adapter_set_params_no_reset_for_non_reload_keys(param, value):
    """
    Test that changing context_length or predict_kwargs via set_params
    updates the attribute but does not reset the cached model, since neither
    is passed to TimesFM3Forecaster.from_pretrained.
    """
    adapter = make_adapter()
    assert adapter._model is not None

    adapter.set_params(**{param: value})

    assert adapter._model is not None
    assert getattr(adapter, param) == value


# ==============================================================================
# Tests TimesFM3Adapter.fit
# ==============================================================================
@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TimesFM3Adapter_fit_output_single_series(context_length, expected_len):
    """
    Test fit on a single series with exog: returns self, sets
    is_fitted=True, stores history trimmed to context_length and stores
    context_exog_, and does not modify the input series.
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter(context_length=context_length)
    y_copy = y.copy()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df, context_length=context_length)
    result = adapter.fit(context=ctx, context_exog=ctx_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    hist = next(iter(adapter.context_.values()))
    assert len(hist) == expected_len
    pd.testing.assert_series_equal(hist, y.iloc[-expected_len:])
    pd.testing.assert_series_equal(y, y_copy)
    assert adapter.context_exog_ is not None
    pd.testing.assert_frame_equal(adapter.context_exog_["sales"], ctx_exog["sales"])


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TimesFM3Adapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input: sets is_fitted=True, stores a dict of
    Series keyed by series names, each trimmed to context_length.
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


# ==============================================================================
# Tests TimesFM3Adapter.predict
# ==============================================================================
@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q={x}"
)
def test_TimesFM3Adapter_predict_ValueError_for_unsupported_quantile(bad_quantile):
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


def test_TimesFM3Adapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: shape
    (steps, 1), values = 0.0 (FakeTimesFM3Forecaster zeros), no covariates
    passed so padding_mode is 'none'.
    """
    fake_model = FakeTimesFM3Forecaster()
    adapter = make_adapter(model=fake_model)
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


def test_TimesFM3Adapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: shape (steps, n_quantiles),
    values matching FakeTimesFM3Forecaster output (q_level at each quantile
    index).
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


def test_TimesFM3Adapter_predict_all_supported_quantiles():
    """
    Test that all 9 supported quantile levels are accepted without error.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    raw = adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=TimesFM3Adapter.SUPPORTED_QUANTILES,
    )
    assert raw["sales"].shape == (3, 9)


def test_TimesFM3Adapter_predict_point_and_quantile_multi_series():
    """
    Test point and quantile forecasts on multi-series input: one array per
    series with the correct shape and values.
    """
    adapter = make_adapter()
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


def test_TimesFM3Adapter_predict_no_horizon_ceiling():
    """
    Test that there is no horizon ceiling: predict succeeds for a horizon far
    larger than the TimesFM 2.5 default max_horizon, since TimesFM 3.0 has no
    compile step.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=600)
    raw = adapter.predict(
        steps=600, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert raw["sales"].shape == (600, 1)


def test_TimesFM3Adapter_predict_forwards_predict_kwargs():
    """
    Test that predict_kwargs are forwarded verbatim to predict_batch.
    """
    fake_model = FakeTimesFM3Forecaster()
    adapter = make_adapter(model=fake_model, predict_kwargs={"use_znorm": True})
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert fake_model.last_kwargs == {"use_znorm": True}


# ==============================================================================
# Tests TimesFM3Adapter.predict: covariate wiring
# ==============================================================================
def test_TimesFM3Adapter_predict_covariates_builds_past_only_and_past_future():
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
    adapter = make_adapter(model=fake_model, context_length=context_length)
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


def test_TimesFM3Adapter_predict_non_numeric_covariate_raises_ValueError():
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

    adapter = make_adapter(context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(
        y, exog=context_exog_df, context_length=context_length
    )
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    err_msg = re.escape(
        "TimesFM3Adapter supports only numeric covariates. Column 'category'"
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=None
        )


def test_TimesFM3Adapter_build_covariates_per_series():
    """
    Test that _build_covariates builds, from a single series' own exog,
    one past-only row per column in `past_only_cols` (shape (n, ctx)) and
    one known-future row per column in `fut_cols` concatenating history and
    future (shape (n, ctx + steps)), and returns None for an empty group.
    """
    ctx_idx = pd.date_range("2023-01-31", periods=5, freq="ME")
    future_idx = pd.date_range("2023-06-30", periods=3, freq="ME")
    context_exog = pd.DataFrame(
        {"known": np.arange(5, dtype=float), "past": np.arange(5, dtype=float) * 10},
        index=ctx_idx,
    )
    exog_future = pd.DataFrame({"known": np.arange(100, 103, dtype=float)}, index=future_idx)

    past_only, past_future = TimesFM3Adapter._build_covariates(
        context_exog=context_exog, exog=exog_future,
        past_only_cols=("past",), fut_cols=("known",),
    )
    assert past_only.shape == (1, 5)
    assert past_future.shape == (1, 5 + 3)
    np.testing.assert_array_almost_equal(past_only[0], context_exog["past"].to_numpy())
    np.testing.assert_array_almost_equal(
        past_future[0],
        np.concatenate([context_exog["known"].to_numpy(), exog_future["known"].to_numpy()]),
    )

    past_only, past_future = TimesFM3Adapter._build_covariates(
        context_exog=context_exog, exog=None, past_only_cols=(), fut_cols=(),
    )
    assert past_only is None
    assert past_future is None


def test_TimesFM3Adapter_predict_single_call_per_homogeneous_batch():
    """
    Test that predict makes exactly one predict_batch call for a batch of
    series sharing the same covariate columns (as guaranteed by
    FoundationModel), building each series' covariate arrays from its own
    exog in the sorted column order of the signature even when the series
    list their columns in a different order or have different lengths, and
    that the output keeps the original series order.
    """
    context_length = 20
    steps = 5
    idx = y.index[-context_length:]
    future_idx = pd.date_range(idx[-1] + pd.DateOffset(months=1), periods=steps, freq="ME")
    p_values = np.arange(context_length, dtype=float)
    context = {
        "full":  pd.Series(p_values, index=idx, name="full"),
        "short": pd.Series(p_values[-8:] * 2, index=idx[-8:], name="short"),
    }
    context_exog = {
        "full":  pd.DataFrame({"p": p_values, "k": p_values * 2}, index=idx),
        "short": pd.DataFrame({"k": p_values[-8:] * 4, "p": p_values[-8:] * 5}, index=idx[-8:]),
    }
    exog = {
        "full":  pd.DataFrame({"k": np.arange(steps, dtype=float) + 100}, index=future_idx),
        "short": pd.DataFrame({"k": np.arange(steps, dtype=float) + 200}, index=future_idx),
    }

    fake_model = FakeTimesFM3Forecaster()
    adapter = make_adapter(context_length=context_length, model=fake_model)
    adapter.fit(context=context, context_exog=context_exog)

    raw = adapter.predict(
        steps=steps, context=context, context_exog=context_exog,
        exog=exog, quantiles=None
    )

    assert list(raw.keys()) == ["full", "short"]
    assert all(arr.shape == (steps, 1) for arr in raw.values())
    assert len(fake_model.calls) == 1

    call = fake_model.calls[0]
    assert len(call["contexts"]) == 2
    assert call["padding_mode"] == "edge"
    np.testing.assert_array_almost_equal(call["past_only_covariates"][0][0], p_values)
    np.testing.assert_array_almost_equal(call["past_only_covariates"][1][0], p_values[-8:] * 5)
    np.testing.assert_array_almost_equal(
        call["past_future_covariates"][0][0],
        np.concatenate([p_values * 2, np.arange(steps, dtype=float) + 100]),
    )
    np.testing.assert_array_almost_equal(
        call["past_future_covariates"][1][0],
        np.concatenate([p_values[-8:] * 4, np.arange(steps, dtype=float) + 200]),
    )


def test_FoundationModel_TimesFM3_predict_levels_matches_batch_covariates():
    """
    Integration test through FoundationModel: with exog on only one series
    of a multi-series batch, each series receives exactly the same
    covariates whether it is predicted with the full batch or alone via
    `levels`, so predictions never depend on the other series' exog.
    """
    from skforecast.foundation import FoundationModel

    fake_model = FakeTimesFM3Forecaster()
    model = FoundationModel(
        model_id="google/timesfm-3.0-pytorch",
        model=fake_model,
    )
    idx = y_wide.index
    exog_s1 = pd.DataFrame({"feat": np.arange(len(idx), dtype=float)}, index=idx)
    model.fit(series=y_wide, exog={"s1": exog_s1})

    steps = 4
    future_idx = pd.date_range(idx[-1] + idx.freq, periods=steps, freq=idx.freq)
    exog_s1_fut = pd.DataFrame(
        {"feat": np.arange(steps, dtype=float)}, index=future_idx
    )

    predictions = model.predict(steps=steps, exog={"s1": exog_s1_fut})
    assert list(predictions["level"].unique()) == ["s1", "s2"]
    assert len(fake_model.calls) == 2
    call_s1, call_s2 = fake_model.calls
    assert len(call_s1["contexts"]) == 1 and len(call_s2["contexts"]) == 1
    assert call_s2["past_future_covariates"] is None
    assert call_s2["past_only_covariates"] is None

    model.predict(steps=steps, levels=["s1"], exog={"s1": exog_s1_fut})
    model.predict(steps=steps, levels=["s2"], exog={"s1": exog_s1_fut})
    call_s1_alone, call_s2_alone = fake_model.calls[2:]

    np.testing.assert_array_equal(
        call_s1["past_future_covariates"][0], call_s1_alone["past_future_covariates"][0]
    )
    np.testing.assert_array_equal(call_s1["contexts"][0], call_s1_alone["contexts"][0])
    assert call_s2_alone["past_future_covariates"] is None
    np.testing.assert_array_equal(call_s2["contexts"][0], call_s2_alone["contexts"][0])


# ==============================================================================
# Tests TimesFM3Adapter.predict: quantile index mapping
# ==============================================================================
def test_TimesFM3Adapter_predict_quantile_index_mapping_uses_model_quantile_grid():
    """
    Test that quantile columns are selected by matching against the model's
    own `config.quantiles` grid rather than a fixed formula: a model whose
    quantile grid is reversed still returns the columns matching the
    requested levels.
    """
    reversed_grid = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    fake_model = FakeTimesFM3Forecaster(quantiles=reversed_grid)
    adapter = make_adapter(model=fake_model)
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


def test_TimesFM3Adapter_predict_ValueError_when_quantile_missing_from_model_grid():
    """
    Test that predict raises ValueError when a requested quantile (valid
    against SUPPORTED_QUANTILES) has no match in the model's own quantile
    grid within tolerance.
    """
    incomplete_grid = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # missing 0.5
    fake_model = FakeTimesFM3Forecaster(quantiles=incomplete_grid)
    adapter = make_adapter(model=fake_model)
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
# Tests TimesFM3Adapter._load_model
# ==============================================================================
def test_TimesFM3Adapter_load_model_LicenseWarning_and_forwards_device():
    """
    Test that _load_model issues a LicenseWarning (non-commercial license),
    forwards the resolved device to TimesFM3Forecaster.from_pretrained, and
    is a no-op once the model is loaded. The real `timesfm` module is mocked
    so no network call happens.
    """

    class _RecordingForecaster(FakeTimesFM3Forecaster):
        last_device = None

        @classmethod
        def from_pretrained(cls, model_id, device=None, **kwargs):
            cls.last_device = device
            return cls()

    mock_timesfm = types.ModuleType("timesfm")
    mock_timesfm.TimesFM3Forecaster = _RecordingForecaster

    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch", device="cpu")
        with pytest.warns(LicenseWarning, match="TimesFM Non-Commercial License"):
            adapter._load_model()
        assert isinstance(adapter._model, _RecordingForecaster)
        assert _RecordingForecaster.last_device == "cpu"

        loaded = adapter._model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter._load_model()
        assert adapter._model is loaded
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original


def test_TimesFM3Adapter_load_model_ImportError_when_timesfm_predates_3(monkeypatch):
    """
    Test that _load_model raises a clear ImportError with an upgrade hint
    when the installed `timesfm` package predates 3.0 (no
    TimesFM3Forecaster attribute and version < 3), and that no LicenseWarning
    is issued since the weights are never loaded.
    """
    import importlib.metadata as ilm

    monkeypatch.setattr(ilm, "version", lambda name: "2.5.0")
    mock_timesfm = types.ModuleType("timesfm")  # no TimesFM3Forecaster

    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = mock_timesfm
    try:
        adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
        err_msg = re.escape("TimesFM 3.0 requires `timesfm>=3.0`")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ImportError, match=err_msg):
                adapter._load_model()
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original


def test_TimesFM3Adapter_load_model_ImportError_when_torch_missing(monkeypatch):
    """
    Test that _load_model raises an ImportError pointing at the missing
    torch backend (not a misleading "upgrade timesfm") when timesfm>=3 is
    installed but `TimesFM3Forecaster` is absent because its backend
    (`timesfm3`, which imports torch) could not be imported.
    """
    import importlib.metadata as ilm

    monkeypatch.setattr(ilm, "version", lambda name: "3.0.1")
    mock_timesfm = types.ModuleType("timesfm")  # no TimesFM3Forecaster

    original = sys.modules.get("timesfm")
    original_t3 = sys.modules.get("timesfm3")
    sys.modules["timesfm"] = mock_timesfm
    sys.modules["timesfm3"] = None  # forces `import timesfm3` to raise ImportError
    try:
        adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ImportError, match="torch is missing"):
                adapter._load_model()
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original
        if original_t3 is None:
            sys.modules.pop("timesfm3", None)
        else:
            sys.modules["timesfm3"] = original_t3


def test_TimesFM3Adapter_load_model_ImportError_when_timesfm_not_installed_no_LicenseWarning():
    """
    Test that _load_model raises ImportError (with no LicenseWarning)
    when `timesfm` itself is not installed, i.e. the failure happens before
    the TimesFM3Forecaster attribute check.
    """
    original = sys.modules.get("timesfm")
    sys.modules["timesfm"] = None  # forces `import timesfm` to raise ImportError
    try:
        adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
        err_msg = re.escape("timesfm is required for TimesFM3Adapter")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ImportError, match=err_msg):
                adapter._load_model()
        assert not any(issubclass(w.category, LicenseWarning) for w in caught)
    finally:
        if original is None:
            del sys.modules["timesfm"]
        else:
            sys.modules["timesfm"] = original
