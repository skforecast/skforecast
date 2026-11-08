# Unit test T0Adapter
# ==============================================================================
import re
import sys
import types
from importlib import metadata
from unittest.mock import Mock, call
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import T0Adapter
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import (
    y, exog,
    FakeT0Forecaster,
    prepare_fit_args, prepare_predict_args
)


# ==============================================================================
# Tests T0Adapter.__init__
# ==============================================================================
def test_T0Adapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    assert adapter.model_id == "theforecastingcompany/t0-alpha"
    assert adapter.context_length == 8192
    assert adapter.device_map == "auto"
    assert adapter.torch_dtype is None
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert T0Adapter.allow_exog is True


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_T0Adapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        T0Adapter(model_id="theforecastingcompany/t0-alpha", context_length=context_length)


# ==============================================================================
# Tests T0Adapter.get_params / set_params
# ==============================================================================
def test_T0Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", context_length=512)
    assert adapter.get_params() == {
        'model_id':       "theforecastingcompany/t0-alpha",
        'context_length': 512,
        'device_map':     "auto",
        'torch_dtype':    None,
    }


def test_T0Adapter_set_params_resets_model_on_loader_keys():
    """
    Test that set_params clears the loaded model when a key baked into the
    model (model_id, device_map, torch_dtype) changes.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=FakeT0Forecaster())
    adapter.set_params(context_length=256)
    assert adapter._model is not None
    adapter.set_params(device_map="cpu")
    assert adapter._model is None
    assert adapter.context_length == 256
    assert adapter.device_map == "cpu"


def test_T0Adapter_set_params_no_reset_when_value_unchanged():
    """
    Test that set_params does not clear the loaded model when reset keys are
    set to their current values (no actual change).
    """
    adapter = T0Adapter(
        model_id="theforecastingcompany/t0-alpha", model=FakeT0Forecaster()
    )
    assert adapter._model is not None

    adapter.set_params(
        model_id="theforecastingcompany/t0-alpha",
        device_map="auto",
        torch_dtype=None,
    )

    assert adapter._model is not None  # not reset because values unchanged


def test_T0Adapter_set_params_ValueError_on_invalid_key():
    """
    Test that set_params raises ValueError for unknown parameters.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    with pytest.raises(ValueError, match=re.escape("Invalid parameter(s) for T0Adapter")):
        adapter.set_params(not_a_param=1)


# ==============================================================================
# Tests T0Adapter.fit
# ==============================================================================
def test_T0Adapter_fit_stores_context():
    """
    Test that fit stores the context and exog dicts and flips is_fitted.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    context, context_exog = prepare_fit_args(y, exog)
    adapter.fit(context, context_exog)
    assert adapter.is_fitted is True
    assert "sales" in adapter.context_
    assert "sales" in adapter.context_exog_


# ==============================================================================
# Tests T0Adapter.predict
# ==============================================================================
@pytest.mark.parametrize("quantiles", [[0.9, 0.1, 0.5], [0.9, 0.1, 0.9, 0.5]])
def test_T0Adapter_predict_returns_steps_by_n_quantiles(quantiles):
    """
    Test that predict returns one (steps, n_quantiles) array per series with
    columns ordered to match the requested quantiles.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=5)
    preds = adapter.predict(
        steps=5, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=quantiles,
    )

    assert preds["sales"].shape == (5, len(quantiles))
    # FakeT0Forecast sets each column equal to its (sorted) quantile level;
    # Return columns in the caller's order, including repeated levels.
    np.testing.assert_allclose(preds["sales"][0], quantiles)
    # T0 was queried with sorted, unique levels.
    assert fake.last_quantiles == [0.1, 0.5, 0.9]


def test_T0Adapter_predict_none_quantiles_returns_median():
    """
    Test that quantiles=None produces a single median (0.5) column.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=4)
    preds = adapter.predict(
        steps=4, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=None,
    )

    assert preds["sales"].shape == (4, 1)
    assert fake.last_quantiles == [0.5]


def test_T0Adapter_predict_builds_future_covariates_from_past_and_future():
    """
    Test that exog past values (context_exog) and future values (exog) are
    concatenated into T0's [1, n_covariates, context_length + steps] stream.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y, exog)
    adapter.fit(context, context_exog)

    steps = 6
    future_exog_values = exog.iloc[:steps].copy()
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps=steps, context=y, context_exog=exog, exog=future_exog_values
    )
    adapter.predict(
        steps=steps, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.5],
    )

    fc = fake.last_future_covariates
    context_length = len(y)
    assert fc.shape == (1, 2, context_length + steps)
    # Context portion equals the historical exog; horizon portion the future.
    np.testing.assert_allclose(fc[0, 0, :context_length], exog["feat_a"].to_numpy())
    np.testing.assert_allclose(fc[0, 0, context_length:], future_exog_values["feat_a"].to_numpy())


def test_T0Adapter_build_future_covariates_dedups_columns_preserving_order():
    """
    Test that _build_future_covariates pools covariate columns across series
    in first-seen order, deduplicating repeated column names.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    steps = 2
    context_length = 3
    idx = pd.date_range("2020-01-01", periods=steps, freq="D")
    exog_future = {
        "s1": pd.DataFrame({"a": [10.0, 10.0], "b": [20.0, 20.0]}, index=idx),
        "s2": pd.DataFrame({"b": [21.0, 21.0], "c": [30.0, 30.0]}, index=idx),
    }

    result = adapter._build_future_covariates(
        series_names=["s1", "s2"],
        context_exog=None,
        exog=exog_future,
        context_length=context_length,
        steps=steps,
    )

    # Pooled columns in first-seen order are (a, b, c): 3 covariates, not 4.
    assert result.shape == (2, 3, context_length + steps)
    # Future values land in the horizon region at the expected column slot
    # (a -> 0, b -> 1, c -> 2), missing columns stay NaN.
    np.testing.assert_array_equal(result[0, 0, context_length:], [10.0, 10.0])
    np.testing.assert_array_equal(result[0, 1, context_length:], [20.0, 20.0])
    assert np.all(np.isnan(result[0, 2, context_length:]))
    np.testing.assert_array_equal(result[1, 1, context_length:], [21.0, 21.0])
    np.testing.assert_array_equal(result[1, 2, context_length:], [30.0, 30.0])
    assert np.all(np.isnan(result[1, 0, context_length:]))


def test_T0Adapter_predict_batches_all_series_in_one_call():
    """
    Test that all series are forecast in a single batched call, rather than
    looped over one at a time, and that shorter series arrive left-padded
    with NaN by FoundationModel (forecast origin aligned at the end).
    """
    fake = FakeT0Forecaster()
    model = FoundationModel("theforecastingcompany/t0-alpha", model=fake)

    long = pd.Series(np.arange(30, dtype=float),
                     index=pd.date_range("2020-01-01", periods=30, freq="ME"), name="long")
    short = pd.Series(np.arange(20, dtype=float),
                      index=pd.date_range("2020-01-01", periods=20, freq="ME"), name="short")
    model.fit(series={"long": long, "short": short})

    preds = model.predict(steps=5)

    # One batched call, batch dimension covers both series.
    assert fake.last_context.shape == (2, 30)
    # Shorter series is left-padded: NaN at the front, data flush to the end.
    assert np.isnan(fake.last_context[1, :10]).all()
    np.testing.assert_allclose(fake.last_context[1, 10:], short.to_numpy())
    assert list(preds["level"].unique()) == ["long", "short"]
    assert preds.shape == (10, 2)


def test_T0Adapter_predict_aligns_unequal_contexts_and_covariates():
    """
    Test shared padding preserves NaNs and aligns covariates at the horizon.
    """
    fake = FakeT0Forecaster()
    model = FoundationModel("theforecastingcompany/t0-alpha", model=fake)
    index = pd.date_range("2024-01-01", periods=8, freq="h")
    series = {
        "long": pd.Series([1., 2., 3., 4., 5., 6.], index=index[:6]),
        "short": pd.Series([10., np.nan, 12.], index=index[3:6]),
    }
    past = {
        "long": pd.DataFrame({"a": [1., 2., 3., 4., 5., 6.]}, index=index[:6]),
        "short": pd.DataFrame({"b": [20., 21., 22.]}, index=index[3:6]),
    }
    future = {
        "long": pd.DataFrame({"a": [7., 8.]}, index=index[6:]),
        "short": pd.DataFrame({"b": [23., 24.]}, index=index[6:]),
    }
    model.fit(series=series, exog=past)

    predictions = model.predict(steps=2, exog=future)

    np.testing.assert_array_equal(
        fake.last_context,
        [[1., 2., 3., 4., 5., 6.],
         [np.nan, np.nan, np.nan, 10., np.nan, 12.]],
    )
    assert fake.last_future_covariates.shape == (2, 2, 8)
    np.testing.assert_array_equal(
        fake.last_future_covariates[0, 0], [1., 2., 3., 4., 5., 6., 7., 8.]
    )
    np.testing.assert_array_equal(
        fake.last_future_covariates[1, 1],
        [np.nan, np.nan, np.nan, 20., 21., 22., 23., 24.],
    )
    assert np.isnan(fake.last_future_covariates[0, 1]).all()
    assert np.isnan(fake.last_future_covariates[1, 0]).all()
    np.testing.assert_array_equal(predictions["pred"], [0.5, 0.5, 0.5, 0.5])


def test_T0Adapter_predict_no_exog_passes_none_covariates():
    """
    Test that a series without future exog is forecast with no covariates.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.5],
    )
    assert fake.last_future_covariates is None


def test_T0Adapter_predict_ValueError_on_non_numeric_covariate():
    """
    Test that a non-numeric future covariate raises a clear ValueError.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    steps = 3
    bad_exog = pd.DataFrame({"feat_str": ["a", "b", "c"]})
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps=steps, context=y, exog=bad_exog
    )
    with pytest.raises(ValueError, match=re.escape("T0Adapter supports only numeric covariates")):
        adapter.predict(
            steps=steps, context=ctx, context_exog=ctx_exog, exog=future_exog,
            quantiles=[0.5],
        )


# ==============================================================================
# Tests T0Adapter._load_model
# ==============================================================================
def test_T0Adapter_load_model_ImportError_when_backend_missing(monkeypatch):
    """
    Test that a missing runtime gives a shell-safe installation command.
    """
    monkeypatch.setitem(sys.modules, "t0", None)
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")

    with pytest.raises(
        ImportError, match=re.escape('pip install "tfc-t0[torch]>=0.6"')
    ):
        adapter._load_model()
    assert adapter._model is None


@pytest.mark.parametrize("installed", ["0.3.2", "0.5.0", "0.6.0rc1"])
def test_T0Adapter_load_model_ImportError_when_backend_too_old(
    monkeypatch, installed
):
    """
    Test that unsupported versions fail before attempting to load weights.
    """
    loader = Mock()
    monkeypatch.setitem(
        sys.modules, "t0", types.SimpleNamespace(T0Forecaster=loader)
    )
    monkeypatch.setattr(metadata, "version", lambda name: installed)
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", device_map="cpu")

    with pytest.raises(
        ImportError, match=re.escape(f"tfc-t0 {installed} is installed")
    ):
        adapter._load_model()
    loader.from_pretrained.assert_not_called()
    assert adapter._model is None


@pytest.mark.parametrize("installed", ["0.6.0", "0.10.0", "1.0.0"])
def test_T0Adapter_load_model_output_when_backend_supported(monkeypatch, installed):
    """
    Test supported versions load the checkpoint and apply device and dtype.
    """
    loaded = Mock()
    loaded.to.return_value = loaded
    loaded.eval.return_value = loaded
    loader = Mock()
    loader.from_pretrained.return_value = loaded
    monkeypatch.setitem(
        sys.modules, "t0", types.SimpleNamespace(T0Forecaster=loader)
    )
    monkeypatch.setattr(metadata, "version", lambda name: installed)
    adapter = T0Adapter(
        model_id="theforecastingcompany/t0-alpha",
        device_map="cpu",
        torch_dtype="test_dtype",
    )

    adapter._load_model()

    loader.from_pretrained.assert_called_once_with(
        "theforecastingcompany/t0-alpha", dtype="test_dtype"
    )
    assert adapter._model is loaded
    assert loaded.to.call_args_list == [call("cpu")]
    loaded.eval.assert_called_once_with()


def test_T0Adapter_load_model_noop_when_already_set():
    """
    Test that _load_model is a no-op when a model is already present.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    adapter._load_model()
    assert adapter._model is fake


def test_T0Adapter_load_model_OSError_when_from_pretrained_raises_TypeError(monkeypatch):
    """
    Test that _load_model converts a TypeError raised by
    T0Forecaster.from_pretrained into a clear OSError. This mirrors the real
    failure mode: huggingface_hub's from_pretrained silently swallows a
    config download failure and falls back to
    instantiating the model with no constructor arguments, raising a
    confusing TypeError about missing hyperparameters. The `t0` package is
    faked so the test does not depend on `tfc-t0` being installed.
    """
    class FakeFailingT0Forecaster:
        @classmethod
        def from_pretrained(cls, model_id, *, dtype=None):
            raise TypeError(
                "T0Forecaster.__init__() missing 8 required positional "
                "arguments: 'embed_dim', 'num_layers', 'num_heads', "
                "'mlp_hidden_dim', 'patch_size', 'group_every_n', 'dropout', "
                "and 'quantile_levels'"
            )

    fake_t0_module = types.ModuleType("t0")
    fake_t0_module.T0Forecaster = FakeFailingT0Forecaster
    monkeypatch.setitem(sys.modules, "t0", fake_t0_module)
    monkeypatch.setattr(metadata, "version", lambda name: "0.6.0")

    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    err_msg = re.escape(
        "Could not load model 'theforecastingcompany/t0-alpha' from the "
        "Hugging Face Hub. Its configuration could not be downloaded: "
        "check the model ID and the network connection."
    )
    with pytest.raises(OSError, match=err_msg):
        adapter._load_model()
    assert adapter._model is None
