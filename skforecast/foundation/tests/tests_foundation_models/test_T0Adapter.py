# Unit test T0Adapter
# ==============================================================================
import re
import sys
import types
import importlib.metadata
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
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
    assert T0Adapter.supports_heterogeneous_covariates is False
    assert T0Adapter.requires_equal_length_context is True
    assert T0Adapter.backend_package == "tfc-t0[torch]>=0.6"


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
def test_T0Adapter_predict_returns_steps_by_n_quantiles():
    """
    Test that predict returns one (steps, n_quantiles) array per series with
    columns ordered to match the requested quantiles, and that T0 is asked to
    read the leading NaN of a series as padding.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=5)
    preds = adapter.predict(
        steps=5, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.9, 0.1, 0.5],
    )

    assert preds["sales"].shape == (5, 3)
    # FakeT0Forecast sets each column equal to its (sorted) quantile level;
    # the adapter must reorder columns back to the requested [0.9, 0.1, 0.5].
    np.testing.assert_allclose(preds["sales"][0], [0.9, 0.1, 0.5])
    # T0 was queried with sorted, unique levels.
    assert fake.last_quantiles == [0.1, 0.5, 0.9]
    assert fake.last_trim_leading_missing is True


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


def test_T0Adapter_build_future_covariates_uses_columns_of_first_series():
    """
    Test that _build_future_covariates takes the covariate columns, and their
    order, from the first series, matches the columns of the other series by
    name, places the historical values flush to the forecast origin and
    leaves NaN where there are no historical values.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    steps = 2
    context_length = 3
    idx = pd.RangeIndex(context_length)
    idx_future = pd.date_range("2020-01-04", periods=steps, freq="D")
    context_exog = {
        "s1": pd.DataFrame(
            {"b": [np.nan, 2.0, 3.0], "a": [np.nan, 5.0, 6.0]}, index=idx
        ),
        "s2": None,
    }
    exog_future = {
        "s1": pd.DataFrame({"b": [10.0, 11.0], "a": [20.0, 21.0]}, index=idx_future),
        "s2": pd.DataFrame({"a": [30.0, 31.0], "b": [40.0, 41.0]}, index=idx_future),
    }

    result = adapter._build_future_covariates(
        series_names=["s1", "s2"],
        context_exog=context_exog,
        exog=exog_future,
        context_length=context_length,
        steps=steps,
    )

    expected = np.array(
        [
            [[np.nan, 2.0, 3.0, 10.0, 11.0], [np.nan, 5.0, 6.0, 20.0, 21.0]],
            [[np.nan, np.nan, np.nan, 40.0, 41.0], [np.nan, np.nan, np.nan, 30.0, 31.0]],
        ],
        dtype=np.float32,
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_T0Adapter_predict_batches_all_series_in_one_call():
    """
    Test that series with the same covariates are forecast in a single
    batched call, rather than looped over one at a time, and that shorter
    series arrive left-padded with NaN by FoundationModel (forecast origin
    aligned at the end), with T0 asked to read that padding as padding.
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
    assert len(fake.calls) == 1
    assert fake.last_context.shape == (2, 30)
    assert fake.last_trim_leading_missing is True
    # Shorter series is left-padded: NaN at the front, data flush to the end.
    assert np.isnan(fake.last_context[1, :10]).all()
    np.testing.assert_allclose(fake.last_context[1, 10:], short.to_numpy())
    assert list(preds["level"].unique()) == ["long", "short"]
    assert preds.shape == (10, 2)


def test_T0Adapter_predict_one_call_per_covariate_signature():
    """
    Test that series with different covariate columns are forecast in
    separate backend calls (supports_heterogeneous_covariates=False), so that
    a series never receives the covariates of another one as rows of missing
    values: the series without exog is called with no covariates.
    """
    fake = FakeT0Forecaster()
    model = FoundationModel("theforecastingcompany/t0-alpha", model=fake)

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    idx_future = pd.date_range("2020-01-31", periods=5, freq="D")
    series = {
        "with_exog": pd.Series(np.arange(30, dtype=float), index=idx),
        "no_exog": pd.Series(np.arange(30, dtype=float), index=idx),
        "other_exog": pd.Series(np.arange(30, dtype=float), index=idx),
    }
    model.fit(
        series=series,
        exog={
            "with_exog": pd.DataFrame({"a": np.arange(30, dtype=float)}, index=idx),
            "no_exog": None,
            "other_exog": pd.DataFrame({"z": np.arange(30, dtype=float)}, index=idx),
        },
    )

    preds = model.predict(
        steps=5,
        exog={
            "with_exog": pd.DataFrame({"a": np.arange(5, dtype=float)}, index=idx_future),
            "no_exog": None,
            "other_exog": pd.DataFrame({"z": np.arange(5, dtype=float)}, index=idx_future),
        },
    )

    assert len(fake.calls) == 3
    assert [call["context"].shape for call in fake.calls] == [(1, 30)] * 3
    assert fake.calls[0]["future_covariates"].shape == (1, 1, 35)
    assert fake.calls[1]["future_covariates"] is None
    assert fake.calls[2]["future_covariates"].shape == (1, 1, 35)
    assert list(preds["level"].unique()) == ["with_exog", "no_exog", "other_exog"]
    assert preds.shape == (15, 2)


@pytest.mark.parametrize(
    "exog_columns",
    [["feat_a"], None],
    ids=lambda exog_columns: f"exog_columns: {exog_columns}",
)
def test_T0Adapter_predict_past_only_exog_removed_by_FoundationModel(exog_columns):
    """
    Test that, through FoundationModel, the historical exog columns without
    future values are not part of the covariates sent to the backend: only
    the column with future values is, and no covariates at all when
    predicting without future exog.
    """
    steps = 3
    fake = FakeT0Forecaster()
    model = FoundationModel("theforecastingcompany/t0-alpha", model=fake)
    model.fit(series=y, exog=exog)
    future_index = pd.date_range(
        y.index[-1] + y.index.freq, periods=steps, freq=y.index.freq
    )
    future_exog = (
        pd.DataFrame({"feat_a": np.arange(steps, dtype=float)}, index=future_index)
        if exog_columns is not None else None
    )

    with pytest.warns(IgnoredArgumentWarning, match="are ignored for series"):
        model.predict(steps=steps, exog=future_exog)

    fc = fake.last_future_covariates
    if exog_columns is None:
        assert fc is None
    else:
        assert fc.shape == (1, 1, len(y) + steps)
        np.testing.assert_allclose(fc[0, 0, :len(y)], exog["feat_a"].to_numpy())
        np.testing.assert_allclose(fc[0, 0, len(y):], np.arange(steps, dtype=float))


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
def test_T0Adapter_load_model_noop_when_already_set():
    """
    Test that _load_model is a no-op when a model is already present.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    adapter._load_model()
    assert adapter._model is fake


def test_T0Adapter_load_model_ImportError_when_tfc_t0_not_installed(monkeypatch):
    """
    Test that _load_model raises an ImportError with the quoted install
    command when the `t0` package cannot be imported.
    """
    monkeypatch.setitem(sys.modules, "t0", None)  # forces `import t0` to fail

    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    err_msg = re.escape(
        "tfc-t0 is required for T0Adapter. Install it with "
        '`pip install "tfc-t0[torch]>=0.6"`.'
    )
    with pytest.raises(ImportError, match=err_msg):
        adapter._load_model()
    assert adapter._model is None


@pytest.mark.parametrize(
    "installed",
    ["0.3.2", "0.5.0"],
    ids=lambda installed: f"tfc-t0: {installed}"
)
def test_T0Adapter_load_model_ImportError_when_tfc_t0_older_than_0_6(
    monkeypatch, installed
):
    """
    Test that _load_model raises an ImportError asking to upgrade when the
    installed `tfc-t0` predates 0.6, whose `T0Forecaster.predict` does not
    accept the arguments used by the adapter. The model is never loaded.
    """

    class FakeOldT0Forecaster:
        @classmethod
        def from_pretrained(cls, model_id, **model_kwargs):
            raise AssertionError("The model must not be loaded.")

    fake_t0_module = types.ModuleType("t0")
    fake_t0_module.T0Forecaster = FakeOldT0Forecaster
    monkeypatch.setitem(sys.modules, "t0", fake_t0_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)

    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    err_msg = re.escape(
        f"T0Adapter requires `tfc-t0>=0.6`, but tfc-t0 {installed} is "
        f'installed. Upgrade with `pip install -U "tfc-t0[torch]>=0.6"`.'
    )
    with pytest.raises(ImportError, match=err_msg):
        adapter._load_model()
    assert adapter._model is None


def test_T0Adapter_load_model_ImportError_when_torch_runtime_missing(monkeypatch):
    """
    Test that _load_model raises an ImportError pointing at the PyTorch
    runtime (not a misleading "tfc-t0 is required") when tfc-t0>=0.6 is
    installed without its `torch` extra: `t0` imports, but `T0Forecaster`
    does not.
    """
    fake_t0_module = types.ModuleType("t0")  # no T0Forecaster
    monkeypatch.setitem(sys.modules, "t0", fake_t0_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.6.0")

    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    err_msg = (
        re.escape("tfc-t0 is installed but its PyTorch runtime could not be imported (")
        + ".*"
        + re.escape('). Install it with `pip install "tfc-t0[torch]>=0.6"`.')
    )
    with pytest.raises(ImportError, match=err_msg):
        adapter._load_model()
    assert adapter._model is None


@pytest.mark.parametrize(
    "installed, torch_dtype, expected_kwargs",
    [
        ("0.6.0", None, {}),
        ("0.6.1", "bfloat16", {"dtype": "bfloat16"}),
        ("0.10.0rc1", "float16", {"dtype": "float16"}),
        (None, None, {}),
    ],
    ids=lambda value: f"{value}"
)
def test_T0Adapter_load_model_passes_torch_dtype_to_from_pretrained(
    monkeypatch, installed, torch_dtype, expected_kwargs
):
    """
    Test that _load_model accepts tfc-t0>=0.6 (also a pre-release, and a
    package without metadata), passes `torch_dtype` to
    `T0Forecaster.from_pretrained` as `dtype` and only moves the model to the
    device: casting the weights with `model.to(dtype)` makes T0 fail at
    predict time.
    """

    class FakeLoadedT0Forecaster:
        def __init__(self, model_id, model_kwargs):
            self.model_id = model_id
            self.model_kwargs = model_kwargs
            self.to_args = []
            self.is_eval = False

        @classmethod
        def from_pretrained(cls, model_id, **model_kwargs):
            return cls(model_id, model_kwargs)

        def to(self, target):
            self.to_args.append(target)
            return self

        def eval(self):
            self.is_eval = True
            return self

    def fake_version(name):
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    fake_t0_module = types.ModuleType("t0")
    fake_t0_module.T0Forecaster = FakeLoadedT0Forecaster
    monkeypatch.setitem(sys.modules, "t0", fake_t0_module)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    adapter = T0Adapter(
        model_id="theforecastingcompany/t0-alpha",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    adapter._load_model()

    assert isinstance(adapter._model, FakeLoadedT0Forecaster)
    assert adapter._model.model_id == "theforecastingcompany/t0-alpha"
    assert adapter._model.model_kwargs == expected_kwargs
    assert adapter._model.to_args == ["cpu"]
    assert adapter._model.is_eval is True


def test_T0Adapter_load_model_OSError_when_from_pretrained_raises_TypeError(monkeypatch):
    """
    Test that _load_model converts a TypeError raised by
    T0Forecaster.from_pretrained into a clear OSError. This mirrors the real
    failure mode: huggingface_hub's from_pretrained silently swallows a
    config download failure and falls back to
    instantiating the model with no constructor arguments, raising a
    confusing TypeError about missing hyperparameters. The `t0` package and
    its version are faked so the test does not depend on `tfc-t0` being
    installed.
    """

    class FakeFailingT0Forecaster:
        @classmethod
        def from_pretrained(cls, model_id, **model_kwargs):
            raise TypeError(
                "T0Forecaster.__init__() missing 8 required positional "
                "arguments: 'embed_dim', 'num_layers', 'num_heads', "
                "'mlp_hidden_dim', 'patch_size', 'group_every_n', 'dropout', "
                "and 'quantile_levels'"
            )

    fake_t0_module = types.ModuleType("t0")
    fake_t0_module.T0Forecaster = FakeFailingT0Forecaster
    monkeypatch.setitem(sys.modules, "t0", fake_t0_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.6.0")

    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    err_msg = re.escape(
        "Could not load model 'theforecastingcompany/t0-alpha' from the "
        "Hugging Face Hub. Its configuration could not be downloaded: "
        "check the model ID and the network connection."
    )
    with pytest.raises(OSError, match=err_msg):
        adapter._load_model()
    assert adapter._model is None
