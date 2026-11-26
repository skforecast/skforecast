# Unit tests for the ci-proto adapter.
# ==============================================================================
import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from skforecast.foundation import FoundationModel, ForecasterFoundation, get_model_info
from skforecast.foundation._adapters import CIprotoAdapter
from .fixtures_adapters import FakeTorch

MODEL_ID = "ContinuousIntelligence/ci-proto"


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    """Use the same lightweight torch stand-in as the other adapter tests."""
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)


class FakeCIprotoModel:
    """Mirror the published Forecaster signature and record per-series inputs."""

    def __init__(self, quantiles=None):
        self.quantiles = np.asarray(
            CIprotoAdapter.SUPPORTED_QUANTILES if quantiles is None else quantiles,
            dtype=np.float32,
        )
        self.calls = []

    def forecast(
        self,
        history,
        horizon=320,
        *,
        past_covariates=None,
        future_covariates=None,
        categorical_past=(),
        categorical_future=(),
        timestamps=None,
        freq=None,
        series_ids=None,
        static_features=None,
    ):
        self.calls.append(
            {
                "history": history.copy(),
                "horizon": horizon,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "categorical_past": categorical_past,
                "categorical_future": categorical_future,
                "timestamps": timestamps,
                "freq": freq,
                "series_ids": series_ids,
            }
        )
        return (
            history[:, -1, None, None]
            + self.quantiles[None, :, None]
            + np.arange(horizon, dtype=np.float32)[None, None, :]
        )


def make_context(length=4, end="2026-01-04", freq="D"):
    """A regular target with a meaningful forecast origin."""
    return pd.Series(
        np.arange(1, length + 1, dtype=float),
        index=pd.date_range(end=end, periods=length, freq=freq),
        name="sales",
    )


def test_defaults_and_clone_without_backend():
    model = FoundationModel(MODEL_ID, revision="abc123")
    model.fit(make_context())
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned.adapter._model is None
    assert not cloned.is_fitted
    assert model.adapter.context_length is None
    assert model.adapter.requires_equal_length_context is False
    assert model.adapter.max_series_per_call is None


@pytest.mark.parametrize("device", ["mps", "cuda:0", None, 1])
def test_invalid_device(device):
    with pytest.raises(ValueError, match="device.*must be"):
        CIprotoAdapter(MODEL_ID, device=device)


@pytest.mark.parametrize("length", [0, -1, "12", 1.5])
def test_invalid_context_length(length):
    with pytest.raises(ValueError, match="positive integer"):
        CIprotoAdapter(MODEL_ID, context_length=length)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("model_id", MODEL_ID + "-alternative"),
        ("device", "cpu"),
        ("revision", "pinned"),
    ],
)
def test_set_params_resets_backend(parameter, value):
    adapter = CIprotoAdapter(MODEL_ID, model=FakeCIprotoModel())
    assert adapter.set_params(**{parameter: value}) is adapter
    assert adapter._model is None


def test_set_params_is_atomic_and_context_does_not_reload():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    adapter.set_params(context_length=20000)
    assert adapter._model is backend
    with pytest.raises(ValueError, match="device"):
        adapter.set_params(context_length=42, device="mps")
    assert adapter.context_length == 20000
    assert adapter._model is backend


def test_backend_load_is_lazy_cached_and_revision_pinned(monkeypatch):
    calls = []
    backend = FakeCIprotoModel()

    def load_model(checkpoint, *, device, revision):
        calls.append((checkpoint, device, revision))
        return backend

    monkeypatch.setitem(
        sys.modules, "continuousintelligence", SimpleNamespace(load_model=load_model)
    )
    adapter = CIprotoAdapter(MODEL_ID, device="cpu", revision="abc123")
    adapter.fit({"sales": make_context()}, None)
    assert calls == []
    for _ in range(2):
        adapter.predict(2, adapter.context_, None, None, None)
    assert calls == [(MODEL_ID, "cpu", "abc123")]


def test_missing_backend_has_install_hint(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "continuousintelligence":
            raise ImportError("backend not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    adapter = CIprotoAdapter(MODEL_ID)
    with pytest.raises(
        ImportError, match='pip install "continuousintelligence>=0.2.0"'
    ):
        adapter.predict(2, {"sales": make_context()}, None, None, None)


@pytest.mark.parametrize("quantiles", [None, [0.9, 0.1, 0.5]])
def test_quantile_selection_uses_loaded_grid(quantiles):
    backend = FakeCIprotoModel([0.9, 0.5, 0.1])
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    prediction = adapter.predict(2, {"sales": make_context()}, None, None, quantiles)
    expected = np.array([0.5] if quantiles is None else quantiles)
    np.testing.assert_allclose(prediction["sales"], 4 + expected[None, :] + [[0], [1]])


def test_unsupported_quantiles_fail_before_loading():
    adapter = CIprotoAdapter(MODEL_ID)
    with pytest.raises(ValueError, match="only supports quantile"):
        adapter.predict(2, {"sales": make_context()}, None, None, [0.05])
    assert adapter._model is None


def test_missing_checkpoint_quantile_fails_before_forecasting():
    backend = FakeCIprotoModel([0.1, 0.9])
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    with pytest.raises(ValueError, match="Quantile 0.5 not found"):
        adapter.predict(2, {"sales": make_context()}, None, None, None)
    assert backend.calls == []


@pytest.mark.parametrize("shape", [(1, 9, 1), (2, 9, 2), (1, 2, 9), (9, 2)])
def test_malformed_backend_output(shape):
    backend = FakeCIprotoModel()
    backend.forecast = lambda **kwargs: np.zeros(shape)
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    with pytest.raises(ValueError, match="unexpected forecast shape"):
        adapter.predict(2, {"sales": make_context()}, None, None, None)


def test_dated_unequal_contexts_keep_their_own_lengths_and_origins():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    context = {"later": make_context(), "earlier": make_context(2, "2026-01-02")}
    context["later"].iloc[1] = np.nan
    before = {name: series.copy() for name, series in context.items()}
    adapter.predict(3, context, None, None, None)
    later, earlier = backend.calls
    assert later["history"].dtype == np.float32
    assert later["history"].shape == (1, 4)
    assert earlier["history"].shape == (1, 2)
    np.testing.assert_allclose(earlier["history"][0], [1, 2])
    assert np.isnan(later["history"][0, 1])
    assert later["freq"] == earlier["freq"] == "D"
    assert later["series_ids"] is earlier["series_ids"] is None
    expected = pd.date_range("2026-01-01", periods=5, freq="D").as_unit("ns").asi8
    np.testing.assert_array_equal(earlier["timestamps"], expected)
    for name in context:
        pd.testing.assert_series_equal(context[name], before[name])


def test_range_index_does_not_fabricate_calendar_dates():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    adapter.predict(2, {"sales": pd.Series([1.0, 2.0])}, None, None, None)
    assert backend.calls[0]["timestamps"] is None
    assert backend.calls[0]["freq"] is None


def test_past_and_future_covariates_keep_each_series_history():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    context = {"long": make_context(), "short": make_context(2)}
    historical, future = {}, {}
    for name, series in context.items():
        historical[name] = pd.DataFrame(
            {
                "past": pd.array(range(len(series)), dtype="Int64"),
                "known": pd.array([True] * len(series), dtype="boolean"),
            },
            index=series.index,
        )
        future[name] = pd.DataFrame({"known": [False, True]})
    historical["short"].iloc[0, 0] = pd.NA
    adapter.predict(2, context, historical, future, None)
    long, short = backend.calls
    assert long["past_covariates"].shape == (1, 4)
    np.testing.assert_allclose(short["past_covariates"][0], [np.nan, 1])
    np.testing.assert_allclose(short["future_covariates"][0], [1, 1, 0, 1])
    assert short["categorical_past"] == ()
    assert short["categorical_future"] == ()


def test_categorical_codes_match_history_and_future_and_keep_missing_labels():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    series = make_context()
    historical = {
        "sales": pd.DataFrame(
            {
                "past": ["a", None, "a", "b"],
                "known": ["red", "blue", "red", None],
            },
            index=series.index,
        )
    }
    future = {"sales": pd.DataFrame({"known": ["blue", "new"]})}
    adapter.predict(2, {"sales": series}, historical, future, None)
    call = backend.calls[0]
    assert call["categorical_past"] == (0,)
    assert call["categorical_future"] == (0,)
    np.testing.assert_allclose(call["past_covariates"][0], [0, np.nan, 0, 1])
    np.testing.assert_allclose(call["future_covariates"][0], [0, 1, 0, np.nan, 1, 2])


def test_complex_covariates_are_rejected():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    series = make_context()
    historical = {"sales": pd.DataFrame({"x": [1j] * 4}, index=series.index)}
    with pytest.raises(ValueError, match="real values"):
        adapter.predict(2, {"sales": series}, historical, None, None)


@pytest.mark.parametrize(
    "history_values,future_values,expected",
    [
        ([1, 2, 1, 2], [2, 3], [0, 1, 0, 1, 1, 2]),
        ([False, True, False, True], [True, True], [0, 1, 0, 1, 1, 1]),
    ],
)
def test_categorical_dtype_survives_different_history_and_future_categories(
    history_values, future_values, expected
):
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    series = make_context()
    historical = {
        "sales": pd.DataFrame(
            {"known": pd.Categorical(history_values)}, index=series.index
        )
    }
    future = {"sales": pd.DataFrame({"known": pd.Categorical(future_values)})}
    adapter.predict(2, {"sales": series}, historical, future, None)
    assert backend.calls[0]["categorical_future"] == (0,)
    np.testing.assert_allclose(backend.calls[0]["future_covariates"][0], expected)


def test_FoundationModel_sends_one_target_per_backend_call():
    backend = FakeCIprotoModel()
    model = FoundationModel(MODEL_ID, model=backend)
    series = {f"s{i}": make_context() for i in range(70)}
    model.fit(series)
    predictions = model.predict(2)
    assert len(backend.calls) == 70
    assert all(call["history"].shape == (1, 4) for call in backend.calls)
    assert all(call["series_ids"] is None for call in backend.calls)
    assert len(predictions) == 140


def test_heterogeneous_covariates_are_forwarded_per_series_without_union_columns():
    backend = FakeCIprotoModel()
    model = FoundationModel(MODEL_ID, model=backend)
    series = {name: make_context() for name in ["a", "b", "bare"]}
    historical = {
        "a": pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=series["a"].index),
        "b": pd.DataFrame({"other": [5.0, 6.0, 7.0, 8.0]}, index=series["b"].index),
    }
    model.fit(series, exog=historical)
    model.predict(2)
    assert len(backend.calls) == 3
    np.testing.assert_allclose(backend.calls[0]["past_covariates"], [[1, 2, 3, 4]])
    np.testing.assert_allclose(backend.calls[1]["past_covariates"], [[5, 6, 7, 8]])
    assert backend.calls[2]["past_covariates"] is None


def test_predict_forwards_requested_horizon():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    prediction = adapter.predict(1000, {"sales": make_context()}, None, None, None)
    assert prediction["sales"].shape == (1000, 1)
    assert backend.calls[0]["horizon"] == 1000


def test_metadata_describes_the_published_model():
    info = get_model_info(MODEL_ID)
    assert info.adapter == "CIprotoAdapter"
    assert info.backend_package == "continuousintelligence>=0.2.0"
    assert info.default_context_length is None
    assert info.supports_categorical_covariates
    assert info.supports_past_only_covariates
    assert info.supports_heterogeneous_covariates
    assert info.license == "ci-proto-licence"
    assert not info.commercial_use_restricted


@pytest.mark.parametrize("method", ["predict", "predict_interval", "predict_quantiles"])
def test_forecaster_public_api_with_covariates(monkeypatch, method):
    """Verify clone, exog alignment, output dates and native uncertainty together."""
    backend = FakeCIprotoModel()
    monkeypatch.setitem(
        sys.modules,
        "continuousintelligence",
        SimpleNamespace(load_model=lambda *args, **kwargs: backend),
    )
    forecaster = ForecasterFoundation(estimator=FoundationModel(MODEL_ID))
    series = make_context()
    historical = pd.DataFrame(
        {"past": [2.0, 3.0, 4.0, 5.0], "known": ["a", "b", "a", "b"]},
        index=series.index,
    )
    future = pd.DataFrame(
        {"known": ["b", "c"]},
        index=pd.date_range("2026-01-05", periods=2, freq="D"),
    )
    forecaster.fit(series=series, exog=historical)
    kwargs = {}
    if method == "predict_interval":
        kwargs["interval"] = [0.1, 0.9]
    elif method == "predict_quantiles":
        kwargs["quantiles"] = [0.9, 0.1]
    result = getattr(forecaster, method)(steps=2, exog=future, **kwargs)
    pd.testing.assert_index_equal(result.index, future.index, check_names=False)
    column = "q_0.9" if method == "predict_quantiles" else "pred"
    np.testing.assert_allclose(
        result[column], [4.9, 5.9] if column == "q_0.9" else [4.5, 5.5]
    )
    assert backend.calls[-1]["categorical_future"] == (0,)
    assert backend.calls[-1]["past_covariates"].shape == (1, 4)


def test_predict_preserves_unequal_history_lengths():
    backend = FakeCIprotoModel()
    model = FoundationModel(MODEL_ID, model=backend, context_length=32768)
    model.fit({"short": make_context(), "long": make_context(20000)})
    model.predict(2)
    assert [call["history"].shape for call in backend.calls] == [(1, 4), (1, 20000)]


def test_monthly_dates_use_calendar_offsets():
    backend = FakeCIprotoModel()
    adapter = CIprotoAdapter(MODEL_ID, model=backend)
    series = make_context(3, end="2026-03-31", freq="ME")
    adapter.predict(2, {"sales": series}, None, None, None)
    expected = pd.DatetimeIndex(["2026-04-30", "2026-05-31"]).as_unit("ns").asi8
    np.testing.assert_array_equal(backend.calls[0]["timestamps"][-2:], expected)


@pytest.mark.parametrize("length", [20000, 2000000])
@pytest.mark.parametrize("use_context_override", [False, True])
def test_default_passes_all_history_and_covariates(length, use_context_override):
    """Neither fit nor predict overrides may truncate history at 16,384."""
    backend = FakeCIprotoModel()
    model = FoundationModel(MODEL_ID, model=backend)
    series = pd.Series(np.arange(length, dtype=float), name="sales")
    historical = pd.DataFrame({"past": series + 1, "known": series + 2})
    future = pd.DataFrame(
        {"known": [length + 2.0, length + 3.0]},
        index=pd.RangeIndex(length, length + 2),
    )
    if use_context_override:
        model.predict(
            2,
            context=series,
            context_exog=historical,
            exog=future,
        )
    else:
        model.fit(series, exog=historical)
        assert len(model.context_["sales"]) == length
        assert len(model.context_exog_["sales"]) == length
        model.predict(2, exog=future)
    call = backend.calls[0]
    assert call["history"].shape == (1, length)
    assert call["history"][0, 0] == 0
    assert call["past_covariates"].shape == (1, length)
    assert call["past_covariates"][0, 0] == 1
    assert call["future_covariates"].shape == (1, length + 2)
    assert call["future_covariates"][0, 0] == 2


def test_explicit_context_limit_is_optional_and_can_be_removed():
    backend = FakeCIprotoModel()
    model = FoundationModel(MODEL_ID, model=backend, context_length=128)
    series = make_context(20000)
    model.fit(series)
    model.predict(2)
    assert backend.calls[-1]["history"].shape == (1, 128)
    model.set_params(context_length=None)
    assert model.adapter._model is backend
    model.fit(series)
    model.predict(2)
    assert backend.calls[-1]["history"].shape == (1, 20000)


def test_forecaster_and_metadata_represent_all_history_without_a_numeric_limit():
    model = FoundationModel(MODEL_ID)
    forecaster = ForecasterFoundation(estimator=model)
    assert forecaster.context_length is None
    assert forecaster.window_size is None
    from skforecast.foundation import list_adapters

    frame = list_adapters(as_frame=True)
    assert frame.loc["CIprotoAdapter", "default_context_length"] is None
