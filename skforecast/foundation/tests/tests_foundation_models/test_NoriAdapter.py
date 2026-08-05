# Unit test NoriAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import NoriAdapter
from .fixtures_adapters import (
    y,
    exog,
    y_dict,
    y_range,
    FakeNoriRegressor,
    prepare_fit_args,
    prepare_predict_args,
)


# ==============================================================================
# Tests NoriAdapter.__init__
# ==============================================================================
def test_NoriAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori")
    assert adapter.model_id == "Synthefy/Nori"
    assert adapter.context_length == 4096
    assert adapter.point_estimate == "mean"
    assert adapter.add_calendar_features is True
    assert adapter.n_fourier_terms == 2
    assert adapter.nori_config == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert NoriAdapter.allow_exog is True


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}",
)
def test_NoriAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(
        ValueError, match=re.escape("`context_length` must be a positive integer")
    ):
        NoriAdapter(model_id="Synthefy/Nori", context_length=context_length)


def test_NoriAdapter_init_ValueError_when_point_estimate_invalid():
    """
    Test that __init__ rejects an unknown point_estimate.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("`point_estimate` must be 'mean', 'median' or 'mode'"),
    ):
        NoriAdapter(model_id="Synthefy/Nori", point_estimate="bad")


def test_NoriAdapter_init_ValueError_when_add_calendar_features_not_bool():
    """
    Test that __init__ rejects a non-bool add_calendar_features.
    """
    with pytest.raises(
        ValueError, match=re.escape("`add_calendar_features` must be a bool")
    ):
        NoriAdapter(model_id="Synthefy/Nori", add_calendar_features="yes")


@pytest.mark.parametrize(
    "n_fourier_terms",
    [-1, 1.5, "x"],
    ids=lambda n: f"n_fourier_terms={n}",
)
def test_NoriAdapter_init_ValueError_when_n_fourier_terms_invalid(n_fourier_terms):
    """
    Test that __init__ rejects non-negative-integer n_fourier_terms.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("`n_fourier_terms` must be a non-negative integer"),
    ):
        NoriAdapter(model_id="Synthefy/Nori", n_fourier_terms=n_fourier_terms)


# ==============================================================================
# Tests NoriAdapter.get_params / set_params
# ==============================================================================
def test_NoriAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys, and nori_config is
    reported as None when no extra config was provided.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", context_length=1024)
    assert adapter.get_params() == {
        "model_id":              "Synthefy/Nori",
        "context_length":        1024,
        "point_estimate":        "mean",
        "add_calendar_features": True,
        "n_fourier_terms":       2,
        "nori_config":           None,
    }


def test_NoriAdapter_set_params_updates_values_and_roundtrips():
    """
    Test that set_params updates parameters and get_params reflects them.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    adapter.set_params(
        context_length=512, point_estimate="median", n_fourier_terms=0
    )
    params = adapter.get_params()
    assert params["context_length"] == 512
    assert params["point_estimate"] == "median"
    assert params["n_fourier_terms"] == 0


def test_NoriAdapter_set_params_resets_model_only_on_loader_keys():
    """
    Test that set_params clears the loaded model when a key baked into the
    NoriRegressor changes (nori_config) but keeps it for featurization-only
    keys (context_length, point_estimate, n_fourier_terms).
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    adapter.set_params(context_length=256, point_estimate="mode", n_fourier_terms=1)
    assert adapter._model is not None
    adapter.set_params(nori_config={"device": "cpu"})
    assert adapter._model is None


def test_NoriAdapter_set_params_no_reset_when_loader_key_value_unchanged():
    """
    Test that set_params does not reset the model when a loader key
    (model_id, nori_config) is set to its current, unchanged value.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    adapter.set_params(model_id="Synthefy/Nori")  # same as current value
    assert adapter._model is not None


def test_NoriAdapter_set_params_ValueError_on_invalid_key():
    """
    Test that set_params raises ValueError for unknown parameters.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori")
    with pytest.raises(
        ValueError, match=re.escape("Invalid parameter(s) for NoriAdapter")
    ):
        adapter.set_params(not_a_param=1)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        (
            {"point_estimate": "bad"},
            "`point_estimate` must be 'mean', 'median' or 'mode'",
        ),
        ({"add_calendar_features": "yes"}, "`add_calendar_features` must be a bool"),
        ({"n_fourier_terms": -1}, "`n_fourier_terms` must be a non-negative integer"),
    ],
    ids=["context_length", "point_estimate", "add_calendar_features", "n_fourier_terms"],
)
def test_NoriAdapter_set_params_ValueError_on_invalid_value(params, match):
    """
    Test that set_params validates parameter values.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


# ==============================================================================
# Tests NoriAdapter.fit
# ==============================================================================
def test_NoriAdapter_fit_stores_context():
    """
    Test that fit stores the context and exog dicts and flips is_fitted.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori")
    context, context_exog = prepare_fit_args(y, exog)
    adapter.fit(context, context_exog)
    assert adapter.is_fitted is True
    assert "sales" in adapter.context_
    assert "sales" in adapter.context_exog_


# ==============================================================================
# Tests NoriAdapter.predict
# ==============================================================================
def test_NoriAdapter_predict_point_forecast_shape():
    """
    Test that a point forecast returns one (steps, 1) array per series and
    forwards point_estimate to the backend.
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(
        model_id="Synthefy/Nori", model=fake, point_estimate="median"
    )
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=7)
    preds = adapter.predict(
        steps=7, context=ctx, context_exog=ctx_exog, exog=future_exog, quantiles=None
    )
    assert preds["sales"].shape == (7, 1)
    assert fake.last_output_type == "median"


def test_NoriAdapter_predict_quantiles_shape_and_column_order():
    """
    Test that quantile forecasts return (steps, n_quantiles) with columns in
    the requested (unsorted) order, and that the backend is queried with
    sorted, unique levels.
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=5)
    preds = adapter.predict(
        steps=5,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=[0.9, 0.1, 0.5],
    )
    assert preds["sales"].shape == (5, 3)
    # The fake sets each column equal to its (sorted) quantile level; the
    # adapter must reorder columns back to the requested [0.9, 0.1, 0.5].
    np.testing.assert_allclose(preds["sales"][0], [0.9, 0.1, 0.5])
    # Nori was queried with sorted, unique levels.
    assert fake.last_quantiles == [0.1, 0.5, 0.9]


def test_NoriAdapter_predict_multiple_series():
    """
    Test that predict returns one array per series in multi-series mode.
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=fake)
    context, context_exog = prepare_fit_args(y_dict)
    adapter.fit(context, context_exog)

    preds = adapter.predict(
        steps=4,
        context=adapter.context_,
        context_exog=adapter.context_exog_,
        exog=None,
        quantiles=None,
    )
    assert set(preds) == {"s1", "s2"}
    assert preds["s1"].shape == (4, 1)
    assert preds["s2"].shape == (4, 1)


@pytest.mark.parametrize(
    "quantiles",
    [[0.0, 0.5], [0.5, 1.0], [-0.1], [1.2]],
    ids=lambda q: f"quantiles={q}",
)
def test_NoriAdapter_predict_ValueError_when_quantiles_out_of_range(quantiles):
    """
    Test that quantiles outside the open interval (0, 1) raise ValueError.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    context, _ = prepare_fit_args(y)
    adapter.fit(context, None)
    with pytest.raises(ValueError, match=re.escape("must lie strictly in (0, 1)")):
        adapter.predict(3, adapter.context_, None, None, quantiles)


def test_NoriAdapter_predict_datetime_builds_calendar_and_fourier_features():
    """
    Test the feature count for a DatetimeIndex series: running index (1) +
    calendar features (6) + Fourier harmonics (4 per term).
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=fake, n_fourier_terms=1)
    context, _ = prepare_fit_args(y)
    adapter.fit(context, None)
    adapter.predict(3, adapter.context_, None, None, None)
    assert fake.n_features_in_ == 1 + 6 + 4


def test_NoriAdapter_predict_rangeindex_uses_index_and_fourier_only():
    """
    Test that RangeIndex series skip calendar features (running index +
    Fourier(index) only) and that known-future exog columns are appended
    while context-only columns are dropped.
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=fake)
    ctx_exog = pd.DataFrame(
        {"temp": np.arange(50, dtype=float), "drop_me": np.zeros(50)}
    )
    fut_exog = pd.DataFrame({"temp": np.arange(50, 55, dtype=float)})
    adapter.fit({"sales": y_range}, {"sales": ctx_exog})
    preds = adapter.predict(
        5, {"sales": y_range}, {"sales": ctx_exog}, {"sales": fut_exog}, None
    )
    assert preds["sales"].shape == (5, 1)
    # running index (1) + Fourier 2 terms (4) + known-future 'temp' (1)
    assert fake.n_features_in_ == 1 + 4 + 1


def test_NoriAdapter_predict_warns_on_rangeindex():
    """
    Test that a RangeIndex series with calendar features enabled issues a
    single warning (calendar features skipped).
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    adapter.fit({"sales": y_range}, None)
    with pytest.warns(UserWarning, match="non-DatetimeIndex"):
        adapter.predict(3, {"sales": y_range}, None, None, None)


def test_NoriAdapter_predict_ValueError_on_non_numeric_covariate():
    """
    Test that a non-numeric known-future covariate raises a clear ValueError.
    """
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=FakeNoriRegressor())
    ctx_exog = pd.DataFrame({"cat": ["a"] * 50})
    fut_exog = pd.DataFrame({"cat": ["a", "b", "c"]})
    adapter.fit({"sales": y_range}, {"sales": ctx_exog})
    with pytest.raises(
        ValueError, match=re.escape("NoriAdapter supports only numeric covariates")
    ):
        adapter.predict(
            3, {"sales": y_range}, {"sales": ctx_exog}, {"sales": fut_exog}, None
        )


def test_NoriAdapter_predict_handles_device_tensor_output():
    """
    Test that a model returning a torch-like tensor (detach / cpu / numpy) is
    converted to a numpy array instead of failing.
    """

    class _TorchLike:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=float)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _TensorNori(FakeNoriRegressor):
        def predict(self, X, *, output_type="mean", quantiles=None):
            out = super().predict(X, output_type=output_type, quantiles=quantiles)
            return _TorchLike(out)

    adapter = NoriAdapter(model_id="Synthefy/Nori", model=_TensorNori())
    adapter.fit({"sales": y}, None)
    preds = adapter.predict(4, {"sales": y}, None, None, None)
    assert isinstance(preds["sales"], np.ndarray)
    assert preds["sales"].shape == (4, 1)


# ==============================================================================
# Tests NoriAdapter._load_model
# ==============================================================================
def test_NoriAdapter_load_model_noop_when_already_set():
    """
    Test that _load_model is a no-op when a model is already present.
    """
    fake = FakeNoriRegressor()
    adapter = NoriAdapter(model_id="Synthefy/Nori", model=fake)
    adapter._load_model()
    assert adapter._model is fake


def test_NoriAdapter_load_model_ImportError_when_backend_missing(monkeypatch):
    """
    Test that predict raises ImportError with an install hint when the
    synthefy-nori backend is not installed.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori":
            raise ImportError("no module named synthefy_nori")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    adapter = NoriAdapter(model_id="Synthefy/Nori")
    adapter.fit({"sales": y}, None)
    with pytest.raises(ImportError, match=re.escape("synthefy-nori is required")):
        adapter.predict(3, {"sales": y}, None, None, None)
