# Unit test NoriAdapter
# ==============================================================================
import numpy as np
import pandas as pd
import pytest

from skforecast.foundation._adapters import NoriAdapter


class _FakeNori:
    """Stand-in for NoriRegressor: records shapes, returns deterministic output."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X, *, output_type="mean", quantiles=None):
        n = len(X)
        if quantiles is None:
            fill = {"mean": self.y_.mean(), "median": np.median(self.y_),
                    "mode": self.y_.mean()}[output_type]
            return np.full(n, fill)
        return np.tile(np.asarray(quantiles, dtype=float).reshape(-1, 1), (1, n))


def _series(n=50, freq="D"):
    return pd.Series(np.arange(n, dtype=float),
                     index=pd.date_range("2020-01-01", periods=n, freq=freq))


def test_point_forecast_shape():
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori()).fit({"s": _series()}, None)
    out = a.predict(7, {"s": _series()}, None, None, None)
    assert out["s"].shape == (7, 1)


def test_quantile_forecast_shape_and_transpose():
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori()).fit({"s": _series()}, None)
    out = a.predict(7, {"s": _series()}, None, None, [0.1, 0.5, 0.9])
    assert out["s"].shape == (7, 3)
    assert np.allclose(out["s"][0], [0.1, 0.5, 0.9])


def test_multiple_series():
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori())
    ctx = {"a": _series(30), "b": _series(40)}
    a.fit(ctx, None)
    out = a.predict(5, ctx, None, None, None)
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (5, 1) and out["b"].shape == (5, 1)


def test_quantiles_out_of_range_raise():
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori()).fit({"s": _series()}, None)
    with pytest.raises(ValueError):
        a.predict(7, {"s": _series()}, None, None, [0.0, 0.5])
    with pytest.raises(ValueError):
        a.predict(7, {"s": _series()}, None, None, [0.5, 1.0])


def test_invalid_constructor_params():
    with pytest.raises(ValueError):
        NoriAdapter("x", context_length=0)
    with pytest.raises(ValueError):
        NoriAdapter("x", point_estimate="bad")
    with pytest.raises(ValueError):
        NoriAdapter("x", n_fourier_terms=-1)


def test_missing_backend_raises_importerror(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori":
            raise ImportError("no module named synthefy_nori")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    a = NoriAdapter("Synthefy/Nori").fit({"s": _series()}, None)
    with pytest.raises(ImportError):
        a.predict(3, {"s": _series()}, None, None, None)


def test_get_set_params_roundtrip_and_reset():
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori())
    assert a.get_params()["context_length"] == 4096
    assert a.get_params()["nori_config"] is None
    a.set_params(context_length=1024, point_estimate="median")
    assert a.get_params()["context_length"] == 1024
    assert a.get_params()["point_estimate"] == "median"
    assert a._model is None


def test_set_params_invalid_key():
    a = NoriAdapter("Synthefy/Nori")
    with pytest.raises(ValueError):
        a.set_params(not_a_param=1)


def test_rangeindex_and_known_future_exog():
    s = pd.Series(np.arange(30, dtype=float))  # RangeIndex
    ctx_x = pd.DataFrame({"temp": np.arange(30, dtype=float), "drop_me": np.zeros(30)})
    fut_x = pd.DataFrame({"temp": np.arange(30, 35, dtype=float)})
    fake = _FakeNori()
    a = NoriAdapter("Synthefy/Nori", model=fake).fit({"s": s}, {"s": ctx_x})
    out = a.predict(5, {"s": s}, {"s": ctx_x}, {"s": fut_x}, None)
    assert out["s"].shape == (5, 1)
    # running index (1) + Fourier 2 terms (4) + known-future 'temp' (1); 'drop_me' excluded
    assert fake.n_features_in_ == 1 + 4 + 1


def test_datetime_calendar_features_present():
    fake = _FakeNori()
    s = _series(60)
    a = NoriAdapter("Synthefy/Nori", model=fake, n_fourier_terms=1).fit({"s": s}, None)
    a.predict(3, {"s": s}, None, None, None)
    assert fake.n_features_in_ == 1 + 6 + 4  # index + calendar(6) + fourier 1 term(4)


def test_rangeindex_warns():
    s = pd.Series(np.arange(20, dtype=float))
    a = NoriAdapter("Synthefy/Nori", model=_FakeNori()).fit({"s": s}, None)
    with pytest.warns(UserWarning):
        a.predict(3, {"s": s}, None, None, None)
