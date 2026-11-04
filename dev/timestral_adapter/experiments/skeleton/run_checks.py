"""
Execute the skeleton of the document with the real backend:

0. Contract and parameters (no weights needed).
1. Smoke test: FoundationModel, ForecasterFoundation and backtesting_foundation.
2. exp3 heterogeneous multi-series check: one call equals each series alone.
3. Unsupported quantile.
4. Dtypes of the covariates: bool, nullable Int64, category, object, Series
   exog, RangeIndex.
5. Warning of mistralai/timestral#7.
6. T0Adapter through the patched FoundationModel (fake backend): the arrays
   sent to T0 are identical to its current inline padding, also with a
   tz-aware daily index across a DST change, a bool exog column and series
   that end on different dates in a zone with DST at midnight.
7. FakeTimestralForecaster (draft for the unit tests) with the skeleton.
8. Positional left padding: values, dtypes, no freq needed.
9. MPS fallback through ForecasterFoundation (torch.backends.mps patched).

Run from this folder after `extract_skeleton.py`, which writes
`timestral_skeleton.py`, `shared_changes.py` and `foundation_loop.txt` from
the document.

`FoundationModel.predict` is patched with the loop of the document
(`foundation_loop.txt`): its source is read, the loop over the groups is
replaced by the one of the document and the method is compiled again with
`_left_pad_context` in its globals. This runs the exact code the document
proposes for `skforecast/foundation/_foundation_model.py`.
"""
import inspect
import sys
import types
import warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import timestral_skeleton as sk
from shared_changes import _left_pad_context
from fake_timestral import FakeTimestralForecaster
from skforecast.exceptions import FeatureOutOfRangeWarning, LicenseWarning
from skforecast.foundation import (
    FoundationModel, ForecasterFoundation, _adapters, _utils,
)
from skforecast.foundation import _foundation_model
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

TimestralAdapter = sk.TimestralAdapter

# Patch FoundationModel.predict with the loop of the document.
_original_predict = FoundationModel.predict
_source = inspect.getsource(_foundation_model)
_start = _source.index("        # Adapter returns dict[str, np.ndarray] with shape (steps, n_q)")
_end = _source.index("        # Build long-format DataFrame from raw predictions")
_loop = open("foundation_loop.txt").read().rstrip("\n") + "\n\n"
_patched = types.ModuleType("_foundation_model_patched")
_patched.__dict__.update(vars(_foundation_model))
_patched.__dict__["_left_pad_context"] = _left_pad_context
exec(compile(_source[:_start] + _loop + _source[_end:], _foundation_model.__file__, "exec"),
     _patched.__dict__)
FoundationModel.predict = _patched.FoundationModel.predict
# Step A4 of the plan: the existing adapters declare the new attribute.
for _cls in set(_adapters._ADAPTER_REGISTRY.values()):
    _cls.requires_equal_length_context = _cls is _adapters.T0Adapter
_adapters._ADAPTER_REGISTRY["mistralai/Timestral"] = TimestralAdapter
_utils._MODEL_LICENSES["mistralai/Timestral"] = (
    "other",
    "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
    True,
    "Timestral Modified MIT License (no rights for companies with a monthly "
    "revenue above USD 20 million)",
)

ok = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))
    ok.append(bool(cond))


# 0. Contract and parameters
a = TimestralAdapter("mistralai/Timestral-28M")
check("get_params", a.get_params() == {
    "model_id": "mistralai/Timestral-28M", "context_length": 4096,
    "device": "auto", "predict_kwargs": None})
for bad, kwargs in [
    ("prefix", dict(model_id="mistralai/Mistral-7B")),
    ("context_length", dict(model_id="mistralai/Timestral-5M", context_length=0)),
    ("reserved", dict(model_id="mistralai/Timestral-5M",
                      predict_kwargs={"known_covariates": None})),
]:
    try:
        TimestralAdapter(**kwargs)
        check(f"init rejects {bad}", False)
    except ValueError as e:
        check(f"init rejects {bad}", True, str(e)[:100])
a._model = object()
a.set_params(context_length=512, predict_kwargs={"symmetric_averaging": False})
check("set_params keeps model", a._model is not None and a.context_length == 512)
a.set_params(device="cpu")
check("set_params device resets model", a._model is None)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dev = sk._resolve_torch_device("mps:0", mps_fallback_message="MPS not supported")
check("explicit mps warns and falls back to cpu", dev == "cpu" and len(w) == 1, str(w[0].message) if w else "")
check("explicit mps allowed by default", sk._resolve_torch_device("mps") == "mps")
saved = sys.modules.get("timestral")
sys.modules["timestral"] = None
try:
    TimestralAdapter("mistralai/Timestral-5M")._load_model()
    check("ImportError", False)
except ImportError as e:
    check("ImportError", 'pip install "timestral @ git+https://github.com/mistralai/timestral.git"' in str(e), str(e))
finally:
    if saved is not None:
        sys.modules["timestral"] = saved
    else:
        del sys.modules["timestral"]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    b = TimestralAdapter("mistralai/Timestral-28M", device="cpu")
    b._load_model()
check("LicenseWarning on load", any(issubclass(x.category, LicenseWarning) for x in w),
      next((str(x.message)[:120] for x in w if issubclass(x.category, LicenseWarning)), ""))
shared_model = b._model

# 1. Smoke test
rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=900, freq="h")
t = np.arange(900)


def make_series(n, phase):
    return pd.Series(
        100 + 20 * np.sin(2 * np.pi * (t[-n:] + phase) / 24) + rng.standard_normal(n).cumsum(),
        index=idx[-n:],
    )


y = make_series(900, 0).rename("y")
full_idx = pd.date_range(idx[0], periods=900 + 24, freq="h")
temp = pd.Series(15 + 8 * np.sin(2 * np.pi * np.arange(924) / 24), index=full_idx, name="temp")
fm = FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=512)
fm.adapter._model = shared_model
fm.fit(series=y, exog=temp.loc[: idx[-1]])
p = fm.predict(steps=24, exog=temp.loc[idx[-1]:].iloc[1:], quantiles=[0.1, 0.5, 0.9])
check("FoundationModel.predict shape", p.shape == (24, 4), str(list(p.columns)))
p0 = fm.predict(steps=24, exog=temp.loc[idx[-1]:].iloc[1:])
check("point forecast is the median", np.allclose(p0["pred"].to_numpy(), p["q_0.5"].to_numpy()),
      str(list(p0.columns)))
fc = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=512))
fc.estimator.adapter._model = shared_model
data = pd.DataFrame({"y": y, "temp": temp.loc[: idx[-1]]})
fc.fit(series=data["y"], exog=data[["temp"]])
pi = fc.predict_interval(steps=12, exog=temp.loc[idx[-1]:].iloc[1:13].to_frame())
check("ForecasterFoundation.predict_interval", pi.shape[0] == 12, str(list(pi.columns)))
cv = TimeSeriesFold(steps=24, initial_train_size=900 - 72, refit=False)
metric, preds = backtesting_foundation(
    forecaster=fc, series=data["y"], exog=data[["temp"]], cv=cv,
    metric="mean_absolute_error", quantiles=[0.1, 0.5, 0.9], show_progress=False,
)
check("backtesting_foundation", len(preds) == 72, f"MAE {metric.iloc[0, -1]:.3f}")

# 2. exp3 heterogeneous multi-series
series = {"a": make_series(900, 0), "b": make_series(400, 5), "c": make_series(700, 9)}
series["c"].iloc[100:120] = np.nan
dow = pd.Series(full_idx.day_name(), index=full_idx, name="dow").astype("category")
exog = {
    "a": pd.concat([temp, dow], axis=1),
    "b": pd.concat([temp, dow], axis=1),
    "c": temp.to_frame(),
}
hist = {k: v.loc[: idx[-1]] for k, v in exog.items()}
fut = {k: v.loc[idx[-1]:].iloc[1:] for k, v in exog.items()}
model = FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=2048)
model.adapter._model = shared_model
model.fit(series=series, exog=hist)
calls = []
inner_forecast = shared_model.forecast


def spy(*args, **kwargs):
    calls.append({k: (None if v is None else np.shape(v)) for k, v in kwargs.items()
                  if k != "prediction_length"})
    return inner_forecast(*args, **kwargs)


shared_model.forecast = spy
together = model.predict(steps=24, exog=fut, quantiles=[0.1, 0.5, 0.9])
shared_model.forecast = inner_forecast
check("one backend call per exog group", len(calls) == 2, str(calls))
max_diff = 0.0
for name in series:
    alone = FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=2048)
    alone.adapter._model = shared_model
    alone.fit(series=series[name], exog=hist[name])
    pa = alone.predict(steps=24, exog=fut[name], quantiles=[0.1, 0.5, 0.9])
    q = together.loc[together["level"] == name, ["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    max_diff = max(max_diff, np.abs(q - pa[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()).max())
check("heterogeneous batch equals each series alone", max_diff < 1e-3, f"max abs diff {max_diff:.2e}")
print("   median means:", together.groupby("level")["q_0.5"].mean().round(2).to_dict())

# 3. Unsupported quantile
try:
    model.predict(steps=24, exog=fut, quantiles=[0.33])
    check("quantile 0.33 rejected", False)
except ValueError as e:
    check("quantile 0.33 rejected", "Timestral only supports quantile levels" in str(e), str(e)[:90])
p_all = model.predict(steps=24, exog=fut, quantiles=TimestralAdapter.SUPPORTED_QUANTILES)
check("all 21 levels accepted", p_all.shape[1] == 22)

# 4. Dtypes of the covariates
n = 300
ri = pd.RangeIndex(n)
yr = pd.Series(np.sin(np.arange(n) / 4) * 10 + 50, index=ri, name="y")
exr = pd.DataFrame({
    "flag":  (np.arange(n + 12) % 7 == 0),
    "promo": pd.array(np.where(np.arange(n + 12) % 11 == 0, None, np.arange(n + 12) % 3), dtype="Int64"),
    "shop":  np.where(np.arange(n + 12) % 2 == 0, "x", "y"),
}, index=pd.RangeIndex(n + 12))
m4 = FoundationModel("mistralai/Timestral-5M", device="cpu")
m4.fit(series={"s1": yr, "s2": yr.iloc[50:].rename("s2")},
       exog={"s1": exr.iloc[:n], "s2": exr.iloc[50:n]})
p4 = m4.predict(steps=12, exog={"s1": exr.iloc[n:], "s2": exr.iloc[n:]}, quantiles=[0.5])
check("bool, Int64 with NA, object, RangeIndex, padded batch", p4["q_0.5"].notna().all() and len(p4) == 24)
m5 = FoundationModel("mistralai/Timestral-5M", device="cpu")
m5.adapter._model = m4.adapter._model
m5.fit(series=y, exog=temp.loc[: idx[-1]])
p5 = m5.predict(steps=6, exog=temp.loc[idx[-1]:].iloc[1:7])
check("exog as pandas Series", len(p5) == 6)

mixed = {"s1": exr.iloc[:n].assign(shop=1.0), "s2": exr.iloc[50:n]}
try:
    FoundationModel("mistralai/Timestral-5M", device="cpu").fit(
        series={"s1": yr, "s2": yr.iloc[50:].rename("s2")}, exog=mixed
    )
    check("FoundationModel rejects a column with different dtypes per series", False)
except TypeError as e:
    check("FoundationModel rejects a column with different dtypes per series", True, str(e)[:80])
ex_inf = exr.iloc[:n].assign(promo=np.where(np.arange(n) == 10, np.inf, 1.0))
m4c = FoundationModel("mistralai/Timestral-5M", device="cpu")
m4c.adapter._model = m4.adapter._model
m4c.fit(series=yr, exog=ex_inf[["promo"]])
p4c = m4c.predict(steps=12, exog=exr.iloc[n:][["promo"]].astype(float).fillna(0))
check("inf in a numeric covariate", p4c["pred"].notna().all())
try:
    m4d = FoundationModel("mistralai/Timestral-5M", device="cpu")
    m4d.fit(series={"ok": yr, "empty": pd.Series(np.nan, index=ri, name="empty")})
    out = m4d.predict(steps=3)
    print("   all-NaN series accepted by FoundationModel:", out.loc[out.level == "empty", "pred"].tolist())
except Exception as e:
    print("   all-NaN series rejected by FoundationModel:", type(e).__name__, str(e)[:120])

# 5. Warning of mistralai/timestral#7
yp = pd.Series(rng.poisson(2, 400).astype(float), index=pd.date_range("2020-01-01", periods=400, freq="D"))
fidx = pd.date_range(yp.index[0], periods=428, freq="D")
price = pd.Series(2.50, index=fidx, name="price")
price.iloc[405:] = 2.56
m6 = FoundationModel("mistralai/Timestral-5M", device="cpu")
m6.adapter._model = m4.adapter._model
m6.fit(series=yp, exog=price.loc[: yp.index[-1]])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m6.predict(steps=28, exog=price.loc[yp.index[-1]:].iloc[1:])
w7 = [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]
check("FeatureOutOfRangeWarning for issue #7", len(w7) == 1, str(w7[0].message)[:140] if w7 else "")
fc7 = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-5M", device="cpu"))
fc7.estimator.adapter._model = m4.adapter._model
fc7.fit(series=yp, exog=price.loc[: yp.index[-1]].to_frame())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fc7.predict(steps=28, exog=price.loc[yp.index[-1]:].iloc[1:].to_frame())
w7f = [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]
check(
    "issue #7 warning through ForecasterFoundation points to the user's call",
    len(w7f) == 1 and w7f[0].filename.endswith("run_checks.py"),
    f"{w7f[0].filename.split('/')[-1]}:{w7f[0].lineno}" if w7f else "",
)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m6.predict(steps=28, exog=pd.Series(2.50, index=fidx[400:], name="price"))
check("no warning when the future stays constant",
      not [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)])

# 6. T0Adapter through the patched FoundationModel (fake backend)
class FakeT0:
    def __init__(self):
        self.calls = []

    def predict(self, context, horizon, quantiles, future_covariates):
        self.calls.append((context.copy(), None if future_covariates is None else future_covariates.copy()))
        out = types.SimpleNamespace()
        out.quantiles = np.zeros((context.shape[0], horizon, len(quantiles)))
        return out


def t0_arrays(predict_method, series, exog_hist, exog_fut):
    fake = FakeT0()
    FoundationModel.predict = predict_method
    try:
        m = FoundationModel("theforecastingcompany/t0-alpha", model=fake)
        m.fit(series=series, exog=exog_hist)
        m.predict(steps=12, exog=exog_fut, quantiles=[0.5])
    finally:
        FoundationModel.predict = _patched.FoundationModel.predict
    return fake.calls[0]


didx = pd.date_range("2020-01-01", periods=212, freq="D", tz="Europe/Madrid")
dexog = pd.DataFrame({"flag": np.arange(212) % 7 == 0, "temp": np.sin(np.arange(212) / 5)}, index=didx)
dy = pd.Series(np.cos(np.arange(200) / 3) + 10, index=didx[:200])
t0_series = {"long": dy.rename("long"), "short": dy.iloc[120:].rename("short")}
t0_hist = {"long": dexog.iloc[:200], "short": dexog.iloc[120:200]}
t0_fut = {"long": dexog.iloc[200:], "short": dexog.iloc[200:]}
before = t0_arrays(_original_predict, t0_series, t0_hist, t0_fut)
after = t0_arrays(_patched.FoundationModel.predict, t0_series, t0_hist, t0_fut)
# Series that end on different dates in a zone with DST at midnight
# (America/Santiago): generating padding timestamps would fail there.
# A daily midnight index cannot cross 2024-09-08 there, so each series has
# its own index: "a" ends before that date and "b" starts after it.
aidx = pd.date_range("2024-04-07", periods=142, freq="D", tz="America/Santiago")
bidx = pd.date_range("2024-09-09", periods=24, freq="D", tz="America/Santiago")
aexog = pd.DataFrame({"flag": np.arange(142) % 7 == 0}, index=aidx)
bexog = pd.DataFrame({"flag": np.arange(24) % 7 == 0}, index=bidx)
s_series = {"a": pd.Series(np.arange(130.0), index=aidx[:130], name="a"),
            "b": pd.Series(np.arange(12.0), index=bidx[:12], name="b")}
s_hist = {"a": aexog.iloc[:130], "b": bexog.iloc[:12]}
s_fut = {"a": aexog.iloc[130:], "b": bexog.iloc[12:]}
before_s = t0_arrays(_original_predict, s_series, s_hist, s_fut)
after_s = t0_arrays(_patched.FoundationModel.predict, s_series, s_hist, s_fut)
check(
    "T0 arrays identical with FoundationModel padding (different end dates, DST at midnight)",
    np.array_equal(before_s[0], after_s[0], equal_nan=True)
    and np.array_equal(before_s[1], after_s[1], equal_nan=True),
    f"context {after_s[0].shape}",
)
check(
    "T0 arrays identical with FoundationModel padding (tz-aware daily across DST, bool exog)",
    np.array_equal(before[0], after[0], equal_nan=True)
    and np.array_equal(before[1], after[1], equal_nan=True)
    and np.isfinite(after[0][1, -80:]).all(),
    f"context {after[0].shape}, covariates {after[1].shape}",
)

# 7. FakeTimestralForecaster (draft for the unit tests) with the skeleton

fake_ts = FakeTimestralForecaster()
mf = FoundationModel("mistralai/Timestral-28M", model=fake_ts)
sh = {"long": yr.iloc[:250], "short": yr.iloc[200:250].rename("short"), "noexog": yr.iloc[:100].rename("noexog")}
ex_hist = {"long": exr.iloc[:250], "short": exr.iloc[200:250]}
ex_fut = {"long": exr.iloc[250:262], "short": exr.iloc[250:262]}
mf.fit(series=sh, exog=ex_hist)
pf = mf.predict(steps=12, exog=ex_fut, quantiles=[0.1, 0.5])
first = fake_ts.calls[0]
check(
    "fake: one call per signature, NaN left padding, categorical codes",
    len(fake_ts.calls) == 2
    and first["context"].shape == (2, 1, 250)
    and np.isnan(first["context"][1, 0, :200]).all()
    and first["categorical_known_covariates"].shape == (2, 1, 262)
    and np.isnan(first["categorical_known_covariates"][1, 0, :200]).all()
    and fake_ts.calls[1]["known_covariates"] is None
    and np.allclose(pf["q_0.1"], 0.1),
    f"calls {[c['context'].shape for c in fake_ts.calls]}",
)

try:
    mf.adapter.predict_kwargs = {"symmetric_averaging": False, "typo": 1}
    mf.predict(steps=12, exog=ex_fut)
    check("fake: unknown predict_kwargs raise TypeError as the backend", False)
except TypeError:
    check("fake: unknown predict_kwargs raise TypeError as the backend", True)

# 8. _left_pad_context: positional padding, values and dtypes
pc, pe = _left_pad_context(
    {"long": s_series["a"], "short": s_series["b"]},
    {"long": s_hist["a"].assign(cat=pd.Categorical(["x"] * 130)), "short": None},
)
check(
    "left padding is positional and keeps the values and dtypes",
    pc["short"].index.equals(pd.RangeIndex(130))
    and pc["long"].index.equals(pd.RangeIndex(130))
    and pc["short"].iloc[:118].isna().all()
    and np.array_equal(pc["short"].iloc[118:].to_numpy(), s_series["b"].to_numpy())
    and pe["short"] is None
    and pe["long"]["flag"].dtype == float
    and isinstance(pe["long"]["cat"].dtype, pd.CategoricalDtype),
)
pc2, _ = _left_pad_context(
    {"a": pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-05"])),
     "b": pd.Series([4.0], index=pd.DatetimeIndex(["2021-06-01"]))},
    None,
)
check("left padding works without freq", pc2["b"].tolist()[-1] == 4.0 and pc2["b"].isna().sum() == 2)

# 9. MPS fallback through ForecasterFoundation
_mps = torch.backends.mps.is_available
torch.backends.mps.is_available = lambda: True
try:
    fc_mps = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-5M", context_length=128))
    fc_mps.fit(series=y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fc_mps.predict(steps=3)
    w_mps = [x for x in w if "MPS device is not supported by Timestral" in str(x.message)]
    check(
        "auto with MPS available: warning at the user's call and model on CPU",
        len(w_mps) == 1 and w_mps[0].filename.endswith("run_checks.py")
        and str(fc_mps.estimator.adapter._model.device) == "cpu",
        f"{w_mps[0].filename.split('/')[-1]}:{w_mps[0].lineno}" if w_mps else "",
    )
finally:
    torch.backends.mps.is_available = _mps

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
