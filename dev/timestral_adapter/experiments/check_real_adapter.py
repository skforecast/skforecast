"""
Check the TimestralAdapter of the package with the real backend and weights
(point 5 of section 13 of the plan): the checks of `skeleton/run_checks.py`
that need the backend, without the skeleton nor any patch of
`FoundationModel`.

0. Contract, ImportError and LicenseWarning.
1. Smoke test: FoundationModel, ForecasterFoundation and backtesting_foundation.
2. Heterogeneous multi-series batch (lengths, exog columns, NaN): one backend
   call per exog group, and each series equals its forecast alone.
3. Quantiles: unsupported level rejected, the 21 levels accepted.
4. Covariate dtypes: bool, nullable Int64, object, category, RangeIndex,
   exog as a Series, inf.
5. Warning of mistralai/timestral#7.
6. MPS fallback through ForecasterFoundation (torch.backends.mps patched).

Run from the repository root, with `timestral` installed:

    HF_HUB_OFFLINE=1 python -W ignore dev/timestral_adapter/experiments/check_real_adapter.py

Uses the 28M and 5M checkpoints (about 2 minutes on CPU).
"""
import sys
import warnings
import numpy as np
import pandas as pd
import torch

import skforecast
from skforecast.exceptions import FeatureOutOfRangeWarning, LicenseWarning
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.foundation._adapters import TimestralAdapter
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

print("skforecast", skforecast.__version__, "from", skforecast.__file__)
print("torch", torch.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)

ok = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))
    ok.append(bool(cond))


# 0. Contract, ImportError and LicenseWarning
saved = sys.modules.get("timestral")
sys.modules["timestral"] = None
try:
    TimestralAdapter("mistralai/Timestral-5M")._load_model()
    check("ImportError without timestral", False)
except ImportError as e:
    check(
        "ImportError without timestral",
        'pip install "timestral @ git+https://github.com/mistralai/timestral.git"' in str(e),
    )
finally:
    if saved is not None:
        sys.modules["timestral"] = saved
    else:
        del sys.modules["timestral"]

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    adapter_28m = TimestralAdapter("mistralai/Timestral-28M", device="cpu")
    adapter_28m._load_model()
check(
    "LicenseWarning on load",
    any(issubclass(x.category, LicenseWarning) for x in w),
    next((str(x.message)[:110] for x in w if issubclass(x.category, LicenseWarning)), ""),
)
model_28m = adapter_28m._model
model_5m = TimestralAdapter("mistralai/Timestral-5M", device="cpu")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", LicenseWarning)
    model_5m._load_model()
model_5m = model_5m._model
check(
    "quantile levels of the checkpoint equal SUPPORTED_QUANTILES",
    np.allclose(model_28m.forecast(np.ones((1, 1, 64)), 1).quantile_levels,
                TimestralAdapter.SUPPORTED_QUANTILES),
)

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
fm = FoundationModel("mistralai/Timestral-28M", model=model_28m, context_length=512)
fm.fit(series=y, exog=temp.loc[: idx[-1]])
p = fm.predict(steps=24, exog=temp.loc[idx[-1]:].iloc[1:], quantiles=[0.1, 0.5, 0.9])
check("FoundationModel.predict shape", p.shape == (24, 4), str(list(p.columns)))
p0 = fm.predict(steps=24, exog=temp.loc[idx[-1]:].iloc[1:])
check("point forecast is the median", np.allclose(p0["pred"], p["q_0.5"]))
check(
    "quantiles ordered",
    bool((p["q_0.1"] <= p["q_0.5"]).all() and (p["q_0.5"] <= p["q_0.9"]).all()),
)
fc = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-28M", context_length=512))
fc.estimator.adapter._model = model_28m
data = pd.DataFrame({"y": y, "temp": temp.loc[: idx[-1]]})
fc.fit(series=data["y"], exog=data[["temp"]])
pi = fc.predict_interval(steps=12, exog=temp.loc[idx[-1]:].iloc[1:13].to_frame(), interval=0.9)
check("ForecasterFoundation.predict_interval(interval=0.9)", pi.shape == (12, 4), str(list(pi.columns)))
cv = TimeSeriesFold(steps=24, initial_train_size=900 - 72, refit=False)
metric, preds = backtesting_foundation(
    forecaster=fc, series=data["y"], exog=data[["temp"]], cv=cv,
    metric="mean_absolute_error", quantiles=[0.1, 0.5, 0.9], show_progress=False,
)
check("backtesting_foundation", len(preds) == 72, f"MAE {metric['mean_absolute_error'].iloc[0]:.3f}")

# 2. Heterogeneous multi-series batch
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
model = FoundationModel("mistralai/Timestral-28M", model=model_28m, context_length=2048)
model.fit(series=series, exog=hist)
calls = []
inner_forecast = model_28m.forecast


def spy(*args, **kwargs):
    calls.append({k: (None if v is None else np.shape(v)) for k, v in kwargs.items()
                  if k != "prediction_length"})
    return inner_forecast(*args, **kwargs)


model_28m.forecast = spy
together = model.predict(steps=24, exog=fut, quantiles=[0.1, 0.5, 0.9])
model_28m.forecast = inner_forecast
check("one backend call per exog group", len(calls) == 2, str(calls))
max_diff = 0.0
for name in series:
    alone = FoundationModel("mistralai/Timestral-28M", model=model_28m, context_length=2048)
    alone.fit(series=series[name], exog=hist[name])
    pa = alone.predict(steps=24, exog=fut[name], quantiles=[0.1, 0.5, 0.9])
    q = together.loc[together["level"] == name, ["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    max_diff = max(max_diff, np.abs(q - pa[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()).max())
check("heterogeneous batch equals each series alone", max_diff < 1e-3, f"max abs diff {max_diff:.2e}")

# 3. Quantiles
try:
    model.predict(steps=24, exog=fut, quantiles=[0.33])
    check("quantile 0.33 rejected", False)
except ValueError as e:
    check("quantile 0.33 rejected", "Timestral only supports quantile levels" in str(e))
p_all = model.predict(steps=24, exog=fut, quantiles=TimestralAdapter.SUPPORTED_QUANTILES)
check("all 21 levels accepted", p_all.shape[1] == 22)

# 4. Covariate dtypes
n = 300
ri = pd.RangeIndex(n)
yr = pd.Series(np.sin(np.arange(n) / 4) * 10 + 50, index=ri, name="y")
exr = pd.DataFrame({
    "flag":  (np.arange(n + 12) % 7 == 0),
    "promo": pd.array(np.where(np.arange(n + 12) % 11 == 0, None, np.arange(n + 12) % 3), dtype="Int64"),
    "shop":  np.where(np.arange(n + 12) % 2 == 0, "x", "y"),
}, index=pd.RangeIndex(n + 12))
m4 = FoundationModel("mistralai/Timestral-5M", model=model_5m)
m4.fit(series={"s1": yr, "s2": yr.iloc[50:].rename("s2")},
       exog={"s1": exr.iloc[:n], "s2": exr.iloc[50:n]})
p4 = m4.predict(steps=12, exog={"s1": exr.iloc[n:], "s2": exr.iloc[n:]}, quantiles=[0.5])
check("bool, Int64 with NA, object, RangeIndex, padded batch", p4["q_0.5"].notna().all() and len(p4) == 24)
m5 = FoundationModel("mistralai/Timestral-5M", model=model_5m)
m5.fit(series=y, exog=temp.loc[: idx[-1]])
p5 = m5.predict(steps=6, exog=temp.loc[idx[-1]:].iloc[1:7])
check("exog as pandas Series", len(p5) == 6)
ex_inf = exr.iloc[:n].assign(promo=np.where(np.arange(n) == 10, np.inf, 1.0))
m4c = FoundationModel("mistralai/Timestral-5M", model=model_5m)
m4c.fit(series=yr, exog=ex_inf[["promo"]])
p4c = m4c.predict(steps=12, exog=exr.iloc[n:][["promo"]].astype(float).fillna(0))
check("inf in a numeric covariate", p4c["pred"].notna().all())

# 5. Warning of mistralai/timestral#7
yp = pd.Series(rng.poisson(2, 400).astype(float), index=pd.date_range("2020-01-01", periods=400, freq="D"))
fidx = pd.date_range(yp.index[0], periods=428, freq="D")
price = pd.Series(2.50, index=fidx, name="price")
price.iloc[405:] = 2.56
fc7 = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-5M"))
fc7.estimator.adapter._model = model_5m
fc7.fit(series=yp, exog=price.loc[: yp.index[-1]].to_frame())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    p7 = fc7.predict_quantiles(steps=28, exog=price.loc[yp.index[-1]:].iloc[1:].to_frame(),
                               quantiles=[0.5, 0.99])
w7 = [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]
check(
    "issue #7 warning through ForecasterFoundation points to the user's call",
    len(w7) == 1 and w7[0].filename == __file__,
    f"q_0.99 max {p7['q_0.99'].max():.1f}",
)
hist_nan = price.loc[: yp.index[-1]].to_frame() * np.nan
fc7.fit(series=yp, exog=hist_nan)
p7n = fc7.predict_quantiles(steps=28, exog=price.loc[yp.index[-1]:].iloc[1:].to_frame(),
                            quantiles=[0.5, 0.99])
check("historical values to NaN avoid it", p7n["q_0.99"].max() < 100, f"q_0.99 max {p7n['q_0.99'].max():.1f}")

# 6. MPS fallback through ForecasterFoundation
_mps, _cuda = torch.backends.mps.is_available, torch.cuda.is_available
torch.backends.mps.is_available = lambda: True
torch.cuda.is_available = lambda: False
try:
    fc_mps = ForecasterFoundation(estimator=FoundationModel("mistralai/Timestral-5M", context_length=128))
    fc_mps.fit(series=y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fc_mps.predict(steps=3)
    w_mps = [x for x in w if "MPS device is not supported by Timestral" in str(x.message)]
    check(
        "auto with MPS available: warning at the user's call and model on CPU",
        len(w_mps) == 1 and w_mps[0].filename == __file__
        and str(fc_mps.estimator.adapter._model.device) == "cpu",
    )
finally:
    torch.backends.mps.is_available, torch.cuda.is_available = _mps, _cuda

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
