"""
Compare NoriAdapter (through FoundationModel) with direct calls to the backend
`synthefy_nori.NoriRegressor` on the same input, and probe the backend
behaviours the adapter relies on.

Checks:
1. Routing of every Synthefy checkpoint on the Hub by `_resolve_adapter`.
2. For the smallest checkpoint of each family (Nori 6M and Nori 1.5): point and
   quantile forecasts of the adapter vs NoriRegressor fitted on an
   independently built feature matrix with the same features; registry name
   (`nori-6m`, `nori-1-5`) vs raw repo id (`Synthefy/Nori`, `Synthefy/Nori-1-5`).
3. Output types and shapes returned by the backend, order of the quantile rows
   when the levels are not sorted, `mode` vs `median` on pinball heads,
   determinism of two consecutive calls.
4. NaN in the feature matrix (context and query rows) accepted by the backend.
5. `Synthefy/Nori-Causal` through FoundationModel.

Run from the repository root:
    ~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/compare_direct.py
"""

import platform
import traceback
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import _resolve_adapter

warnings.simplefilter("ignore")

print("python         :", platform.python_version())
print("platform       :", platform.platform())
print("cpu count      :", torch.get_num_threads(), "torch threads")
for pkg in ["synthefy-nori", "synthefy", "torch", "numpy", "pandas", "huggingface-hub", "skforecast"]:
    print(f"{pkg:15s}:", version(pkg))
print()

# 1. Routing ------------------------------------------------------------------
print("== 1. _resolve_adapter on the Synthefy Hub checkpoints")
for repo in [
    "Synthefy/Nori",
    "Synthefy/Nori-1-5",
    "Synthefy/Nori-30M",
    "Synthefy/Nori-100M",
    "Synthefy/Nori-Causal",
    "Synthefy/migas-1.5",
]:
    try:
        print(f"  {repo:22s} -> {_resolve_adapter(repo).__name__}")
    except ValueError as exc:
        print(f"  {repo:22s} -> ValueError: {str(exc)[:80]}")
print()

# Data --------------------------------------------------------------------------
N, STEPS, CTX = 400, 24, 300
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=N + STEPS, freq="h")
hour = idx.hour.to_numpy()
exog_all = pd.DataFrame({"temp": np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(idx))}, index=idx)
y_all = pd.Series(
    50 + 0.02 * np.arange(len(idx)) + 5 * np.sin(2 * np.pi * hour / 24)
    + 2 * exog_all["temp"].to_numpy() + rng.normal(0, 0.5, len(idx)),
    index=idx, name="y",
)
y, exog, exog_fut = y_all.iloc[:N], exog_all.iloc[:N], exog_all.iloc[N:]
ctx_y, ctx_exog = y.iloc[-CTX:], exog.iloc[-CTX:]


def features(ts: pd.DatetimeIndex, offset: int, exog_block: pd.DataFrame) -> np.ndarray:
    """Independent re-implementation of the adapter default features."""
    n = len(ts)
    doy = ts.dayofyear.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)
    cols = [
        np.arange(offset, offset + n, dtype=float),
        ts.month.to_numpy(dtype=float), ts.day.to_numpy(dtype=float), dow, doy,
        ts.quarter.to_numpy(dtype=float), ts.hour.to_numpy(dtype=float),
    ]
    for k in (1, 2):
        cols += [
            np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25),
            np.sin(2 * np.pi * k * dow / 7.0), np.cos(2 * np.pi * k * dow / 7.0),
        ]
    cols.append(exog_block["temp"].to_numpy(dtype=float))
    return np.column_stack(cols).astype(np.float32)


X_ctx = features(ctx_y.index, 0, ctx_exog)
X_fut = features(exog_fut.index, CTX, exog_fut)
y_ctx = ctx_y.to_numpy(dtype=float)

from synthefy_nori import NoriRegressor  # noqa: E402

QS = [0.9, 0.1, 0.5]

for model_id, registry_name in [("Synthefy/Nori", "nori-6m"), ("Synthefy/Nori-1-5", "nori-1-5")]:
    print(f"== 2-3. {model_id} (registry name {registry_name!r})")
    try:
        fm = FoundationModel(model_id=model_id, context_length=CTX)
        fm.fit(series=y, exog=exog)
        p_ad = fm.predict(steps=STEPS, exog=exog_fut)["pred"].to_numpy(dtype=float)
        q_ad = fm.predict(steps=STEPS, exog=exog_fut, quantiles=QS)
        q_ad = q_ad[[f"q_{q}" for q in QS]].to_numpy(dtype=float)

        reg = NoriRegressor(model=registry_name, device="cpu").fit(X_ctx, y_ctx)
        p_dir = reg.predict(X_fut)
        p_dir2 = reg.predict(X_fut)
        q_dir = reg.predict(X_fut, output_type="quantiles", quantiles=QS)
        q_dir_sorted = reg.predict(X_fut, output_type="quantiles", quantiles=sorted(QS))
        med = reg.predict(X_fut, output_type="median")
        mode = reg.predict(X_fut, output_type="mode")
        head = reg._get_predictor().regression_head

        reg_raw = NoriRegressor(model=model_id, device="cpu").fit(X_ctx, y_ctx)
        p_raw = reg_raw.predict(X_fut)
        q_raw = reg_raw.predict(X_fut, output_type="quantiles", quantiles=QS)

        print(f"  backend point type/dtype/shape : {type(p_dir).__name__} {p_dir.dtype} {p_dir.shape}")
        print(f"  backend quantiles type/shape   : {type(q_dir).__name__} {q_dir.dtype} {q_dir.shape} (levels {QS})")
        print(f"  regression_head                : {head}")
        print(f"  device_                        : {reg.device_}")
        print(f"  max|adapter - direct| point    : {np.max(np.abs(p_ad - p_dir)):.3e}")
        print(f"  max|adapter - direct| quantiles: {np.max(np.abs(q_ad - q_dir.T)):.3e}")
        print(f"  max|registry - raw repo| point : {np.max(np.abs(p_dir - p_raw)):.3e}")
        print(f"  max|registry - raw repo| quant : {np.max(np.abs(q_dir - q_raw)):.3e}")
        print(f"  determinism max|call1 - call2| : {np.max(np.abs(p_dir - p_dir2)):.3e}")
        reorder = [sorted(QS).index(q) for q in QS]
        print(f"  unsorted levels keep order     : {np.allclose(q_dir, q_dir_sorted[reorder])}")
        print(f"  mode == median                 : {np.array_equal(mode, med)}")
        print(f"  q0.5 == median (point)         : max diff {np.max(np.abs(q_dir[2] - med)):.3e}")
        print(f"  first 3 steps adapter point    : {np.round(p_ad[:3], 4).tolist()}")
    except Exception:
        traceback.print_exc()
    print()

# 4. NaN in features ---------------------------------------------------------
print("== 4. NaN in the feature matrix (nori-6m)")
try:
    Xc = X_ctx.copy()
    Xf = X_fut.copy()
    Xc[::10, -1] = np.nan
    Xf[::5, -1] = np.nan
    reg = NoriRegressor(model="nori-6m", device="cpu").fit(Xc, y_ctx)
    p = reg.predict(Xf)
    q = reg.predict(Xf, output_type="quantiles", quantiles=[0.1, 0.9])
    print(f"  fit/predict with NaN in X: ok, finite point={np.isfinite(p).all()}, finite quantiles={np.isfinite(q).all()}")
except Exception as exc:
    print(f"  NaN in X raised {type(exc).__name__}: {str(exc)[:200]}")
try:
    yc = y_ctx.copy()
    yc[5] = np.nan
    NoriRegressor(model="nori-6m", device="cpu").fit(X_ctx, yc)
    print("  NaN in y: accepted")
except Exception as exc:
    print(f"  NaN in y raised {type(exc).__name__}: {str(exc)[:200]}")
print()

# 5. Nori-Causal ---------------------------------------------------------------
print("== 5. FoundationModel('Synthefy/Nori-Causal')")
try:
    fm = FoundationModel(model_id="Synthefy/Nori-Causal", context_length=CTX)
    print("  adapter:", type(fm.adapter).__name__)
    fm.fit(series=y)
    fm.predict(steps=STEPS)
    print("  predict: ok (unexpected)")
except Exception as exc:
    print(f"  predict raised {type(exc).__name__}: {str(exc).splitlines()[0][:250]}")
