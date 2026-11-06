"""
Independent re-check of the NoriAdapter review findings.

Checks (no timing, no accuracy comparison):
A. Routing of Synthefy repos by `_resolve_adapter` and failure mode of
   `Synthefy/Nori-Causal` through FoundationModel and NoriRegressor.
B. Adapter (FoundationModel) vs direct `NoriRegressor` on an independently
   built feature matrix, with a different data set from the investigator
   (two series, weekly + daily seasonality, two covariates), for
   `Synthefy/Nori` and `Synthefy/Nori-1-5`; registry name vs raw repo id.
C. Backend output types; quantile rows with unsorted and duplicated levels;
   `mode` vs `median`; stored inference config in each checkpoint.
D. NaN in covariates: rows the current adapter passes to `fit`, backend
   acceptance of NaN in X, and the "keep NaN covariate rows" variant vs a
   direct backend call.
E. Default device chosen by the backend on this machine.

Run from the repository root:
    ~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/verification/verify_nori.py
"""

import contextlib
import io
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import NoriAdapter, _resolve_adapter

warnings.simplefilter("ignore")

print("python         :", platform.python_version())
print("platform       :", platform.platform())
for pkg in ["synthefy-nori", "synthefy", "torch", "numpy", "pandas", "huggingface-hub", "skforecast"]:
    print(f"{pkg:15s}:", version(pkg))
print()

from synthefy_nori import NoriRegressor  # noqa: E402

# A. Routing ------------------------------------------------------------------
print("== A. routing")
for repo in ["Synthefy/Nori", "Synthefy/Nori-1-5", "Synthefy/Nori-30M",
             "Synthefy/Nori-100M", "Synthefy/Nori-Causal", "Synthefy/migas-1.5"]:
    try:
        print(f"  {repo:22s} -> {_resolve_adapter(repo).__name__}")
    except ValueError as exc:
        print(f"  {repo:22s} -> ValueError: {str(exc)[:60]}")

# Data ------------------------------------------------------------------------
N, STEPS, CTX = 350, 12, 256
rng = np.random.default_rng(123)
idx = pd.date_range("2025-03-01", periods=N + STEPS, freq="D")
t = np.arange(len(idx))
exog_all = pd.DataFrame(
    {
        "x1": np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, len(idx)),
        "x2": (idx.dayofweek >= 5).astype(float),
    },
    index=idx,
)
y_all = pd.Series(
    20 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 30)
    + 1.5 * exog_all["x1"].to_numpy() - 2 * exog_all["x2"].to_numpy()
    + rng.normal(0, 0.4, len(idx)),
    index=idx, name="s1",
)
y, exog, exog_fut = y_all.iloc[:N], exog_all.iloc[:N], exog_all.iloc[N:]
ctx_y, ctx_exog = y.iloc[-CTX:], exog.iloc[-CTX:]


def features(ts, offset, block):
    """Independent re-implementation of the adapter default features."""
    n = len(ts)
    doy = np.asarray(ts.dayofyear, dtype=float)
    dow = np.asarray(ts.dayofweek, dtype=float)
    cols = [np.arange(offset, offset + n, dtype=float),
            np.asarray(ts.month, dtype=float), np.asarray(ts.day, dtype=float), dow, doy,
            np.asarray(ts.quarter, dtype=float), np.asarray(ts.hour, dtype=float)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25),
                 np.sin(2 * np.pi * k * dow / 7.0), np.cos(2 * np.pi * k * dow / 7.0)]
    cols += [block["x1"].to_numpy(dtype=float), block["x2"].to_numpy(dtype=float)]
    return np.column_stack(cols).astype(np.float32)


X_ctx = features(ctx_y.index, 0, ctx_exog)
X_fut = features(exog_fut.index, CTX, exog_fut)
y_ctx = ctx_y.to_numpy(dtype=float)
QS = [0.8, 0.2, 0.5, 0.05]

# B and C ---------------------------------------------------------------------
for model_id, name in [("Synthefy/Nori", "nori-6m"), ("Synthefy/Nori-1-5", "nori-1-5")]:
    print(f"\n== B/C. {model_id}")
    for pe in ["mean", "median", "mode"]:
        fm = FoundationModel(model_id=model_id, context_length=CTX, point_estimate=pe)
        fm.fit(series=y, exog=exog)
        p_ad = fm.predict(steps=STEPS, exog=exog_fut)["pred"].to_numpy(dtype=float)
        reg = NoriRegressor(model=model_id).fit(X_ctx, y_ctx)
        p_dir = reg.predict(X_fut, output_type=pe)
        print(f"  point_estimate={pe:6s} max|adapter - direct| = {np.max(np.abs(p_ad - p_dir)):.3e}")
    fm = FoundationModel(model_id=model_id, context_length=CTX)
    fm.fit(series=y, exog=exog)
    q_ad = fm.predict(steps=STEPS, exog=exog_fut, quantiles=QS)
    q_ad = q_ad[[f"q_{q}" for q in QS]].to_numpy(dtype=float)
    reg = NoriRegressor(model=model_id).fit(X_ctx, y_ctx)
    q_dir = reg.predict(X_fut, output_type="quantiles", quantiles=QS)
    print(f"  quantiles {QS}: max|adapter - direct.T| = {np.max(np.abs(q_ad - q_dir.T)):.3e}")
    p_mean = reg.predict(X_fut)
    print(f"  backend point: {type(p_mean).__name__} {p_mean.dtype} {p_mean.shape}; "
          f"quantiles: {type(q_dir).__name__} {q_dir.dtype} {q_dir.shape}")
    q_sorted = reg.predict(X_fut, output_type="quantiles", quantiles=sorted(QS))
    reorder = [sorted(QS).index(q) for q in QS]
    print(f"  unsorted levels returned in requested order: {np.array_equal(q_dir, q_sorted[reorder])}")
    dup = [0.5, 0.9, 0.5, 0.1]
    q_dup = reg.predict(X_fut, output_type="quantiles", quantiles=dup)
    q_ref = reg.predict(X_fut, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    print(f"  duplicated levels {dup}: shape {q_dup.shape}, rows match "
          f"{np.array_equal(q_dup, q_ref[[1, 2, 1, 0]])}")
    med = reg.predict(X_fut, output_type="median")
    mode = reg.predict(X_fut, output_type="mode")
    pred = reg._get_predictor()
    stored = getattr(pred._bare_model(), "stored_inference_config", None)
    print(f"  regression_head={pred.regression_head}; mode == median: {np.array_equal(mode, med)}")
    print(f"  stored inference config in checkpoint: "
          f"{None if stored is None else stored.get('name')}")
    print(f"  device_ chosen with device=None: {reg.device_}")
    reg_name = NoriRegressor(model=name).fit(X_ctx, y_ctx)
    print(f"  registry name {name!r} vs raw id: point diff "
          f"{np.max(np.abs(reg_name.predict(X_fut) - p_mean)):.3e}, quantile diff "
          f"{np.max(np.abs(reg_name.predict(X_fut, output_type='quantiles', quantiles=QS) - q_dir)):.3e}")

# D. NaN in covariates ------------------------------------------------------------
print("\n== D. NaN in covariates (Synthefy/Nori)")
exog_nan = exog.copy()
nan_mask = rng.random(N) < 0.3
exog_nan.loc[nan_mask, "x1"] = np.nan
n_nan_ctx = int(exog_nan["x1"].iloc[-CTX:].isna().sum())
print(f"  context rows with NaN in x1: {n_nan_ctx} of {CTX}")

seen = {}
orig_fit = NoriRegressor.fit


def spy_fit(self, X, yy):
    seen["rows"] = len(X)
    seen["nan_in_X"] = bool(np.isnan(np.asarray(X, dtype=float)).any())
    return orig_fit(self, X, yy)


NoriRegressor.fit = spy_fit
fm = FoundationModel(model_id="Synthefy/Nori", context_length=CTX)
fm.fit(series=y, exog=exog_nan)
p_cur = fm.predict(steps=STEPS, exog=exog_fut)["pred"].to_numpy(dtype=float)
print(f"  current adapter: rows passed to fit = {seen['rows']}, NaN in X = {seen['nan_in_X']}")

# Variant: keep rows with NaN covariates, only drop NaN target rows.
orig_isnan = np.isnan


class _KeepNanCovariates:
    """Patch NoriAdapter.predict so valid_rows ignores NaN in the features."""

    def __enter__(self):
        import skforecast.foundation._adapters as mod
        self.mod = mod
        self.orig = mod.NoriAdapter.predict
        src_np = mod.np

        class _NpProxy:
            def __getattr__(self, item):
                return getattr(src_np, item)

            @staticmethod
            def isnan(a):
                arr = np.asarray(a)
                if arr.ndim == 2:
                    return np.zeros_like(arr, dtype=bool)
                return orig_isnan(arr)

        self.src_np = src_np
        mod.np = _NpProxy()
        return self

    def __exit__(self, *exc):
        self.mod.np = self.src_np


with _KeepNanCovariates():
    fm2 = FoundationModel(model_id="Synthefy/Nori", context_length=CTX)
    fm2.fit(series=y, exog=exog_nan)
    p_keep = fm2.predict(steps=STEPS, exog=exog_fut)["pred"].to_numpy(dtype=float)
print(f"  keep-NaN variant: rows passed to fit = {seen['rows']}, NaN in X = {seen['nan_in_X']}")
NoriRegressor.fit = orig_fit

X_ctx_nan = features(ctx_y.index, 0, exog_nan.iloc[-CTX:])
reg = NoriRegressor(model="Synthefy/Nori").fit(X_ctx_nan, y_ctx)
p_dir = reg.predict(X_fut)
print(f"  keep-NaN variant vs direct backend call on X with NaN: max diff "
      f"{np.max(np.abs(p_keep - p_dir)):.3e}; finite {np.isfinite(p_dir).all()}")
valid = ~np.isnan(X_ctx_nan).any(axis=1)
reg_drop = NoriRegressor(model="Synthefy/Nori").fit(X_ctx_nan[valid], y_ctx[valid])
print(f"  current adapter vs direct call on the rows without NaN: max diff "
      f"{np.max(np.abs(p_cur - reg_drop.predict(X_fut))):.3e}")
X_fut_nan = X_fut.copy()
X_fut_nan[::3, -2] = np.nan
q = reg.predict(X_fut_nan, output_type="quantiles", quantiles=[0.1, 0.9])
print(f"  NaN in query rows accepted: finite quantiles {np.isfinite(q).all()}")
try:
    yy = y_ctx.copy()
    yy[3] = np.nan
    NoriRegressor(model="Synthefy/Nori").fit(X_ctx, yy)
    print("  NaN in y: accepted")
except ValueError as exc:
    print(f"  NaN in y: ValueError: {exc}")

# A (continued). Nori-Causal ------------------------------------------------------
print("\n== A2. Synthefy/Nori-Causal")
fm = FoundationModel(model_id="Synthefy/Nori-Causal", context_length=CTX)
print(f"  FoundationModel adapter: {type(fm.adapter).__name__} (construction ok)")
fm.fit(series=y)
try:
    fm.predict(steps=STEPS)
except Exception as exc:
    print(f"  predict raised {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")

# E. stdout noise ---------------------------------------------------------------
print("\n== E. stdout printed by a fresh model load on CPU")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    NoriAdapter("Synthefy/Nori")._load_model()
    r = NoriRegressor(model="Synthefy/Nori").fit(X_ctx, y_ctx)
    r.predict(X_fut[:2])
print(f"  captured stdout: {buf.getvalue().strip()!r}")
