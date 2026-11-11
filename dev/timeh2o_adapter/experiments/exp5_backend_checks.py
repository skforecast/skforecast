"""
Experiment 5: checks of the h2ofm backend called directly (v1 checkpoint).

1. NaN in the context (inside, at the start) and a context that is all NaN,
   also through `FoundationModel` with the prototype.
2. float32 against float64 input, and two identical calls.
3. Maximum context: a series longer than 16384 points against its last
   16384 points, and the `context_length` option above the maximum.
4. Long horizons: one pass (`autoregressive=False`, the default of the
   checkpoint) against `autoregressive=True`, and the time per call.
5. Dependence of the forecast on `horizon`, without covariates and with a
   future covariate.
6. `batch_size`: same forecasts, different time.
7. Devices: `"mps"`, `"cuda"` without CUDA, `"auto"`.
8. `clip_nonnegative` and `snap_integer` on counts, and `mean` against the
   median.

Usage: HF_HUB_OFFLINE=1 python exp5_backend_checks.py > exp5.log 2>&1
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from h2ofm import TimeH2O

sys.path.insert(0, ".")
from common import hourly_series, versions  # noqa: E402
from timeh2o_proto import register  # noqa: E402
from skforecast.foundation import FoundationModel  # noqa: E402

register()
print(versions(), flush=True)
V1 = "h2oai/timeh2o-v1"
model = TimeH2O.from_pretrained(V1, device="cpu")
STEPS = 24


def max_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# 1. NaN in the context
# ==============================================================================
y = hourly_series(2000)
base = model.forecast(y, horizon=STEPS).quantiles
inside = y.copy()
inside[500:700] = np.nan
leading = y.copy()
leading[:800] = np.nan
fc_inside = model.forecast(inside, horizon=STEPS).quantiles
fc_leading = model.forecast(leading, horizon=STEPS).quantiles
fc_tail = model.forecast(y[800:], horizon=STEPS).quantiles
print(f"1. 200 NaN inside: finite {np.isfinite(fc_inside).all()} | max abs diff vs no NaN {max_diff(fc_inside, base):.3f}")
print(f"1. 800 leading NaN vs the last 1200 points alone: max abs diff {max_diff(fc_leading, fc_tail):.2e}")
all_nan = model.forecast(np.full(500, np.nan), horizon=STEPS)
print(f"1. all-NaN context: median min/max {all_nan.point.min():.3f}/{all_nan.point.max():.3f} "
      f"| q0.1 min {all_nan.quantiles[:, 0].min():.3f} | q0.9 max {all_nan.quantiles[:, -1].max():.3f}")
idx = pd.date_range("2024-01-01", periods=2000, freq="h")
s = pd.Series(y, index=idx, name="y")
s_tail_nan = s.copy()
s_tail_nan.iloc[-600:] = np.nan
fm = FoundationModel(V1, context_length=500, device="cpu")
fm.adapter._model = model
fm.fit(series=s_tail_nan)
pred = fm.predict(steps=STEPS)
print(f"1. FoundationModel.fit with the last 600 values NaN and context_length=500: "
      f"fit accepted, predictions min/max {pred['pred'].min():.3f}/{pred['pred'].max():.3f}")
try:
    fm.predict(steps=STEPS, context=pd.Series(np.nan, index=idx[:500], name="y"))
    print("1. FoundationModel.predict with an all-NaN context: no error")
except Exception as exc:
    print(f"1. FoundationModel.predict with an all-NaN context: {type(exc).__name__}: {str(exc)[:120]}")

# 2. dtype and determinism
# ==============================================================================
fc32 = model.forecast(y.astype(np.float32), horizon=STEPS).quantiles
fc64 = model.forecast(y.astype(np.float64), horizon=STEPS).quantiles
again = model.forecast(y, horizon=STEPS).quantiles
print(f"2. float32 vs float64 input: max abs diff {max_diff(fc32, fc64):.2e} | two identical calls: {max_diff(base, again):.2e} "
      f"| output dtype {base.dtype}")

# 3. Maximum context
# ==============================================================================
long = hourly_series(20000)
fc_long = model.forecast(long, horizon=STEPS).quantiles
fc_last = model.forecast(long[-16384:], horizon=STEPS).quantiles
print(f"3. 20000 points vs the last 16384: max abs diff {max_diff(fc_long, fc_last):.2e}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model.forecast(long, horizon=STEPS, context_length=20000)
print(f"3. context_length=20000 option: warnings {[str(x.message)[:120] for x in w]}")

# 4. Long horizons
# ==============================================================================
y_ctx = hourly_series(4096)
for horizon in (1024, 2048, 4096):
    start = time.perf_counter()
    one_pass = model.forecast(y_ctx, horizon=horizon, autoregressive=False).quantiles
    seconds = time.perf_counter() - start
    print(f"4. horizon {horizon}, one pass: shape {one_pass.shape} finite {np.isfinite(one_pass).all()} "
          f"{seconds:.2f} s", flush=True)
start = time.perf_counter()
ar = model.forecast(y_ctx, horizon=2048, autoregressive=True).quantiles
seconds = time.perf_counter() - start
one_pass = model.forecast(y_ctx, horizon=2048, autoregressive=False).quantiles
print(f"4. horizon 2048, autoregressive=True: {seconds:.2f} s | first 1024 steps equal to one pass "
      f"{max_diff(ar[:1024], one_pass[:1024]):.2e} | steps 1024+ max abs diff {max_diff(ar[1024:], one_pass[1024:]):.3f}")

# 5. Dependence on the horizon
# ==============================================================================
short = model.forecast(y, horizon=24).quantiles
longer = model.forecast(y, horizon=100).quantiles
print(f"5. no covariates, horizon 24 vs first 24 of horizon 100: max abs diff {max_diff(short, longer[:24]):.2e}")
cov = 15 + 8 * np.sin(2 * np.pi * np.arange(2100) / 24)
y_cov = y + 0.5 * cov[:2000]
short = model.forecast(y_cov, horizon=24, future_covariates=cov[:2024]).quantiles
longer = model.forecast(y_cov, horizon=100, future_covariates=cov[:2100]).quantiles
diff_steps = np.max(np.abs(short - longer[:24]), axis=1)
print(f"5. future covariate, horizon 24 vs first 24 of horizon 100: max abs diff {diff_steps.max():.3f} "
      f"| first step with diff > 1e-3: {int(np.argmax(diff_steps > 1e-3)) if (diff_steps > 1e-3).any() else None} "
      f"| level {np.mean(np.abs(short)):.1f}")

# 6. batch_size
# ==============================================================================
batch = [hourly_series(512, seed=i) for i in range(40)]
results = {}
for batch_size in (8, 256):
    m = TimeH2O.from_pretrained(V1, device="cpu", batch_size=batch_size)
    start = time.perf_counter()
    out = m.forecast_many(batch, horizon=STEPS, patch_ensemble=False)
    results[batch_size] = (np.stack([f.quantiles for f in out]), time.perf_counter() - start)
print(f"6. 40 series x 512, patch_ensemble=False: batch_size 8 {results[8][1]:.2f} s | 256 "
      f"{results[256][1]:.2f} s | max abs diff {max_diff(results[8][0], results[256][0]):.2e}")

# 7. Devices
# ==============================================================================
for device in ("mps", "cuda", "auto"):
    try:
        m = TimeH2O.from_pretrained(V1, device=device)
        print(f"7. device={device!r}: loaded on {m.device}")
    except ValueError as exc:
        print(f"7. device={device!r}: ValueError: {exc}")

# 8. clip_nonnegative, snap_integer and mean
# ==============================================================================
rng = np.random.default_rng(3)
counts = rng.poisson(0.7, 1000).astype(float)
default = model.forecast(counts, horizon=STEPS)
clipped = model.forecast(counts, horizon=STEPS, clip_nonnegative=True)
snapped = model.forecast(counts, horizon=STEPS, clip_nonnegative=True, snap_integer=True)
print(f"8. Poisson(0.7) counts: q0.1 min default {default.quantiles[:, 0].min():.3f} "
      f"| clip_nonnegative=True {clipped.quantiles[:, 0].min():.3f} "
      f"| snap_integer=True values {sorted(set(np.unique(snapped.quantiles).round(3)))[:6]}")
print(f"8. mean vs median (hourly series): max abs diff {max_diff(model.forecast(y, horizon=STEPS).mean, model.forecast(y, horizon=STEPS).point):.3f}")
