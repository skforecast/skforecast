"""
Experiment 1: smoke test of the h2ofm backend called directly (no skforecast).

1. Output of `forecast` and `forecast_many`: types, shapes, levels, point and
   median.
2. A batch of series of different lengths passed to `forecast_many` as a list
   (the backend pads each internal batch itself) against each series alone.
3. The same batch left-padded with NaN by the caller to a common length (what
   `FoundationModel` does for adapters with `requires_equal_length_context`)
   against each series alone.
4. Load time and time per call on CPU for each checkpoint (24 steps, no
   covariates, median of 3 calls after a warm-up).

Usage: HF_HUB_OFFLINE=1 python exp1_smoke.py > exp1.log
"""
import time
import warnings
import numpy as np
from h2ofm import TimeH2O
from common import CHECKPOINTS, hourly_series, timeit, versions

warnings.simplefilter("always")
print(versions())
STEPS = 24

for model_id in CHECKPOINTS:
    print(f"\n=== {model_id}")
    start = time.perf_counter()
    model = TimeH2O.from_pretrained(model_id, device="cpu")
    print(f"load: {time.perf_counter() - start:.2f} s | device {model.device} "
          f"| batch_size {model.batch_size} | patch_sizes {model.config.patch_sizes} "
          f"| context_length {model.config.context_length} "
          f"| max_horizon {model.config.max_horizon} "
          f"| autoregressive {model.config.autoregressive} "
          f"| variate_attention {model.config.variate_attention} "
          f"| flip_equivariant {model.config.flip_equivariant} "
          f"| quantiles {model.config.quantiles}")
    n_params = sum(p.numel() for p in model._pipeline.model.parameters())
    print(f"parameters: {n_params:,}")

    # 1. Output
    y = hourly_series(500)
    fc = model.forecast(y, horizon=STEPS)
    print(f"forecast: type {type(fc).__name__} | point {fc.point.shape} {fc.point.dtype} "
          f"| quantiles {fc.quantiles.shape} {fc.quantiles.dtype} "
          f"| levels {fc.quantile_levels} | mean {fc.mean.shape}")
    print(f"point == median column: {np.array_equal(fc.point, fc.quantiles[:, 4])}")
    fc2 = model.forecast(y, horizon=STEPS, quantile_levels=[0.9, 0.5, 0.01])
    print(f"requested [0.9, 0.5, 0.01]: levels {fc2.quantile_levels} "
          f"| q0.5 == point {np.array_equal(fc2.quantiles[:, 1], fc.point)} "
          f"| q0.9 equal to default grid {np.array_equal(fc2.quantiles[:, 0], fc.quantiles[:, 8])}")

    # 2. and 3. Batch of different lengths
    lengths = [300, 3000, 9000]
    series = [hourly_series(n, seed=i) for i, n in enumerate(lengths)]
    alone = [model.forecast(s, horizon=STEPS).quantiles for s in series]
    batch = model.forecast_many(series, horizon=STEPS)
    diffs = [float(np.max(np.abs(b.quantiles - a))) for b, a in zip(batch, alone)]
    print(f"forecast_many list (lengths {lengths}) vs alone, max abs diff per series: "
          f"{[f'{d:.2e}' for d in diffs]}")
    width = max(lengths)
    padded = np.full((len(series), width), np.nan)
    for row, s in zip(padded, series):
        row[width - len(s):] = s
    batch_padded = model.forecast_many(padded, horizon=STEPS)
    diffs = [float(np.max(np.abs(b.quantiles - a))) for b, a in zip(batch_padded, alone)]
    rel = [
        float(np.max(np.abs(b.quantiles - a)) / np.mean(np.abs(a)))
        for b, a in zip(batch_padded, alone)
    ]
    print(f"caller-padded array (width {width}) vs alone, max abs diff per series: "
          f"{[f'{d:.2e}' for d in diffs]} | relative to the mean level: "
          f"{[f'{r:.2e}' for r in rel]}")
    single_padded = [
        model.forecast(row, horizon=STEPS).quantiles for row in padded
    ]
    diffs = [
        float(np.max(np.abs(p - a))) for p, a in zip(single_padded, alone)
    ]
    print(f"each series alone, left-padded to {width} vs unpadded: "
          f"{[f'{d:.2e}' for d in diffs]}")

    # 4. Time per call
    rows = []
    for n_series, length in [(1, 512), (100, 512), (1, 4096), (1, 8192), (1, 16384)]:
        data = [hourly_series(length, seed=i) for i in range(n_series)]
        seconds = timeit(lambda: model.forecast_many(data, horizon=STEPS))
        seconds_no_ens = timeit(
            lambda: model.forecast_many(data, horizon=STEPS, patch_ensemble=False)
        )
        rows.append((n_series, length, seconds, seconds_no_ens))
    for n_series, length, seconds, seconds_no_ens in rows:
        print(f"time {n_series:>3} series x {length:>5}: {seconds:.3f} s "
              f"| patch_ensemble=False: {seconds_no_ens:.3f} s")
