"""
Experiment 6: cost of the internal padding and of large batches (v1).

h2ofm splits every call into chunks of at most `batch_size` rows (targets
plus covariates) and left-pads the series of each chunk to the longest one
(`_data/dataset.py`, `iter_batches`). This script measures:

1. Mixed lengths in one call: 200 series with lengths between 500 and 4096
   in random order, the same series sorted by length, and 200 series of
   4096 points (all with `patch_ensemble=False`, and 40 series with the
   default `True`).
2. Large batches without covariates: 100, 400 and 1000 series of 512 points
   (`patch_ensemble=False`), time and peak RSS.
3. `batch_size` 64, 256 and 1024 with 400 series of 512 points and two
   future covariates.

Peak RSS is the maximum of the process so far, so the cases run in
increasing order of size.

Usage: HF_HUB_OFFLINE=1 python exp6_large_batches.py > exp6.log 2>&1
"""
import sys
import time
import numpy as np
from h2ofm import TimeH2O

sys.path.insert(0, ".")
from common import hourly_series, peak_rss_gb, versions  # noqa: E402

print(versions(), flush=True)
V1 = "h2oai/timeh2o-v1"
STEPS = 24
model = TimeH2O.from_pretrained(V1, device="cpu")
print(f"after load: peak RSS {peak_rss_gb():.2f} GB", flush=True)


def run(contexts, covariates=None, model=model, **kwargs):
    start = time.perf_counter()
    out = model.forecast_many(contexts, horizon=STEPS, future_covariates=covariates, **kwargs)
    return out, time.perf_counter() - start


# 1. Mixed lengths
# ==============================================================================
rng = np.random.default_rng(0)
lengths = rng.integers(500, 4097, 200)
mixed = [hourly_series(int(n), seed=i) for i, n in enumerate(lengths)]
order = np.argsort(lengths)
sorted_series = [mixed[i] for i in order]
equal = [hourly_series(4096, seed=i) for i in range(200)]
out_mixed, t_mixed = run(mixed, patch_ensemble=False)
out_sorted, t_sorted = run(sorted_series, patch_ensemble=False)
_, t_equal = run(equal, patch_ensemble=False)
diff = max(
    float(np.max(np.abs(out_mixed[i].quantiles - out_sorted[k].quantiles)))
    for k, i in enumerate(order)
)
print(f"1. 200 series, patch_ensemble=False: mixed lengths 500-4096 (mean {lengths.mean():.0f}) "
      f"{t_mixed:.1f} s | sorted by length {t_sorted:.1f} s (max abs diff {diff:.2e}) "
      f"| all 4096 {t_equal:.1f} s | peak RSS {peak_rss_gb():.2f} GB", flush=True)
_, t_mixed40 = run(mixed[:40])
_, t_sorted40 = run(sorted(mixed[:40], key=len))
_, t_equal40 = run(equal[:40])
print(f"1. 40 series, patch_ensemble=True: mixed {t_mixed40:.1f} s | sorted {t_sorted40:.1f} s "
      f"| all 4096 {t_equal40:.1f} s | peak RSS {peak_rss_gb():.2f} GB", flush=True)

# 2. Large batches without covariates
# ==============================================================================
for n_series in (100, 400, 1000):
    data = [hourly_series(512, seed=i) for i in range(n_series)]
    _, seconds = run(data, patch_ensemble=False)
    print(f"2. {n_series:>4} series x 512, no covariates: {seconds:.1f} s "
          f"({1000 * seconds / n_series:.1f} ms per series) | peak RSS {peak_rss_gb():.2f} GB", flush=True)

# 3. batch_size with covariates
# ==============================================================================
data = [hourly_series(512, seed=i) for i in range(400)]
t = np.arange(512 + STEPS)
cov = [
    [15 + 8 * np.sin(2 * np.pi * t / 24), np.where(t % 168 < 120, "work", "weekend").astype(object)]
    for _ in range(400)
]
for batch_size in (64, 256, 1024):
    m = TimeH2O.from_pretrained(V1, device="cpu", batch_size=batch_size)
    _, seconds = run(data, cov, model=m, patch_ensemble=False)
    print(f"3. 400 series x 512, 2 covariates, batch_size {batch_size:>4}: {seconds:.1f} s "
          f"| peak RSS {peak_rss_gb():.2f} GB", flush=True)
