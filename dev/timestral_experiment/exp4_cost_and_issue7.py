"""
1. Cost of one call versus chunked calls with covariates (209M).
2. mistralai/timestral#7: which part of the blow-up comes from the flat
   covariate and which mitigations work (backend called directly).
"""
import time
import resource
import numpy as np
import torch
from timestral import TimestralForecaster

rng = np.random.default_rng(2)
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# 1. One call vs chunks of 100 series, 1 target + 2 known covariates, 512 steps
t = np.arange(536)
for n_series in [100, 400]:
    ctx = (100 + 20 * np.sin(2 * np.pi * t[:512] / 24) + rng.standard_normal((n_series, 512)).cumsum(-1))[:, None]
    known = np.stack([np.stack([15 + 8 * np.sin(2 * np.pi * t / 24), (t % 7).astype(float)])] * n_series)
    t0 = time.perf_counter()
    fc.forecast(ctx, prediction_length=24, known_covariates=known)
    one = time.perf_counter() - t0
    print(f"1. {n_series} series, one call: {one:6.1f}s  peak RSS {rss():,.0f} MB", flush=True)
    t0 = time.perf_counter()
    for i in range(0, n_series, 100):
        fc.forecast(ctx[i:i + 100], prediction_length=24, known_covariates=known[i:i + 100])
    chunked = time.perf_counter() - t0
    print(f"1. {n_series} series, chunks of 100: {chunked:6.1f}s  peak RSS {rss():,.0f} MB", flush=True)

# 2. Issue #7: Poisson(2) daily counts, price known covariate
T, H = 1024, 28
y = rng.poisson(2, (32, 1, T)).astype(float)
cases = {}
flat = np.full((32, 1, T + H), 2.50)
flat[:, :, T + 5:] = 2.56                               # flat in the context, moves in the horizon
moving = np.full((32, 1, T + H), 2.50)
moving[:, :, T - 300:] = 2.56                           # moves inside the context
moving[:, :, T + 5:] = 2.60
masked = flat.copy()
masked[:, :, :T] = np.nan                               # workaround of the issue: hide the flat history
cases = {
    "no covariate": dict(),
    "flat price": dict(known_covariates=flat),
    "flat price, symmetric_averaging=False": dict(known_covariates=flat, symmetric_averaging=False),
    "price that moves in the context": dict(known_covariates=moving),
    "flat price, history set to NaN": dict(known_covariates=masked),
}
for name, kwargs in cases.items():
    q = fc.forecast(y, prediction_length=H, **kwargs).quantiles[:, 0].numpy()
    print(f"2. {name:40s} median max {q[..., 10].max():9.2f}  q99 max {q[..., -1].max():9.2f}  "
          f"series with q99 > 100: {(q[..., -1].max(-1) > 100).sum()}/32", flush=True)

fc_cap = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", quantile_real_cap_k=10)
q = fc_cap.forecast(y, prediction_length=H, known_covariates=flat).quantiles[:, 0].numpy()
print(f"2. {'flat price, quantile_real_cap_k=10':40s} median max {q[..., 10].max():9.2f}  q99 max {q[..., -1].max():9.2f}  "
      f"series with q99 > 100: {(q[..., -1].max(-1) > 100).sum()}/32")
print("torch", torch.__version__, "threads", torch.get_num_threads())
