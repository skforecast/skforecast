"""
Backend checks that back the claims of the report not covered by exp1-exp4
(backend called directly, no skforecast):

1. mistralai/timestral#7 with the data of exp4 on the 5M, 28M and 209M:
   default, symmetric_averaging=False and the flat history set to NaN.
2. The same blow-up with a 0/1 indicator that is 0 over the context.
3. With known covariates, the first steps depend on prediction_length.
4. An all-NaN covariate row is not the same as no covariate.
5. Cost of symmetric averaging (209M, 50 series x 500).
6. float32 and float64 inputs, repeated calls.
7. All-NaN context.
"""
import time
import numpy as np
import torch
from timestral import TimestralForecaster

T, H = 1024, 28
# Same random draws as exp4_cost_and_issue7.py, so `y` is the same data.
rng = np.random.default_rng(2)
rng.standard_normal((100, 512))
rng.standard_normal((400, 512))
y = rng.poisson(2, (32, 1, T)).astype(float)

flat = np.full((32, 1, T + H), 2.50)
flat[:, :, T + 5:] = 2.56
flat_masked = flat.copy()
flat_masked[:, :, :T] = np.nan
indicator = np.zeros((32, 1, T + H))
indicator[:, :, T + 10:T + 12] = 1.0
indicator_masked = indicator.copy()
indicator_masked[:, :, :T] = np.nan


def summary(fc, **kwargs):
    q = fc.forecast(y, prediction_length=H, **kwargs).quantiles[:, 0].numpy()
    return (
        f"median max {q[..., 10].max():9.2f}  q99 max {q[..., -1].max():9.2f}  "
        f"series with q99 > 100: {(q[..., -1].max(-1) > 100).sum()}/32"
    )


models = {}
for size in ["5M", "28M", "209M"]:
    models[size] = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")

# 1. Issue #7 with a flat price, per checkpoint
for size, fc in models.items():
    for name, kwargs in {
        "no covariate": {},
        "flat price": {"known_covariates": flat},
        "flat price, symmetric_averaging=False": {"known_covariates": flat, "symmetric_averaging": False},
        "flat price, history set to NaN": {"known_covariates": flat_masked},
    }.items():
        print(f"1. {size:4s} {name:40s} {summary(fc, **kwargs)}", flush=True)

# 2. A 0/1 indicator that is 0 over the context and 1 at two horizon steps
for size in ["5M", "28M", "209M"]:
    fc = models[size]
    for name, kwargs in {
        "indicator": {"known_covariates": indicator},
        "indicator, symmetric_averaging=False": {"known_covariates": indicator, "symmetric_averaging": False},
        "indicator, history set to NaN": {"known_covariates": indicator_masked},
    }.items():
        print(f"2. {size:4s} {name:40s} {summary(fc, **kwargs)}", flush=True)

# 3. Dependence on prediction_length (28M, one smooth series)
fc = models["28M"]
t = np.arange(600)
series = 50 + 10 * np.sin(2 * np.pi * t[:500] / 24) + np.random.default_rng(0).standard_normal(500)
cov = np.sin(2 * np.pi * t / 7)[None, None, :]
for name, kwargs in {
    "no covariate": lambda h: {},
    "past covariate": lambda h: {"past_covariates": cov[..., :500]},
    "known covariate": lambda h: {"known_covariates": cov[..., :500 + h]},
}.items():
    q24 = fc.forecast(series, prediction_length=24, **kwargs(24)).quantiles[0, 0].numpy()
    q100 = fc.forecast(series, prediction_length=100, **kwargs(100)).quantiles[0, 0].numpy()
    print(f"3. 28M {name:16s} max |h=24 - h=100| on the first 24 steps: {np.abs(q24 - q100[:24]).max():.2e}", flush=True)

# 4. All-NaN known covariate row vs no covariate (28M)
nan_row = np.full((1, 1, 524), np.nan)
q_none = fc.forecast(series, prediction_length=24).quantiles[0, 0].numpy()
q_nan = fc.forecast(series, prediction_length=24, known_covariates=nan_row).quantiles[0, 0].numpy()
print(f"4. 28M all-NaN known covariate vs none: max abs diff {np.abs(q_none - q_nan).max():.3f}, "
      f"median level {np.median(q_none[:, 10]):.1f}", flush=True)

# 5. Cost of symmetric averaging (209M, 50 series x 500, 24 steps)
fc = models["209M"]
x50 = np.stack([series + np.random.default_rng(i).standard_normal(500) for i in range(50)])
fc.forecast(x50[:2], prediction_length=24)
for sym in [True, False]:
    t0 = time.perf_counter()
    fc.forecast(x50, prediction_length=24, symmetric_averaging=sym)
    print(f"5. 209M 50 series x 500, symmetric_averaging={sym}: {time.perf_counter() - t0:.2f}s", flush=True)

# 6. float32 vs float64 input, repeated calls (5M)
fc = models["5M"]
q64 = fc.forecast(series.astype(np.float64), prediction_length=24).quantiles.numpy()
q32 = fc.forecast(series.astype(np.float32), prediction_length=24).quantiles.numpy()
q64b = fc.forecast(series.astype(np.float64), prediction_length=24).quantiles.numpy()
print(f"6. 5M float32 vs float64 identical: {np.array_equal(q32, q64)}; repeated call identical: {np.array_equal(q64, q64b)}", flush=True)

# 7. All-NaN context (5M)
q_empty = fc.forecast(np.full(200, np.nan), prediction_length=6).quantiles[0, 0].numpy()
print(f"7. 5M all-NaN context: no error, max |quantile| {np.abs(q_empty).max():.3f}", flush=True)
print("torch", torch.__version__, "threads", torch.get_num_threads())
