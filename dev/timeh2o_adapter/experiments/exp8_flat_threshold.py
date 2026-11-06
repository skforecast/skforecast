"""
Experiment 8: how flat must a future covariate be over the context for
TimeH2O to blow up (threshold for the warning of the adapter).

Same data as part 5 of exp4 (32 daily Poisson(2) series, 1024 points of
context, 28 steps):

1. A price of 2.50 plus Gaussian noise of standard deviation `s` over the
   context, that moves to 2.56 in the horizon, for several `s`: the ratio
   between the change in the horizon and the variability of the context is
   0.06 / s.
2. A 0/1 indicator with `k` ones spread over the context and 1 at two
   horizon steps, for several `k`.

Usage: HF_HUB_OFFLINE=1 python -W ignore exp8_flat_threshold.py > exp8.log 2>&1
"""
import sys
import warnings
import numpy as np
from h2ofm import TimeH2O

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from common import versions  # noqa: E402

print(versions(), flush=True)
model = TimeH2O.from_pretrained("h2oai/timeh2o-v1", device="cpu")
levels = [0.1, 0.5, 0.9, 0.99]
rng = np.random.default_rng(7)
T, H = 1024, 28
counts = rng.poisson(2, (32, T)).astype(float)


def report(name, cov):
    fc = model.forecast_many(
        list(counts), horizon=H, future_covariates=list(cov), quantile_levels=levels
    )
    q = np.stack([f.quantiles for f in fc])
    print(f"{name:48s} median max {q[..., 1].max():8.2f} | q0.9 max {q[..., 2].max():8.2f} "
          f"| q0.99 max {q[..., 3].max():8.2f} | series with q0.99 > 100: "
          f"{int((q[..., 3].max(-1) > 100).sum())}/32", flush=True)


print("Reference without covariate (exp4): median max 2.09 | q0.9 max 4.29 | q0.99 max 6.07")
noise_rng = np.random.default_rng(11)
for s in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
    price = np.full((32, T + H), 2.50)
    price[:, :T] += noise_rng.normal(0, s, (32, T))
    price[:, T + 5:] = 2.56
    report(f"1. price noise s={s:g} (change / s = {0.06 / s:g})", price)
for k in (1, 4, 16, 64):
    indicator = np.zeros((32, T + H))
    positions = np.linspace(0, T - 1, k + 2)[1:-1].astype(int)
    indicator[:, positions] = 1.0
    indicator[:, T + 10:T + 12] = 1.0
    std = indicator[0, :T].std()
    report(f"2. indicator with {k} ones in the context (std {std:.3f})", indicator)
