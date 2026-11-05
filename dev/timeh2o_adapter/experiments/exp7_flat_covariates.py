"""
Experiment 7: flat future covariates (the case of mistralai/timestral#7) in
TimeH2O, beyond the synthetic data of exp4.

1. Mechanism: a price flat over the context that moves in the horizon,
   against the same price with a tiny noise in the context (1e-6), with a
   tiny step in the context, and with the step only in the horizon but
   smaller (2.50 to 2.5001). If the causal scaler of the covariate row is
   the cause (zero variance, scale clamped to 1e-6), any non-zero variance
   in the context removes the blow-up.
2. Remedies with the price flat in the context: `patch_ensemble=False`,
   the price relative to its last context value (0 in the context), and
   the price history set to NaN (the backend then drops the row, which is
   the same as no covariate).
3. The user guide: in the backtesting of `vic_electricity` with 500 points
   of context, the folds where `Holiday` is constant over the context and
   changes in the horizon, and the forecast of TimeH2O there with Holiday
   as float and as bool (categorical for h2ofm), with Temperature only and
   without exog.

Usage: HF_HUB_OFFLINE=1 python -W ignore exp7_flat_covariates.py > exp7.log 2>&1
"""
import sys
import warnings
import numpy as np
from h2ofm import TimeH2O

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from common import versions  # noqa: E402
from skforecast.datasets import fetch_dataset  # noqa: E402

print(versions(), flush=True)
model = TimeH2O.from_pretrained("h2oai/timeh2o-v1", device="cpu")
levels = [0.1, 0.5, 0.9, 0.99]

# Same data as part 5 of exp4
rng = np.random.default_rng(7)
T, H = 1024, 28
counts = rng.poisson(2, (32, T)).astype(float)


def report(name, cov, **kwargs):
    fc = model.forecast_many(
        list(counts), horizon=H,
        future_covariates=None if cov is None else list(cov),
        quantile_levels=levels, **kwargs,
    )
    q = np.stack([f.quantiles for f in fc])
    print(f"{name:52s} median max {q[..., 1].max():8.2f} | q0.9 max {q[..., 2].max():8.2f} "
          f"| q0.99 max {q[..., 3].max():8.2f} | series with q0.99 > 100: "
          f"{int((q[..., 3].max(-1) > 100).sum())}/32", flush=True)


# 1. Mechanism
flat = np.full((32, T + H), 2.50)
flat[:, T + 5:] = 2.56
noisy = flat + np.r_[rng.normal(0, 1e-6, T), np.zeros(H)]
step_in_context = flat.copy()
step_in_context[:, T - 2:T] = 2.5000001
tiny_step = np.full((32, T + H), 2.50)
tiny_step[:, T + 5:] = 2.5001
report("1. flat price (2.50 to 2.56 in the horizon)", flat)
report("1. same, noise of 1e-6 in the context", noisy)
report("1. same, a step of 1e-7 at the end of the context", step_in_context)
report("1. flat price, step of 1e-4 in the horizon", tiny_step)
# 2. Remedies
report("2. flat price, patch_ensemble=False", flat, patch_ensemble=False)
relative = flat - 2.50
report("2. price relative to the last context value", relative)
masked = flat.copy()
masked[:, :T] = np.nan
report("2. price history set to NaN (row dropped)", masked)
report("2. no covariate", None)

# 3. The folds of the user guide where Holiday is flat in the context
data = fetch_dataset(name="vic_electricity", verbose=False).drop(columns="Date")
data = data.resample(rule="h", closed="left", label="right").agg(
    {"Demand": "mean", "Temperature": "mean", "Holiday": "mean"}
)
data = data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00"]
start = data.index.get_loc(data.loc["2014-12-01 00:00:00":].index[0])
for context_length in (500, 4096):
    flat_folds = []
    for fold in range(30):
        origin = start + 24 * fold
        holiday = data["Holiday"].to_numpy()
        history = holiday[origin - context_length:origin]
        future = holiday[origin:origin + 24]
        if np.unique(history).size == 1 and (future != history[0]).any():
            flat_folds.append(fold)
    print(f"3. context {context_length}: folds with Holiday flat in the context and "
          f"changing in the horizon: {flat_folds}", flush=True)
    for fold in flat_folds:
        origin = start + 24 * fold
        y = data["Demand"].to_numpy()[origin - context_length:origin]
        truth = data["Demand"].to_numpy()[origin:origin + 24]
        cov = [
            data[col].to_numpy()[origin - context_length:origin + 24]
            for col in ("Temperature", "Holiday")
        ]
        with_exog = model.forecast(y, horizon=24, future_covariates=cov, quantile_levels=[0.1, 0.5, 0.9])
        with_bool = model.forecast(
            y, horizon=24, future_covariates=[cov[0], cov[1].astype(bool)], quantile_levels=[0.1, 0.5, 0.9]
        )
        temp_only = model.forecast(y, horizon=24, future_covariates=cov[:1], quantile_levels=[0.1, 0.5, 0.9])
        no_exog = model.forecast(y, horizon=24, quantile_levels=[0.1, 0.5, 0.9])
        for name, fc in (
            ("Temperature and Holiday", with_exog),
            ("Holiday as bool", with_bool),
            ("Temperature only", temp_only),
            ("no exog", no_exog),
        ):
            q = fc.quantiles
            print(f"3. fold {fold} (origin {data.index[origin]}), {name:23s}: MAE {np.mean(np.abs(q[:, 1] - truth)):7.2f} "
                  f"| mean width 80% {np.mean(q[:, 2] - q[:, 0]):7.1f} | max q0.9 {q[:, 2].max():8.1f} "
                  f"| max demand {truth.max():7.1f}", flush=True)
