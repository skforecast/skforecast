"""
Experiment 9: the folds of the user guide backtesting (vic_electricity,
December 2014, 30 folds of 24 steps) where a future covariate deviates from
its context by many standard deviations, to calibrate the warning of the
adapter for flat or nearly flat future covariates.

For each context length, the ratio between the largest deviation of the
future values from the mean of the context and the standard deviation of
the context is computed per fold and covariate (0 when the future does not
change, infinite when the context is constant and the future changes). In
the folds where the ratio of Holiday is above 5, TimeH2O-v1 is run with
Temperature and Holiday (floats, as in the guide), with Temperature only
and without exog.

Usage: HF_HUB_OFFLINE=1 python -W ignore exp9_guide_folds.py > exp9.log 2>&1
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
data = fetch_dataset(name="vic_electricity", verbose=False).drop(columns="Date")
data = data.resample(rule="h", closed="left", label="right").agg(
    {"Demand": "mean", "Temperature": "mean", "Holiday": "mean"}
)
data = data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00"]
start = data.index.get_loc(data.loc["2014-12-01 00:00:00":].index[0])


def ratio(history, future):
    deviation = np.abs(future - history.mean()).max()
    if deviation == 0:
        return 0.0
    spread = history.std()
    return np.inf if spread == 0 else deviation / spread


for context_length in (500, 4096, 16384):
    ratios = {col: [] for col in ("Temperature", "Holiday")}
    for fold in range(30):
        origin = start + 24 * fold
        for col in ratios:
            x = data[col].to_numpy()
            ratios[col].append(ratio(x[origin - context_length:origin], x[origin:origin + 24]))
    for col, values in ratios.items():
        values = np.array(values)
        print(f"context {context_length}, {col}: max ratio {values.max():.1f} | folds with ratio > 5: "
              f"{np.flatnonzero(values > 5).tolist()} | > 10: {np.flatnonzero(values > 10).tolist()}",
              flush=True)
    for fold in np.flatnonzero(np.array(ratios["Holiday"]) > 5):
        origin = start + 24 * fold
        y = data["Demand"].to_numpy()[origin - context_length:origin]
        truth = data["Demand"].to_numpy()[origin:origin + 24]
        cov = [data[col].to_numpy()[origin - context_length:origin + 24] for col in ("Temperature", "Holiday")]
        cases = {
            "Temperature and Holiday": cov,
            "Temperature only": cov[:1],
            "no exog": None,
        }
        for name, rows in cases.items():
            q = model.forecast(y, horizon=24, future_covariates=rows, quantile_levels=[0.1, 0.5, 0.9]).quantiles
            print(f"  fold {fold} (origin {data.index[origin]}, Holiday ratio {ratios['Holiday'][fold]:.1f}), "
                  f"{name:23s}: MAE {np.mean(np.abs(q[:, 1] - truth)):7.2f} | mean width 80% "
                  f"{np.mean(q[:, 2] - q[:, 0]):7.1f} | max q0.9 {q[:, 2].max():8.1f} | max demand {truth.max():7.1f}",
                  flush=True)
