"""
Experiment 3: which `y_train` does each path pass to the scaled metrics?

For every path, implied scale = MAE / MASE over the same points. It is matched
against the naive scale of every slice y[a:b] (a in 0..60, b in the possible
training ends), and the slice that reproduces it exactly is reported.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

from skforecast.datasets import load_demo_dataset
from skforecast.recursive import (
    ForecasterRecursive, ForecasterEquivalentDate, ForecasterStats,
    ForecasterRecursiveMultiSeries,
)
from skforecast.direct import ForecasterDirect
from skforecast.stats import Ets, Sarimax
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster, backtesting_stats,
    backtesting_forecaster_multiseries, grid_search_forecaster,
    grid_search_forecaster_multiseries,
)
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error

warnings.filterwarnings("ignore")
print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
      f"scikit-learn {sklearn.__version__}")

y = load_demo_dataset(verbose=False)
N = len(y)
n_train = N - 36
METRICS = ["mean_absolute_error", mean_absolute_scaled_error]


def find_slice(values, implied, ends):
    values = np.asarray(values, dtype=float)
    hits = []
    for a in range(0, 61):
        for b in ends:
            if b - a < 3:
                continue
            s = np.nanmean(np.abs(np.diff(values[a:b])))
            if abs(s - implied) < 1e-10:
                hits.append(f"[{a}:{b}]")
    return ", ".join(hits) if hits else "no match"


ENDS = list(range(n_train - 30, N + 1))


def show(name, ws, mae, mase, values=None, ends=None):
    values = y if values is None else values
    implied = mae / mase
    print(f"| {name} | {ws} | {mase:.4f} | {implied:.5f} | {find_slice(values, implied, ends or ENDS)} |")


print(f"\nh2o, N={N}, n_train={n_train} (folds of 12: train ends {n_train}, {n_train+12}, {n_train+24})")
print("| Path | window_size | MASE | implied scale | y_train slice (iloc) |")
print("|---|---|---|---|---|")


def bt(f, name, **cv_kw):
    kw = dict(steps=12, initial_train_size=n_train, refit=False)
    kw.update(cv_kw)
    m, _ = backtesting_forecaster(forecaster=f, y=y, cv=TimeSeriesFold(**kw), metric=METRICS, show_progress=False)
    show(name, f.window_size, m.iat[0, 0], m.iat[0, 1])


bt(ForecasterRecursive(Ridge(), lags=24), "backtesting_forecaster Recursive lags=24, refit=False")
bt(ForecasterRecursive(Ridge(), lags=24), "  refit=True", refit=True)
bt(ForecasterRecursive(Ridge(), lags=24), "  refit=True, fixed_train_size=True", refit=True, fixed_train_size=True)
bt(ForecasterRecursive(Ridge(), lags=24), "  refit=2 (intermittent)", refit=2)
bt(ForecasterRecursive(Ridge(), lags=24), "  gap=5, refit=False", gap=5)
bt(ForecasterRecursive(Ridge(), lags=24, differentiation=1), "  differentiation=1 (window_size 25)")
bt(ForecasterRecursive(Ridge(), lags=3), "Recursive lags=3")
bt(ForecasterDirect(Ridge(), lags=24, steps=12), "ForecasterDirect lags=24")
bt(ForecasterEquivalentDate(offset=12), "ForecasterEquivalentDate offset=12")
bt(ForecasterEquivalentDate(offset=12, n_offsets=3), "ForecasterEquivalentDate offset=12, n_offsets=3")

for est, name, refit in [(Sarimax(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)), "backtesting_stats Sarimax, refit=False", False),
                         (Sarimax(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)), "backtesting_stats Sarimax, refit=True", True),
                         (Ets(m=12), "backtesting_stats Ets (refit forced True)", False)]:
    f = ForecasterStats(estimator=est)
    m, _ = backtesting_stats(forecaster=f, y=y, cv=TimeSeriesFold(steps=12, initial_train_size=n_train, refit=refit),
                             metric=METRICS, show_progress=False, suppress_warnings=True)
    show(name, f.window_size, m.iat[0, 0], m.iat[0, 1])


def osa(f, name):
    r = grid_search_forecaster(forecaster=f, y=y, cv=OneStepAheadFold(initial_train_size=n_train),
                               param_grid={"alpha": [1.0]}, metric=METRICS, return_best=False,
                               n_jobs=1, verbose=False, show_progress=False)
    show(name, f.window_size, r["mean_absolute_error"].iat[0], r["mean_absolute_scaled_error"].iat[0])


osa(ForecasterRecursive(Ridge(), lags=24), "grid search OneStepAheadFold, Recursive lags=24")
osa(ForecasterRecursive(Ridge(), lags=24, differentiation=1), "  differentiation=1")
osa(ForecasterDirect(Ridge(), lags=24, steps=12), "  ForecasterDirect lags=24")

r = grid_search_forecaster(forecaster=ForecasterRecursive(Ridge(), lags=24), y=y,
                           cv=TimeSeriesFold(steps=12, initial_train_size=n_train, refit=False),
                           param_grid={"alpha": [1.0]}, metric=METRICS, return_best=False,
                           n_jobs=1, verbose=False, show_progress=False)
show("grid search TimeSeriesFold, Recursive lags=24", 24, r["mean_absolute_error"].iat[0], r["mean_absolute_scaled_error"].iat[0])

# Multi-series --------------------------------------------------------------
rng = np.random.default_rng(1)
s1 = y.rename("s1")
s2 = (2 * y + rng.normal(scale=0.05, size=N)).rename("s2")
s3 = (y * 1.5).rename("s3").iloc[30:]          # starts 30 dates later
wide = pd.concat([s1, s2, s3], axis=1)          # s3 padded with NaN
as_dict = {"s1": s1, "s2": s2, "s3": s3}
as_dict_pad = {"s1": s1, "s2": s2, "s3": wide["s3"]}  # s3 with leading NaN in the dict
print("\nMulti-series (s3 starts at iloc 30 of the span, values 1.5*y)")
print("| Path | level | MASE | implied scale | y_train slice (iloc of the span) |")
print("|---|---|---|---|---|")


def show_ms(name, metrics):
    for _, row in metrics.iterrows():
        lvl = row["levels"]
        if lvl in ("average", "weighted_average", "pooling"):
            print(f"| {name} | {lvl} | {row['mean_absolute_scaled_error']:.4f} | - | - |")
            continue
        vals = wide[lvl]
        implied = row["mean_absolute_error"] / row["mean_absolute_scaled_error"]
        print(f"| {name} | {lvl} | {row['mean_absolute_scaled_error']:.4f} | {implied:.5f} | {find_slice(vals, implied, ENDS)} |")


for data, dname in [(wide, "wide DataFrame"), (as_dict, "dict"), (as_dict_pad, "dict, s3 with leading NaN")]:
    f = ForecasterRecursiveMultiSeries(Ridge(), lags=24)
    m, _ = backtesting_forecaster_multiseries(
        forecaster=f, series=data, cv=TimeSeriesFold(steps=12, initial_train_size=n_train, refit=False),
        metric=METRICS, show_progress=False)
    show_ms(f"backtesting_multiseries lags=24, {dname}", m)

r = grid_search_forecaster_multiseries(
    forecaster=ForecasterRecursiveMultiSeries(Ridge(), lags=24), series=wide,
    cv=OneStepAheadFold(initial_train_size=n_train), param_grid={"alpha": [1.0]},
    metric=METRICS, aggregate_metric=["average"], return_best=False, n_jobs=1, verbose=False, show_progress=False)
print("(grid multiseries OSA returns only aggregated metrics; per level below via backtesting of the internals is not exposed)")
print(r.filter(like="mean_absolute").round(5).to_string())

# Pooled aggregation: check the scale used by `pooling`
f = ForecasterRecursiveMultiSeries(Ridge(), lags=24)
m, p = backtesting_forecaster_multiseries(
    forecaster=f, series=wide, cv=TimeSeriesFold(steps=12, initial_train_size=n_train, refit=False),
    metric=METRICS, show_progress=False, add_aggregated_metric=True)
pooled = m.set_index("levels").loc["pooling"]
diffs_ws = np.concatenate([np.diff(wide[c].dropna().iloc[: (n_train - (30 if c == "s3" else 0))].to_numpy()[24:]) for c in wide])
diffs_full = np.concatenate([np.diff(wide[c].dropna().iloc[: (n_train - (30 if c == "s3" else 0))].to_numpy()) for c in wide])
print(f"\npooling: MAE {pooled['mean_absolute_error']:.5f}, MASE {pooled['mean_absolute_scaled_error']:.5f}, "
      f"implied scale {pooled['mean_absolute_error'] / pooled['mean_absolute_scaled_error']:.5f}; "
      f"pooled diffs without first 24 of each level {np.mean(np.abs(diffs_ws)):.5f}, with all {np.mean(np.abs(diffs_full)):.5f}")
