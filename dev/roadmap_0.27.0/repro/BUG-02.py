"""
BUG-02: with `fold_stride < steps` the backtesting metric only measures the
shortest horizons, and the hyperparameter searches select by that error.

The metric is computed after `index.duplicated(keep='last')`, so for every
date only the prediction of the most recent origin (shortest horizon) is
kept. With `fold_stride=1` the reported value is the 1 step ahead error,
not the error of the configured `steps`.

Run: python dev/roadmap_0.27.0/repro/BUG-02.py
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import skforecast
import sklearn
import lightgbm
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    grid_search_forecaster,
)

warnings.filterwarnings("ignore")

print(
    f"python {sys.version.split()[0]} | skforecast {skforecast.__version__} | "
    f"pandas {pd.__version__} | numpy {np.__version__} | "
    f"scikit-learn {sklearn.__version__} | lightgbm {lightgbm.__version__}"
)

# Data: bike sharing, hourly users
# ==============================================================================
data = fetch_dataset("bike_sharing", verbose=False)
y = data["users"].astype(float)
end_train = "2012-08-31 23:00:00"
end_test = "2012-09-30 23:00:00"  # 30 days of test (720 hours)
y = y.loc[:end_test]
n_train = len(y.loc[:end_train])
STEPS = 24


def add_step(preds):
    """Horizon step of each prediction (gap=0): position inside its fold."""
    preds = preds.copy()
    preds["step"] = preds.groupby("fold").cumcount() + 1
    preds["abs_err"] = (y.loc[preds.index].to_numpy() - preds["pred"].to_numpy())
    preds["abs_err"] = preds["abs_err"].abs()
    return preds


def make_forecaster(lags=24):
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=123, verbose=-1, n_estimators=200),
        lags=lags,
    )


# 1. Single series: reported metric vs real 24 step metric
# ==============================================================================
print("\n1) backtesting_forecaster, LGBM lags=24, steps=24, refit=False")
print(f"{'fold_stride':>11} | {'reported MAE':>12} | {'MAE all pairs':>13} | "
      f"{'MAE steps 1..stride':>19} | {'MAE step 1':>10} | {'MAE step 24':>11} | "
      f"{'n preds':>7} | {'n used':>6} | steps in metric")
for stride in [24, 6, 1]:
    cv = TimeSeriesFold(
        steps=STEPS, initial_train_size=n_train, refit=False, fold_stride=stride
    )
    t0 = time.perf_counter()
    metric, preds = backtesting_forecaster(
        forecaster=make_forecaster(), y=y, cv=cv,
        metric="mean_absolute_error", show_progress=False,
    )
    elapsed = time.perf_counter() - t0
    preds = add_step(preds)
    used = preds.loc[~preds.index.duplicated(keep="last")]
    steps_used = sorted(used["step"].unique())
    steps_used_txt = (
        f"{steps_used[0]}..{steps_used[-1]}" if len(steps_used) > 1 else f"{steps_used[0]}"
    )
    share_short = (used["step"] <= stride).mean()
    print(
        f"{stride:>11} | {metric.iloc[0, 0]:>12.2f} | {preds['abs_err'].mean():>13.2f} | "
        f"{preds.loc[preds['step'] <= stride, 'abs_err'].mean():>19.2f} | "
        f"{preds.loc[preds['step'] == 1, 'abs_err'].mean():>10.2f} | "
        f"{preds.loc[preds['step'] == STEPS, 'abs_err'].mean():>11.2f} | "
        f"{len(preds):>7} | {len(used):>6} | {steps_used_txt} "
        f"({share_short:.1%} rows with step<=stride) [{elapsed:.1f}s]"
    )
    if stride == 1:
        mae_by_step = preds.groupby("step")["abs_err"].mean()
        print("   MAE by horizon step (fold_stride=1):",
              " ".join(f"{s}:{v:.0f}" for s, v in mae_by_step.items()))

print(
    "   EXPECTED: with steps=24 the metric should measure the 24 step horizon "
    "(the 'MAE all pairs' column, ~equal for every fold_stride).\n"
    "   OBSERVED: the reported MAE drops as fold_stride decreases; with "
    "fold_stride=1 it equals the 1 step ahead MAE, without any warning."
)

# 2. Grid search: ranking with fold_stride=1 vs real ranking
# ==============================================================================
print("\n2) grid_search_forecaster, steps=24, fold_stride=1, refit=False")
lags_grid = {
    "lags_3": 3,
    "lags_24": 24,
    "lags_1_2_24_168": [1, 2, 24, 168],
    "lags_seasonal": [1, 2, 3, 23, 24, 25, 167, 168, 169],
}
cv1 = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=False, fold_stride=1)
results = grid_search_forecaster(
    forecaster=make_forecaster(), y=y, cv=cv1,
    param_grid={"n_estimators": [200]}, lags_grid=lags_grid,
    metric="mean_absolute_error", return_best=False,
    show_progress=False, verbose=False,
)
rows = []
for label, lags in lags_grid.items():
    reported = results.loc[results["lags_label"] == label, "mean_absolute_error"].iloc[0]
    _, p1 = backtesting_forecaster(
        forecaster=make_forecaster(lags), y=y, cv=cv1,
        metric="mean_absolute_error", show_progress=False,
    )
    cv24 = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=False)
    m24, _ = backtesting_forecaster(
        forecaster=make_forecaster(lags), y=y, cv=cv24,
        metric="mean_absolute_error", show_progress=False,
    )
    rows.append({
        "lags": label,
        "search MAE (fold_stride=1)": round(reported, 2),
        "real MAE all pairs (fold_stride=1)": round(add_step(p1)["abs_err"].mean(), 2),
        "MAE fold_stride=24 (no overlap)": round(m24.iloc[0, 0], 2),
    })
table = pd.DataFrame(rows).set_index("lags")
table["rank search"] = table["search MAE (fold_stride=1)"].rank().astype(int)
table["rank real"] = table["real MAE all pairs (fold_stride=1)"].rank().astype(int)
print(table.to_string())
best_search = table["search MAE (fold_stride=1)"].idxmin()
best_real = table["real MAE all pairs (fold_stride=1)"].idxmin()
print(f"   search picks: {best_search} | best for the 24 step horizon: {best_real}")
print(
    "   EXPECTED: the search ranks candidates by their 24 step error.\n"
    "   OBSERVED: it ranks them by the 1 step error "
    + ("(ranking changes)." if best_search != best_real else
       "(same winner here, but the gaps between candidates are distorted).")
)

# 3. Multiseries: same deduplication
# ==============================================================================
print("\n3) backtesting_forecaster_multiseries, 2 series, steps=24, fold_stride=1")
series = pd.DataFrame({
    "users": y,
    "users_x2_noise": y * 2 + np.random.default_rng(0).normal(0, 20, len(y)),
})
fc_ms = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=123, verbose=-1, n_estimators=200),
    lags=24, encoding="ordinal",
)
cv_ms = TimeSeriesFold(
    steps=STEPS, initial_train_size=len(y) - 24 * 10, refit=False, fold_stride=1
)
m_ms, p_ms = backtesting_forecaster_multiseries(
    forecaster=fc_ms, series=series, cv=cv_ms, metric="mean_absolute_error",
    add_aggregated_metric=False, show_progress=False,
)
p_ms = p_ms.copy()
p_ms["step"] = p_ms.groupby(["fold", "level"]).cumcount() + 1
p_ms["abs_err"] = [
    abs(series.at[idx, lvl] - pr) for idx, lvl, pr in
    zip(p_ms.index, p_ms["level"], p_ms["pred"])
]
for lvl in series.columns:
    sub = p_ms[p_ms["level"] == lvl]
    reported = m_ms.loc[m_ms["levels"] == lvl, "mean_absolute_error"].iloc[0]
    print(
        f"   {lvl:>15}: reported {reported:8.2f} | all pairs {sub['abs_err'].mean():8.2f} | "
        f"step 1 {sub.loc[sub['step'] == 1, 'abs_err'].mean():8.2f}"
    )
