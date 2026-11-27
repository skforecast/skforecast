"""Peak memory of one operation in a fresh process (peak working set).

Usage (through run.sh): perf_mem.py <estimator> <scenario> <op> [n_series] [n_obs]
Prints the RSS before the operation and the peak of the process after it.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import warnings
import numpy as np
import pandas as pd
import psutil
import skforecast
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)

est_name, scenario, op = sys.argv[1:4]
n_series = int(sys.argv[4]) if len(sys.argv) > 4 else 300
n_obs = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
warnings.simplefilter("ignore")


def make_estimator(name):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=50, n_jobs=1, random_state=1)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=30, random_state=1)
    if name == "ridge":
        return Ridge()
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=30, verbose=0, random_seed=1,
                                 thread_count=1, allow_writing_files=False)
    raise ValueError(name)


rng = np.random.default_rng(0)
n_fut = 24
idx = pd.date_range("2018-01-01", periods=n_obs + n_fut, freq="D")
series = pd.DataFrame(
    rng.normal(size=(n_obs, n_series)).cumsum(axis=0),
    index=idx[:n_obs], columns=[f"s{i}" for i in range(n_series)]
)
exog_all = pd.DataFrame(
    rng.normal(size=(n_obs + n_fut, 10)), index=idx,
    columns=[f"x{i}" for i in range(10)]
)
if scenario == "mixed":
    exog_all["x5"] = rng.integers(0, 100, size=len(idx)).astype("int64")
    exog_all["x6"] = rng.integers(0, 7, size=len(idx)).astype("int32")
    exog_all["x7"] = rng.integers(0, 2, size=len(idx)).astype(bool)
    exog_all["x8"] = pd.Categorical(rng.integers(0, 12, size=len(idx)))
    exog_all["x9"] = pd.Categorical(
        np.array(["a", "b", "c", "d"])[rng.integers(0, 4, size=len(idx))]
    )
exog, exog_fut = exog_all.iloc[:n_obs], exog_all.iloc[n_obs:]
enc = "ordinal_category" if scenario == "mixed" else "ordinal"
n_train = n_obs - 100
GRID = {"lgbm": {"num_leaves": [15, 31]}, "xgb": {"max_depth": [4, 6]},
        "hgb": {"max_depth": [4, None]}, "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "catboost": {"depth": [4, 6]}}[est_name]

f = ForecasterRecursiveMultiSeries(
    estimator=make_estimator(est_name), lags=24, encoding=enc,
)
# warm-up on a tiny problem: imports, numba compilation
fw = ForecasterRecursiveMultiSeries(estimator=make_estimator(est_name), lags=24, encoding=enc)
fw.fit(series=series.iloc[:60, :3], exog=exog.iloc[:60])
del fw

proc = psutil.Process()
rss_before = proc.memory_info().rss
peak_before = proc.memory_info().peak_wset
t = time.perf_counter()
if op == "fit":
    f.fit(series=series, exog=exog)
elif op == "search":
    grid_search_forecaster_multiseries(
        forecaster=f, series=series, exog=exog,
        cv=OneStepAheadFold(initial_train_size=n_train), param_grid=GRID,
        lags_grid=[24], metric="mean_absolute_error",
        aggregate_metric="weighted_average", return_best=False, n_jobs=1,
        show_progress=False, verbose=False,
    )
elif op == "backtest":
    backtesting_forecaster_multiseries(
        forecaster=f, series=series, exog=exog,
        cv=TimeSeriesFold(steps=10, initial_train_size=n_train, refit=False),
        metric="mean_absolute_error", n_jobs=1, show_progress=False,
    )
elapsed = time.perf_counter() - t
peak_after = proc.memory_info().peak_wset
mb = 1024 ** 2
print(
    f"MEM {os.environ.get('TREE'):6} {est_name:8} {scenario:6} {op:8} "
    f"rss_before {rss_before / mb:8.1f} MB  peak {peak_after / mb:8.1f} MB  "
    f"peak-rss_before {(peak_after - rss_before) / mb:8.1f} MB  time {elapsed:6.2f} s"
)
