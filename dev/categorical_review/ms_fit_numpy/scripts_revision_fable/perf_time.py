"""Timing, base vs branch in the same process, alternating blocks.

Usage: perf_time.py <estimator> <scenario> [n_series] [n_obs] [reps]
Pinned to one core, single thread. Prints min / median / max per operation.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import time
import warnings
import importlib
import numpy as np
import pandas as pd

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
import common as C
C.pin_core(12)

est_name = sys.argv[1]
scenario = sys.argv[2]
n_series = int(sys.argv[3]) if len(sys.argv) > 3 else 300
n_obs = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 5

PATHS = {
    "base": os.path.join(W, "base"),
    "branch": "<repo>",
}


def load(tag):
    for k in list(sys.modules):
        if k == "skforecast" or k.startswith("skforecast."):
            del sys.modules[k]
    sys.path.insert(0, PATHS[tag])
    import skforecast
    rec = importlib.import_module("skforecast.recursive")
    ms = importlib.import_module("skforecast.model_selection")
    sys.path.remove(PATHS[tag])
    print(tag, skforecast.__file__, flush=True)
    return {"rec": rec, "ms": ms}


mods = {"base": load("base"), "branch": load("branch")}


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


def make_data(scenario, n_series, n_obs, n_fut=24):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=n_obs + n_fut, freq="D")
    series = pd.DataFrame(
        rng.normal(size=(n_obs, n_series)).cumsum(axis=0),
        index=idx[:n_obs], columns=[f"s{i}" for i in range(n_series)]
    )
    exog = pd.DataFrame(
        rng.normal(size=(n_obs + n_fut, 10)), index=idx,
        columns=[f"x{i}" for i in range(10)]
    )
    if scenario == "mixed":
        exog["x5"] = rng.integers(0, 100, size=len(idx)).astype("int64")
        exog["x6"] = rng.integers(0, 7, size=len(idx)).astype("int32")
        exog["x7"] = rng.integers(0, 2, size=len(idx)).astype(bool)
        exog["x8"] = pd.Categorical(rng.integers(0, 12, size=len(idx)))
        exog["x9"] = pd.Categorical(
            np.array(["a", "b", "c", "d"])[rng.integers(0, 4, size=len(idx))]
        )
    return series, exog.iloc[:n_obs], exog.iloc[n_obs:]


series, exog, exog_fut = make_data(scenario, n_series, n_obs)
enc = "ordinal_category" if scenario == "mixed" else "ordinal"
n_train = n_obs - 100
GRID = {"lgbm": {"num_leaves": [15, 31]}, "xgb": {"max_depth": [4, 6]},
        "hgb": {"max_depth": [4, None]}, "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "catboost": {"depth": [4, 6]}}[est_name]


def new(tag):
    return mods[tag]["rec"].ForecasterRecursiveMultiSeries(
        estimator=make_estimator(est_name), lags=24, encoding=enc,
    )


def op_fit(tag):
    f = new(tag)
    t = time.perf_counter()
    f.fit(series=series, exog=exog)
    return time.perf_counter() - t, f


def op_predict(tag, f):
    t = time.perf_counter()
    f.predict(steps=24, exog=exog_fut)
    return time.perf_counter() - t


def op_backtest(tag):
    ms = mods[tag]["ms"]
    f = new(tag)
    cv = ms.TimeSeriesFold(steps=10, initial_train_size=n_train, refit=False)
    t = time.perf_counter()
    ms.backtesting_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv,
        metric="mean_absolute_error", n_jobs=1, show_progress=False,
    )
    return time.perf_counter() - t


def op_search(tag):
    ms = mods[tag]["ms"]
    f = new(tag)
    cv = ms.OneStepAheadFold(initial_train_size=n_train)
    t = time.perf_counter()
    ms.grid_search_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv, param_grid=GRID,
        lags_grid=[24], metric="mean_absolute_error",
        aggregate_metric="weighted_average", return_best=False, n_jobs=1,
        show_progress=False, verbose=False,
    )
    return time.perf_counter() - t


times = {(op, tag): [] for op in ("fit", "predict", "backtest", "search") for tag in PATHS}
warnings.simplefilter("ignore")
# warm-up (numba, imports), not recorded
for tag in PATHS:
    _, f = op_fit(tag)
    op_predict(tag, f)
for r in range(reps):
    order = ["base", "branch"] if r % 2 == 0 else ["branch", "base"]
    for tag in order:
        t, f = op_fit(tag)
        times[("fit", tag)].append(t)
        times[("predict", tag)].append(op_predict(tag, f))
        del f
    for tag in order:
        times[("backtest", tag)].append(op_backtest(tag))
    for tag in order:
        times[("search", tag)].append(op_search(tag))

import sklearn
print(f"python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__} "
      f"sklearn {sklearn.__version__} | est={est_name} scenario={scenario} "
      f"n_series={n_series} n_obs={n_obs} rows~{n_series * (n_obs - 24)} reps={reps}")
for op in ("fit", "predict", "backtest", "search"):
    b = np.array(times[(op, "base")])
    n = np.array(times[(op, "branch")])
    print(
        f"RESULT {est_name:8} {scenario:6} {op:9} "
        f"base min {b.min():7.3f} med {np.median(b):7.3f} max {b.max():7.3f} | "
        f"branch min {n.min():7.3f} med {np.median(n):7.3f} max {n.max():7.3f} | "
        f"ratio(min) {n.min() / b.min():5.3f} ratio(med) {np.median(n) / np.median(b):5.3f}"
    )
