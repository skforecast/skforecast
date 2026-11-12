#!/usr/bin/env python
"""
Benchmark of the parallelization strategy (`n_jobs`) of skforecast.

It measures, for the estimators most used with skforecast (Ridge, LightGBM,
XGBoost, HistGradientBoosting, CatBoost and RandomForest), when running the
folds of a backtesting or of a hyperparameter search in parallel pays off, and
compares it with what `n_jobs='auto'` chooses.

Usage (see README.md):

    python bench_n_jobs.py check             # environment and plan, 1 minute
    python bench_n_jobs.py run --smoke       # every code path, tiny sizes
    python bench_n_jobs.py run               # the benchmark (resumable)
    python bench_n_jobs.py summary           # tables from the results

Method
------
- Each measure runs in a new process (`child`), which calls the operation
  several times: the first call pays the start of the workers, as a script
  does ("first"), and the following ones reuse them, as the cells of a
  notebook do ("next").
- Smaller machines are emulated by pinning the process to k CPUs (one thread
  per physical core while there are cores left), before any library is
  imported: joblib, OpenMP, LightGBM and CatBoost then see a machine of k CPUs.
- Configurations of `n_jobs`: `1`, `max` (`cpu_count() - 1`, what 'auto' uses
  when it runs in parallel), `auto`, and `half` / `all` in some cases.
- The measures are ordered in stages (one repetition of everything first) and
  shuffled inside each stage. The run stops at the time budget and can be
  resumed: every measure is appended to `results.jsonl` when it finishes.

Nothing is written outside the folder of this script.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
SCHEMA = 1

# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------
END_TRAIN = "2012-03-31 23:00:00"
END_VALIDATION = "2012-08-31 23:00:00"
START = {"S": "2011-09-01 00:00:00", "L": "2011-01-08 00:00:00"}
EXOG = [
    "month_sin", "month_cos", "week_day_sin", "week_day_cos",
    "hour_day_sin", "hour_day_cos", "temp", "holiday",
]
LAGS = 48
WINDOW = 72

# Estimators. `*_1t`: one thread in the estimator. `rf_all`: all the threads.
# `lgbm`, `xgb`, `hgb` and `catboost` use the default threads of each library
# (all the processors); `rf` uses the default of scikit-learn (one thread).
ESTIMATORS = {
    "ridge": {},
    "lgbm": {},
    "lgbm_1t": {"n_jobs": 1},
    "xgb": {},
    "xgb_1t": {"n_jobs": 1},
    "hgb": {},
    "catboost": {},
    "catboost_1t": {"thread_count": 1},
    "rf": {},
    "rf_all": {"n_jobs": -1},
}
# The estimators use the default parameters of their libraries, except the
# number of trees: `--trees` in the boosting models and `--rf-trees` in
# RandomForest (its default, since its trees are fully grown and each fit
# takes much longer).
N_TREES = 500
RF_TREES = 100
TREE_PARAM = {
    "lgbm": "n_estimators", "xgb": "n_estimators", "hgb": "max_iter",
    "catboost": "iterations", "rf": "n_estimators",
}
CORE = ["ridge", "lgbm", "lgbm_1t", "xgb", "hgb", "catboost", "rf"]
THREAD_VARIANTS = ["xgb_1t", "catboost_1t", "rf_all"]
MULTISERIES = ["ridge", "lgbm", "lgbm_1t", "xgb", "hgb"]
DIRECT = ["ridge", "lgbm", "lgbm_1t", "xgb", "hgb", "rf"]
PREDICT = ["lgbm", "xgb", "hgb", "catboost"]
SMALL_AUTO = ["lgbm_1t", "xgb", "hgb"]
# Training rows of the `lgbm_1t` ladder: a dial of the work of a backtesting
# with refit (16 folds), from a short series to a medium one, to find where
# running in parallel starts to pay off.
LADDER = [200, 300, 400, 550, 800]

PARAM_GRIDS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "lgbm": {"learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "xgb": {"learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "hgb": {"learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "catboost": {"learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "rf": {"min_samples_leaf": [1, 2, 4, 8]},
}

# `n_val`: rows after the training set that are predicted (None: up to
# END_VALIDATION, 3672 hours). `calls`: calls in each process.
CASES = {
    # 100 folds of 24 steps, the forecaster is fitted once.
    "bt_norefit": {"op": "bt", "steps": 24, "refit": False, "n_val": 2400, "calls": 3},
    # 16 folds of 240 steps, 15 refits: enough tasks for 15 workers.
    "bt_refit": {"op": "bt", "steps": 240, "refit": True, "n_val": None, "calls": 2},
    # 6 folds of 168 steps, 5 refits: fewer tasks than workers.
    "bt_refit_small": {"op": "bt", "steps": 168, "refit": True, "n_val": 1008, "calls": 3},
    # Grid search of 3 candidates, 40 folds of 24 steps each, without refit.
    "gs_norefit": {"op": "gs", "steps": 24, "refit": False, "n_val": 960, "calls": 2,
                   "n_candidates": 3},
    # Grid search of 3 candidates, 6 folds with refit each.
    "gs_refit_small": {"op": "gs", "steps": 168, "refit": True, "n_val": 1008,
                       "calls": 2, "n_candidates": 3},
    # 50 daily series. 52 folds of 7 steps without refit.
    "ms_norefit": {"op": "ms", "steps": 7, "refit": False, "n_val": 364, "calls": 2},
    # 50 daily series. 8 folds of 28 steps, 7 refits.
    "ms_refit": {"op": "ms", "steps": 28, "refit": True, "n_val": 224, "calls": 2},
    # Fit of a ForecasterDirect of 12 steps (12 estimators).
    "direct_fit": {"op": "direct", "steps": 12, "refit": True, "n_val": None,
                   "calls": 3},
}
for _rows in LADDER:
    CASES[f"ladder_{_rows:04d}"] = {
        "op": "bt", "steps": 24, "refit": True, "n_val": 384, "calls": 3,
        "n_train_rows": _rows,
    }

LONG_CALL = 8.0         # a first call longer than this is followed by one call
CELL_TIMEOUT = 900      # seconds
RISKY_TIMEOUT = 240     # LightGBM with its threads inside the workers
STRIP_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_PROC_BIND", "OMP_PLACES",
    "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "LOKY_MAX_CPU_COUNT",
)


def base_estimator(name: str) -> str:
    return name.split("_")[0]


# ----------------------------------------------------------------------------
# Machine
# ----------------------------------------------------------------------------
def allowed_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return list(range(os.cpu_count() or 1))


def read_cores() -> list[list[int]]:
    """Logical CPUs of each physical core, ordered by their first CPU."""
    cores = {}
    for cpu in allowed_cpus():
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                int((topology / "physical_package_id").read_text()),
                int((topology / "core_id").read_text()),
            )
        except (OSError, ValueError):
            key = (-1, cpu)
        cores.setdefault(key, []).append(cpu)
    return sorted((sorted(c) for c in cores.values()), key=lambda c: c[0])


def cpuset_for(n_cpus: int, cores: list[list[int]]) -> list[int]:
    """
    CPUs that emulate a machine of `n_cpus`: one thread per physical core
    while there are cores left, then their siblings.
    """
    every = sorted(cpu for core in cores for cpu in core)
    if n_cpus >= len(every):
        return every
    cpus = []
    depth = 0
    while len(cpus) < n_cpus and depth < 16:
        for core in cores:
            if len(core) > depth and len(cpus) < n_cpus:
                cpus.append(core[depth])
        depth += 1
    return sorted(cpus)


def loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError):
        return None


def cpu_busy(seconds: float = 0.25) -> float | None:
    """
    Fraction of the CPUs of the machine in use during `seconds`. Sampled before
    each measure, when no process of the benchmark is running, it tells
    whether something else was using the machine.
    """
    def read():
        fields = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return idle, sum(fields)

    try:
        idle_0, total_0 = read()
        time.sleep(seconds)
        idle_1, total_1 = read()
        elapsed = total_1 - total_0
        return 1 - (idle_1 - idle_0) / elapsed if elapsed > 0 else None
    except (OSError, ValueError, IndexError):
        return None


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "os_cpu_count": os.cpu_count(),
        "allowed_cpus": allowed_cpus(),
        "cores": read_cores(),
        "env": {k: os.environ[k] for k in STRIP_ENV if k in os.environ},
    }
    try:
        info["lscpu"] = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        info["lscpu"] = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["mem_total_gb"] = round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=HERE, timeout=10,
        ).stdout.strip() or None
        info["git_status"] = subprocess.run(
            ["git", "status", "--short", "--", "../../skforecast"],
            capture_output=True, text=True, cwd=HERE, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


# ----------------------------------------------------------------------------
# Data, estimators and forecasters (child processes only)
# ----------------------------------------------------------------------------
def _cached(name: str, build):
    import pandas as pd

    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        build().to_pickle(tmp)
        os.replace(tmp, path)
    return pd.read_pickle(path)


def load_bike():
    """Hourly users of a bike sharing system and 8 exogenous variables."""
    def build():
        from skforecast.datasets import fetch_dataset

        data = fetch_dataset("bike_sharing_extended_features", verbose=False)
        return data.loc[:END_VALIDATION, ["users"] + EXOG].astype("float64")

    return _cached("bike_sharing.pkl", build)


def load_sales():
    """Daily sales of the 50 items of one store (wide format)."""
    def build():
        from skforecast.datasets import fetch_dataset

        sales = fetch_dataset("store_sales", raw=False, verbose=False)
        sales = sales.query("store == 1").pivot(columns="item", values="sales")
        sales = sales.asfreq("D").astype("float64")
        sales.columns = [f"item_{item}" for item in sales.columns]
        sales.columns.name = None
        return sales

    return _cached("store_sales.pkl", build)


def make_estimator(name: str, trees: int | None = None):
    params = dict(ESTIMATORS[name])
    base = base_estimator(name)
    if base in TREE_PARAM:
        params[TREE_PARAM[base]] = trees or (RF_TREES if base == "rf" else N_TREES)
    if base == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**{"alpha": 1.0, **params})
    if base == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**{"random_state": 123, "verbose": -1, **params})
    if base == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(**{"random_state": 123, **params})
    if base == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(**{"random_state": 123, **params})
    if base == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **{"random_seed": 123, "verbose": 0, "allow_writing_files": False,
               **params}
        )
    if base == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**{"random_state": 123, **params})
    raise ValueError(name)


def make_recursive(est: str, trees: int | None = None, window_features=True):
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursive

    return ForecasterRecursive(
        estimator       = make_estimator(est, trees),
        lags            = LAGS,
        window_features = (
            RollingFeatures(stats=["mean"], window_sizes=WINDOW)
            if window_features else None
        ),
    )


def signature(values) -> list:
    """Numbers that identify a result, compared with a tolerance."""
    import numpy as np

    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [0]
    positions = np.linspace(0, array.size - 1, 7).astype(int)
    return [int(array.size), float(array.mean()), float(array.std())] + [
        float(v) for v in array[positions]
    ]


def case_trees(spec: dict) -> int:
    """Trees of the estimator of a case (at most 20 in smoke mode)."""
    if base_estimator(spec["est"]) == "rf":
        trees = spec.get("rf_trees") or RF_TREES
    else:
        trees = spec.get("trees") or N_TREES
    return min(trees, 20) if spec.get("smoke") else trees


def build_case(spec: dict) -> dict:
    """
    Return the functions of a case: `run(n_jobs)` executes the skforecast call
    and returns the signature of its result, `fit_once()` fits the forecaster
    on the initial training set (None if the case has no such fit), and `info`
    describes the case.
    """
    from skforecast.model_selection import TimeSeriesFold

    op = spec["op"]
    est = spec["est"]
    trees = case_trees(spec)
    steps = spec["steps"]
    refit = spec["refit"]
    metric = "mean_absolute_error"
    common = {"verbose": False, "show_progress": False}

    if op in ("bt", "gs", "direct"):
        data = load_bike().loc[START[spec.get("size", "S")]:]
        n_train = len(data.loc[:END_TRAIN])
        if spec.get("n_train_rows"):
            data = data.iloc[n_train - spec["n_train_rows"]:]
            n_train = spec["n_train_rows"]
        if spec.get("n_val"):
            data = data.iloc[: n_train + spec["n_val"]]
        y = data["users"]
        exog = data.drop(columns="users")
        n_val = len(data) - n_train
    else:
        series = load_sales()
        n_val = spec["n_val"]
        n_train = len(series) - n_val

    n_folds = math.ceil(n_val / steps)
    info = {
        "n_train": n_train, "n_val": n_val, "n_tasks": n_folds,
        "n_fits": n_folds - 1 if refit else 0, "trees": trees,
    }

    def new_cv():
        return TimeSeriesFold(
            steps=steps, initial_train_size=n_train, refit=refit, verbose=False
        )

    if op in ("bt", "gs"):
        from skforecast.model_selection import (
            backtesting_forecaster,
            grid_search_forecaster,
        )
        from skforecast.model_selection._utils import select_n_jobs_backtesting

        def new_forecaster():
            return make_recursive(est, trees)

        if op == "bt":
            def run(n_jobs):
                _, predictions = backtesting_forecaster(
                    forecaster=new_forecaster(), y=y, exog=exog, cv=new_cv(),
                    metric=metric, n_jobs=n_jobs, **common,
                )
                return signature(predictions["pred"])
        else:
            param_grid = {
                name: values[: spec.get("n_candidates") or len(values)]
                for name, values in PARAM_GRIDS[base_estimator(est)].items()
            }
            info["n_candidates"] = len(next(iter(param_grid.values())))

            def run(n_jobs):
                results = grid_search_forecaster(
                    forecaster=new_forecaster(), y=y, exog=exog, cv=new_cv(),
                    param_grid=param_grid, metric=metric, return_best=False,
                    n_jobs=n_jobs, **common,
                )
                return signature(results[metric])

        def fit_once():
            forecaster = new_forecaster()
            start = time.perf_counter()
            forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
            return time.perf_counter() - start, forecaster

        info["auto_rule"] = select_n_jobs_backtesting(
            forecaster=new_forecaster(), refit=refit
        )

    elif op == "ms":
        from skforecast.model_selection import backtesting_forecaster_multiseries
        from skforecast.model_selection._utils import select_n_jobs_backtesting
        from skforecast.recursive import ForecasterRecursiveMultiSeries

        def new_forecaster():
            return ForecasterRecursiveMultiSeries(
                estimator=make_estimator(est, trees), lags=14,
                encoding="ordinal",
            )

        def run(n_jobs):
            _, predictions = backtesting_forecaster_multiseries(
                forecaster=new_forecaster(), series=series, cv=new_cv(),
                metric=metric, n_jobs=n_jobs, **common,
            )
            return signature(predictions["pred"])

        def fit_once():
            forecaster = new_forecaster()
            start = time.perf_counter()
            forecaster.fit(series=series.iloc[:n_train])
            return time.perf_counter() - start, forecaster

        info["n_series"] = series.shape[1]
        info["auto_rule"] = select_n_jobs_backtesting(
            forecaster=new_forecaster(), refit=refit
        )

    elif op == "direct":
        from skforecast.direct import ForecasterDirect
        from skforecast.preprocessing import RollingFeatures
        from skforecast.utils.utils import select_n_jobs_fit_forecaster

        y_train = y.iloc[:n_train]
        exog_train = exog.iloc[:n_train]
        exog_next = exog.iloc[n_train: n_train + steps]

        def run(n_jobs):
            forecaster = ForecasterDirect(
                estimator=make_estimator(est, trees), steps=steps, lags=LAGS,
                window_features=RollingFeatures(stats=["mean"], window_sizes=WINDOW),
                n_jobs=n_jobs,
            )
            forecaster.fit(y=y_train, exog=exog_train)
            return signature(forecaster.predict(exog=exog_next))

        fit_once = None
        info.update({"n_tasks": steps, "n_fits": steps, "n_val": steps})
        info["auto_rule"] = select_n_jobs_fit_forecaster(
            forecaster_name="ForecasterDirect",
            estimator=make_estimator(est, trees),
        )
    else:
        raise ValueError(op)

    return {"run": run, "fit_once": fit_once, "info": info}


# ----------------------------------------------------------------------------
# Instrumentation (main process of the child only; the workers import the
# original classes, so nothing is recorded or slowed down in them)
# ----------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.parallel = []   # n_jobs of each joblib.Parallel of skforecast
        self.fits = []       # seconds of each fit of a forecaster
        self.limits = []     # calls to limit_n_jobs_by_workload


def install_instrumentation(recorder: Recorder) -> None:
    import importlib

    import joblib

    class RecordingParallel(joblib.Parallel):
        def __init__(self, *args, **kwargs):
            recorder.parallel.append(kwargs.get("n_jobs", args[0] if args else None))
            super().__init__(*args, **kwargs)

    for name in (
        "skforecast.model_selection._validation",
        "skforecast.direct._forecaster_direct",
        "skforecast.direct._forecaster_direct_multivariate",
    ):
        module = importlib.import_module(name)
        if hasattr(module, "Parallel"):
            module.Parallel = RecordingParallel

    validation = importlib.import_module("skforecast.model_selection._validation")
    if hasattr(validation, "limit_n_jobs_by_workload"):
        original_limit = validation.limit_n_jobs_by_workload

        def limit_n_jobs_by_workload(*args, **kwargs):
            result = original_limit(*args, **kwargs)
            recorder.limits.append(
                {**{k: v for k, v in kwargs.items()}, "result": result}
            )
            return result

        validation.limit_n_jobs_by_workload = limit_n_jobs_by_workload

    from skforecast.direct import ForecasterDirect
    from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

    for cls in (ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterDirect):
        def patch(cls=cls):
            original_fit = cls.fit

            @functools.wraps(original_fit)
            def fit(self, *args, **kwargs):
                start = time.perf_counter()
                try:
                    return original_fit(self, *args, **kwargs)
                finally:
                    recorder.fits.append(time.perf_counter() - start)

            cls.fit = fit

        patch()


def resolve_config(config: str, n_cpus: int):
    if config == "auto":
        return "auto"
    if config == "max":
        return max(1, n_cpus - 1)
    if config == "all":
        return n_cpus
    if config == "half":
        return max(2, n_cpus // 2)
    return int(config)


def process_memory() -> dict:
    """Memory of this process and of its descendants (the workers), in MB."""
    def status(pid, field):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    out = {
        "main_rss_mb": status("self", "VmRSS"),
        "main_peak_mb": status("self", "VmHWM"),
    }
    try:
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    stat = Path(f"/proc/{entry}/stat").read_text()
                    parents[int(entry)] = int(stat.rsplit(")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
        family = {os.getpid()}
        grew = True
        while grew:
            grew = False
            for pid, parent in parents.items():
                if parent in family and pid not in family:
                    family.add(pid)
                    grew = True
        family.discard(os.getpid())
        rss = [status(pid, "VmRSS") for pid in family]
        out["n_descendants"] = len(family)
        out["descendants_rss_mb"] = sum(r for r in rss if r)
    except OSError:
        pass
    return out


def thread_pools() -> list:
    from threadpoolctl import threadpool_info

    return [
        [lib["user_api"], Path(lib["filepath"]).name, lib["num_threads"]]
        for lib in threadpool_info()
    ]


def os_threads() -> int | None:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


# ----------------------------------------------------------------------------
# Operations of the child processes
# ----------------------------------------------------------------------------
def op_timed(spec: dict) -> dict:
    """Backtesting, search or fit: one first call and the following ones."""
    import pickle

    import joblib

    start = time.perf_counter()
    case = build_case(spec)
    recorder = Recorder()
    install_instrumentation(recorder)
    setup_seconds = time.perf_counter() - start

    n_cpus = joblib.cpu_count()
    n_jobs = resolve_config(spec["config"], n_cpus)
    calls = []
    n_calls = spec["calls"]
    while len(calls) < n_calls:
        recorder.reset()
        executor_before = _executor_state()
        start = time.perf_counter()
        result = case["run"](n_jobs)
        seconds = time.perf_counter() - start
        calls.append({
            "seconds": seconds,
            "executor_before": executor_before,
            "executor_after": _executor_state(),
            "signature": result,
            "parallel_n_jobs": sorted(set(recorder.parallel)),
            "n_parallel": len(recorder.parallel),
            "n_fits_main": len(recorder.fits),
            "fit_first": recorder.fits[0] if recorder.fits else None,
            "fits_main_sum": sum(recorder.fits),
            "limits": recorder.limits[:3],
        })
        if len(calls) == 1 and seconds > LONG_CALL:
            n_calls = min(n_calls, 2)

    post = process_memory()
    post["os_threads"] = os_threads()
    if case["fit_once"] is not None:
        post["fit_after"], fitted = case["fit_once"]()
        start = time.perf_counter()
        blob = pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)
        post["pickle_seconds"] = time.perf_counter() - start
        post["pickle_mb"] = len(blob) / 1e6
    post["thread_pools"] = thread_pools()

    return {
        "n_cpus": n_cpus,
        "n_jobs": n_jobs,
        "setup_seconds": setup_seconds,
        "calls": calls,
        "info": case["info"],
        "post": post,
    }


def _start_task(forecaster, seconds):
    """Task of the start block: one prediction, then a short wait."""
    forecaster.predict(steps=2)
    time.sleep(seconds)
    return os.getpid()


def _small_forecaster(window_features: bool):
    data = load_bike()
    forecaster = make_recursive("lgbm_1t", 20, window_features=window_features)
    forecaster.fit(y=data["users"].iloc[:2000])
    return forecaster


def _executor_state() -> dict:
    """State of the reusable executor of loky (private attributes)."""
    try:
        from joblib.externals.loky import reusable_executor

        executor = reusable_executor._executor
        if executor is None:
            return {"executor_id": None, "n_processes": 0}
        processes = list(executor._processes.values())
        return {
            "executor_id": executor.executor_id,
            "n_processes": sum(process.is_alive() for process in processes),
            "pids": sorted(process.pid for process in processes),
        }
    except Exception as error:  # the attributes are private
        return {"error": repr(error)}


def op_start(spec: dict) -> dict:
    """
    Cost of starting k workers that receive a fitted forecaster and predict
    with it: every worker runs one task (they wait 0.2 seconds, so an idle
    worker takes the next one).
    """
    import joblib
    from joblib import Parallel, delayed

    forecaster = _small_forecaster(spec.get("window_features", True))
    n_cpus = joblib.cpu_count()
    n_jobs = resolve_config(spec["config"], n_cpus)
    wait = 0.2
    calls = []
    for _ in range(spec["calls"]):
        start = time.perf_counter()
        pids = Parallel(n_jobs=n_jobs)(
            delayed(_start_task)(forecaster, wait) for _ in range(n_jobs)
        )
        calls.append({
            "seconds": time.perf_counter() - start,
            "n_workers_used": len(set(pids)),
        })
    return {"n_cpus": n_cpus, "n_jobs": n_jobs, "wait": wait, "calls": calls,
            "post": process_memory()}


def op_resize(spec: dict) -> dict:
    """
    Cost of changing the number of workers between calls. loky reuses the
    workers, but it starts them all again when `cpu_count() // n_jobs` (the
    threads it gives to each worker) changes.
    """
    import joblib
    from joblib import Parallel, delayed

    forecaster = _small_forecaster(True)
    n_cpus = joblib.cpu_count()
    n_max = max(2, n_cpus - 1)
    sequence = [n_max, n_max, max(2, n_cpus // 2), n_max, max(2, n_max - 3),
                n_max, 2, n_max]
    wait = 0.1
    calls = []
    for n_jobs in sequence:
        start = time.perf_counter()
        pids = Parallel(n_jobs=n_jobs)(
            delayed(_start_task)(forecaster, wait) for _ in range(n_jobs)
        )
        calls.append({
            "n_jobs": n_jobs,
            "seconds": time.perf_counter() - start,
            "threads_per_worker": max(n_cpus // n_jobs, 1),
            "n_workers_used": len(set(pids)),
            **_executor_state(),
        })
    return {"n_cpus": n_cpus, "wait": wait, "calls": calls}


def op_predict(spec: dict) -> dict:
    """
    Threads of the prediction. `micro`: microseconds per call of the predict
    method of the estimator by number of rows and threads. `loop`: seconds of
    the predictions of 40 folds of 24 steps with the forecaster, as skforecast
    predicts today and with one thread.
    """
    import joblib
    import numpy as np
    import skforecast.utils.utils as sk_utils
    from threadpoolctl import ThreadpoolController

    est = spec["est"]
    base = base_estimator(est)
    data = load_bike().loc[START[spec.get("size", "S")]:]
    n_train = len(data.loc[:END_TRAIN])
    y = data["users"]
    exog = data.drop(columns="users")
    forecaster = make_recursive(est, case_trees(spec))
    forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
    X_train, _ = forecaster.create_train_X_y(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
    X = np.ascontiguousarray(X_train.to_numpy(dtype=float))
    estimator = forecaster.estimator
    controller = ThreadpoolController()
    n_cpus = joblib.cpu_count()
    thread_counts = [k for k in (1, 2, 4, 8) if k < n_cpus]
    reps = spec.get("reps_inner", 4)

    def limited(function, k):
        def call(X_rows):
            with controller.limit(limits=k, user_api="openmp"):
                return function(X_rows)
        return call

    # ---- micro ------------------------------------------------------------
    variants = {}
    if base == "xgb":
        booster = estimator.get_booster()
        variants["default"] = booster.inplace_predict
        for k in thread_counts:
            copy_k = booster.copy()
            copy_k.set_param({"nthread": k})
            variants[f"native={k}"] = copy_k.inplace_predict
            variants[f"limit={k}"] = limited(booster.inplace_predict, k)
    elif base == "hgb":
        variants["default"] = estimator.predict
        for k in thread_counts:
            variants[f"limit={k}"] = limited(estimator.predict, k)
    elif base == "lgbm":
        booster = estimator.booster_
        # Threads of `LGBMRegressor.predict`, and of skforecast, with the
        # default `n_jobs`: the physical cores.
        default_threads = joblib.cpu_count(only_physical_cores=True)
        variants["default"] = functools.partial(
            booster.predict, num_threads=default_threads
        )
        for k in thread_counts:
            variants[f"native={k}"] = functools.partial(booster.predict, num_threads=k)
    elif base == "catboost":
        variants["default"] = estimator.predict
        for k in thread_counts:
            variants[f"native={k}"] = functools.partial(estimator.predict, thread_count=k)
    else:
        raise ValueError(est)

    micro = {}
    for n_rows in spec.get("rows", (1, 5, 10, 25, 50, 100, 500)):
        X_rows = X[:n_rows].copy()
        reference = np.asarray(variants["default"](X_rows)).ravel()
        start = time.perf_counter()
        for _ in range(3):
            variants["default"](X_rows)
        per_call = (time.perf_counter() - start) / 3
        n_calls = int(min(2000, max(5, 0.08 / max(per_call, 1e-6))))
        times = {name: [] for name in variants}
        for _ in range(reps):
            names = list(variants)
            random.shuffle(names)
            for name in names:
                function = variants[name]
                start = time.perf_counter()
                for _ in range(n_calls):
                    function(X_rows)
                times[name].append((time.perf_counter() - start) / n_calls * 1e6)
        micro[str(n_rows)] = {
            name: {
                "median": statistics.median(values),
                "min": min(values),
                "same": bool(np.allclose(
                    np.asarray(variants[name](X_rows)).ravel(), reference,
                    rtol=1e-6, atol=1e-6,
                )),
            }
            for name, values in times.items()
        }

    out = {
        "n_cpus": n_cpus,
        "physical_cores": joblib.cpu_count(only_physical_cores=True),
        "n_features": int(X.shape[1]),
        "trees": case_trees(spec),
        "micro": micro,
        "thread_pools": thread_pools(),
    }
    if not spec.get("loop", True):
        return out

    # ---- loop -------------------------------------------------------------
    state = {"mode": "current"}
    copies = {}

    def install(cls, method, native):
        original = getattr(cls, method)

        def patched(self, data, *args, **kwargs):
            mode = state["mode"]
            if mode == "no_estimator":
                return np.zeros(data.shape[0])
            if mode == "native=1" and native is not None:
                return native(original, self, data, args, kwargs)
            return original(self, data, *args, **kwargs)

        setattr(cls, method, patched)
        return original

    if base == "xgb":
        import xgboost

        def native(original, self, data, args, kwargs):
            if id(self) not in copies:
                copy_1 = self.copy()
                copy_1.set_param({"nthread": 1})
                copies[id(self)] = copy_1
            return original(copies[id(self)], data, *args, **kwargs)

        patched = (xgboost.Booster, "inplace_predict",
                   install(xgboost.Booster, "inplace_predict", native))
    elif base == "lgbm":
        import lightgbm

        def native(original, self, data, args, kwargs):
            return original(self, data, *args, **{**kwargs, "num_threads": 1})

        patched = (lightgbm.Booster, "predict",
                   install(lightgbm.Booster, "predict", native))
    elif base == "catboost":
        import catboost

        def native(original, self, data, args, kwargs):
            return original(self, data, *args, **{**kwargs, "thread_count": 1})

        patched = (catboost.CatBoostRegressor, "predict",
                   install(catboost.CatBoostRegressor, "predict", native))
    else:
        patched = (type(estimator), "predict",
                   install(type(estimator), "predict", None))

    steps = 24
    window_size = forecaster.window_size
    n_folds = spec.get("loop_folds", 30)
    starts = [n_train + i * steps for i in range(n_folds)]
    hgb_rows = getattr(sk_utils, "_HGB_MAX_ROWS_ONE_THREAD", None)

    def predict_folds(mode):
        outer_limit = mode == "outer_limit=1"
        no_inner_limit = mode in ("outer_limit=1", "all_threads")
        state["mode"] = mode if mode in ("native=1", "no_estimator") else "current"
        if no_inner_limit and hgb_rows is not None:
            sk_utils._HGB_MAX_ROWS_ONE_THREAD = -1
        predictions = []
        try:
            start = time.perf_counter()
            if outer_limit:
                with controller.limit(limits=1, user_api="openmp"):
                    for i in starts:
                        predictions.append(forecaster.predict(
                            steps=steps, last_window=y.iloc[i - window_size: i],
                            exog=exog.iloc[i: i + steps],
                        ).to_numpy())
            else:
                for i in starts:
                    predictions.append(forecaster.predict(
                        steps=steps, last_window=y.iloc[i - window_size: i],
                        exog=exog.iloc[i: i + steps],
                    ).to_numpy())
            seconds = time.perf_counter() - start
        finally:
            if hgb_rows is not None:
                sk_utils._HGB_MAX_ROWS_ONE_THREAD = hgb_rows
            state["mode"] = "current"
        return seconds, np.concatenate(predictions)

    # `current`: skforecast as it is. `outer_limit=1`: OpenMP limited to one
    # thread once, around the folds, and not in every call (XGBoost and
    # HistGradientBoosting). `all_threads`: HistGradientBoosting without its
    # limit of one thread up to 100 rows. `native=1`: one thread with the
    # parameter of the library. `no_estimator`: the loop without the estimator.
    modes = ["current", "no_estimator"]
    if base in ("xgb", "hgb"):
        modes.append("outer_limit=1")
    if base == "hgb":
        modes.append("all_threads")
    if base in ("xgb", "lgbm", "catboost"):
        modes.append("native=1")
    _, reference = predict_folds("current")
    times = {mode: [] for mode in modes}
    same = {}
    for _ in range(reps):
        order = list(modes)
        random.shuffle(order)
        for mode in order:
            seconds, predictions = predict_folds(mode)
            times[mode].append(seconds)
            same[mode] = bool(
                np.allclose(predictions, reference, rtol=1e-6, atol=1e-6)
            )
    setattr(patched[0], patched[1], patched[2])
    loop = {
        mode: {"median": statistics.median(values), "min": min(values),
               "same": same[mode]}
        for mode, values in times.items()
    }
    out["n_steps"] = n_folds * steps
    out["loop"] = loop
    return out


def op_info(spec: dict) -> dict:
    """Versions, and how each library sees the CPUs of this process."""
    import importlib

    import joblib
    import numpy as np

    out = {
        "n_cpus": joblib.cpu_count(),
        "physical_cores": joblib.cpu_count(only_physical_cores=True),
        "os_cpu_count": os.cpu_count(),
        "affinity": allowed_cpus(),
        "versions": {},
        "os_threads_after_fit": {},
    }
    for name in ("skforecast", "numpy", "pandas", "sklearn", "scipy", "joblib",
                 "threadpoolctl", "lightgbm", "xgboost", "catboost", "numba",
                 "optuna"):
        try:
            out["versions"][name] = importlib.import_module(name).__version__
        except Exception as error:
            out["versions"][name] = f"ERROR: {error!r}"
    try:
        import skforecast

        out["skforecast_file"] = skforecast.__file__
    except Exception:
        pass

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 20))
    y = X[:, 0] + rng.normal(size=3000)
    for est in ("ridge", "lgbm", "xgb", "hgb", "catboost", "rf", "rf_all"):
        before = os_threads()
        try:
            make_estimator(est, 50).fit(X, y)
            out["os_threads_after_fit"][est] = [before, os_threads()]
        except Exception as error:
            out["os_threads_after_fit"][est] = f"ERROR: {error!r}"
    out["thread_pools"] = thread_pools()

    # Cost of the first use of the window features (numba) in a new process.
    try:
        from skforecast.preprocessing import RollingFeatures

        window = np.arange(float(WINDOW))
        features = RollingFeatures(stats=["mean"], window_sizes=WINDOW)
        start = time.perf_counter()
        features.transform(window)
        out["window_features_first_call"] = time.perf_counter() - start
        start = time.perf_counter()
        features.transform(window)
        out["window_features_second_call"] = time.perf_counter() - start
    except Exception as error:
        out["window_features_first_call"] = f"ERROR: {error!r}"
    return out


def op_prepare(spec: dict) -> dict:
    """Download the data and warm up the caches of the disk (not timed)."""
    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

    data = load_bike()
    sales = load_sales()
    y = data["users"].iloc[:1500]
    exog = data.drop(columns="users").iloc[:1500]
    for est in ESTIMATORS:
        backtesting_forecaster(
            forecaster=make_recursive(est, 20),
            y=y, exog=exog,
            cv=TimeSeriesFold(steps=24, initial_train_size=1400, refit=True,
                              verbose=False),
            metric="mean_absolute_error", n_jobs=2, verbose=False,
            show_progress=False,
        )
    return {"bike": list(data.shape), "sales": list(sales.shape)}


OPS = {
    "bt": op_timed, "gs": op_timed, "ms": op_timed, "direct": op_timed,
    "start": op_start, "resize": op_resize, "predict": op_predict,
    "info": op_info, "prepare": op_prepare,
}


def child_main(spec: dict) -> None:
    if spec.get("cpuset"):
        os.sched_setaffinity(0, spec["cpuset"])
    import warnings

    warnings.filterwarnings("ignore")
    start = time.perf_counter()
    out = OPS[spec["op"]](spec)
    out["child_seconds"] = time.perf_counter() - start
    out["affinity"] = allowed_cpus()
    print("RESULT " + json.dumps(out), flush=True)


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
# Stage of a measure by priority and repetition. The plan runs stage by stage,
# so that a run cut by the time budget keeps the widest coverage:
# 1: the whole machine, 2: emulated machines, 3: extended cases on the whole
# machine, 4: extended cases on the emulated machines.
STAGES = {
    (1, 0): 1, (2, 0): 2, (3, 0): 3, (1, 1): 4, (2, 1): 5, (4, 0): 6,
    (1, 2): 7, (3, 1): 8, (2, 2): 9, (3, 2): 10, (4, 1): 11,
}


def cell_key(spec: dict) -> str:
    return "|".join(str(spec[k]) for k in ("case", "est", "config", "cpus", "rep"))


def smoke_subset(cells: list[dict], total: int) -> list[dict]:
    """
    Measures of the smoke test: on the whole machine, one configuration of
    every case and estimator (the parallel one when there is one) and all of
    them for `lgbm_1t`; on the smallest emulated machine, a few cases.
    """
    emulated = sorted({c["cpus"] for c in cells if c["cpus"] < total})
    few = {
        ("bt_norefit", "xgb", "max"), ("bt_norefit", "xgb", "1"),
        ("bt_refit_small", "hgb", "auto"), ("predict", "xgb", "inproc"),
        ("predict_t100", "hgb", "inproc"),
    }
    order = {"max": 0, "auto": 1, "1": 2}
    keep, seen = [], set()
    for cell in sorted(cells, key=lambda c: order.get(c["config"], 3)):
        case, est, config = cell["case"], cell["est"], cell["config"]
        if cell["cpus"] == total:
            if case.startswith("ladder") and case != f"ladder_{LADDER[0]:04d}":
                continue
            if case == "start" and config != str(max(2, total - 1)):
                continue
            if est == "lgbm_1t" or (case, est) not in seen:
                keep.append(cell)
                seen.add((case, est))
        elif emulated and cell["cpus"] == emulated[0]:
            if (case, est, config) in few or (
                case == "start" and config == str(cell["cpus"])
            ):
                keep.append(cell)
    return keep


def build_plan(levels: list[int], total: int, args) -> list[dict]:
    cells = []
    cores = read_cores()

    def add(priority, reps, case, est, config, cpus, **extra):
        for rep in range(reps):
            spec = {
                "case": case, "est": est, "config": str(config), "cpus": cpus,
                "rep": rep, "priority": priority,
                "stage": STAGES.get((priority, rep), 20 + rep),
                "cpuset": cpuset_for(cpus, cores) if cpus < total else None,
                "size": args.size, "smoke": bool(args.smoke),
                "timeout": CELL_TIMEOUT, "trees": args.trees,
                "rf_trees": args.rf_trees,
            }
            if case in CASES:
                spec.update(CASES[case])
                if args.smoke:
                    spec["calls"] = 2
                    if spec.get("op") != "direct":
                        spec["n_val"] = min(spec["n_val"] or 10**6, 10 * spec["steps"])
                        if spec["refit"]:
                            spec["n_val"] = min(spec["n_val"], 4 * spec["steps"])
            spec.update(extra)
            cells.append(spec)

    def timed(priority, reps, case, estimators, configs, cpus):
        for est in estimators:
            for config in configs:
                risky = (
                    est == "lgbm" and CASES[case]["refit"]
                    and config in ("max", "half", "all")
                )
                if risky:
                    continue
                # RandomForest with its default parameters and one thread takes
                # too long in the cases with more fits; `bt_refit_small` covers it.
                if est == "rf" and case in ("bt_refit", "gs_refit_small"):
                    continue
                if config != "1" and config != "auto" and cpus < 3:
                    continue
                add(priority, reps, case, est, config, cpus)

    for cpus in levels:
        whole = cpus == total
        core_priority = 1 if whole else 2
        extended_priority = 3 if whole else 4
        reps = args.reps or (1 if args.smoke else 3 if whole else 2)
        extended_reps = args.reps or 1
        n_max = max(1, cpus - 1)

        # Start of the workers by their number, and change of their number.
        for k in sorted({k for k in (2, 4, 8, n_max, cpus) if 2 <= k <= cpus}):
            add(core_priority, reps, "start", "lgbm_1t", k, cpus, op="start", calls=4)
        if cpus >= 3:
            add(core_priority, reps, "start_no_wf", "lgbm_1t", n_max, cpus,
                op="start", calls=4, window_features=False)
            add(core_priority, min(reps, 2), "resize", "lgbm_1t", "seq", cpus,
                op="resize")

        # Threads of the prediction.
        # Threads of the prediction: by rows and threads with 100, the default
        # and 1000 trees, and the loop of the forecaster with the default.
        small = {"rows": (1, 50), "loop_folds": 5, "reps_inner": 2} if args.smoke else {}
        for est in PREDICT:
            add(core_priority, 1, "predict", est, "inproc", cpus, op="predict", **small)
            for trees in (100, 1000):
                if trees != args.trees:
                    add(core_priority, 1, f"predict_t{trees}", est, "inproc", cpus,
                        op="predict", trees=trees, loop=False, **small)

        # Core cases: the six estimators with their default threads, and
        # LightGBM with one thread.
        timed(core_priority, reps, "bt_norefit", CORE, ["1", "max"], cpus)
        timed(core_priority, reps, "bt_refit", CORE, ["1", "max"], cpus)
        timed(core_priority, reps, "bt_refit_small", CORE, ["1", "max"], cpus)
        timed(core_priority, reps, "bt_refit_small", SMALL_AUTO, ["auto"], cpus)
        timed(core_priority, reps, "gs_norefit", CORE, ["1", "max"], cpus)
        for rows in LADDER:
            timed(core_priority, reps, f"ladder_{rows:04d}", ["lgbm_1t"],
                  ["1", "auto", "max"], cpus)

        # Extended cases.
        p, r = extended_priority, extended_reps
        timed(p, r, "gs_refit_small", CORE, ["1", "max"], cpus)
        for case in ("bt_norefit", "bt_refit", "gs_norefit"):
            timed(p, r, case, THREAD_VARIANTS, ["1", "max"], cpus)
        timed(p, r, "bt_refit", ["hgb", "lgbm_1t", "xgb"], ["half", "all"], cpus)
        timed(p, r, "ms_norefit", MULTISERIES, ["1", "max"], cpus)
        timed(p, r, "ms_refit", MULTISERIES, ["1", "max"], cpus)
        if args.direct:
            timed(p, r, "direct_fit", DIRECT, ["1", "auto", "max"], cpus)
        # 'auto' end to end, to check the decisions computed from the rules.
        timed(p, 1, "bt_norefit", ["xgb", "hgb"], ["auto"], cpus)
        timed(p, 1, "bt_refit", ["lgbm", "xgb", "hgb"], ["auto"], cpus)
        timed(p, 1, "gs_norefit", ["lgbm_1t", "xgb", "hgb", "catboost"], ["auto"], cpus)
        if whole and cpus >= 3:
            # LightGBM with its threads inside the workers: the case that the
            # rule of 'auto' avoids. One call, short timeout.
            add(p, 1, "bt_refit_small", "lgbm", "max", cpus, calls=1,
                timeout=RISKY_TIMEOUT)

    if args.smoke:
        cells = [c for c in cells if c["rep"] == 0]
        if not args.smoke_full:
            cells = smoke_subset(cells, total)
    if args.blocks:
        cells = [c for c in cells if any(c["case"].startswith(b) for b in args.blocks)]
    if args.estimators:
        cells = [c for c in cells if c["est"] in args.estimators]
    if args.max_priority:
        cells = [c for c in cells if c["priority"] <= args.max_priority]

    rng = random.Random(args.seed)
    by_stage = {}
    for cell in cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    plan = []
    for stage in sorted(by_stage):
        rng.shuffle(by_stage[stage])
        plan.extend(by_stage[stage])
    return plan


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def launch(spec: dict, timeout: float) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    env.update(PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", "--spec",
           json.dumps(spec)]
    start = time.perf_counter()
    # NOTE: A new session, so that the workers can be killed with the child.
    process = subprocess.Popen(
        cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_group(process.pid)
        stdout, stderr = process.communicate()
    finally:
        kill_group(process.pid)
    out = {"spec": spec, "cell_seconds": time.perf_counter() - start}
    if timed_out:
        return {**out, "ok": False, "timeout": True}
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT ")]
    if process.returncode != 0 or not lines:
        return {**out, "ok": False, "timeout": False,
                "returncode": process.returncode, "error": stderr[-3000:]}
    return {**out, "ok": True, **json.loads(lines[-1][len("RESULT "):])}


def read_results(path: Path) -> list[dict]:
    results = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def describe(result: dict) -> str:
    if not result["ok"]:
        if result.get("timeout"):
            return "TIMEOUT"
        return "ERROR " + result.get("error", "").strip().splitlines()[-1][:150]
    calls = result.get("calls")
    if not calls:
        return "ok"
    first = calls[0]["seconds"]
    following = [c["seconds"] for c in calls[1:]]
    text = f"first {first:6.2f}"
    if following:
        text += f"  next {statistics.median(following):6.2f}"
    if "parallel_n_jobs" in calls[0]:
        text += f"  n_jobs={calls[0]['parallel_n_jobs']}"
    return text


def default_out(args=None) -> Path:
    name = "".join(c if c.isalnum() else "-" for c in platform.node()) or "run"
    if args is not None and getattr(args, "size", "S") != "S":
        name += f"_{args.size}"
    if args is not None and getattr(args, "trees", N_TREES) != N_TREES:
        name += f"_t{args.trees}"
    if args is not None and getattr(args, "rf_trees", RF_TREES) != RF_TREES:
        name += f"_rf{args.rf_trees}"
    return HERE / "results" / name


def acquire_lock():
    """Refuse to run while another run of the benchmark uses the machine."""
    import fcntl

    lock = open(HERE / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another run of the benchmark is in progress in this folder. The"
              " measures must run one at a time: wait for it to finish.")
        sys.exit(1)
    return lock


def worker_checks(result: dict) -> list[str]:
    """
    Check that the first call of a measure started its workers (no workers
    existed before it) and that the following calls reused them (same
    executor and same processes, all alive). Return the problems found.
    """
    problems = []
    calls = result.get("calls", [])
    for i, call in enumerate(calls):
        before, after = call.get("executor_before"), call.get("executor_after")
        if before is None or after is None or "error" in before or "error" in after:
            continue
        parallel = max(call.get("parallel_n_jobs") or [1]) > 1
        if i == 0:
            if before["executor_id"] is not None:
                problems.append("workers existed before the first call")
        elif parallel and max(calls[i - 1].get("parallel_n_jobs") or [1]) > 1:
            previous = calls[i - 1]["executor_after"]
            if before["executor_id"] != previous["executor_id"]:
                problems.append(f"call {i + 1}: the executor changed between calls")
            elif before.get("pids") != previous.get("pids"):
                problems.append(f"call {i + 1}: the workers changed between calls")
            elif after["executor_id"] != before["executor_id"]:
                problems.append(f"call {i + 1}: the workers were restarted")
            elif before["n_processes"] < len(before.get("pids", [])):
                problems.append(f"call {i + 1}: some workers were not alive")
    return problems


def run(args) -> None:
    out_dir = Path(args.out) if args.out else default_out(args)
    if args.smoke and not args.out:
        out_dir = HERE / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    cores = read_cores()
    total = sum(len(core) for core in cores)
    levels = args.cpus or [total] + [k for k in (8, 4) if k < total]
    levels = [k for k in levels if k <= total]
    plan = build_plan(levels, total, args)

    done = {}
    for result in read_results(results_file):
        spec = result.get("spec", {})
        if "case" in spec:
            done[cell_key(spec)] = result
    # A measure that failed (not by timeout) is tried again when resuming.
    pending = [
        cell for cell in plan
        if cell_key(cell) not in done
        or (not done[cell_key(cell)]["ok"] and not done[cell_key(cell)].get("timeout"))
    ]
    timeouts = {
        cell_key(r["spec"]).rsplit("|", 1)[0]
        for r in done.values() if r.get("timeout")
    }

    stages = {}
    for cell in plan:
        stages[cell["stage"]] = stages.get(cell["stage"], 0) + 1
    print(f"Results: {results_file}")
    print(f"CPUs: {total} ({len(cores)} physical cores). Levels: {levels}")
    for cpus in levels:
        print(f"  {cpus} CPUs -> {cpuset_for(cpus, cores)}")
    print(f"Plan: {len(plan)} measures, {len(pending)} pending. By stage: "
          + ", ".join(f"{s}: {n}" for s, n in sorted(stages.items())))
    if args.list:
        counts = {}
        for cell in plan:
            key = (cell["stage"], cell["cpus"], cell["case"])
            counts[key] = counts.get(key, 0) + 1
        for (stage, cpus, case), n in sorted(counts.items()):
            print(f"  stage {stage:>2}  cpus {cpus:>3}  {case:<16} {n}")
        return

    stripped = [k for k in STRIP_ENV if k in os.environ]
    if stripped:
        print(f"WARNING: these variables are removed for the measures: {stripped}")
    lock = acquire_lock()  # held until the process ends

    machine_file = out_dir / "machine.json"
    if not machine_file.exists() or args.refresh_info:
        info = machine_info()
        info["levels"] = {str(k): cpuset_for(k, cores) for k in levels}
        info["schema"] = SCHEMA
        info["args"] = {k: v for k, v in vars(args).items() if k != "func"}
        print("Preparing the data and the caches...", flush=True)
        prepared = launch({"op": "prepare"}, timeout=1800)
        if not prepared["ok"]:
            print("ERROR preparing the data:\n" + prepared.get("error", "timeout"))
            sys.exit(1)
        info["prepare"] = {k: prepared[k] for k in ("bike", "sales", "cell_seconds")}
        info["libraries"] = {}
        for cpus in levels:
            spec = {"op": "info",
                    "cpuset": cpuset_for(cpus, cores) if cpus < total else None}
            result = launch(spec, timeout=600)
            info["libraries"][str(cpus)] = (
                {k: v for k, v in result.items() if k != "spec"}
            )
        machine_file.write_text(json.dumps(info, indent=1))

    budget = args.budget_minutes * 60
    start = time.perf_counter()
    n_run = 0
    for i, cell in enumerate(pending, start=1):
        elapsed = time.perf_counter() - start
        if elapsed > budget:
            print(f"Time budget reached ({args.budget_minutes} min): "
                  f"{len(pending) - i + 1} measures left. Run again to continue.")
            break
        if cell_key(cell).rsplit("|", 1)[0] in timeouts:
            continue
        load = loadavg()
        busy = cpu_busy()
        result = launch(cell, timeout=cell["timeout"])
        result["load_before"] = load
        result["busy_before"] = busy
        result["wall_time"] = time.time()
        with open(results_file, "a") as file:
            file.write(json.dumps(result) + "\n")
        if result.get("timeout"):
            timeouts.add(cell_key(cell).rsplit("|", 1)[0])
        n_run += 1
        problems = worker_checks(result) if result["ok"] else []
        if busy is not None and busy > 0.05:
            problems.append(f"the machine was busy before the measure ({busy:.0%})")
        print(
            f"[{i}/{len(pending)} s{cell['stage']}] cpus={cell['cpus']:<3}"
            f" {cell['case']:<15} {cell['est']:<12} {cell['config']:<5}"
            f" rep{cell['rep']}: {describe(result)}"
            f"  ({result['cell_seconds']:.0f} s,"
            f" busy {'?' if busy is None else format(busy, '.0%')},"
            f" elapsed {(time.perf_counter() - start) / 60:.1f} min)"
            + "".join(f"\n    WARNING: {problem}" for problem in problems),
            flush=True,
        )
    else:
        print("Plan finished.")
    print(f"{n_run} measures in {(time.perf_counter() - start) / 60:.1f} min.")
    lock.close()
    summarize(out_dir)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2, suffix=""):
    return "" if value is None else f"{value:.{digits}f}{suffix}"


def first_seconds(cells):
    return med([c["calls"][0]["seconds"] for c in cells])


def next_seconds(cells):
    return med([
        med([call["seconds"] for call in c["calls"][1:]])
        for c in cells if len(c["calls"]) > 1
    ])


def same_signature(a, b, tolerance=1e-5) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(
        abs(x - y) <= tolerance * max(1.0, abs(x), abs(y)) for x, y in zip(a, b)
    )


def workload_limit(n_jobs, n_tasks, n_fits, fit_seconds):
    """`limit_n_jobs_by_workload` of the installed skforecast."""
    try:
        from skforecast.model_selection._utils import limit_n_jobs_by_workload
    except Exception:
        return None
    return limit_n_jobs_by_workload(
        n_jobs=n_jobs, n_tasks=n_tasks, n_fits=n_fits, fit_seconds=fit_seconds
    )


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return lines


CASE_NOTES = {
    "ladder": "Ladder of work: backtesting with refit of LightGBM with one "
              "thread (16 folds), by training rows",
    "bt_norefit": "Backtesting without refit (100 folds of 24 steps)",
    "bt_refit": "Backtesting with refit (16 folds, 15 refits)",
    "bt_refit_small": "Backtesting with refit, few folds (6 folds, 5 refits)",
    "gs_norefit": "Grid search without refit (3 candidates, 40 folds each)",
    "gs_refit_small": "Grid search with refit (3 candidates, 6 folds each)",
    "ms_norefit": "Multi-series backtesting without refit (50 series, 52 folds)",
    "ms_refit": "Multi-series backtesting with refit (50 series, 8 folds)",
    "direct_fit": "Fit of ForecasterDirect (12 steps)",
}


def worker_summary(ok: list[dict]) -> str:
    timed = [r for r in ok if r["spec"].get("op") in ("bt", "gs", "ms", "direct")]
    parallel = [
        r for r in timed
        if max(r["calls"][0].get("parallel_n_jobs") or [1]) > 1
    ]
    bad = [(cell_key(r["spec"]), worker_checks(r)) for r in timed]
    bad = [(key, problems) for key, problems in bad if problems]
    text = (
        f"{len(timed)} timed measures, {len(parallel)} in parallel. In all of"
        " them the first call started its workers and the following calls"
        " reused the same ones"
    )
    if not bad:
        return text + "."
    return (
        f"{len(bad)} of {len(timed)} timed measures did NOT follow the first /"
        " next protocol: " + "; ".join(f"{key} ({', '.join(p)})" for key, p in bad)
    )


def summarize(out_dir: Path) -> None:
    results = read_results(out_dir / "results.jsonl")
    latest = {}
    for result in results:
        spec = result.get("spec", {})
        if "case" in spec:
            latest[cell_key(spec)] = result
    results = list(latest.values())
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    machine = {}
    if (out_dir / "machine.json").exists():
        machine = json.loads((out_dir / "machine.json").read_text())

    lines = ["# Benchmark of n_jobs in skforecast", ""]
    libraries = machine.get("libraries", {})
    if machine:
        first = next(iter(libraries.values()), {})
        lines += [
            f"- Machine: {machine.get('platform')}, {machine.get('os_cpu_count')}"
            f" logical CPUs, {len(machine.get('cores', []))} physical cores,"
            f" {machine.get('mem_total_gb')} GB.",
            f"- Python {machine.get('python')}; "
            + ", ".join(f"{k} {v}" for k, v in first.get("versions", {}).items())
            + ".",
            f"- skforecast commit: {machine.get('git_commit')}"
            + (" (with local changes in skforecast/)" if machine.get("git_status")
               else "") + ".",
            f"- Measures: {len(ok)} finished, {len(bad)} failed or timed out.",
            "- Workers: " + worker_summary(ok),
            "- CPU in use by other processes before each measure: median "
            + fmt(med([r.get("busy_before") for r in results]), 3) + ", maximum "
            + fmt(max([r.get("busy_before") or 0 for r in results], default=None), 3)
            + " (fraction of the machine).",
            "",
            "Times in seconds, median of the processes. `first`: first call of a"
            " new process (it pays the start of the workers). `next`: median of"
            " the following calls (workers already started). `seq`: `n_jobs=1`."
            " `par`: `n_jobs=cpu_count() - 1`.",
            "",
        ]

    grouped = {}
    for result in ok:
        spec = result["spec"]
        if spec.get("op") in ("bt", "gs", "ms", "direct"):
            grouped.setdefault(
                (spec["cpus"], spec["case"], spec["est"]), {}
            ).setdefault(spec["config"], []).append(result)

    levels = sorted(
        {r["spec"]["cpus"] for r in ok if "cpus" in r["spec"]}, reverse=True
    )
    policy = {}

    for cpus in levels:
        lines += [f"## {cpus} CPUs", ""]
        library = libraries.get(str(cpus), {})
        if library.get("ok"):
            lines += [
                f"joblib sees {library.get('n_cpus')} CPUs"
                f" ({library.get('physical_cores')} physical cores). Thread pools:"
                f" {library.get('thread_pools')}. First use of the window features"
                f" in a process: {fmt(library.get('window_features_first_call'))} s.",
                "",
            ]

        # ---- start of the workers ------------------------------------------
        rows = []
        for case in ("start", "start_no_wf"):
            by_k = {}
            for result in ok:
                spec = result["spec"]
                if spec["case"] == case and spec["cpus"] == cpus:
                    by_k.setdefault(result["n_jobs"], []).append(result)
            for k in sorted(by_k):
                cells = by_k[k]
                first, following = first_seconds(cells), next_seconds(cells)
                rows.append([
                    k, "yes" if case == "start" else "no", fmt(first),
                    fmt(following),
                    fmt(first - following if first and following else None),
                ])
        if rows:
            lines += ["### Start of the workers", ""] + table(
                ["workers", "window features", "first", "next", "start cost"], rows
            ) + [""]
        resize = [r for r in ok if r["spec"]["case"] == "resize"
                  and r["spec"]["cpus"] == cpus]
        if resize:
            rows = [
                [call["n_jobs"], call["threads_per_worker"],
                 fmt(med([r["calls"][i]["seconds"] for r in resize])),
                 call.get("executor_id"), call.get("n_processes")]
                for i, call in enumerate(resize[0]["calls"])
            ]
            lines += ["Consecutive calls changing the number of workers:", ""]
            lines += table(
                ["n_jobs", "threads per worker", "seconds", "executor id",
                 "processes"], rows
            ) + [""]

        # ---- timed cases ----------------------------------------------------
        present = {
            "ladder" if key[1].startswith("ladder") else key[1]
            for key in grouped if key[0] == cpus
        }
        titles = [title for title in CASE_NOTES if title in present]
        titles += sorted(present - set(titles))
        for title in titles:
            rows = []
            for (level, case, est), configs in sorted(grouped.items()):
                if level != cpus or not (
                    case == title or (title == "ladder" and case.startswith("ladder"))
                ):
                    continue
                seq = configs.get("1", [])
                par = configs.get("max", [])
                auto = configs.get("auto", [])
                any_cell = (seq or par or auto or next(iter(configs.values())))[0]
                info = any_cell["info"]
                op = any_cell["spec"]["op"]
                seq_first, seq_next = first_seconds(seq) if seq else None, (
                    next_seconds(seq) if seq else None)
                par_first, par_next = first_seconds(par) if par else None, (
                    next_seconds(par) if par else None)
                speedup = seq_next / par_next if seq_next and par_next else None
                start_cost = (
                    par_first - par_next if par_first and par_next else None
                )
                break_even = (
                    start_cost * speedup / (speedup - 1)
                    if speedup and start_cost is not None and speedup > 1.05
                    and start_cost > 0 else None
                )

                # What 'auto' chooses: measured when the case has 'auto'
                # measures, computed from the rules otherwise.
                n_par = par[0]["n_jobs"] if par else None
                rule = info.get("auto_rule")
                decision = {}
                for label, index in (("first", 0), ("next", 1)):
                    if auto:
                        chosen = [
                            max(c["calls"][min(index, len(c["calls"]) - 1)]
                                ["parallel_n_jobs"] or [1])
                            for c in auto
                        ]
                        decision[label] = max(set(chosen), key=chosen.count)
                    elif op in ("bt", "ms") and seq:
                        fit_seconds = med([
                            c["calls"][min(index, len(c["calls"]) - 1)]["fit_first"]
                            for c in seq
                        ])
                        decision[label] = workload_limit(
                            rule, info["n_tasks"], info["n_fits"], fit_seconds
                        )
                    else:
                        decision[label] = rule
                auto_time = {}
                for label, seq_t, par_t, auto_t in (
                    ("first", seq_first, par_first,
                     first_seconds(auto) if auto else None),
                    ("next", seq_next, par_next,
                     next_seconds(auto) if auto else None),
                ):
                    if auto_t is not None:
                        auto_time[label] = auto_t
                    elif decision[label] == 1:
                        auto_time[label] = seq_t
                    elif decision[label] is not None and decision[label] == n_par:
                        auto_time[label] = par_t
                    else:
                        auto_time[label] = None
                best = {
                    "first": min([t for t in (seq_first, par_first) if t], default=None),
                    "next": min([t for t in (seq_next, par_next) if t], default=None),
                }
                regret = {
                    label: (auto_time[label] / best[label]
                            if auto_time[label] and best[label] else None)
                    for label in ("first", "next")
                }
                if seq and par and not case.startswith("ladder") and (
                    est in CORE or op in ("ms", "direct")
                ):
                    for label, seq_t, par_t in (("first", seq_first, par_first),
                                                ("next", seq_next, par_next)):
                        if seq_t and par_t and auto_time[label]:
                            entry = policy.setdefault((cpus, label), [])
                            entry.append({
                                "case": case, "est": est, "seq": seq_t,
                                "par": par_t, "auto": auto_time[label],
                                "best": min(seq_t, par_t),
                            })

                estimate = None
                if op in ("bt", "ms") and seq and info["n_fits"]:
                    fit_first = med([c["calls"][0]["fit_first"] for c in seq])
                    if fit_first and seq_first:
                        estimate = info["n_fits"] * fit_first / seq_first
                fit_seq = med([c["post"].get("fit_after") for c in seq])
                fit_par = med([c["post"].get("fit_after") for c in par])
                signatures = [
                    c["calls"][0]["signature"] for cs in configs.values() for c in cs
                ]
                same = all(same_signature(signatures[0], s) for s in signatures)
                extra = "".join(
                    f" {name}: {fmt(first_seconds(cells))}/{fmt(next_seconds(cells))}"
                    f" (n_jobs={cells[0]['n_jobs']})"
                    for name, cells in sorted(configs.items())
                    if name not in ("1", "max", "auto")
                )
                label = est
                if case.startswith("ladder"):
                    label = f"{int(case.split('_')[1])} rows"
                rows.append([
                    label, f"{len(seq)}/{len(par)}", fmt(seq_first), fmt(seq_next),
                    fmt(par_first),
                    fmt(par_next), fmt(speedup, 2, "x"), fmt(start_cost),
                    fmt(break_even), rule,
                    f"{decision['first']}/{decision['next']}"
                    + (" (measured)" if auto else ""),
                    fmt(regret["first"], 2, "x"), fmt(regret["next"], 2, "x"),
                    fmt(estimate), fmt(fit_par / fit_seq if fit_seq and fit_par
                                       else None, 2, "x"),
                    fmt(med([c["post"].get("pickle_mb")
                             for cs in configs.values() for c in cs]), 1),
                    fmt(med([c["post"].get("descendants_rss_mb") for c in par]), 0),
                    "yes" if same else "NO", extra,
                ])
            lines += [f"### {CASE_NOTES.get(title, title)}", ""] + table(
                ["estimator", "processes seq/par", "seq first", "seq next",
                 "par first", "par next",
                 "speedup next", "start cost", "break-even", "rule", "auto first/next",
                 "auto vs best first", "auto vs best next", "estimate / real",
                 "fit after par / seq", "model MB", "workers MB", "same result",
                 "other configs (first/next)"], rows
            ) + [""]

        # ---- threads of the prediction ---------------------------------------
        predict = sorted(
            (r for r in ok if r["spec"]["case"].startswith("predict")
             and r["spec"]["cpus"] == cpus),
            key=lambda r: (r["spec"]["est"], r.get("trees", 0)),
        )
        if predict:
            lines += [
                "### Threads of the prediction", "",
                "Microseconds per call of the predict method of the estimator, by"
                " rows. `native=k`: k threads with the parameter of the library."
                " `limit=k`: k OpenMP threads with threadpoolctl in every call.", "",
            ]
        for result in predict:
            variants = list(next(iter(result["micro"].values())))
            rows = [
                [n_rows] + [fmt(values[name]["median"], 0) for name in variants]
                for n_rows, values in result["micro"].items()
            ]
            lines += [f"{result['spec']['est']}, {result.get('trees')} trees:", ""]
            lines += table(["rows"] + variants, rows) + [""]
            if "loop" not in result:
                continue
            base = result["loop"]["current"]["median"]
            rows = [
                [mode, fmt(values["median"], 3), fmt(values["min"], 3),
                 fmt(values["median"] / result["n_steps"] * 1e3, 3),
                 fmt(values["median"] / base, 2, "x"),
                 "" if mode == "no_estimator" else "yes" if values["same"] else "NO"]
                for mode, values in result["loop"].items()
            ]
            lines += [
                f"Predictions of {result['n_steps']} steps with the forecaster."
                " `current`: skforecast as it is. `outer_limit=1`: one OpenMP"
                " thread, set once around the folds. `native=1`: one thread with"
                " the parameter of the library. `all_threads`: without the limit"
                " of one thread of HistGradientBoosting.", "",
            ]
            lines += table(
                ["mode", "median s", "min s", "ms per step", "vs current",
                 "same predictions"], rows
            ) + [""]

    # ---- policies -----------------------------------------------------------
    if policy:
        lines += [
            "## Policies", "",
            "Sum of the times of the cases measured with `n_jobs=1` and"
            " `n_jobs=cpu_count() - 1`, with each policy, and worst case against"
            " the best of the two for each case. `auto`: what skforecast chooses.",
            "",
        ]
        rows = []
        for (cpus, label), entries in sorted(policy.items(), key=lambda i: (-i[0][0], i[0][1])):
            row = [cpus, label, len(entries)]
            for name in ("seq", "par", "auto", "best"):
                row.append(fmt(sum(e[name] for e in entries), 1))
            for name in ("seq", "par", "auto"):
                worst = max(entries, key=lambda e: e[name] / e["best"])
                row.append(
                    f"{worst[name] / worst['best']:.2f}x ({worst['case']},"
                    f" {worst['est']})"
                )
            rows.append(row)
        lines += table(
            ["CPUs", "call", "cases", "always 1", "always parallel", "auto",
             "best", "worst always 1", "worst always parallel", "worst auto"], rows
        ) + [""]

    if bad:
        lines += ["## Failed measures", ""]
        rows = [
            [cell_key(r["spec"]), "timeout" if r.get("timeout") else
             r.get("error", "").strip().splitlines()[-1][:200]
             if r.get("error", "").strip() else "error"]
            for r in bad
        ]
        lines += table(["measure", "reason"], rows) + [""]

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    print(f"Summary: {out_dir / 'summary.md'}")


# ----------------------------------------------------------------------------
# Check
# ----------------------------------------------------------------------------
def check(args) -> None:
    cores = read_cores()
    total = sum(len(core) for core in cores)
    print(f"Python {platform.python_version()} ({sys.executable})")
    print(f"{platform.platform()}")
    print(f"CPUs: {total} logical, {len(cores)} physical cores")
    problems = []
    if not hasattr(os, "sched_setaffinity"):
        problems.append("os.sched_setaffinity is not available (Linux only):"
                        " smaller machines cannot be emulated.")
    stripped = [k for k in STRIP_ENV if k in os.environ]
    if stripped:
        print(f"NOTE: {stripped} are set; they are removed for the measures.")
    load = loadavg()
    if load is not None and load > 0.5:
        problems.append(f"The machine is busy (load average {load}).")

    levels = [total] + [k for k in (8, 4) if k < total]
    for cpus in levels:
        cpuset = cpuset_for(cpus, cores)
        result = launch(
            {"op": "info", "cpuset": cpuset if cpus < total else None}, timeout=600
        )
        if not result["ok"]:
            problems.append(f"info with {cpus} CPUs failed: {result.get('error')}")
            continue
        print(f"\n{cpus} CPUs -> {cpuset}")
        print(f"  joblib.cpu_count() = {result['n_cpus']},"
              f" physical cores = {result['physical_cores']}")
        if result["n_cpus"] != cpus:
            problems.append(f"joblib sees {result['n_cpus']} CPUs, expected {cpus}.")
        print(f"  thread pools: {result['thread_pools']}")
        print(f"  OS threads before/after a fit: {result['os_threads_after_fit']}")
        print(f"  first use of the window features: "
              f"{result['window_features_first_call']}")
        if cpus == total:
            print("  versions: " + ", ".join(
                f"{k} {v}" for k, v in result["versions"].items()))
            print(f"  skforecast from: {result.get('skforecast_file')}")
            for name, version in result["versions"].items():
                if str(version).startswith("ERROR") and name != "optuna":
                    problems.append(f"{name} cannot be imported: {version}")
            for est, threads in result["os_threads_after_fit"].items():
                if isinstance(threads, str):
                    problems.append(f"{est} cannot be fitted: {threads}")

    print("\nDownloading the data and warming up (about a minute)...", flush=True)
    prepared = launch({"op": "prepare"}, timeout=1800)
    if prepared["ok"]:
        print(f"  bike sharing {prepared['bike']}, store sales {prepared['sales']}")
    else:
        problems.append("The data could not be prepared: "
                        + prepared.get("error", "timeout")[-500:])

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nOK. Next: python bench_n_jobs.py run --smoke")


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check the environment.").set_defaults(func=check)

    p_run = sub.add_parser("run", help="Run the benchmark (resumable).")
    p_run.add_argument("--budget-minutes", type=float, default=120)
    p_run.add_argument("--out", help="Folder of the results.")
    p_run.add_argument("--smoke", action="store_true",
                       help="Tiny sizes and one repetition, to test every path.")
    p_run.add_argument("--smoke-full", action="store_true",
                       help="With --smoke: every measure of the plan, once.")
    p_run.add_argument("--cpus", type=int, nargs="+",
                       help="Machine sizes (default: all the CPUs, 8 and 4).")
    p_run.add_argument("--reps", type=int, help="Repetitions (default 3 and 2).")
    p_run.add_argument("--size", choices=list(START), default="S",
                       help="Training rows: S (5,112) or L (10,776).")
    p_run.add_argument("--trees", type=int, default=N_TREES,
                       help="Trees of the boosting models (default 500).")
    p_run.add_argument("--rf-trees", type=int, default=RF_TREES,
                       help="Trees of RandomForest (default 100).")
    p_run.add_argument("--direct", action="store_true",
                       help="Include the fit of ForecasterDirect.")
    p_run.add_argument("--blocks", nargs="+",
                       help="Only the cases that start with these names.")
    p_run.add_argument("--estimators", nargs="+", choices=list(ESTIMATORS))
    p_run.add_argument("--max-priority", type=int, choices=[1, 2, 3, 4])
    p_run.add_argument("--seed", type=int, default=123)
    p_run.add_argument("--list", action="store_true", help="Print the plan and exit.")
    p_run.add_argument("--refresh-info", action="store_true")
    p_run.set_defaults(func=run)

    p_summary = sub.add_parser("summary", help="Write summary.md.")
    p_summary.add_argument("--out", help="Folder of the results.")
    p_summary.set_defaults(
        func=lambda a: summarize(Path(a.out) if a.out else default_out())
    )

    p_child = sub.add_parser("child")
    p_child.add_argument("--spec", required=True)
    p_child.set_defaults(func=lambda a: child_main(json.loads(a.spec)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
