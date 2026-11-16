"""
Short check of the `n_jobs` strategy on Windows (or any platform).

`bench_n_jobs.py` needs Linux (CPU affinity, process groups). This script
measures, with the skforecast installed in the active environment, the cases
that decide the changes of the branch `perf/n-jobs-auto-strategy`:

- `bt_norefit_100` and `bt_norefit_40`: backtesting without refit with a
  `HistGradientBoostingRegressor` (100 and 40 folds), `n_jobs` 1, 'auto' and
  `cpu_count() - 1`. Checks the minimum prediction time to run in parallel
  (12 s on Windows): 'auto' should be as fast as the best of the other two.
- `gs_refit_small` (XGBoost, 3 candidates of 6 folds with refit) and
  `bt_refit_small` (XGBoost, 6 folds with refit): the forecaster serialized
  once and no more workers than folds. `fit_after` times a fit of XGBoost in
  the main process after the calls (it was slower after running in parallel).
- `ms_norefit` and `ms_refit`: the same in multi-series backtesting.

Each measure runs in a new Python process, so its first call starts the
workers, as in a new session. The following calls reuse them.

Usage, with the skforecast to measure installed in the active environment
(`pip install -e <clone>`; with an editable install, checking out another
commit of the clone is enough to measure it):

    python bench_windows.py run --out results_win/before
    python bench_windows.py run --out results_win/after
    python bench_windows.py compare results_win/before results_win/after

`run` appends every measure to `<out>/results.jsonl` and skips the ones
already there, so it can be stopped and run again. `--cases` and `--reps`
narrow or widen the plan. CatBoost is measured if it is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache_win"

END_TRAIN = "2012-03-31 23:00:00"
END_VALIDATION = "2012-08-31 23:00:00"
START = "2011-09-01 00:00:00"
EXOG = [
    "month_sin", "month_cos", "week_day_sin", "week_day_cos",
    "hour_day_sin", "hour_day_cos", "temp", "holiday",
]
N_TREES = 500

# op, steps, refit, rows predicted after the training set, calls per process.
CASES = {
    "bt_norefit_100": {"op": "bt", "steps": 24, "refit": False, "n_val": 2400},
    "bt_norefit_40": {"op": "bt", "steps": 24, "refit": False, "n_val": 960},
    "gs_refit_small": {"op": "gs", "steps": 168, "refit": True, "n_val": 1008},
    "bt_refit_small": {"op": "bt", "steps": 168, "refit": True, "n_val": 1008},
    "ms_norefit": {"op": "ms", "steps": 7, "refit": False, "n_val": 364},
    "ms_refit": {"op": "ms", "steps": 28, "refit": True, "n_val": 224},
}
PLAN = [
    ("bt_norefit_100", "hgb", ["1", "auto", "max"]),
    ("bt_norefit_40", "hgb", ["1", "auto", "max"]),
    ("bt_norefit_100", "catboost", ["1", "auto", "max"]),
    ("gs_refit_small", "xgb", ["1", "auto", "max"]),
    ("bt_refit_small", "xgb", ["1", "auto", "max"]),
    ("ms_norefit", "xgb", ["1", "max"]),
    ("ms_refit", "xgb", ["1", "auto", "max"]),
]
CALLS = 3


# ----------------------------------------------------------------------------
# Data and forecasters
# ----------------------------------------------------------------------------
def _cached(name, build):
    import pandas as pd

    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        build().to_pickle(path)
    return pd.read_pickle(path)


def load_bike():
    def build():
        from skforecast.datasets import fetch_dataset

        data = fetch_dataset("bike_sharing_extended_features", verbose=False)
        return data.loc[START:END_VALIDATION, ["users"] + EXOG].astype("float64")

    return _cached("bike_sharing.pkl", build)


def load_sales():
    def build():
        from skforecast.datasets import fetch_dataset

        sales = fetch_dataset("store_sales", raw=False, verbose=False)
        sales = sales.query("store == 1").pivot(columns="item", values="sales")
        sales = sales.asfreq("D").astype("float64")
        sales.columns = [f"item_{item}" for item in sales.columns]
        sales.columns.name = None
        return sales

    return _cached("store_sales.pkl", build)


def make_estimator(name, learning_rate=None):
    extra = {} if learning_rate is None else {"learning_rate": learning_rate}
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=N_TREES, random_state=123, **extra)
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=N_TREES, random_state=123, **extra)
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=N_TREES, random_seed=123, verbose=0,
            allow_writing_files=False, **extra
        )
    raise ValueError(name)


def available(est):
    try:
        make_estimator(est)
    except ImportError:
        return False
    return True


def build_case(case, est):
    """Return `run(n_jobs)` (signature of the result) and `fit_once()` (seconds)."""
    import numpy as np
    from skforecast.model_selection import (
        TimeSeriesFold,
        backtesting_forecaster,
        backtesting_forecaster_multiseries,
        grid_search_forecaster,
    )
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import (
        ForecasterRecursive,
        ForecasterRecursiveMultiSeries,
    )

    spec = CASES[case]
    common = {"metric": "mean_absolute_error", "show_progress": False}

    def signature(values):
        array = np.asarray(values, dtype=float).ravel()
        array = array[np.isfinite(array)]
        positions = np.linspace(0, array.size - 1, 7).astype(int)
        return [int(array.size), float(array.mean())] + [float(v) for v in array[positions]]

    if spec["op"] == "ms":
        series = load_sales()
        n_train = len(series) - spec["n_val"]

        def new_forecaster():
            return ForecasterRecursiveMultiSeries(
                estimator=make_estimator(est), lags=14, encoding="ordinal"
            )

        def new_cv():
            return TimeSeriesFold(steps=spec["steps"], initial_train_size=n_train,
                                  refit=spec["refit"], verbose=False)

        def run(n_jobs):
            _, predictions = backtesting_forecaster_multiseries(
                forecaster=new_forecaster(), series=series, cv=new_cv(),
                n_jobs=n_jobs, **common
            )
            return signature(predictions["pred"])

        def fit_once():
            forecaster = new_forecaster()
            start = time.perf_counter()
            forecaster.fit(series=series.iloc[:n_train])
            return time.perf_counter() - start

        return run, fit_once

    data = load_bike()
    n_train = len(data.loc[:END_TRAIN])
    data = data.iloc[: n_train + spec["n_val"]]
    y, exog = data["users"], data.drop(columns="users")

    def new_forecaster(learning_rate=None):
        return ForecasterRecursive(
            estimator=make_estimator(est, learning_rate), lags=48,
            window_features=RollingFeatures(stats=["mean"], window_sizes=72),
        )

    def new_cv():
        return TimeSeriesFold(steps=spec["steps"], initial_train_size=n_train,
                              refit=spec["refit"], verbose=False)

    if spec["op"] == "bt":
        def run(n_jobs):
            _, predictions = backtesting_forecaster(
                forecaster=new_forecaster(), y=y, exog=exog, cv=new_cv(),
                n_jobs=n_jobs, **common
            )
            return signature(predictions["pred"])
    else:
        def run(n_jobs):
            results = grid_search_forecaster(
                forecaster=new_forecaster(), y=y, exog=exog, cv=new_cv(),
                param_grid={"learning_rate": [0.03, 0.05, 0.1]},
                return_best=False, verbose=False, n_jobs=n_jobs, **common
            )
            return signature(results["mean_absolute_error"])

    def fit_once():
        forecaster = new_forecaster()
        start = time.perf_counter()
        forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
        return time.perf_counter() - start

    return run, fit_once


# ----------------------------------------------------------------------------
# Child process: one measure
# ----------------------------------------------------------------------------
def child(case, est, config):
    from joblib import cpu_count

    import skforecast
    import skforecast.model_selection._validation as validation

    # `n_jobs` of every `Parallel` of the backtesting (main process only).
    recorded = []
    original = validation.Parallel

    class RecordingParallel(original):
        def __init__(self, *args, **kwargs):
            recorded.append(kwargs.get("n_jobs", args[0] if args else None))
            super().__init__(*args, **kwargs)

    validation.Parallel = RecordingParallel

    n_jobs = {"1": 1, "auto": "auto", "max": max(1, cpu_count() - 1)}[config]
    run, fit_once = build_case(case, est)
    fit_before = fit_once() if est == "xgb" else None

    calls = []
    for _ in range(CALLS):
        recorded.clear()
        start = time.perf_counter()
        result = run(n_jobs)
        calls.append({
            "seconds": time.perf_counter() - start,
            "n_jobs": sorted(set(recorded), key=str),
            "signature": result,
        })

    fit_after = fit_once() if est == "xgb" else None
    print("RESULT " + json.dumps({
        "calls": calls, "fit_before": fit_before, "fit_after": fit_after,
        "skforecast_file": skforecast.__file__,
    }))


# ----------------------------------------------------------------------------
# Main process
# ----------------------------------------------------------------------------
def versions():
    from importlib.metadata import PackageNotFoundError, version

    from joblib import cpu_count

    out = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "cpu_count": cpu_count(),
        "os_cpu_count": os.cpu_count(),
    }
    for package in ["skforecast", "numpy", "pandas", "scikit-learn", "joblib",
                    "xgboost", "catboost"]:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    import skforecast

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(skforecast.__file__).parent,
        capture_output=True, text=True
    )
    out["skforecast_commit"] = commit.stdout.strip() or None
    return out


def read_results(path):
    results = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    machine = versions()
    (out / "machine.json").write_text(json.dumps(machine, indent=2))
    print(json.dumps(machine, indent=2))

    results_file = out / "results.jsonl"
    done = {(r["case"], r["est"], r["config"], r["rep"])
            for r in read_results(results_file) if r.get("ok")}
    plan = [
        (case, est, config, rep)
        for rep in range(args.reps)
        for case, est, configs in PLAN
        if (not args.cases or case in args.cases) and available(est)
        for config in configs
    ]
    pending = [cell for cell in plan if cell not in done]
    print(f"Plan: {len(plan)} measures, {len(pending)} pending.")

    # Prepare the data once, so that no measure pays the download.
    load_bike()
    load_sales()

    for i, (case, est, config, rep) in enumerate(pending, start=1):
        start = time.perf_counter()
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "child", case, est, config],
            capture_output=True, text=True, timeout=args.timeout
        )
        lines = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
        record = {"case": case, "est": est, "config": config, "rep": rep,
                  "ok": process.returncode == 0 and bool(lines)}
        if record["ok"]:
            record.update(json.loads(lines[-1][len("RESULT "):]))
        else:
            record["error"] = process.stderr[-2000:]
        with open(results_file, "a") as file:
            file.write(json.dumps(record) + "\n")
        times = " ".join(f"{c['seconds']:.1f}" for c in record.get("calls", []))
        n_jobs = record["calls"][0]["n_jobs"] if record["ok"] else "-"
        print(f"[{i}/{len(pending)}] {case:<15} {est:<9} {config:<5} n_jobs={n_jobs} "
              f"calls: {times or 'FAILED'} ({time.perf_counter() - start:.0f} s)",
              flush=True)
    print("Done.")


def summary(path):
    """Median of the repetitions of each cell: first call, next calls, n_jobs."""
    cells = {}
    for r in read_results(Path(path) / "results.jsonl"):
        if r.get("ok"):
            cells.setdefault((r["case"], r["est"], r["config"]), []).append(r)
    out = {}
    for key, reps in cells.items():
        out[key] = {
            "first": statistics.median(r["calls"][0]["seconds"] for r in reps),
            "next": statistics.median(
                statistics.median(c["seconds"] for c in r["calls"][1:]) for r in reps
            ),
            "n_jobs": reps[0]["calls"][0]["n_jobs"],
            "signature": reps[0]["calls"][0]["signature"],
            "fit_before": reps[0].get("fit_before"),
            "fit_after": reps[0].get("fit_after"),
        }
    return out


def same(a, b, tolerance=1e-6):
    return len(a) == len(b) and all(
        abs(x - y) <= tolerance * max(1.0, abs(x), abs(y)) for x, y in zip(a, b)
    )


def compare(args):
    before, after = summary(args.before), summary(args.after)
    for name, path in [("before", args.before), ("after", args.after)]:
        machine = json.loads((Path(path) / "machine.json").read_text())
        print(f"{name}: skforecast {machine.get('skforecast')} "
              f"commit {str(machine.get('skforecast_commit'))[:9]}")
    print("\nSeconds: first call / median of the next calls.\n")
    print(f"{'case':<15}{'est':<9}{'cfg':<5}| {'n_jobs b/a':<11}| {'before':>13} | "
          f"{'after':>13} | {'ratio':>11} | {'fit after b/a':>13} | same")
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if not b or not a:
            print(f"{key[0]:<15}{key[1]:<9}{key[2]:<5}| only in one of the runs")
            continue
        fits = (f"{b['fit_after']:5.1f}/{a['fit_after']:5.1f}"
                if b["fit_after"] is not None and a["fit_after"] is not None else "")
        n_jobs = "/".join(str(r["n_jobs"][0]) if r["n_jobs"] else "-" for r in (b, a))
        print(f"{key[0]:<15}{key[1]:<9}{key[2]:<5}| {n_jobs:<11}| "
              f"{b['first']:5.1f} / {b['next']:5.1f} | {a['first']:5.1f} / {a['next']:5.1f} | "
              f"{a['first'] / b['first']:4.2f} / {a['next'] / b['next']:4.2f} | {fits:>13} | "
              f"{'yes' if same(b['signature'], a['signature']) else 'NO'}")

    print("\n'auto' against the best of 1 and cpu_count() - 1 (after):")
    for (case, est, config), a in sorted(after.items()):
        if config != "auto":
            continue
        options = [after.get((case, est, c)) for c in ("1", "max")]
        options = [o for o in options if o]
        if not options:
            continue
        best_first = min(o["first"] for o in options)
        best_next = min(o["next"] for o in options)
        print(f"  {case:<15}{est:<9} n_jobs={a['n_jobs']}: first {a['first'] / best_first:4.2f}x, "
              f"next {a['next'] / best_next:4.2f}x of the best")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--reps", type=int, default=1)
    p_run.add_argument("--cases", nargs="+", choices=list(CASES))
    p_run.add_argument("--timeout", type=float, default=1800)
    p_child = sub.add_parser("child")
    p_child.add_argument("case")
    p_child.add_argument("est")
    p_child.add_argument("config")
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("before")
    p_compare.add_argument("after")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "child":
        child(args.case, args.est, args.config)
    else:
        compare(args)


if __name__ == "__main__":
    main()
