"""
Does XGBoost predict the single rows of the recursive loop faster with one
OpenMP thread than with all the threads of the machine?

Hypothesis: part of the gain of running the folds of a backtesting without
refit in parallel (skforecast 0.26.0) comes from each worker predicting with
one thread (joblib sets OMP_NUM_THREADS = cpu_count // n_jobs in the workers),
not from sharing the folds. If so, limiting the threads of the prediction in
the main process gives the gain without starting workers.

Usage (nothing is written outside the folder of this script):

    python bench_xgb_predict_threads.py micro
    python bench_xgb_predict_threads.py loop
    python bench_xgb_predict_threads.py macro --reps 3
    python bench_xgb_predict_threads.py all

`micro`: time per call of `Booster.inplace_predict` by number of rows and
threads, in one process.
`loop`: the predictions of the 102 folds (36 steps each) with a fitted
forecaster, without the fit, in one process: default threads, 1 thread
(threadpoolctl or a copy of the booster with nthread=1) and the loop without
the estimator.
`macro`: `backtesting_forecaster` without refit (steps=36, 102 folds), each
measure in a new process that calls it 4 times (first call and following ones):

    seq_default  n_jobs=1, current code
    seq_1thread  n_jobs=1, `inplace_predict` limited to 1 OpenMP thread up to
                 100 rows (threadpoolctl, as `HistGradientBoostingRegressor`)
    parallel     n_jobs=cpu_count() - 1, current code ('auto' of 0.26.0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_bench_cache"
DATA_FILE = CACHE / "bike_sharing.pkl"
RESULTS_FILE = HERE / "bench_xgb_predict_threads_results.jsonl"

END_TRAIN = "2012-03-31 23:00:00"
END_VALIDATION = "2012-08-31 23:00:00"
SIZES = {"large": "2011-01-08 00:00:00", "small": "2011-09-01 00:00:00"}
PREFERRED_EXOG = [
    "month_sin", "month_cos", "week_day_sin", "week_day_cos",
    "hour_day_sin", "hour_day_cos", "temp", "holiday", "weather",
    "hum", "windspeed",
]
N_EXOG = 8
VARIANTS = ["seq_default", "seq_1thread", "parallel"]
MAX_ROWS_ONE_THREAD = 100


def load_data():
    """Dataset of the report, downloaded once and cached next to the script."""
    import pandas as pd

    if DATA_FILE.exists():
        return pd.read_pickle(DATA_FILE)

    from skforecast.datasets import fetch_dataset

    data = fetch_dataset("bike_sharing_extended_features", verbose=False)
    exog_cols = [c for c in PREFERRED_EXOG if c in data.columns][:N_EXOG]
    exog_cols += [
        c for c in data.columns if c not in exog_cols and c != "users"
    ][: N_EXOG - len(exog_cols)]
    data = data.loc[:END_VALIDATION, ["users"] + exog_cols].astype("float64")
    CACHE.mkdir(exist_ok=True)
    data.to_pickle(DATA_FILE)
    return data


def make_forecaster():
    from xgboost import XGBRegressor
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursive

    return ForecasterRecursive(
        estimator       = XGBRegressor(n_estimators=300, random_state=123),
        lags            = 48,
        window_features = RollingFeatures(stats=["mean"], window_sizes=72),
    )


def versions() -> dict:
    import joblib
    import numpy
    import pandas
    import sklearn
    import threadpoolctl
    import xgboost
    import skforecast

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "joblib_cpu_count": joblib.cpu_count(),
        "physical_cores": joblib.cpu_count(only_physical_cores=True),
        "skforecast": skforecast.__version__,
        "xgboost": xgboost.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def limit_inplace_predict_threads():
    """
    Patch `Booster.inplace_predict` to use 1 OpenMP thread up to 100 rows.
    Prototype of the change in `_build_predict_function`; only affects the
    process where it is called (the main one).
    """
    import xgboost
    from threadpoolctl import ThreadpoolController

    controller = ThreadpoolController()
    original = xgboost.Booster.inplace_predict

    def inplace_predict(self, data, *args, **kwargs):
        if data.shape[0] <= MAX_ROWS_ONE_THREAD:
            with controller.limit(limits=1, user_api="openmp"):
                return original(self, data, *args, **kwargs)
        return original(self, data, *args, **kwargs)

    xgboost.Booster.inplace_predict = inplace_predict


# ----------------------------------------------------------------------------
# micro
# ----------------------------------------------------------------------------
def time_calls(function, n_calls: int) -> float:
    """Microseconds per call."""
    start = time.perf_counter()
    for _ in range(n_calls):
        function()
    return (time.perf_counter() - start) / n_calls * 1e6


def run_micro(reps: int) -> None:
    import numpy as np
    from threadpoolctl import ThreadpoolController

    data = load_data()
    train = data.loc[SIZES["large"]:END_TRAIN]
    forecaster = make_forecaster()
    forecaster.fit(y=train["users"], exog=train.drop(columns="users"))
    X_train, _ = forecaster.create_train_X_y(
        y=train["users"], exog=train.drop(columns="users")
    )
    X = np.ascontiguousarray(X_train.to_numpy(dtype=float))
    booster = forecaster.estimator.get_booster()

    controller = ThreadpoolController()
    print("\nVersions:", json.dumps(versions(), indent=1))
    print("OpenMP runtimes seen by threadpoolctl:")
    for lib in controller.info():
        if lib["user_api"] == "openmp":
            print(f"  {Path(lib['filepath']).name}: num_threads={lib['num_threads']}")
    print(f"Matrix of predictors: {X.shape}, model: 300 trees, max_depth=6")

    n_cpu = os.cpu_count()
    thread_counts = sorted({t for t in (1, 2, 4, 8, 16, n_cpu) if t <= n_cpu})
    booster_1 = booster.copy()
    booster_1.set_param({"nthread": 1})

    for n_rows in (1, 10, 50, 100, 250, 1000, 5000):
        X_rows = X[:n_rows]
        n_calls = max(20, min(2000, 20000 // n_rows))
        reference = booster.inplace_predict(X_rows)

        def default():
            return booster.inplace_predict(X_rows)

        def nthread_param():
            return booster_1.inplace_predict(X_rows)

        configs = {"default": default, "copy nthread=1": nthread_param}
        for threads in thread_counts:
            def limited(threads=threads):
                with controller.limit(limits=threads, user_api="openmp"):
                    return booster.inplace_predict(X_rows)

            configs[f"limit={threads}"] = limited

        for name, function in configs.items():
            assert np.array_equal(function(), reference), name

        times = {name: [] for name in configs}
        for _ in range(reps):
            names = list(configs)
            random.shuffle(names)
            for name in names:
                times[name].append(time_calls(configs[name], n_calls))

        base = statistics.median(times["default"])
        print(f"\nrows={n_rows} ({n_calls} calls x {reps} reps), microseconds per call")
        for name in configs:
            median = statistics.median(times[name])
            print(
                f"  {name:<16} {median:9.1f}  (min {min(times[name]):9.1f})"
                f"  x{median / base:5.2f} of default"
            )


# ----------------------------------------------------------------------------
# loop
# ----------------------------------------------------------------------------
def run_loop(reps: int) -> None:
    """Predictions of the folds of the backtesting, without the fit."""
    import numpy as np
    import xgboost
    from threadpoolctl import ThreadpoolController

    data = load_data()
    y = data["users"]
    exog = data.drop(columns="users")
    n_train = len(data.loc[:END_TRAIN])
    forecaster = make_forecaster()
    forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
    window_size = forecaster.window_size
    steps = 36
    starts = list(range(n_train, len(y), steps))

    controller = ThreadpoolController()
    original = xgboost.Booster.inplace_predict
    state = {"mode": "default"}
    copies = {}
    n_cpu = os.cpu_count()
    limits = sorted({t for t in (1, 2, 4, 8) if t < n_cpu})

    def inplace_predict(self, data, *args, **kwargs):
        mode = state["mode"]
        if mode == "default":
            return original(self, data, *args, **kwargs)
        if mode == "no estimator":
            return np.zeros(data.shape[0], dtype=np.float32)
        if mode == "copy nthread=1":
            if id(self) not in copies:
                booster = self.copy()
                booster.set_param({"nthread": 1})
                copies[id(self)] = booster
            return original(copies[id(self)], data, *args, **kwargs)
        with controller.limit(limits=int(mode.split("=")[1]), user_api="openmp"):
            return original(self, data, *args, **kwargs)

    xgboost.Booster.inplace_predict = inplace_predict

    def predict_folds():
        predictions = []
        start_time = time.perf_counter()
        for start in starts:
            predictions.append(
                forecaster.predict(
                    steps       = steps,
                    last_window = y.iloc[start - window_size:start],
                    exog        = exog.iloc[start:start + steps],
                ).to_numpy()
            )
        return time.perf_counter() - start_time, np.concatenate(predictions)

    modes = ["default"] + [f"limit={t}" for t in limits] + [
        "copy nthread=1", "no estimator"
    ]
    _, reference = predict_folds()
    times = {mode: [] for mode in modes}
    for _ in range(reps):
        order = list(modes)
        random.shuffle(order)
        for mode in order:
            state["mode"] = mode
            seconds, predictions = predict_folds()
            if mode != "no estimator":
                assert np.array_equal(predictions, reference), mode
            times[mode].append(seconds)
    xgboost.Booster.inplace_predict = original

    n_steps = len(reference)
    print("\nVersions:", json.dumps(versions(), indent=1))
    print(
        f"\nPredictions of {len(starts)} folds of {steps} steps ({n_steps} steps),"
        f" {n_train} training rows, {reps} reps.\nSeconds: median (min), and"
        " milliseconds per step."
    )
    base = statistics.median(times["default"])
    for mode in modes:
        median = statistics.median(times[mode])
        print(
            f"  {mode:<16} {median:6.3f} ({min(times[mode]):6.3f})"
            f"  {median / n_steps * 1e3:6.3f} ms/step  x{median / base:4.2f}"
        )


# ----------------------------------------------------------------------------
# macro
# ----------------------------------------------------------------------------
def run_child(variant: str, size: str, n_calls: int) -> None:
    import joblib
    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

    if variant == "seq_1thread":
        limit_inplace_predict_threads()

    data = load_data().loc[SIZES[size]:]
    y = data["users"]
    exog = data.drop(columns="users")
    n_train = len(data.loc[:END_TRAIN])
    n_jobs = max(1, joblib.cpu_count() - 1) if variant == "parallel" else 1

    def fit_seconds() -> float:
        forecaster = make_forecaster()
        start = time.perf_counter()
        forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
        return time.perf_counter() - start

    fit_before = fit_seconds()
    calls, digests, n_folds = [], [], None
    for _ in range(n_calls):
        cv = TimeSeriesFold(steps=36, initial_train_size=n_train, refit=False)
        forecaster = make_forecaster()
        start = time.perf_counter()
        _, predictions = backtesting_forecaster(
            forecaster    = forecaster,
            y             = y,
            exog          = exog,
            cv            = cv,
            metric        = "mean_absolute_error",
            n_jobs        = n_jobs,
            verbose       = False,
            show_progress = False,
        )
        calls.append(time.perf_counter() - start)
        digests.append(
            hashlib.md5(predictions["pred"].to_numpy().round(7).tobytes()).hexdigest()
        )
        n_folds = int(predictions["fold"].nunique())
    fit_after = fit_seconds()

    print("RESULT " + json.dumps({
        "variant": variant, "size": size, "n_jobs": n_jobs, "n_train": n_train,
        "n_folds": n_folds, "n_predictions": len(predictions), "calls": calls,
        "fit_before": fit_before, "fit_after": fit_after,
        "digest": digests[0], "same_digest": len(set(digests)) == 1,
    }))


def run_macro(reps: int, n_calls: int, sizes: list[str]) -> None:
    load_data()
    print("\nVersions:", json.dumps(versions(), indent=1))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    results = []
    for rep in range(reps):
        cells = [(variant, size) for variant in VARIANTS for size in sizes]
        random.shuffle(cells)
        for variant, size in cells:
            output = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "child",
                 "--variant", variant, "--size", size, "--calls", str(n_calls)],
                capture_output=True, text=True, env=env, timeout=900,
            )
            lines = [l for l in output.stdout.splitlines() if l.startswith("RESULT ")]
            if not lines:
                print(f"FAILED {variant} {size}:\n{output.stderr[-2000:]}")
                continue
            result = json.loads(lines[-1][len("RESULT "):])
            result["rep"] = rep
            results.append(result)
            with open(RESULTS_FILE, "a") as file:
                file.write(json.dumps(result) + "\n")
            print(
                f"rep {rep} {variant:<12} {size:<6} n_jobs={result['n_jobs']:<3}"
                f" calls={[round(c, 2) for c in result['calls']]}"
                f" fit={result['fit_before']:.2f}/{result['fit_after']:.2f}",
                flush=True,
            )
    summarize(results)


def summarize(results: list[dict]) -> None:
    def med(values):
        return statistics.median(values) if values else float("nan")

    print(
        "\nBacktesting without refit, seconds: median (min) over the processes."
        "\n'first' = first call of a new process, 'next' = median of the"
        " following calls,\n'fit' = a fit alone before / after the calls."
    )
    for size in SIZES:
        rows = [r for r in results if r["size"] == size]
        if not rows:
            continue
        print(
            f"\nsize={size}: {rows[0]['n_train']} training rows,"
            f" {rows[0]['n_folds']} folds, {rows[0]['n_predictions']} predictions"
        )
        digests = {r["digest"] for r in rows}
        print(f"  predictions identical in all the variants: {len(digests) == 1}")
        base = None
        for variant in VARIANTS:
            cells = [r for r in rows if r["variant"] == variant]
            if not cells:
                continue
            first = [r["calls"][0] for r in cells]
            following = [med(r["calls"][1:]) for r in cells]
            if variant == "seq_default":
                base = med(following)
            ratio = f"x{med(following) / base:4.2f}" if base else ""
            print(
                f"  {variant:<12} n_jobs={cells[0]['n_jobs']:<3}"
                f" first {med(first):5.2f} ({min(first):5.2f})"
                f"  next {med(following):5.2f} ({min(following):5.2f}) {ratio}"
                f"  fit {med([r['fit_before'] for r in cells]):.2f}"
                f" / {med([r['fit_after'] for r in cells]):.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["micro", "loop", "macro", "all", "child"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--size", choices=list(SIZES))
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    args = parser.parse_args()

    if args.command == "child":
        run_child(args.variant, args.size, args.calls)
    else:
        if args.command in ("micro", "all"):
            run_micro(reps=max(args.reps, 5))
        if args.command in ("loop", "all"):
            run_loop(reps=max(args.reps, 9))
        if args.command in ("macro", "all"):
            run_macro(reps=args.reps, n_calls=args.calls, sizes=args.sizes)
