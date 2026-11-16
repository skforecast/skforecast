"""
Benchmark of the fixes of the final review of PR #1503 (n_jobs strategy).

Each scenario runs in a new process, with PYTHONPATH pointing to the version
of skforecast under test, and calls the operation twice: the first call starts
the workers ("first"), the second reuses them ("next"). It also samples the
peak use of /dev/shm (joblib memmaps) during the calls and hashes the output,
to check that the versions return the same results.

    python bench_review.py run          # every version and scenario
    python bench_review.py child NAME   # one scenario (used by run)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = {
    "0.28.x": os.path.join(HERE, "wt-028"),  # git worktree of origin/0.28.x
    "fd48114": os.path.join(HERE, "wt-fd"),  # git worktree of fd48114
    "head": os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")),
}
SCENARIOS = ["knn40", "knn480", "rf40", "rf480", "lgbm2000",
             "xgb_grid_refit", "hgb_norefit100"]
SKIP = {("0.28.x", "rf480")}  # Fills /dev/shm in 0.28.x (final review).
N_JOBS = 3


def child(name):
    import numpy as np
    import pandas as pd
    import skforecast
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import (
        TimeSeriesFold, backtesting_forecaster, grid_search_forecaster
    )

    rng = np.random.default_rng(123)

    def make_y(n):
        t = np.arange(n)
        return pd.Series(
            10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(size=n).cumsum() * 0.05
            + rng.normal(size=n),
            name="y",
        )

    def backtesting(estimator, n_train, lags, steps, n_folds, n_jobs, refit=False):
        y = make_y(n_train + steps * n_folds)
        cv = TimeSeriesFold(steps=steps, initial_train_size=n_train, refit=refit,
                            verbose=False)

        def run():
            forecaster = ForecasterRecursive(estimator=estimator, lags=lags)
            _, pred = backtesting_forecaster(
                forecaster=forecaster, y=y, cv=cv, metric="mean_absolute_error",
                n_jobs=n_jobs, use_in_sample_residuals=False, show_progress=False,
            )
            return pred
        return run

    if name.startswith("knn"):
        from sklearn.neighbors import KNeighborsRegressor
        run = backtesting(KNeighborsRegressor(), n_train=20_000, lags=240, steps=5,
                          n_folds=int(name[3:]), n_jobs=N_JOBS)
    elif name.startswith("rf"):
        from sklearn.ensemble import RandomForestRegressor
        run = backtesting(RandomForestRegressor(n_estimators=100, random_state=123),
                          n_train=8_000, lags=20, steps=5, n_folds=int(name[2:]),
                          n_jobs=N_JOBS)
    elif name == "lgbm2000":
        from lightgbm import LGBMRegressor
        run = backtesting(LGBMRegressor(n_estimators=500, random_state=123,
                                        n_jobs=1, verbose=-1),
                          n_train=10_000, lags=20, steps=1, n_folds=2000,
                          n_jobs=N_JOBS)
    elif name == "hgb_norefit100":
        from sklearn.ensemble import HistGradientBoostingRegressor
        run = backtesting(HistGradientBoostingRegressor(max_iter=200, random_state=123),
                          n_train=6_000, lags=24, steps=36, n_folds=100,
                          n_jobs="auto")
    elif name == "xgb_grid_refit":
        from xgboost import XGBRegressor
        y = make_y(6_000 + 24 * 6)
        cv = TimeSeriesFold(steps=24, initial_train_size=6_000, refit=True,
                            verbose=False)

        def run():
            forecaster = ForecasterRecursive(
                estimator=XGBRegressor(n_estimators=200, max_depth=6,
                                       random_state=123), lags=24)
            return grid_search_forecaster(
                forecaster=forecaster, y=y, cv=cv,
                param_grid={"learning_rate": [0.05, 0.1, 0.2]},
                metric="mean_absolute_error", return_best=False, n_jobs="auto",
                verbose=False, show_progress=False,
            )
    else:
        raise ValueError(name)

    peak = {"shm": 0}
    stop = threading.Event()
    base = shutil.disk_usage("/dev/shm").used

    def sample():
        while not stop.is_set():
            peak["shm"] = max(peak["shm"], shutil.disk_usage("/dev/shm").used - base)
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    times = []
    for _ in range(2):
        start = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - start)
    stop.set()
    sampler.join()

    if name == "xgb_grid_refit":
        out = out[["learning_rate", "mean_absolute_error"]]
    digest = int(pd.util.hash_pandas_object(out.round(8), index=True).sum())
    print(json.dumps({
        "scenario": name, "first": times[0], "next": times[1],
        "shm_peak_mb": peak["shm"] / 1e6, "hash": digest,
        "skforecast_file": skforecast.__file__,
    }))


def run_all():
    results = []
    out_path = os.path.join(HERE, "results.jsonl")
    for scenario in SCENARIOS:
        for version, path in VERSIONS.items():
            if (version, scenario) in SKIP:
                continue
            env = dict(os.environ, PYTHONPATH=path)
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, __file__, "child", scenario], env=env,
                capture_output=True, text=True, timeout=1200,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = {"scenario": scenario, "error": proc.stderr[-500:]}
            record["version"] = version
            record["wall"] = time.perf_counter() - start
            results.append(record)
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        run_all()
