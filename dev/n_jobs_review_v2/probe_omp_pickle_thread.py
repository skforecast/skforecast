"""
Does pickling a fitted model from a thread that stays alive (as loky's
QueueFeederThread does) slow down later fits in the main thread?
Usage: python probe_omp_pickle_thread.py ESTIMATOR ACTION
ESTIMATOR: xgb, xgb1, lgbm, lgbm1, hgb, catboost. ACTION: pickle, predict.
"""

import os
import pickle
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402


def make(name):
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=6)
    if name == "xgb1":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=6, n_jobs=1)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, verbose=-1)
    if name == "lgbm1":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, n_jobs=1, verbose=-1)
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=100)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=200, verbose=0)
    raise ValueError(name)


def n_threads():
    return len(os.listdir("/proc/self/task"))


def time_fits(name, X, y, k=5):
    times = []
    for _ in range(k):
        t0 = time.perf_counter()
        make(name).fit(X, y)
        times.append(time.perf_counter() - t0)
    return round(float(np.median(times)) * 1000)


name = sys.argv[1]
action = sys.argv[2]
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 30))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.1, size=5000)

model = make(name).fit(X, y)
before = time_fits(name, X, y)
threads_before = n_threads()

stop = threading.Event()


def feeder():
    if action == "pickle":
        pickle.dumps(model)
    else:
        model.predict(X[:24])
    stop.wait()


t = threading.Thread(target=feeder, daemon=True)
t.start()
time.sleep(0.5)
threads_alive = n_threads()
during = time_fits(name, X, y)

stop.set()
t.join()
time.sleep(0.5)
after = time_fits(name, X, y)

print(
    f"{name}, {action}: fit ms before {before}, with live thread {during} "
    f"({during / before:.2f}x), after thread ends {after} | threads "
    f"{threads_before} -> {threads_alive} -> {n_threads()}"
)
