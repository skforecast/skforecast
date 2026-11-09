"""
Effect of limiting the OpenMP threads of HistGradientBoostingRegressor to 1
only while predicting (recursive loop), in a backtesting with n_jobs=1.
Variants are interleaved in random order in the same process (warm).
"""

import random
import statistics
import sys
import time

from threadpoolctl import threadpool_limits

import cases
from skforecast.model_selection import backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

refit = sys.argv[1] == "refit" if len(sys.argv) > 1 else False
est = sys.argv[2] if len(sys.argv) > 2 else "hgb"
y = cases.make_y(2000)
from skforecast.model_selection import TimeSeriesFold  # noqa: E402

cv = TimeSeriesFold(steps=24, initial_train_size=2000 - 24 * 20, refit=refit)
original = ForecasterRecursive._recursive_predict


def limited(self, *args, **kwargs):
    with threadpool_limits(limits=1, user_api="openmp"):
        return original(self, *args, **kwargs)


def run(variant):
    ForecasterRecursive._recursive_predict = limited if variant == "limit" else original
    f = ForecasterRecursive(estimator=cases.make_estimator(est), lags=24)
    t0 = time.perf_counter()
    m, p = backtesting_forecaster(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error", n_jobs=1,
        show_progress=False,
    )
    return time.perf_counter() - t0, cases.fingerprint(p)


res = {"default": [], "limit": []}
fps = set()
for v in res:
    run(v)
rng = random.Random(0)
for rep in range(5):
    order = list(res)
    rng.shuffle(order)
    for v in order:
        dt, fp = run(v)
        res[v].append(dt)
        fps.add(fp)
for v, secs in res.items():
    print(f"{est} refit={refit} {v}: median {statistics.median(secs):.3f} s, min {min(secs):.3f} s")
print("same predictions:", len(fps) == 1)
