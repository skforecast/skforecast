"""
H2 prototype: send an unfitted copy of the forecaster to the folds that refit
it (fold[5] is True), instead of the forecaster fitted in the main process.
The joblib.Parallel of skforecast's backtesting is replaced by a subclass that
rewrites the arguments of those tasks. Variants interleaved in one process:

- current: skforecast as is, n_jobs=3.
- unfitted: same, with the unfitted copy in the refit folds.
- seq: n_jobs=1, timed after each parallel variant to see the slowdown of the
  later sequential fits (XGBoost, second libgomp team).

Usage: python probe_unfitted_refit.py EST VARIANT (current or unfitted)
"""

import random
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib  # noqa: E402

import cases  # noqa: E402
import skforecast.model_selection._validation as val  # noqa: E402
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster  # noqa: E402
from skforecast.recursive import ForecasterRecursive  # noqa: E402
from skforecast.utils import deepcopy_forecaster  # noqa: E402

est = sys.argv[1]
y = cases.make_y(2000)
cv = TimeSeriesFold(steps=24, initial_train_size=2000 - 24 * 20, refit=True)
MODE = {"unfitted": False}


class UnfittedParallel(joblib.Parallel):
    def __call__(self, iterable):
        if not MODE["unfitted"]:
            return super().__call__(iterable)
        copies = {}

        def rewrite(task):
            func, args, kwargs = task
            fold = kwargs.get("fold")
            forecaster = kwargs.get("forecaster")
            if fold is not None and forecaster is not None and fold[5] is True:
                key = id(forecaster)
                if key not in copies:
                    copies[key] = deepcopy_forecaster(forecaster)
                kwargs = {**kwargs, "forecaster": copies[key]}
            return func, args, kwargs

        return super().__call__(rewrite(t) for t in iterable)


val.Parallel = UnfittedParallel


def bt(n_jobs):
    f = ForecasterRecursive(estimator=cases.make_estimator(est), lags=24)
    t0 = time.perf_counter()
    _, p = backtesting_forecaster(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error", n_jobs=n_jobs,
        show_progress=False,
    )
    return time.perf_counter() - t0, cases.fingerprint(p)


# NOTE: The second libgomp team lives as long as the loky executor, so each
# variant runs in its own process: clean sequential runs first, then parallel
# and sequential runs alternated.
variant = sys.argv[2]
MODE["unfitted"] = variant == "unfitted"
res = {"seq_clean": [], "parallel": [], "seq_after_parallel": []}
fps = set()
for rep in range(3):
    dt, fp = bt(1)
    res["seq_clean"].append(dt)
    fps.add(fp)
bt(3)  # warm-up of the workers
for rep in range(5):
    for key, n_jobs in (("parallel", 3), ("seq_after_parallel", 1)):
        dt, fp = bt(n_jobs)
        res[key].append(dt)
        fps.add(fp)
print(
    f"{est} {variant}: " + "; ".join(
        f"{k} {statistics.median(v):.2f} (min {min(v):.2f})" for k, v in res.items()
    ) + f"; same predictions: {len(fps) == 1}"
)
