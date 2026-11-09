"""
Diagnose why a sequential XGBoost backtesting (n_jobs=1) is ~2x slower in a
process after a parallel run. Usage: python probe_xgb_after_parallel.py TRIGGER
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import cases  # noqa: E402
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster  # noqa: E402
from skforecast.recursive import ForecasterRecursive  # noqa: E402

y = cases.make_y(2000)
cv = TimeSeriesFold(steps=24, initial_train_size=2000 - 24 * 10, refit=True)


def bt(n_jobs, est="xgb"):
    f = ForecasterRecursive(estimator=cases.make_estimator(est), lags=24)
    t0 = time.perf_counter()
    backtesting_forecaster(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error", n_jobs=n_jobs,
        show_progress=False,
    )
    return round(time.perf_counter() - t0, 2)


def trivial(i):
    return i


def xgb_fit(i):
    from xgboost import XGBRegressor
    rng = np.random.default_rng(i)
    X = rng.normal(size=(1500, 24))
    XGBRegressor(n_estimators=50).fit(X, X[:, 0])
    return i


trigger = sys.argv[1]
before = [bt(1), bt(1)]
if trigger == "trivial":
    Parallel(n_jobs=3)(delayed(trivial)(i) for i in range(6))
elif trigger == "xgb_fits":
    Parallel(n_jobs=3)(delayed(xgb_fit)(i) for i in range(6))
elif trigger == "bt_ridge":
    bt(3, "ridge")
elif trigger == "bt_xgb":
    bt(3, "xgb")
elif trigger == "threads_xgb":
    Parallel(n_jobs=3, prefer="threads")(delayed(xgb_fit)(i) for i in range(6))
after = [bt(1), bt(1)]
print(f"{trigger}: before {before}, after {after}")
