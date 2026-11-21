"""
PERF-njobs-auto-first-fold: with `n_jobs='auto'`, a backtesting without refit
of a `HistGradientBoostingRegressor` decides whether to run in parallel from
the time of the first fold, predicted in the main process. In a fresh process
the first `predict` takes about 10 ms and the next ones about 0.6 ms, so with
about 600 folds or more (1200 on Windows) the first call of a session always
goes parallel, whatever the real cost of the predictions.

`Parallel` is replaced by a sequential one: no workers are started, the script
only records the `n_jobs` the backtesting asks for. Run it in a new process.

Usage: OMP_NUM_THREADS=1 python njobs_auto_first_fold.py

Result (2026-11-21, 4 CPUs): first call, first fold 12.5 ms, median of the
other folds 0.91 ms, estimate 12.48 s, n_jobs [3]; second call in the same
process, first fold 1.5 ms, estimate 1.45 s, n_jobs [1].
"""
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from _common import versions
from joblib import Parallel
from sklearn.ensemble import HistGradientBoostingRegressor

import skforecast.model_selection._validation as validation
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

print(versions())
rng = np.random.default_rng(0)
n = 2000
y = pd.Series(
    np.cumsum(rng.normal(size=n)),
    index=pd.date_range("2020-01-01", periods=n, freq="h"),
    name="y",
)
times, calls = [], []
real_fit_predict = validation._fit_predict_forecaster


def timed(**kwargs):
    start = time.perf_counter()
    out = real_fit_predict(**kwargs)
    times.append(time.perf_counter() - start)
    return out


def spy(**kwargs):
    calls.append(kwargs.get("n_jobs"))
    return Parallel(n_jobs=1)


forecaster = ForecasterRecursive(
    HistGradientBoostingRegressor(max_iter=50, random_state=0), lags=24
)
cv = TimeSeriesFold(steps=1, initial_train_size=1000, refit=False)
for label in ["first call", "second call"]:
    times.clear()
    calls.clear()
    with (
        patch.object(validation, "_fit_predict_forecaster", side_effect=timed),
        patch.object(validation, "Parallel", side_effect=spy),
    ):
        backtesting_forecaster(
            forecaster, y=y, cv=cv, metric="mean_absolute_error",
            n_jobs="auto", show_progress=False,
        )
    t = np.array(times)
    print(
        f"{label}: {len(t)} folds, first fold {t[0] * 1000:.1f} ms, "
        f"median of the others {np.median(t[1:]) * 1000:.2f} ms, "
        f"estimate {t[0] * (len(t) - 1):.2f} s, n_jobs passed to Parallel {calls}"
    )
