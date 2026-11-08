"""
PERF-contiguous-lags, script 05: peak memory of `ForecasterRecursive.fit` with
the training matrix as it is today (view, flag off) and C-contiguous
(prototype, flag on).

Peak measured with `tracemalloc` (numpy reports its allocations to it). The
allocations made in C/C++ by LightGBM, XGBoost or CatBoost are not traced, but
they are the same in both versions; the copy of the prototype is a numpy array
and is traced.

Run from the root of the repository (`F` measures the F-ordered variant):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_05_memory.py
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_05_memory.py F
"""

import gc
import sys
import platform
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

import skforecast.recursive._forecaster_recursive as m_rec
from skforecast.recursive import ForecasterRecursive

warnings.filterwarnings("ignore")

ON_VALUE = "F" if len(sys.argv) > 1 and sys.argv[1] == "F" else True


def make_y(n: int) -> pd.Series:
    rng = np.random.default_rng(123)
    return pd.Series(rng.normal(size=n).cumsum(), index=pd.RangeIndex(n), name="y")


ESTIMATORS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=50),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=20),
    "LightGBM": lambda: LGBMRegressor(n_estimators=20, verbose=-1),
}

print(f"Python {platform.python_version()}, numpy {np.__version__}, pandas "
      f"{pd.__version__}, scikit-learn {sklearn.__version__}, {platform.platform()}")
print(f"Version under test: {'F order' if ON_VALUE == 'F' else 'C order'}")
print("| rows x lags | matrix (MB) | estimator | peak off (MB) | peak on (MB) "
      "| difference (MB) |")
print("|---|---|---|---|---|---|")
for n_rows, lags in [(20_000, 168), (200_000, 24), (200_000, 168)]:
    y = make_y(n_rows + lags)
    matrix_mb = n_rows * lags * 8 / 2**20
    for name, make_est in ESTIMATORS.items():
        peaks = {}
        for flag in (False, True, False, True):
            m_rec._CONTIGUOUS_X_TRAIN = ON_VALUE if flag else False
            f = ForecasterRecursive(estimator=make_est(), lags=lags)
            gc.collect()
            tracemalloc.start()
            f.fit(y=y)
            peak = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
            peaks[flag] = min(peaks.get(flag, np.inf), peak)
            del f
        m_rec._CONTIGUOUS_X_TRAIN = True
        print(f"| {n_rows:,} x {lags} | {matrix_mb:.0f} | {name} | {peaks[False]:.0f} "
              f"| {peaks[True]:.0f} | {peaks[True] - peaks[False]:+.0f} |", flush=True)
