"""
Small timings behind the verdicts of REVISION.md (run alone, OMP_NUM_THREADS=1):
D-NP1 (cost per fold of a single-series backtesting), P2 (bool exog in the fit
of ForecasterRecursive), P7 (`training_range_`) and P17 (empty `drop`).
"""
import _env  # noqa: F401
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")


def best(fn, n=5):
    out = np.inf
    for _ in range(n):
        t = time.perf_counter(); fn(); out = min(out, time.perf_counter() - t)
    return out


rng = np.random.default_rng(0)
n = 3000
idx = pd.date_range("2020-01-01", periods=n, freq="h")
y = pd.Series(np.cumsum(rng.normal(size=n)), index=idx, name="y")
exog = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["a", "b", "c"])
print("D-NP1: single-series backtesting, 50 folds of 24 steps, no refit, exog of 3 float columns")
for name, est in (("Ridge", Ridge()), ("LightGBM(50)", LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1))):
    f = ForecasterRecursive(est, lags=24)
    cv = TimeSeriesFold(steps=24, initial_train_size=n - 50 * 24, refit=False, verbose=False)
    t_bt = best(lambda: backtesting_forecaster(f, y=y, exog=exog, cv=cv, metric="mean_absolute_error", n_jobs=1, show_progress=False), 3)
    f.fit(y.iloc[: n - 1200], exog=exog.iloc[: n - 1200])
    t_fit = best(lambda: f.fit(y.iloc[: n - 1200], exog=exog.iloc[: n - 1200]), 3)
    print(f"  {name:<13} whole call {t_bt * 1e3:7.1f} ms, fit {t_fit * 1e3:6.1f} ms -> {(t_bt - t_fit) / 50 * 1e3:.2f} ms per fold")

print("P2: fit of ForecasterRecursive(Ridge), 100000 rows, 3 float columns, with and without one bool column")
n2 = 100_000
idx2 = pd.date_range("2000-01-01", periods=n2, freq="h")
y2 = pd.Series(rng.normal(size=n2), index=idx2)
ex_f = pd.DataFrame(rng.normal(size=(n2, 3)), index=idx2, columns=["a", "b", "c"])
ex_b = ex_f.assign(flag=rng.random(n2) < 0.5)
ex_bf = ex_f.assign(flag=(rng.random(n2) < 0.5).astype(float))
for name, ex in (("float only", ex_f), ("float + bool", ex_b), ("float + bool cast to float by the user", ex_bf)):
    f = ForecasterRecursive(Ridge(), lags=24)
    print(f"  {name:<40} {best(lambda: f.fit(y2, exog=ex), 3) * 1e3:7.1f} ms")

print("P7: v[[0, -1]] against v[np.array([0, -1])] on a DatetimeIndex")
pos = np.array([0, -1])
t_a = best(lambda: [idx[[0, -1]] for _ in range(1000)]) / 1000
t_b = best(lambda: [idx[pos] for _ in range(1000)]) / 1000
print(f"  list {t_a * 1e6:.1f} us, array {t_b * 1e6:.1f} us, identical={idx[[0, -1]].identical(idx[pos])}")

print("P17: DataFrame.drop(columns=[]) on a window of 24 rows x 10 columns")
lw = pd.DataFrame(rng.normal(size=(3000, 10)), index=idx).iloc[-24:]
print(f"  {best(lambda: [lw.drop(columns=[]) for _ in range(1000)]) / 1000 * 1e6:.1f} us per call")
print(_env.versions())
