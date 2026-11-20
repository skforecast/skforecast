"""
A grid search over a multi-series backtesting without refit: today, and with
the prototype (`proto_bt_blocks.py`) called once per candidate, without and
with the preparation shared by all candidates (`static`) and numpy metrics.

Usage: python time_search.py N_SERIES EXOG EST(ridge|lgbm) N_CANDIDATES N_FOLDS [STEPS]
Run alone, OMP_NUM_THREADS=1.
"""
import _env  # noqa: F401
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from bench_data import make_data
from proto_bt_blocks import backtesting_blocks, prepare_static
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
n_cand = int(sys.argv[4]); n_folds = int(sys.argv[5])
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 24
n_rows = 1500
series, exog = make_data(n_series, exog_kind, n_rows)
if est == "ridge":
    estimator, grid = Ridge(), {"alpha": list(np.logspace(-3, 3, n_cand))}
else:
    estimator = LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
    grid = {"learning_rate": list(np.linspace(0.02, 0.3, n_cand))}
forecaster = ForecasterRecursiveMultiSeries(estimator, lags=24, encoding="ordinal")
cv = TimeSeriesFold(steps=steps, initial_train_size=n_rows - n_folds * steps, refit=False, verbose=False)
key = next(iter(grid))

t = time.perf_counter()
res = grid_search_forecaster_multiseries(
    forecaster=deepcopy(forecaster), series=series, exog=exog, cv=cv, param_grid=grid,
    metric="mean_absolute_error", aggregate_metric="weighted_average",
    return_best=False, n_jobs=1, show_progress=False,
)
t_cur = time.perf_counter() - t
col = [c for c in res.columns if "mean_absolute_error" in c][0]
current = dict(zip(res[key], res[col]))


def proto(shared, fast):
    t = time.perf_counter()
    static = None
    if shared:
        cv2 = deepcopy(cv)
        cv2.set_params({"window_size": forecaster.window_size})
        static = prepare_static(series, exog, cv2)
    out, fit = {}, 0.0
    for value in grid[key]:
        f = deepcopy(forecaster)
        f.set_params({key: value})
        tm = {}
        m, _ = backtesting_blocks(
            f, series=series, exog=exog, cv=cv, metric="mean_absolute_error",
            static=static, fast_metrics=fast, timings=tm,
        )
        fit += tm["fit"]
        out[value] = m.loc[m["levels"] == "weighted_average", "mean_absolute_error"].iloc[0]
    total = time.perf_counter() - t
    diff = max(abs(out[v] - current[v]) for v in out)
    return total, fit, diff


print(f"n_series={n_series} exog={exog_kind} est={est} candidates={n_cand} folds={n_folds} steps={steps}")
print(f"CURRENT grid search: {t_cur:7.2f} s ({t_cur / n_cand * 1e3:.0f} ms per candidate)")
for label, shared, fast in (
    ("prototype per candidate, nothing shared, metrics of today", False, False),
    ("prototype, preparation shared, metrics of today", True, False),
    ("prototype, preparation shared, numpy metrics", True, True),
):
    total, fit, diff = proto(shared, fast)
    print(f"{label:<60} {total:7.2f} s ({t_cur / total:4.1f}x; fits {fit:5.2f} s, rest {total - fit:5.2f} s; max metric diff {diff:.2g})")
print(_env.versions())
