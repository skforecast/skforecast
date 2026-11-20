"""
Variant of py59 (`real_data.py`, `bdg2`) that is not in the article: the
hyperparameter search with a backtesting (`TimeSeriesFold`, steps 7, the 61
days of validation, 9 folds) instead of `OneStepAheadFold`. Grid of 4
candidates over the default LightGBM (lags 14), one thread. Today
(`grid_search_forecaster_multiseries`) against the prototype called once per
candidate, with the preparation shared by the candidates and numpy metrics.

Usage: REAL_DATA_DIR=... python time_real_search.py
Run it alone on the machine (OMP_NUM_THREADS=1).
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import time
import warnings
from copy import deepcopy

from lightgbm import LGBMRegressor

from proto_bt_blocks import backtesting_blocks, prepare_static
from real_data import _bdg2_data
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster_multiseries
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
end_validation = "2017-10-31 23:59:00"
series_dict, exog_dict = _bdg2_data()
series = {k: v.loc[:end_validation] for k, v in series_dict.items()}
exog = {k: v.loc[:end_validation] for k, v in exog_dict.items()}
forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=8520, verbose=-1, n_jobs=1),
    lags=14,
    window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
    categorical_features="auto",
    encoding="ordinal",
)
cv = TimeSeriesFold(initial_train_size="2017-08-31", steps=7, refit=False, verbose=False)
grid = {"n_estimators": [100, 200], "num_leaves": [31, 63]}
candidates = [{"n_estimators": n, "num_leaves": leaves}
              for n in grid["n_estimators"] for leaves in grid["num_leaves"]]

FIT = [0.0]
_orig_fit = ForecasterRecursiveMultiSeries.fit


def _timed_fit(self, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return _orig_fit(self, *args, **kwargs)
    finally:
        FIT[0] += time.perf_counter() - t0


ForecasterRecursiveMultiSeries.fit = _timed_fit
t = time.perf_counter()
res = grid_search_forecaster_multiseries(
    forecaster=deepcopy(forecaster), series=series, exog=exog, cv=cv, param_grid=grid,
    metric="mean_absolute_error", aggregate_metric="weighted_average",
    return_best=False, n_jobs=1, show_progress=False, suppress_warnings=True,
)
t_cur = time.perf_counter() - t
fit_cur = FIT[0]
col = [c for c in res.columns if "mean_absolute_error" in c][0]
current = {tuple(sorted(p.items())): v for p, v in zip(res["params"], res[col])}

t = time.perf_counter()
cv2 = deepcopy(cv)
cv2.set_params({"window_size": forecaster.window_size})
static = prepare_static(series, exog, cv2)
out, fit = {}, 0.0
for params in candidates:
    f = deepcopy(forecaster)
    f.estimator.set_params(**params)
    tm = {}
    m, _ = backtesting_blocks(
        f, series=series, exog=exog, cv=cv, metric="mean_absolute_error", static=static,
        fast_metrics=True, suppress_warnings=True, timings=tm,
    )
    fit += tm["fit"]
    out[tuple(sorted(params.items()))] = (
        m.loc[m["levels"] == "weighted_average", "mean_absolute_error"].iloc[0]
    )
t_new = time.perf_counter() - t
diff = max(abs(out[k] - current[k]) for k in out)
n_folds = len(cv2.split(X=series, as_pandas=False))
print(f"bdg2 search: {len(series)} series, {len(candidates)} candidates, {n_folds} folds of 7 steps, "
      f"LightGBM (n_jobs=1)")
print(f"  today      {t_cur:7.2f} s ({t_cur / len(candidates):.1f} s per candidate; fits {fit_cur:.2f} s, rest {t_cur - fit_cur:.2f} s)")
print(f"  prototype  {t_new:7.2f} s (x{t_cur / t_new:.2f}; fits {fit:.2f} s, rest {t_new - fit:.2f} s); "
      f"max diff of the metric {diff:.2g}")
print(" ", _env.versions())
