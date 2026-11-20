"""
Where the time of a multi-series backtesting without refit goes today, and how
close a "validate once" implementation (`proto_bt_once.py`) gets to the ceiling
of the study. Same data as `scripts/ceiling_bt_ms.py`.

Usage: python time_bt_once.py N_SERIES EXOG(none|float|mixed) EST(ridge|lgbm) [N_FOLDS] [STEPS]

Run it alone on the machine (OMP_NUM_THREADS=1).
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

import skforecast.model_selection._validation as val
import skforecast.recursive._forecaster_recursive_multiseries as frm
from proto_bt_once import backtesting_once
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
n_folds = int(sys.argv[4]) if len(sys.argv) > 4 else 50
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 24
n_rows = 1500; lags = 24
rng = np.random.default_rng(1)
idx = pd.date_range("2020-01-01", periods=n_rows, freq="h")
names = [f"series_{i:04d}" for i in range(n_series)]
values = np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) + 100
series = {k: pd.Series(values[:, j], index=idx, name=k) for j, k in enumerate(names)}
DOW = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])


def make_exog(j):
    df = pd.DataFrame({"temp": rng.normal(size=n_rows), "promo": rng.normal(size=n_rows)}, index=idx)
    if exog_kind == "mixed":
        df["hour"] = idx.hour.to_numpy().astype("int64")
        df["dow"] = pd.Categorical(DOW[idx.dayofweek.to_numpy()], categories=list(DOW))
        df["shop_type"] = np.where(rng.random(n_rows) < 0.5, "big", "small")
    return df


exog = None if exog_kind == "none" else {k: make_exog(j) for j, k in enumerate(names)}
estimator = Ridge() if est == "ridge" else LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
forecaster = ForecasterRecursiveMultiSeries(estimator, lags=lags, encoding="ordinal")
cv = TimeSeriesFold(steps=steps, initial_train_size=n_rows - n_folds * steps, refit=False, verbose=False)
kwargs = dict(forecaster=forecaster, series=series, exog=exog, cv=cv, metric="mean_absolute_error")

# ----------------------------------------------------------------------
# Current implementation, plain (best of 3) and with stage timers (one run).
# ----------------------------------------------------------------------
best = np.inf
for _ in range(3):
    t = time.perf_counter()
    m_cur, p_cur = backtesting_forecaster_multiseries(
        **kwargs, n_jobs=1, show_progress=False, suppress_warnings=False
    )
    best = min(best, time.perf_counter() - t)

acc = {}


def timed(name, fn, is_generator=False):
    def wrapper(*args, **kw):
        t = time.perf_counter()
        out = fn(*args, **kw)
        if is_generator:
            out = iter(list(out))
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return out
    return wrapper


for module, attr, name, gen in [
    (val, "check_preprocess_series", "preprocess (once, per series)", False),
    (val, "check_preprocess_exog_multiseries", "preprocess (once, per series)", False),
    (val, "deepcopy_forecaster", "copy of the forecaster", False),
    (val, "_extract_data_folds_multiseries", "extract folds (per fold and series)", True),
    (val, "_get_y_true_multiseries", "y_true", False),
    (val, "_calculate_metrics_backtesting_multiseries", "metrics", False),
    (val, "_warn_levels_not_predicted_multiseries", "warn levels not predicted", False),
    (frm, "check_predict_input", "predict: check_predict_input", False),
]:
    setattr(module, attr, timed(name, getattr(module, attr), gen))
C = ForecasterRecursiveMultiSeries
orig_inputs = C._create_predict_inputs
C.fit = timed("fit", C.fit)
C._create_predict_inputs = timed("predict: _create_predict_inputs (with check)", C._create_predict_inputs)
C._recursive_predict = timed("predict: _recursive_predict", C._recursive_predict)
C.predict = timed("predict: total", C.predict)
t = time.perf_counter()
backtesting_forecaster_multiseries(**kwargs, n_jobs=1, show_progress=False, suppress_warnings=False)
total_instr = time.perf_counter() - t
print(f"n_series={n_series} exog={exog_kind} est={est} folds={n_folds} steps={steps}")
print(f"CURRENT total: best of 3 {best * 1000:9.1f} ms; instrumented run {total_instr * 1000:9.1f} ms")
fit = acc.pop("fit")
pred_total = acc.pop("predict: total")
inputs = acc.pop("predict: _create_predict_inputs (with check)")
check = acc.pop("predict: check_predict_input")
rec = acc.pop("predict: _recursive_predict")
rows = [
    ("fit (first fold)", fit),
    *sorted(acc.items()),
    ("predict: check_predict_input", check),
    ("predict: rest of _create_predict_inputs", inputs - check),
    ("predict: _recursive_predict", rec),
    ("predict: inverse transform + output DataFrame", pred_total - inputs - rec),
]
other = total_instr - sum(v for _, v in rows)
rows.append(("rest (cv.split, concat, MultiIndex, y_true NaN, ...)", other))
no_fit = total_instr - fit
for name, v in rows:
    share = "" if name.startswith("fit") else f"{v / no_fit * 100:5.1f} % of the time without fit"
    print(f"   {name:<52}{v * 1000:9.1f} ms   {share}")
print(f"   -> without fit: {no_fit * 1000:.1f} ms = {no_fit / n_folds * 1000:.2f} ms per fold")
per_fold = (
    acc["extract folds (per fold and series)"] + pred_total
)
print(f"   -> truly per fold (extract + predict): {per_fold / n_folds * 1000:.2f} ms per fold; "
      f"once per backtesting: {(no_fit - per_fold) * 1000:.1f} ms")

# ----------------------------------------------------------------------
# Prototype: validate and prepare once, numpy folds, same outputs.
# ----------------------------------------------------------------------
best_t = None
for _ in range(3):
    tm = {}
    m_new, p_new = backtesting_once(**kwargs, suppress_warnings=False, timings=tm)
    if best_t is None or tm["total"] < best_t["total"]:
        best_t = tm
pd.testing.assert_frame_equal(m_cur, m_new, check_exact=True)
pd.testing.assert_frame_equal(p_cur, p_new, check_exact=True)
tm = best_t
no_fit_new = tm["total"] - tm["fit"]
print(f"VALIDATE ONCE (identical metrics and predictions): total {tm['total'] * 1000:.1f} ms")
for k in ("preprocess", "prologue", "fit", "prepare", "folds", "output", "metrics"):
    print(f"   {k:<52}{tm[k] * 1000:9.1f} ms")
print(f"   -> without fit: {no_fit_new * 1000:.1f} ms = {no_fit_new / n_folds * 1000:.2f} ms per fold "
      f"(folds only: {tm['folds'] / n_folds * 1000:.2f} ms per fold)")
print(f"SPEED-UP without fit: {no_fit / no_fit_new:.1f}x (instrumented run of the current code); "
      f"whole call {best / tm['total']:.1f}x")
print(_env.versions())
