"""
BUG-09: `backtesting_stats` fails when `fold_stride != steps` in two cases.

(a) Several estimators (long output, one row per date and estimator) with
    `fold_stride < steps`: the overlap filter `~index.duplicated(keep='last')`
    keeps a single estimator per date, so the metrics have one row and
    `metric_values.insert(0, 'estimator_id', ...)` raises.
(b) Sarimax with `refit=False` (or intermittent refit, `refit=int`):
    `last_window` is built as `y[test_start - steps : test_start]`, which only
    starts where the forecaster stopped when `fold_stride == steps`.

The script prints the observed behaviour and validates a candidate fix by
patching the two private functions and comparing with independent references.

Run: python dev/roadmap_0.27.0/repro/BUG-09.py
"""
import inspect
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn
import statsmodels

import skforecast
import skforecast.model_selection._validation as val
from skforecast.model_selection import TimeSeriesFold, backtesting_stats, grid_search_stats
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima, Ets, Sarimax

warnings.filterwarnings("ignore")

print(
    f"python {sys.version.split()[0]} | skforecast {skforecast.__version__} | "
    f"pandas {pd.__version__} | numpy {np.__version__} | "
    f"scikit-learn {sklearn.__version__} | statsmodels {statsmodels.__version__}"
)

# Data: synthetic daily series with weekly seasonality and a random walk trend
# ==============================================================================
rng = np.random.default_rng(123)
n = 300
idx = pd.date_range("2025-01-01", periods=n, freq="D")
y = pd.Series(
    100 + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + np.cumsum(rng.normal(0, 1, n)),
    index=idx,
    name="y",
)
n_train = 258  # 42 days (6 weeks) to backtest
STEPS = 7


def run(forecaster, cv):
    """Run backtesting_stats and return (metric, preds) or the exception."""
    try:
        metric, preds = backtesting_stats(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            n_jobs=1,
            suppress_warnings=True,
            show_progress=False,
        )
        return metric, preds, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e).splitlines()[0]}"


# ==============================================================================
# (a) Several estimators with overlapping folds (fold_stride < steps)
# ==============================================================================
print("\n" + "=" * 78)
print("(a) ForecasterStats([Arima(1,1,1), Ets(AAN)]), steps=7, refit=True")
print("=" * 78)


def make_multi():
    return ForecasterStats(estimator=[Arima(order=(1, 1, 1)), Ets(model="AAN")])


results_a = {}
for stride in [7, 10, 2, 1]:
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=True,
                        fold_stride=stride)
    t0 = time.perf_counter()
    metric, preds, err = run(make_multi(), cv)
    elapsed = time.perf_counter() - t0
    results_a[stride] = err
    if err:
        print(f"fold_stride={stride:>2}: ERROR -> {err}  ({elapsed:.1f} s)")
    else:
        print(f"fold_stride={stride:>2}: OK -> "
              f"{dict(zip(metric['estimator_id'], metric['mean_absolute_error'].round(4)))}"
              f"  ({elapsed:.1f} s)")

# Expected: the metric of each estimator equals the single estimator backtesting
print("\nExpected (each estimator backtested alone, fold_stride=2):")
expected_a = {}
for est in [Arima(order=(1, 1, 1)), Ets(model="AAN")]:
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=True, fold_stride=2)
    f = ForecasterStats(estimator=est)
    metric, _, err = run(f, cv)
    expected_a[f.estimator_ids[0]] = metric.iloc[0, 0]
    print(f"    {f.estimator_ids[0]:<18} MAE = {metric.iloc[0, 0]:.4f}")

# ==============================================================================
# (b) Sarimax with refit=False (and related schemes without refit in every fold)
# ==============================================================================
print("\n" + "=" * 78)
print("(b) ForecasterStats(Sarimax(order=(1,1,1), seasonal_order=(1,0,0,7))), steps=7")
print("=" * 78)


def make_sarimax():
    return ForecasterStats(
        estimator=Sarimax(order=(1, 1, 1), seasonal_order=(1, 0, 0, 7), maxiter=200)
    )


cases_b = [
    ("refit=False, fold_stride=7 (control)", dict(refit=False, fold_stride=7)),
    ("refit=False, fold_stride=1 (daily origin)", dict(refit=False, fold_stride=1)),
    ("refit=False, fold_stride=3", dict(refit=False, fold_stride=3)),
    ("refit=False, fold_stride=10", dict(refit=False, fold_stride=10)),
    ("refit=3 (intermittent), fold_stride=7", dict(refit=3, fold_stride=7)),
]
results_b = {}
for label, kw in cases_b:
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, verbose=False, **kw)
    t0 = time.perf_counter()
    metric, preds, err = run(make_sarimax(), cv)
    elapsed = time.perf_counter() - t0
    results_b[label] = err
    if err:
        print(f"{label:<42}: ERROR -> {err}  ({elapsed:.1f} s)")
    else:
        print(f"{label:<42}: OK -> MAE {metric.iloc[0, 0]:.4f}  ({elapsed:.1f} s)")

# The same failure reaches grid_search_stats (it calls _backtesting_stats)
cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=False, fold_stride=1,
                    verbose=False)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    results_gs = grid_search_stats(
        forecaster=make_sarimax(), y=y, cv=cv,
        param_grid={"order": [(1, 1, 1), (2, 1, 1)]},
        metric="mean_absolute_error", n_jobs=1, show_progress=False,
        suppress_warnings=True, return_best=False,
    )
gs_msgs = [str(x.message).splitlines()[0][:110] for x in w
           if issubclass(x.category, RuntimeWarning)]
gs_err = results_gs.empty
print(f"grid_search_stats, refit=False, fold_stride=1  : no exception, "
      f"{len(results_gs)} valid rows, RuntimeWarnings:")
for msg in gs_msgs:
    print(f"    - {msg}")


# ==============================================================================
# Candidate fix (patched copies of the two private functions)
# ==============================================================================
print("\n" + "=" * 78)
print("Candidate fix")
print("=" * 78)

src_fit = inspect.getsource(val._fit_predict_forecaster_stats)
old_lw = (
    "    if refit:\n"
    "        last_window_iloc_start = fold[1][1]  # Same as train_iloc_end\n"
    "        last_window_iloc_end   = fold[2][1]\n"
    "    else:\n"
    "        last_window_iloc_end   = fold[3][0]  # test_iloc_start\n"
    "        last_window_iloc_start = last_window_iloc_end - steps\n"
)
new_lw = (
    "    # FIX: the forecaster has seen the series up to the origin of the\n"
    "    # previous fold, whatever fold_stride or refit.\n"
    "    fold_position = [f[0] for f in folds].index(fold[0])\n"
    "    previous_fold = folds[fold_position - 1] if fold_position > 0 else fold\n"
    "    last_window_iloc_start = previous_fold[3][0]\n"
    "    last_window_iloc_end   = fold[3][0]  # test_iloc_start\n"
)
assert old_lw in src_fit, "last_window block not found, code changed"
src_fit = src_fit.replace(old_lw, new_lw)

src_bt = inspect.getsource(val._backtesting_stats)
old_dd = (
    "    if overlapping_folds:\n"
    "        backtest_predictions_for_metrics = (\n"
    "            backtest_predictions_for_metrics\n"
    "            .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]\n"
    "        )\n"
)
new_dd = (
    "    if overlapping_folds:\n"
    "        if forecaster.n_estimators == 1:\n"
    "            is_duplicated = backtest_predictions_for_metrics.index.duplicated(keep='last')\n"
    "        else:\n"
    "            is_duplicated = (\n"
    "                backtest_predictions_for_metrics\n"
    "                .set_index('estimator_id', append=True)\n"
    "                .index.duplicated(keep='last')\n"
    "            )\n"
    "        backtest_predictions_for_metrics = backtest_predictions_for_metrics.loc[~is_duplicated]\n"
)
assert old_dd in src_bt, "dedup block not found, code changed"
src_bt = src_bt.replace(old_dd, new_dd)

original_fit = val._fit_predict_forecaster_stats
original_bt = val._backtesting_stats
exec(src_fit, val.__dict__)
exec(src_bt, val.__dict__)

# (a) with the fix
print("\n(a) fold_stride=2 with the fix vs each estimator alone:")
cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=True, fold_stride=2)
metric, preds, err = run(make_multi(), cv)
assert err is None, err
for est_id, mae in zip(metric["estimator_id"], metric["mean_absolute_error"]):
    print(f"    {est_id:<18} fixed MAE = {mae:.4f} | alone = {expected_a[est_id]:.4f} | "
          f"diff = {abs(mae - expected_a[est_id]):.2e}")

# (b) with the fix vs an independent reference: in each fold the reference uses a
# copy of the forecaster fitted on the training set of the last fold that is fitted
# (fold 0 for refit=False) and predicts from the fold origin appending
# last_window = y[end of that training set : origin] in a single call.
print("\n(b) with the fix vs independent reference (single append per origin):")


def reference_preds(folds):
    out = []
    fitted = None
    for fold in folds:
        train_start, train_end = fold[1]
        origin = fold[3][0]
        if fitted is None or fold[5]:
            fitted = make_sarimax()
            fitted.fit(y=y.iloc[train_start:train_end], suppress_warnings=True)
            fit_end = train_end
        f = deepcopy(fitted)
        steps = min(STEPS, n - origin)
        lw = None if origin == fit_end else y.iloc[fit_end:origin]
        p = f.predict(steps=steps, last_window=lw, suppress_warnings=True)
        out.append(pd.DataFrame({"fold": fold[0], "pred": p}))
    return pd.concat(out)


for label, kw in cases_b:
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, verbose=False, **kw)
    metric, preds, err = run(make_sarimax(), cv)
    if err:
        print(f"{label:<42}: still ERROR -> {err}")
        continue
    folds = cv.split(X=y, as_pandas=False)
    ref = reference_preds(folds)
    max_diff = np.max(np.abs(preds["pred"].to_numpy() - ref["pred"].to_numpy()))
    print(f"{label:<42}: OK, MAE {metric.iloc[0, 0]:.4f}, "
          f"max |fixed - reference| = {max_diff:.2e} over {len(preds)} predictions")

cv = TimeSeriesFold(steps=STEPS, initial_train_size=n_train, refit=False, fold_stride=1,
                    verbose=False)
results_gs_fixed = grid_search_stats(
    forecaster=make_sarimax(), y=y, cv=cv,
    param_grid={"order": [(1, 1, 1), (2, 1, 1)]},
    metric="mean_absolute_error", n_jobs=1, show_progress=False,
    suppress_warnings=True, return_best=False,
)
print("\ngrid_search_stats, refit=False, fold_stride=1 with the fix:")
print(results_gs_fixed[["params", "mean_absolute_error"]].to_string(index=False))

val._fit_predict_forecaster_stats = original_fit
val._backtesting_stats = original_bt

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 78)
print("Summary (current code)")
print("=" * 78)
for stride, err in results_a.items():
    print(f"(a) multi-estimator fold_stride={stride:>2}: "
          f"{'FAILS' if err else 'works'} (expected: works)")
for label, err in results_b.items():
    print(f"(b) Sarimax {label:<42}: {'FAILS' if err else 'works'} (expected: works)")
print(f"(b) grid_search_stats Sarimax refit=False, fold_stride=1: "
      f"{'FAILS (all candidates skipped, empty result)' if gs_err else 'works'} "
      f"(expected: works)")
