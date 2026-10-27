"""
BUG-08: `freeze_params=True` (default) does not freeze anything in
`backtesting_stats` (nor in `grid_search_stats` / `random_search_stats`, which
force `freeze_params=True`).

`_backtesting_stats` freezes the selected parameters on `forecaster.estimators_`
(the fitted clones), but every refit fold calls `forecaster.fit`, which rebuilds
`estimators_` from the unfitted templates `forecaster.estimators`. The automatic
search (auto_arima / auto_ets) is therefore repeated in every fold.

The script counts the calls to `auto_arima` / `auto_ets`, measures the runtime
and compares the metric with `freeze_params=True` and `False`. Then it applies
the proposed fix (propagate `best_params_` to the templates) through a wrapper
and runs again.

Self-contained (synthetic daily series with weekly seasonality).
Run: python dev/roadmap_0.27.0/repro/BUG-08.py
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

import skforecast
import skforecast.model_selection._validation as validation_module
import skforecast.stats._arima as arima_module
import skforecast.stats._ets as ets_module
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_stats,
    grid_search_stats,
)
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima, Ets

warnings.filterwarnings("ignore")

# Counting wrappers around the automatic selection routines
CALLS = {"auto_arima": 0, "auto_ets": 0}
_orig_auto_arima = arima_module.auto_arima
_orig_auto_ets = ets_module.auto_ets


def _counting_auto_arima(*args, **kwargs):
    CALLS["auto_arima"] += 1
    return _orig_auto_arima(*args, **kwargs)


def _counting_auto_ets(*args, **kwargs):
    CALLS["auto_ets"] += 1
    return _orig_auto_ets(*args, **kwargs)


arima_module.auto_arima = _counting_auto_arima
ets_module.auto_ets = _counting_auto_ets


def make_series(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = (
        50
        + 0.02 * t
        + 5 * np.sin(2 * np.pi * t / 7)
        + np.cumsum(rng.normal(0, 0.3, n))
        + rng.normal(0, 1.5, n)
    )
    return pd.Series(y, index=idx, name="y")


y = make_series()
cv = TimeSeriesFold(steps=15, initial_train_size=500, refit=True, verbose=False)
n_folds = len(cv.split(X=y, as_pandas=False))


def make_estimator(kind: str):
    if kind == "Arima":
        return Arima(order=None, seasonal_order=None, m=7)
    return Ets(m=7, model="ZZZ")


def run(kind: str, freeze_params: bool) -> dict:
    CALLS["auto_arima"] = 0
    CALLS["auto_ets"] = 0
    forecaster = ForecasterStats(estimator=make_estimator(kind))
    start = time.perf_counter()
    metric, preds = backtesting_stats(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_absolute_error",
        freeze_params=freeze_params,
        show_progress=False,
    )
    elapsed = time.perf_counter() - start
    calls = CALLS["auto_arima"] if kind == "Arima" else CALLS["auto_ets"]
    return {
        "calls": calls,
        "seconds": elapsed,
        "mae": metric.loc[0, "mean_absolute_error"],
        "preds": preds["pred"].to_numpy(),
        "params_col": (
            preds["estimator_params"].unique().tolist()
            if "estimator_params" in preds
            else None
        ),
    }


# Proposed fix, simulated with a wrapper around the per-fold function. The real
# fix goes in `_backtesting_stats`, right after freezing `estimators_` and before
# `Parallel`, so that it also works with n_jobs > 1.
_orig_fit_predict = validation_module._fit_predict_forecaster_stats


def _fit_predict_with_fix(fold, forecaster, freeze_params, refit, **kwargs):
    if freeze_params and refit:
        for est, est_ in zip(forecaster.estimators, forecaster.estimators_):
            if getattr(est_, "best_params_", None):
                est.set_params(**est_.best_params_)
    return _orig_fit_predict(
        fold=fold,
        forecaster=forecaster,
        freeze_params=freeze_params,
        refit=refit,
        **kwargs,
    )


print("=" * 78)
print(
    f"Python {sys.version.split()[0]} | skforecast {skforecast.__version__} | "
    f"numpy {np.__version__} | pandas {pd.__version__}"
)
print(f"Series: {len(y)} daily points, m=7, steps=15, refit=True, folds={n_folds}")
print("Expected with freeze_params=True: 1 automatic search (initial fit only)")
print("Expected with freeze_params=False: 1 + n_folds - 1 = n_folds searches")
print("=" * 78)

# Warm-up (numba compilation) so that timings are comparable
for kind in ("Arima", "Ets"):
    ForecasterStats(estimator=make_estimator(kind)).fit(y=y.iloc[:300])

results = {}
for kind in ("Arima", "Ets"):
    r_true = run(kind, freeze_params=True)
    r_false = run(kind, freeze_params=False)
    validation_module._fit_predict_forecaster_stats = _fit_predict_with_fix
    try:
        r_fix = run(kind, freeze_params=True)
    finally:
        validation_module._fit_predict_forecaster_stats = _orig_fit_predict
    results[kind] = (r_true, r_false, r_fix)

    same_preds = np.allclose(r_true["preds"], r_false["preds"])
    print(f"\n[{kind} automatic] calls to the automatic search / time / MAE")
    print(
        f"  freeze_params=True  (current) : calls={r_true['calls']:>3}  "
        f"time={r_true['seconds']:6.2f} s  MAE={r_true['mae']:.6f}"
    )
    print(
        f"  freeze_params=False (current) : calls={r_false['calls']:>3}  "
        f"time={r_false['seconds']:6.2f} s  MAE={r_false['mae']:.6f}"
    )
    print(
        f"  freeze_params=True  (fixed)   : calls={r_fix['calls']:>3}  "
        f"time={r_fix['seconds']:6.2f} s  MAE={r_fix['mae']:.6f}"
    )
    print(f"  Predictions True vs False identical (current): {same_preds}")
    print(f"  Models selected per fold with False: {r_false['params_col']}")
    first_fold_equal = np.allclose(r_fix["preds"][:15], r_true["preds"][:15])
    print(f"  First fold identical, fixed vs current: {first_fold_equal}")
    print(
        f"  OBSERVED: freeze_params=True runs {r_true['calls']} searches "
        f"(expected 1)."
    )
    print(
        f"  Speed-up of the fix vs current True: "
        f"{r_true['seconds'] / r_fix['seconds']:.1f}x"
    )

# grid_search_stats always calls backtesting_stats with freeze_params=True
CALLS["auto_arima"] = 0
start = time.perf_counter()
grid_search_stats(
    forecaster=ForecasterStats(estimator=make_estimator("Arima")),
    y=y,
    cv=cv,
    param_grid={"max_p": [2, 3]},
    metric="mean_absolute_error",
    return_best=False,
    show_progress=False,
)
elapsed = time.perf_counter() - start
print("\n[grid_search_stats] Arima automatic, 2 candidates, freeze_params forced True")
print(
    f"  OBSERVED: auto_arima calls={CALLS['auto_arima']} ({elapsed:.1f} s); "
    f"EXPECTED: 2 (one per candidate)"
)

print("\n" + "=" * 78)
bug = all(results[k][0]["calls"] == n_folds for k in results)
print(
    "BUG REPRODUCED: freeze_params=True repeats the automatic search in every "
    "fold" if bug else "Bug NOT reproduced"
)
