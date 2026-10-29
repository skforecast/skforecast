"""
BUG-search-short-series, experiment 6: which warnings and errors reach the
user in the hyperparameter searches, and how many times.

Every scenario runs inside `warnings.catch_warnings(record=True)` WITHOUT
changing the filters, so the count is what a user sees with the default
Python filters. "generated" repeats the run with `simplefilter("always")`
to count how many were raised in total.
"""
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold,
    grid_search_forecaster, bayesian_search_forecaster,
    grid_search_forecaster_multiseries, random_search_forecaster_multiseries,
    bayesian_search_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

optuna.logging.set_verbosity(optuna.logging.WARNING)
rng = np.random.default_rng(0)
idx = pd.date_range("2016-01-01", periods=96, freq="MS")


def ar1(n):
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + rng.normal()
    return 100 + y


SERIES_SHORT = {
    "long_1": pd.Series(ar1(96), index=idx),
    "long_2": pd.Series(ar1(96), index=idx),
    "short": pd.Series(ar1(40), index=idx[-40:]),     # 13 obs in the initial train set
}
EXOG_GAP = {k: pd.DataFrame({"x": rng.normal(size=96)}, index=idx) for k in SERIES_SHORT}
EXOG_GAP["long_1"].iloc[70, 0] = np.nan                # gap in exog inside the test set


def run(fn):
    out = {}
    for mode in ["seen", "generated"]:
        with warnings.catch_warnings(record=True) as caught:
            if mode == "generated":
                warnings.simplefilter("always")
            try:
                fn()
                err = ""
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:110]}"
        counts = pd.Series([w.category.__name__ for w in caught]).value_counts().to_dict()
        out[mode] = counts
        out["error"] = err
    return out


def fmt(d):
    return ", ".join(f"{k} x{v}" for k, v in d.items()) or "none"


def multiseries(estimator=None):
    return ForecasterRecursiveMultiSeries(estimator or Ridge(), lags=3, encoding="ordinal")


cv_tsf = TimeSeriesFold(steps=12, initial_train_size="2021-07-01", refit=False)
cv_osa = OneStepAheadFold(initial_train_size="2021-07-01")

SCENARIOS = {
    # 1. Short series left out by the larger window (#1415 warning)
    "grid multi, short series, lags [3, 24] (short window first)": lambda: grid_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_tsf, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        lags_grid=[3, 24], return_best=False, show_progress=False),
    "grid multi, short series, lags [24, 3] (long window first)": lambda: grid_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_tsf, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        lags_grid=[24, 3], return_best=False, show_progress=False),
    "grid multi OneStepAhead, short series, lags [3, 24]": lambda: grid_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_osa, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        lags_grid=[3, 24], return_best=False, show_progress=False),
    "random multi, short series, 6 iters": lambda: random_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_tsf, {"alpha": [0.01, 0.1, 1.0, 10.0]}, "mean_absolute_error",
        lags_grid=[3, 12, 24], n_iter=6, return_best=False, show_progress=False),
    "bayesian multi, short series, lags in search_space, 10 trials": lambda: bayesian_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_tsf,
        lambda t: {"lags": t.suggest_categorical("lags", [3, 12, 24]),
                   "alpha": t.suggest_float("alpha", 0.01, 10, log=True)},
        "mean_absolute_error", n_trials=10, return_best=False, show_progress=False),
    # 2. #1432: linear model with a gap in exog -> NaN predictions warning
    "grid multi, Ridge + exog gap (#1432), 3 lags x 2 alpha": lambda: grid_search_forecaster_multiseries(
        multiseries(), {k: v for k, v in SERIES_SHORT.items() if k != "short"}, cv_tsf,
        {"alpha": [0.1, 1.0]}, "mean_absolute_error", exog={k: v for k, v in EXOG_GAP.items() if k != "short"},
        lags_grid=[3, 6, 12], return_best=False, show_progress=False),
    "bayesian multi, Ridge + exog gap (#1432), 10 trials": lambda: bayesian_search_forecaster_multiseries(
        multiseries(), {k: v for k, v in SERIES_SHORT.items() if k != "short"}, cv_tsf,
        lambda t: {"lags": t.suggest_categorical("lags", [3, 6, 12]),
                   "alpha": t.suggest_float("alpha", 0.01, 10, log=True)},
        "mean_absolute_error", exog={k: v for k, v in EXOG_GAP.items() if k != "short"},
        n_trials=10, return_best=False, show_progress=False),
    # 3. Candidates that fail: lags longer than the training set of every series
    "grid multi, 2 of 3 lags fail (lags > series)": lambda: grid_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_tsf, {"alpha": [0.1, 1.0, 10.0]}, "mean_absolute_error",
        lags_grid=[3, 80, 90], return_best=False, show_progress=False),
    "grid multi OneStepAhead, 1 of 2 lags has nothing to train": lambda: grid_search_forecaster_multiseries(
        multiseries(), SERIES_SHORT, cv_osa, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        lags_grid=[3, 80], return_best=False, show_progress=False),
    # 4. LongTrainingWarning with many refits (single series)
    "grid single, refit=True with 60 folds, 3 lags x 2 alpha": lambda: grid_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), SERIES_SHORT["long_1"],
        TimeSeriesFold(steps=1, initial_train_size=36, refit=True),
        {"alpha": [0.1, 1.0]}, "mean_absolute_error", lags_grid=[3, 6, 12],
        return_best=False, show_progress=False),
    "bayesian single, refit=True with 60 folds, 6 trials": lambda: bayesian_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), SERIES_SHORT["long_1"],
        TimeSeriesFold(steps=1, initial_train_size=36, refit=True),
        lambda t: {"lags": t.suggest_categorical("lags", [3, 6, 12]),
                   "alpha": t.suggest_float("alpha", 0.01, 10, log=True)},
        "mean_absolute_error", n_trials=6, return_best=False, show_progress=False),
}

rows = []
for name, fn in SCENARIOS.items():
    r = run(fn)
    rows.append({"scenario": name, "seen by the user": fmt(r["seen"]),
                 "generated": fmt(r["generated"]), "error": r["error"]})
    print(f"\n{name}\n  seen     : {fmt(r['seen'])}\n  generated: {fmt(r['generated'])}"
          + (f"\n  ERROR    : {r['error']}" if r["error"] else ""), flush=True)
