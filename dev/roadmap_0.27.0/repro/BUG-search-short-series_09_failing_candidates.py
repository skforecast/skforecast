"""
BUG-search-short-series, PR 2b: candidates that fail in the hyperparameter
searches and other pending issues of the searches (section 12.5 of
BUG-search-short-series-transparency.md).

One scenario per point of 12.5:

1. One-step-ahead: a candidate without data to train, or without any level of
   `levels` with points in the test set, aborts the whole search.
2. Bayesian: a trial whose evaluation fails aborts the study. A configuration
   error (`search_space` keys) must keep aborting.
3. `ValueError` of `initial_train_size`: text with a date and the limit
   `initial_train_size == window_size`.
4. Refit of the best candidate with a series of `window_size` observations or
   fewer: the search fails at the end.
5. Single series, `OneStepAheadFold`, `dropna_from_series=True` and NaN in `y`:
   `n_evaluated` changes between candidates, without a warning.

Each scenario prints the warnings seen with the default filters (category and
first line), the error raised, if any, and a short summary of the results.
"""
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold,
    backtesting_forecaster, backtesting_forecaster_multiseries,
    grid_search_forecaster, random_search_forecaster, bayesian_search_forecaster,
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


Y = pd.Series(ar1(96), index=idx)
SERIES = {
    "long_1": pd.Series(ar1(96), index=idx),
    "long_2": pd.Series(ar1(96), index=idx),
    "short": pd.Series(ar1(40), index=idx[-40:]),     # 13 obs in the initial train set
}
# `tiny` has 6 values: 5 in the training set of initial_train_size=45, 1 in the test.
idx_60 = pd.RangeIndex(60)
SERIES_TINY = {
    "l1": pd.Series(10 + np.sin(np.arange(60) / 3), index=idx_60),
    "l2": pd.Series(20 + 2 * np.cos(np.arange(60) / 2), index=idx_60),
    "tiny": pd.Series(5 + np.arange(6) % 3, index=idx_60[40:46], dtype=float),
}


def run(name, fn, summary=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        try:
            out = fn()
            err = ""
        except Exception as e:
            out = None
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
    print(f"\n### {name}")
    seen = {}
    for w in caught:
        if issubclass(w.category, DeprecationWarning) or "optimizer 'adam'" in str(w.message):
            continue
        key = (w.category.__name__, str(w.message).splitlines()[0][:150])
        seen[key] = seen.get(key, 0) + 1
    for (category, first), n in seen.items():
        print(f"  warning {category}{f' x{n}' if n > 1 else ''}: {first}")
    if err:
        print(f"  ERROR {err}")
    elif summary is not None:
        print(f"  {summary(out)}")


def multi(lags=3):
    return ForecasterRecursiveMultiSeries(Ridge(), lags=lags, encoding="ordinal")


def rows(results):
    results = results[0] if isinstance(results, tuple) else results
    cols = [c for c in ["lags", "n_evaluated"] if c in results]
    return f"{len(results)} rows: " + "; ".join(
        f"{len(r['lags']) if r['lags'] is not None else None} lags, n={r['n_evaluated']}"
        for _, r in results[cols].iterrows()
    )


cv_osa = OneStepAheadFold(initial_train_size="2021-07-01")      # 67 obs in train
cv_tsf = TimeSeriesFold(steps=12, initial_train_size="2021-07-01", refit=False)

print("=" * 80)
print("1. One-step-ahead: candidate without data to train or to evaluate")
print("=" * 80)

run("grid, one series, lags_grid [3, 80] (80 > initial_train_size)", lambda: grid_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, cv_osa, {"alpha": [0.1, 1.0]},
    "mean_absolute_error", lags_grid=[3, 80], return_best=False, show_progress=False),
    rows)
run("random, one series, lags_grid [3, 80]", lambda: random_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, cv_osa, {"alpha": [0.1, 1.0]},
    "mean_absolute_error", lags_grid=[3, 80], n_iter=2, return_best=False,
    show_progress=False), rows)
run("grid, multi, lags_grid [3, 80] (no series to train)", lambda: grid_search_forecaster_multiseries(
    multi(), SERIES, cv_osa, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
    lags_grid=[3, 80], return_best=False, show_progress=False), rows)
run("random, multi, lags_grid [3, 80]", lambda: random_search_forecaster_multiseries(
    multi(), SERIES, cv_osa, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
    lags_grid=[3, 80], n_iter=2, return_best=False, show_progress=False), rows)
run("grid, multi, levels=['short'], lags_grid [3, 24] (short not trained with 24)",
    lambda: grid_search_forecaster_multiseries(
        multi(), SERIES, cv_osa, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        levels=["short"], lags_grid=[3, 24], return_best=False, show_progress=False),
    rows)
run("bayesian, multi, OSA, lags in [3, 80]", lambda: bayesian_search_forecaster_multiseries(
    multi(), SERIES, cv_osa,
    lambda t: {"lags": t.suggest_categorical("lags", [3, 80])},
    "mean_absolute_error", n_trials=6, return_best=False, show_progress=False), rows)
run("bayesian, one series, OSA, lags in [3, 80]", lambda: bayesian_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, cv_osa,
    lambda t: {"lags": t.suggest_categorical("lags", [3, 80])},
    "mean_absolute_error", n_trials=6, return_best=False, show_progress=False), rows)

print()
print("=" * 80)
print("2. Bayesian: a trial that fails, and a configuration error")
print("=" * 80)

run("bayesian, one series, TSF, lags in [3, 80]", lambda: bayesian_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, cv_tsf,
    lambda t: {"lags": t.suggest_categorical("lags", [3, 80])},
    "mean_absolute_error", n_trials=6, return_best=False, show_progress=False), rows)
run("bayesian, multi, TSF, lags in [3, 80]", lambda: bayesian_search_forecaster_multiseries(
    multi(), SERIES, cv_tsf,
    lambda t: {"lags": t.suggest_categorical("lags", [3, 80])},
    "mean_absolute_error", n_trials=6, return_best=False, show_progress=False), rows)
run("bayesian, one series, search_space keys mismatch (config error)",
    lambda: bayesian_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), Y, cv_tsf,
        lambda t: {"alpha_x": t.suggest_float("alpha", 0.1, 1)},
        "mean_absolute_error", n_trials=3, return_best=False, show_progress=False), rows)
run("bayesian, one series, search_space raises (user bug)",
    lambda: bayesian_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), Y, cv_tsf,
        lambda t: {"alpha": t.suggest_float("alpha", 0.1, 1) / 0},
        "mean_absolute_error", n_trials=3, return_best=False, show_progress=False), rows)
run("bayesian, one series, invalid estimator param (alpha='x')",
    lambda: bayesian_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), Y, cv_tsf,
        lambda t: {"alpha": t.suggest_categorical("alpha", ["x", 1.0])},
        "mean_absolute_error", n_trials=4, return_best=False, show_progress=False), rows)
run("bayesian, one series, every trial fails, return_best=True",
    lambda: bayesian_search_forecaster(
        ForecasterRecursive(Ridge(), lags=3), Y, cv_tsf,
        lambda t: {"lags": t.suggest_categorical("lags", [80, 90])},
        "mean_absolute_error", n_trials=3, return_best=True, show_progress=False), rows)

print()
print("=" * 80)
print("3. ValueError of initial_train_size")
print("=" * 80)

run("backtesting, lags=80, initial_train_size date (67 obs)", lambda: backtesting_forecaster(
    ForecasterRecursive(Ridge(), lags=80), Y, cv_tsf, "mean_absolute_error",
    show_progress=False))
run("backtesting, lags=100 > len(y)=96, initial_train_size=50", lambda: backtesting_forecaster(
    ForecasterRecursive(Ridge(), lags=100), Y,
    TimeSeriesFold(steps=12, initial_train_size=50), "mean_absolute_error",
    show_progress=False))
run("backtesting, initial_train_size=96 (= len(y))", lambda: backtesting_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y,
    TimeSeriesFold(steps=12, initial_train_size=96), "mean_absolute_error",
    show_progress=False))
run("backtesting, initial_train_size == window_size (3)", lambda: backtesting_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y,
    TimeSeriesFold(steps=12, initial_train_size=3), "mean_absolute_error",
    show_progress=False), lambda out: f"metric: {out[0].iloc[0].to_dict()}")
run("backtesting, initial_train_size == window_size + 1 (4)", lambda: backtesting_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y,
    TimeSeriesFold(steps=12, initial_train_size=4), "mean_absolute_error",
    show_progress=False), lambda out: f"metric: {out[0].iloc[0].to_dict()}")
run("backtesting multi, initial_train_size == window_size (3)",
    lambda: backtesting_forecaster_multiseries(
        multi(), SERIES, TimeSeriesFold(steps=12, initial_train_size=3),
        "mean_absolute_error", show_progress=False),
    lambda out: f"metric rows: {len(out[0])}")
run("grid OSA, initial_train_size == window_size (3)", lambda: grid_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, OneStepAheadFold(initial_train_size=3),
    {"alpha": [1.0]}, "mean_absolute_error", return_best=False, show_progress=False), rows)
run("grid OSA, initial_train_size=2 < window_size (3)", lambda: grid_search_forecaster(
    ForecasterRecursive(Ridge(), lags=3), Y, OneStepAheadFold(initial_train_size=2),
    {"alpha": [1.0]}, "mean_absolute_error", return_best=False, show_progress=False), rows)

print()
print("=" * 80)
print("4. Refit of the best candidate with a series too short for its window")
print("=" * 80)

cv_tiny = TimeSeriesFold(initial_train_size=45, steps=5, refit=False)
run("grid multi, lags_grid [8], 'tiny' has 6 values", lambda: grid_search_forecaster_multiseries(
    multi(), SERIES_TINY, cv_tiny, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
    lags_grid=[8], return_best=True, show_progress=False), rows)
run("grid multi, lags_grid [2, 8], return_best", lambda: grid_search_forecaster_multiseries(
    multi(), SERIES_TINY, cv_tiny, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
    lags_grid=[2, 8], return_best=True, show_progress=False), rows)
run("random multi, lags_grid [8]", lambda: random_search_forecaster_multiseries(
    multi(), SERIES_TINY, cv_tiny, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
    lags_grid=[8], n_iter=2, return_best=True, show_progress=False), rows)
run("bayesian multi, lags=8", lambda: bayesian_search_forecaster_multiseries(
    multi(8), SERIES_TINY, cv_tiny,
    lambda t: {"alpha": t.suggest_float("alpha", 0.1, 1)},
    "mean_absolute_error", n_trials=3, return_best=True, show_progress=False), rows)
run("grid multi OSA, lags_grid [8]", lambda: grid_search_forecaster_multiseries(
    multi(), SERIES_TINY, OneStepAheadFold(initial_train_size=45), {"alpha": [0.1, 1.0]},
    "mean_absolute_error", lags_grid=[8], return_best=True, show_progress=False), rows)

print()
print("=" * 80)
print("5. One series, OSA, dropna_from_series=True, NaN in y")
print("=" * 80)

idx_200 = pd.date_range("2020-01-01", periods=200, freq="D")
Y_NAN = pd.Series(
    50 + 10 * np.sin(2 * np.pi * np.arange(200) / 7) + rng.normal(0, 2, 200),
    index=idx_200
)
Y_NAN.iloc[[150, 165, 180]] = np.nan                    # 3 NaN in the test set
cv_nan = OneStepAheadFold(initial_train_size=143)       # 57 points in the test set
for search in ["grid", "random", "bayesian"]:
    def fn(search=search):
        f = ForecasterRecursive(Ridge(), lags=3, dropna_from_series=True)
        if search == "grid":
            return grid_search_forecaster(
                f, Y_NAN, cv_nan, {"alpha": [1.0]}, "mean_absolute_error",
                lags_grid=[3, 14], return_best=False, show_progress=False)
        if search == "random":
            return random_search_forecaster(
                f, Y_NAN, cv_nan, {"alpha": [1.0]}, "mean_absolute_error",
                lags_grid=[3, 14], n_iter=1, return_best=False, show_progress=False)
        return bayesian_search_forecaster(
            f, Y_NAN, cv_nan, lambda t: {"lags": t.suggest_categorical("lags", [3, 14])},
            "mean_absolute_error", n_trials=6, return_best=False, show_progress=False)
    run(f"{search}, lags [3, 14]", fn, rows)

print()
print("=" * 80)
print("3b. initial_train_size == window_size in the other forecasters (TSF, refit=False)")
print("=" * 80)

from sklearn.linear_model import LogisticRegression
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.recursive import (
    ForecasterEquivalentDate, ForecasterRecursiveClassifier, ForecasterStats
)
from skforecast.stats import Arima

Y_CLASS = pd.Series(np.where(Y.diff().fillna(0) > 0, "up", "down"), index=idx)
SERIES_WIDE = pd.DataFrame({"long_1": SERIES["long_1"], "long_2": SERIES["long_2"]})


def at_limit(make, data, multi_wide=False, offset=0):
    def fn():
        f = make()
        ws = f.window_size
        cv = TimeSeriesFold(steps=3, initial_train_size=ws + offset, refit=False)
        if multi_wide:
            return backtesting_forecaster_multiseries(
                f, data, cv, "mean_absolute_error", show_progress=False)
        if type(f).__name__ == "ForecasterStats":
            from skforecast.model_selection import backtesting_stats
            return backtesting_stats(f, data, cv, "mean_absolute_error", show_progress=False)
        metric = "accuracy_score" if isinstance(data.iloc[0], str) else "mean_absolute_error"
        return backtesting_forecaster(f, data, cv, metric, show_progress=False)
    return fn


CASES = {
    "ForecasterDirect(lags=3, steps=3)": (lambda: ForecasterDirect(Ridge(), lags=3, steps=3), Y, False),
    "ForecasterDirectMultiVariate(lags=3, steps=3)": (
        lambda: ForecasterDirectMultiVariate(Ridge(), level="long_1", lags=3, steps=3),
        SERIES_WIDE, True),
    "ForecasterEquivalentDate(offset=3)": (lambda: ForecasterEquivalentDate(offset=3), Y, False),
    "ForecasterRecursiveClassifier(lags=3)": (
        lambda: ForecasterRecursiveClassifier(LogisticRegression(), lags=3), Y_CLASS, False),
    "ForecasterStats(Arima(order=(1, 0, 0)))": (
        lambda: ForecasterStats(Arima(order=(1, 0, 0))), Y, False),
}
try:
    from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
    CASES["ForecasterRnn(lags=3, steps=3)"] = (
        lambda: ForecasterRnn(
            create_and_compile_model(SERIES_WIDE, lags=3, steps=3, levels=["long_1"],
                                     recurrent_units=4, dense_units=4),
            levels=["long_1"], lags=3, fit_kwargs={"epochs": 1, "verbose": 0}),
        SERIES_WIDE, True)
except ImportError as e:
    print(f"  ForecasterRnn not available: {e}")

for name, (make, data, wide) in CASES.items():
    for offset in [0, 1]:
        run(f"{name}, initial_train_size = window_size + {offset}",
            at_limit(make, data, wide, offset),
            lambda out: f"ok, metric: {out[0].iloc[0].to_dict()}")

print()
print("=" * 80)
print("4b. Refit with a short series: ForecasterDirectMultiVariate and ForecasterRnn")
print("=" * 80)

SERIES_TINY_WIDE = pd.DataFrame(SERIES_TINY)          # 'tiny' NaN except 6 values
run("grid DirectMultiVariate, level='l1', lags_grid [8], wide with 'tiny'",
    lambda: grid_search_forecaster_multiseries(
        ForecasterDirectMultiVariate(Ridge(), level="l1", lags=3, steps=5),
        SERIES_TINY_WIDE, cv_tiny, {"alpha": [0.1, 1.0]}, "mean_absolute_error",
        lags_grid=[8], return_best=True, show_progress=False), rows)
run("grid DirectMultiVariate, level='l1', lags_grid [8], wide without 'tiny'",
    lambda: grid_search_forecaster_multiseries(
        ForecasterDirectMultiVariate(Ridge(), level="l1", lags=3, steps=5),
        SERIES_TINY_WIDE[["l1", "l2"]], cv_tiny, {"alpha": [0.1, 1.0]},
        "mean_absolute_error", lags_grid=[8], return_best=True, show_progress=False), rows)
if "ForecasterRnn(lags=3, steps=3)" in CASES:
    def rnn(series):
        return ForecasterRnn(
            create_and_compile_model(series, lags=8, steps=5, levels=["l1"],
                                     recurrent_units=4, dense_units=4),
            levels=["l1"], lags=8, fit_kwargs={"epochs": 1, "verbose": 0})
    for name, data in [("with", SERIES_TINY_WIDE), ("without", SERIES_TINY_WIDE[["l1", "l2"]])]:
        run(f"grid ForecasterRnn, lags 8, wide {name} 'tiny'",
            lambda data=data: grid_search_forecaster_multiseries(
                rnn(data), data, cv_tiny, {"dense_units": [4]}, "mean_absolute_error",
                return_best=True, show_progress=False), rows)
