"""
BUG-backtesting-nan-y-true: single series backtesting with missing values in
`y` inside the test set (roadmap 0.27.0, section 4).

`backtesting_forecaster` does not drop the dates with `y_true` NaN before the
metrics, as `backtesting_forecaster_multiseries` does, so the scikit-learn
metric raises `ValueError: Input contains NaN.`

Scenarios:

1. `backtesting_forecaster`: 200 days, NaN at positions 150, 170 and 185,
   `TimeSeriesFold(steps=10, initial_train_size=140)`. Ridge with
   `dropna_from_series=True` (lags 3 and 14) and LightGBM with
   `dropna_from_series=False`, with `refit=False` and `refit=True`.
2. Single series searches (grid, random and bayesian) with `TimeSeriesFold`.
3. `backtesting_stats` and `backtesting_foundation` with a single series.
4. Scaled metrics (MASE, RMSSE) with NaN in `y_train`.
5. The same case with `backtesting_forecaster_multiseries` (one and two series).

Simulations (the package is not modified, only patched at run time):

- S1, the fix of the roadmap: a metric that leaves out the dates with `y_true`
  NaN. `_check_nan_predictions_backtesting` runs before the metrics, so it
  shows what the fix gives (in particular, interaction with #1432).
- S2, aligning a single series with multi-series: S1 plus
  `_check_nan_predictions_backtesting` replaced by a check that does not
  raise (the NaN predictions are left out). It shows whether the warning of
  `n_evaluated` of the searches (#1456) fires and with which text.

Each scenario prints the warnings seen with the default filters (category and
first line), the error raised, if any, and the metrics and `n_evaluated`.
"""
import warnings
from contextlib import contextmanager
from unittest import mock

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import skforecast.model_selection._validation as validation_module
from skforecast.metrics import (
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
)
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    backtesting_stats,
    bayesian_search_forecaster,
    grid_search_forecaster,
    random_search_forecaster,
)
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveMultiSeries,
    ForecasterStats,
)
from skforecast.stats import Arima, Ets, Sarimax

optuna.logging.set_verbosity(optuna.logging.WARNING)
rng = np.random.default_rng(0)
N = 200
NAN_POSITIONS = [150, 170, 185]
idx = pd.date_range("2022-01-01", periods=N, freq="D")


def ar1(n):
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + rng.normal()
    return 100 + y + 3 * np.sin(np.arange(n) * 2 * np.pi / 7)


Y = pd.Series(ar1(N), index=idx, name="y")
Y_NAN = Y.copy()
Y_NAN.iloc[NAN_POSITIONS] = np.nan
Y_2 = pd.Series(ar1(N), index=idx, name="y_2")
# NaN only in the training set (positions 60 and 100), for the scaled metrics.
Y_NAN_TRAIN = Y.copy()
Y_NAN_TRAIN.iloc[[60, 100]] = np.nan
# NaN in the training and in the test set.
Y_NAN_BOTH = Y_NAN.copy()
Y_NAN_BOTH.iloc[[60, 100]] = np.nan

N_OBSERVED_TEST = int(Y_NAN.iloc[140:].notna().sum())


def cv(refit=False):
    return TimeSeriesFold(steps=10, initial_train_size=140, refit=refit)


def ridge(lags=3):
    return ForecasterRecursive(Ridge(), lags=lags, dropna_from_series=True)


def lgbm(lags=3):
    return ForecasterRecursive(
        LGBMRegressor(n_estimators=50, verbose=-1, random_state=0),
        lags=lags,
        dropna_from_series=False,
    )


def mae_ignore_nan(y_true, y_pred, y_train=None):
    """Simulation S1: MAE without the dates with `y_true` or `y_pred` NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return mean_absolute_error(y_true[mask], y_pred[mask])


mae_ignore_nan.__name__ = "mean_absolute_error"


def _check_nan_predictions_no_raise(*args, **kwargs):
    """Simulation S2: leave the NaN predictions out instead of raising."""
    return None


@contextmanager
def align_with_multiseries():
    with mock.patch.object(
        validation_module,
        "_check_nan_predictions_backtesting",
        _check_nan_predictions_no_raise,
    ):
        yield


def run(name, fn, summary=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        try:
            out = fn()
            err = ""
        except Exception as e:
            out = None
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"
    print(f"\n### {name}")
    seen = {}
    for w in caught:
        if issubclass(w.category, DeprecationWarning):
            continue
        key = (w.category.__name__, str(w.message).splitlines()[0][:220])
        seen[key] = seen.get(key, 0) + 1
    for (category, first), n in seen.items():
        print(f"  warning {category}{f' x{n}' if n > 1 else ''}: {first}")
    if err:
        print(f"  ERROR {err}")
    elif summary is not None:
        print(f"  {summary(out)}")


def summary_backtesting(out):
    metrics, predictions = out
    row = metrics.iloc[0].to_dict()
    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
    n_nan_pred = int(predictions["pred"].isna().sum())
    n_nan_true = int(predictions["y_true"].isna().sum())
    return (
        f"metrics: {row} | predictions: {len(predictions)} rows, "
        f"y_true NaN: {n_nan_true}, pred NaN: {n_nan_pred}"
    )


def summary_multiseries(out):
    metrics, predictions = out
    records = metrics.round(4).to_dict(orient="records")
    return f"metrics: {records} | predictions: {len(predictions)} rows"


def summary_search(out):
    if out is None or len(out) == 0:
        return "results: EMPTY"
    metric_col = [
        c for c in out.columns
        if c.startswith(("mean_absolute", "root_mean")) and "scaled" in c
        or c == "mean_absolute_error"
    ][:1]
    cols = ["lags", "n_evaluated"] + metric_col
    rows = out[cols].copy()
    rows["lags"] = rows["lags"].apply(lambda x: int(len(x)))
    return f"{len(out)} rows: {rows.round(4).to_dict(orient='records')}"


print(f"y: {N} days, NaN at positions {NAN_POSITIONS}; test from position 140, "
      f"{N - 140} dates, {N_OBSERVED_TEST} with an observed value")

# =============================================================================
# 1. backtesting_forecaster
# =============================================================================
print("\n" + "=" * 79 + "\n1. backtesting_forecaster\n" + "=" * 79)
for refit in (False, True):
    for label, make in [
        ("Ridge lags=3, dropna_from_series=True", lambda: ridge(3)),
        ("Ridge lags=14, dropna_from_series=True", lambda: ridge(14)),
        ("LightGBM lags=3, dropna_from_series=False", lambda: lgbm(3)),
        ("LightGBM lags=14, dropna_from_series=False", lambda: lgbm(14)),
    ]:
        run(
            f"1 | {label} | refit={refit} | metric='mean_absolute_error'",
            lambda: backtesting_forecaster(
                make(), y=Y_NAN, cv=cv(refit), metric="mean_absolute_error",
                show_progress=False,
            ),
            summary_backtesting,
        )
        run(
            f"1 S1 | {label} | refit={refit} | metric ignoring y_true NaN",
            lambda: backtesting_forecaster(
                make(), y=Y_NAN, cv=cv(refit), metric=mae_ignore_nan,
                show_progress=False,
            ),
            summary_backtesting,
        )

for refit in (False, True):
    def fn_s2(refit=refit):
        with align_with_multiseries():
            return backtesting_forecaster(
                ridge(14), y=Y_NAN, cv=cv(refit), metric=mae_ignore_nan,
                show_progress=False,
            )
    run(
        f"1 S2 | Ridge lags=14, dropna_from_series=True | refit={refit} | "
        f"NaN predictions left out (as multi-series)",
        fn_s2,
        summary_backtesting,
    )

# Interval metrics: already leave the NaN out (`_calculate_interval_metrics`).
run(
    "1 S1 | LightGBM lags=3 | interval=[0.1, 0.9] | metric ignoring y_true NaN",
    lambda: backtesting_forecaster(
        lgbm(3), y=Y_NAN, cv=cv(), metric=mae_ignore_nan, interval=[0.1, 0.9],
        interval_method="bootstrapping", n_boot=50, show_progress=False,
    ),
    summary_backtesting,
)

# =============================================================================
# 2. Single series searches with TimeSeriesFold
# =============================================================================
print("\n" + "=" * 79 + "\n2. Single series searches, TimeSeriesFold\n" + "=" * 79)
LAGS_GRID = [3, 14]
PARAM_GRID = {"alpha": [0.1, 1.0]}


def search_space(trial):
    return {
        "lags": trial.suggest_categorical("lags", LAGS_GRID),
        "alpha": trial.suggest_float("alpha", 0.1, 1.0),
    }


def search_space_lgbm(trial):
    return {
        "lags": trial.suggest_categorical("lags", LAGS_GRID),
        "n_estimators": trial.suggest_int("n_estimators", 20, 50),
    }


searches = [
    ("grid", lambda f, m: grid_search_forecaster(
        f, y=Y_NAN, cv=cv(), param_grid=PARAM_GRID if "Ridge" in str(f.estimator)
        else {"n_estimators": [20, 50]}, lags_grid=LAGS_GRID, metric=m,
        return_best=False, show_progress=False)),
    ("random", lambda f, m: random_search_forecaster(
        f, y=Y_NAN, cv=cv(), param_distributions=PARAM_GRID
        if "Ridge" in str(f.estimator) else {"n_estimators": [20, 50]},
        lags_grid=LAGS_GRID, n_iter=2, metric=m, return_best=False,
        random_state=0, show_progress=False)),
    ("bayesian", lambda f, m: bayesian_search_forecaster(
        f, y=Y_NAN, cv=cv(), search_space=search_space
        if "Ridge" in str(f.estimator) else search_space_lgbm,
        n_trials=6, metric=m, return_best=False, random_state=0,
        show_progress=False)[0]),
]
for search_name, search in searches:
    for label, make in [("Ridge", lambda: ridge(3)), ("LightGBM", lambda: lgbm(3))]:
        run(
            f"2 | {search_name} | {label} | metric='mean_absolute_error'",
            lambda: search(make(), "mean_absolute_error"),
            summary_search,
        )
        run(
            f"2 S1 | {search_name} | {label} | metric ignoring y_true NaN",
            lambda: search(make(), mae_ignore_nan),
            summary_search,
        )


def run_s2_search(search_name, search):
    def fn():
        with align_with_multiseries():
            return search(ridge(3), mae_ignore_nan)
    run(
        f"2 S2 | {search_name} | Ridge | NaN predictions left out (as multi-series)",
        fn,
        summary_search,
    )


for search_name, search in searches:
    run_s2_search(search_name, search)

# =============================================================================
# 3. backtesting_stats and backtesting_foundation
# =============================================================================
print("\n" + "=" * 79 + "\n3. backtesting_stats and backtesting_foundation\n" + "=" * 79)
for refit in (False, True):
    for label, make in [
        ("Arima(1, 0, 0)", lambda: ForecasterStats(Arima(order=(1, 0, 0)))),
        ("Ets('ANN')", lambda: ForecasterStats(Ets(model="ANN"))),
        ("Sarimax(1, 0, 0)", lambda: ForecasterStats(Sarimax(order=(1, 0, 0)))),
    ]:
        run(
            f"3 | backtesting_stats | {label} | refit={refit} | NaN only in the test",
            lambda: backtesting_stats(
                make(), y=Y_NAN, cv=cv(refit), metric="mean_absolute_error",
                show_progress=False,
            ),
            summary_backtesting,
        )
        run(
            f"3 S1 | backtesting_stats | {label} | refit={refit} | metric ignoring NaN",
            lambda: backtesting_stats(
                make(), y=Y_NAN, cv=cv(refit), metric=mae_ignore_nan,
                show_progress=False,
            ),
            summary_backtesting,
        )

try:
    from skforecast.foundation import FoundationModel, ForecasterFoundation
    from skforecast.model_selection import backtesting_foundation

    def foundation():
        model = FoundationModel(
            model_id="autogluon/chronos-2-small", context_length=256, device_map="cpu"
        )
        return ForecasterFoundation(estimator=model)

    for metric_label, metric in [
        ("metric='mean_absolute_error'", "mean_absolute_error"),
        ("S1 metric ignoring NaN", mae_ignore_nan),
    ]:
        run(
            f"3 | backtesting_foundation | chronos-2-small | {metric_label}",
            lambda: backtesting_foundation(
                foundation(), series=Y_NAN, cv=cv(), metric=metric,
                show_progress=False,
            ),
            summary_backtesting,
        )
except ImportError as e:
    print(f"\n### 3 | backtesting_foundation: not available ({e})")

# =============================================================================
# 4. Scaled metrics with NaN in y_train
# =============================================================================
print("\n" + "=" * 79 + "\n4. Scaled metrics (MASE, RMSSE)\n" + "=" * 79)
SCALED = [
    "mean_absolute_error", mean_absolute_scaled_error, root_mean_squared_scaled_error
]
for y_label, y in [
    ("NaN only in y_train (60, 100)", Y_NAN_TRAIN),
    ("NaN in y_train and in the test", Y_NAN_BOTH),
    ("NaN only in the test", Y_NAN),
]:
    for refit in (False, True):
        run(
            f"4 | LightGBM lags=3 | {y_label} | refit={refit} | MAE, MASE, RMSSE",
            lambda: backtesting_forecaster(
                lgbm(3), y=y, cv=cv(refit), metric=SCALED, show_progress=False,
            ),
            summary_backtesting,
        )
run(
    "4 | LightGBM lags=3 | NaN only in the test | MASE, RMSSE (no MAE)",
    lambda: backtesting_forecaster(
        lgbm(3), y=Y_NAN, cv=cv(), metric=SCALED[1:], show_progress=False,
    ),
    summary_backtesting,
)
run(
    "4 | grid | LightGBM | NaN only in the test | MASE (ranking with NaN metrics)",
    lambda: grid_search_forecaster(
        lgbm(3), y=Y_NAN, cv=cv(), param_grid={"n_estimators": [20, 50]},
        lags_grid=LAGS_GRID, metric=mean_absolute_scaled_error,
        return_best=False, show_progress=False,
    ),
    summary_search,
)
run(
    "4 | backtesting_stats | Sarimax(1, 0, 0) | NaN only in the test | MASE, RMSSE",
    lambda: backtesting_stats(
        ForecasterStats(Sarimax(order=(1, 0, 0))), y=Y_NAN, cv=cv(),
        metric=SCALED[1:], show_progress=False,
    ),
    summary_backtesting,
)

# =============================================================================
# 5. backtesting_forecaster_multiseries, for comparison
# =============================================================================
print("\n" + "=" * 79 + "\n5. backtesting_forecaster_multiseries\n" + "=" * 79)


def multi(estimator, lags, dropna):
    return ForecasterRecursiveMultiSeries(
        estimator, lags=lags, encoding="ordinal", dropna_from_series=dropna
    )


for n_series, series in [
    (1, pd.DataFrame({"y": Y_NAN})),
    (2, pd.DataFrame({"y": Y_NAN, "y_2": Y_2})),
]:
    for refit in (False, True):
        for label, make in [
            ("Ridge lags=3, dropna=True", lambda: multi(Ridge(), 3, True)),
            ("Ridge lags=14, dropna=True", lambda: multi(Ridge(), 14, True)),
            ("LightGBM lags=3, dropna=False", lambda: multi(
                LGBMRegressor(n_estimators=50, verbose=-1, random_state=0), 3, False)),
        ]:
            run(
                f"5 | {n_series} series | {label} | refit={refit} | MAE, MASE",
                lambda: backtesting_forecaster_multiseries(
                    make(), series=series, cv=cv(refit),
                    metric=["mean_absolute_error", mean_absolute_scaled_error],
                    n_jobs=1, show_progress=False,
                ),
                summary_multiseries,
            )
