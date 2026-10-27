"""
BUG-03: the hyperparameter searches decide the sort order and the Optuna
direction from the forecaster task (regression -> minimize, classification ->
maximize), never from the metric. With a "greater is better" callable such as
`r2_score`, the searches rank the worst model first and `return_best=True`
refits the user's forecaster with it.

Cases:
  A. grid_search_forecaster, metric=r2_score, return_best=True
  B. bayesian_search_forecaster, metric=r2_score (default direction)
  C. bayesian_search_forecaster, metric=r2_score and the documented escape
     hatch kwargs_create_study={'direction': 'maximize'}: Optuna optimizes in
     the right direction but the results table and return_best do not follow it
  D. grid_search_forecaster_multiseries, metric=r2_score
  E. ForecasterRecursiveClassifier + grid_search_forecaster, metric=zero_one_loss
     (symmetric case: a loss in a classification task is maximized)

Self-contained: uses the h2o demo dataset (downloaded) and falls back to a
synthetic monthly series if there is no network.

Run: python dev/roadmap_0.27.0/repro/BUG-03.py
"""

import warnings

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, zero_one_loss

import skforecast
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
    grid_search_forecaster,
    grid_search_forecaster_multiseries,
)
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_y() -> tuple[pd.Series, str]:
    try:
        from skforecast.datasets import load_demo_dataset

        return load_demo_dataset(verbose=False), "h2o (load_demo_dataset)"
    except Exception:
        rng = np.random.default_rng(0)
        idx = pd.date_range("1991-07-01", periods=204, freq="MS")
        t = np.arange(204)
        y = (
            0.4
            + 0.004 * t
            + 0.15 * np.sin(2 * np.pi * t / 12)
            + rng.normal(0, 0.05, 204)
        )
        return pd.Series(y, index=idx, name="y"), "synthetic"


def header(txt: str) -> None:
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


y, source = load_y()
print(
    f"skforecast {skforecast.__version__} | sklearn {sklearn.__version__} | "
    f"optuna {optuna.__version__} | pandas {pd.__version__} | data: {source} "
    f"(n={len(y)})"
)
cv = TimeSeriesFold(steps=12, initial_train_size=len(y) - 60, refit=False)
param_grid = {"alpha": [0.01, 1, 100]}

# ---------------------------------------------------------------------------
header("A. grid_search_forecaster(metric=[r2_score, 'mean_absolute_error'])")
forecaster = ForecasterRecursive(estimator=Ridge(), lags=12)
results = grid_search_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    param_grid=param_grid,
    metric=[r2_score, "mean_absolute_error"],
    return_best=True,
    show_progress=False,
)
print(results[["params", "r2_score", "mean_absolute_error"]].to_string())
best_r2_row = results.loc[results["r2_score"].idxmax()]
print(f"\nObserved: row 0 (the 'best') = {results.loc[0, 'params']}, "
      f"R2 = {results.loc[0, 'r2_score']:.3f}")
print(f"Observed: forecaster.estimator.alpha after return_best = "
      f"{forecaster.estimator.alpha}")
print(f"Expected: {best_r2_row['params']} (R2 = {best_r2_row['r2_score']:.3f}), "
      f"which is also the lowest MAE "
      f"({results['mean_absolute_error'].min():.4f})")

# ---------------------------------------------------------------------------
header("B. bayesian_search_forecaster(metric=r2_score), default direction")
forecaster = ForecasterRecursive(estimator=Ridge(), lags=12)


def search_space(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


results_b, study_b = bayesian_search_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    search_space=search_space,
    metric=r2_score,
    n_trials=20,
    random_state=123,
    return_best=True,
    show_progress=False,
)
print(f"Optuna study direction: {study_b.direction.name}")
print(results_b[["params", "r2_score"]].head(3).to_string())
print(f"Observed: refitted alpha = {forecaster.estimator.alpha:.3f}, "
      f"R2 = {results_b.loc[0, 'r2_score']:.3f}")
print(f"Expected: alpha with max R2 among trials = "
      f"{results_b.loc[results_b['r2_score'].idxmax(), 'alpha']:.4f} "
      f"(R2 = {results_b['r2_score'].max():.3f}); the sampler should explore "
      f"high R2, not low")

# ---------------------------------------------------------------------------
header(
    "C. bayesian_search_forecaster(metric=r2_score, "
    "kwargs_create_study={'direction': 'maximize'})"
)
forecaster = ForecasterRecursive(estimator=Ridge(), lags=12)
results_c, study_c = bayesian_search_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    search_space=search_space,
    metric=r2_score,
    n_trials=20,
    random_state=123,
    return_best=True,
    show_progress=False,
    kwargs_create_study={"direction": "maximize"},
)
print(f"Optuna study direction: {study_c.direction.name}")
print(f"study.best_value = {study_c.best_value:.3f}, "
      f"study.best_params = {study_c.best_params}")
print(results_c[["params", "r2_score"]].head(3).to_string())
print(f"Observed: results row 0 R2 = {results_c.loc[0, 'r2_score']:.3f}; "
      f"refitted alpha = {forecaster.estimator.alpha:.3f}")
print(f"Expected: refitted alpha = {study_c.best_params['alpha']:.4f} "
      f"(study.best_value = {study_c.best_value:.3f})")

# ---------------------------------------------------------------------------
header("D. grid_search_forecaster_multiseries(metric=r2_score)")
rng = np.random.default_rng(1)
series = pd.DataFrame(
    {"s1": y.to_numpy(), "s2": y.to_numpy() * 1.5 + rng.normal(0, 0.03, len(y))},
    index=y.index,
)
forecaster_ms = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=12)
results_d = grid_search_forecaster_multiseries(
    forecaster=forecaster_ms,
    series=series,
    cv=cv,
    param_grid=param_grid,
    metric=r2_score,
    aggregate_metric="weighted_average",
    return_best=True,
    show_progress=False,
)
print(results_d[["params", "r2_score__weighted_average"]].to_string())
print(f"Observed: refitted alpha = {forecaster_ms.estimator.alpha}")
print(f"Expected: refitted alpha = "
      f"{results_d.loc[results_d['r2_score__weighted_average'].idxmax(), 'alpha']}")

# ---------------------------------------------------------------------------
header("E. ForecasterRecursiveClassifier + grid_search(metric=zero_one_loss)")
y_class = pd.Series(
    np.where(y.diff().fillna(0) > 0, "up", "down"), index=y.index, name="y"
)
param_grid_clf = {"C": [1e-4, 1.0]}
forecaster_clf = ForecasterRecursiveClassifier(
    estimator=LogisticRegression(max_iter=1000), lags=12
)
results_e = grid_search_forecaster(
    forecaster=forecaster_clf,
    y=y_class,
    cv=cv,
    param_grid=param_grid_clf,
    metric=[zero_one_loss, "accuracy_score"],
    return_best=True,
    show_progress=False,
)
print(results_e[["params", "zero_one_loss", "accuracy_score"]].to_string())
print(f"Observed: refitted C = {forecaster_clf.estimator.C}")
print(f"Expected: refitted C = "
      f"{results_e.loc[results_e['zero_one_loss'].idxmin(), 'C']} "
      f"(lowest zero_one_loss)")

# ---------------------------------------------------------------------------
header("Sanity check: the metric values are right, only the ranking is wrong")
for alpha in param_grid["alpha"]:
    f = ForecasterRecursive(estimator=Ridge(alpha=alpha), lags=12)
    m, _ = backtesting_forecaster(
        forecaster=f, y=y, cv=cv, metric=r2_score, show_progress=False
    )
    print(f"alpha={alpha:>6}: backtesting R2 = {m.iloc[0, 0]:.3f}")
