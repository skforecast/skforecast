"""
BUG-05: grid_search / random_search (single series and multiseries) turn a
deterministic configuration error into N RuntimeWarnings, return an empty
DataFrame (with the 'params' column duplicated) and leave the forecaster
unfitted (or fitted with the OLD parameters if it was already fitted), while
bayesian_search raises the original error.

Self-contained: synthetic data, Ridge estimator (fast).

Run: python dev/roadmap_0.27.0/repro/BUG-05.py
"""

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

import skforecast
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    bayesian_search_forecaster,
    bayesian_search_forecaster_multiseries,
    grid_search_forecaster,
    grid_search_forecaster_multiseries,
    random_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

print(
    f"skforecast {skforecast.__version__} | pandas {pd.__version__} | "
    f"numpy {np.__version__} | scikit-learn {sklearn.__version__}"
)

rng = np.random.default_rng(123)
n = 200
idx = pd.date_range("2020-01-01", periods=n, freq="D")
t = np.arange(n)
y = pd.Series(
    10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n),
    index=idx,
    name="y",
)
series = pd.DataFrame(
    {"s1": y.to_numpy(), "s2": y.to_numpy() * 1.5 + rng.normal(0, 0.5, n)},
    index=idx,
)
param_grid = {"alpha": [0.01, 0.1, 1, 10]}  # 4 combinations


def run_search(label, func, forecaster, **kwargs):
    """Run a search, report warnings, result shape and forecaster state."""
    print(f"\n=== {label} ===")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            results = func(forecaster=forecaster, **kwargs)
        except Exception as e:
            print(f"  OBSERVED: raises {type(e).__name__}: {str(e)[:150]}")
            return None
    skipped = [x for x in w if "Parameters skipped" in str(x.message)]
    final = [x for x in w if "No valid parameter combinations" in str(x.message)]
    print(f"  OBSERVED: no exception; {len(skipped)} 'Parameters skipped' warnings")
    if skipped:
        print(f"    first: {str(skipped[0].message)[:170]}")
    print(f"    final 'No valid parameter combinations' warning: {len(final)}")
    print(f"    results.shape = {results.shape}; columns = {list(results.columns)}")
    print(f"    duplicated columns: {list(results.columns[results.columns.duplicated()])}")
    print(f"    forecaster.is_fitted = {forecaster.is_fitted}")
    try:
        forecaster.predict(steps=3)
        print(
            f"    forecaster.predict(3) works (estimator alpha = "
            f"{forecaster.estimator.alpha}) -> predictions from the UNTUNED model"
        )
    except Exception as e:
        print(f"    forecaster.predict(3) -> {type(e).__name__}: {str(e)[:90]}")
    return results


# Case A: differentiation in forecaster but not in cv (common mistake)
cv_bad_diff = TimeSeriesFold(steps=7, initial_train_size=150, refit=False)
run_search(
    "A. grid_search_forecaster: differentiation=1 in forecaster, not in cv",
    grid_search_forecaster,
    ForecasterRecursive(estimator=Ridge(), lags=7, differentiation=1),
    y=y, cv=cv_bad_diff, param_grid=param_grid,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case B: initial_train_size larger than the data
run_search(
    "B. grid_search_forecaster: initial_train_size=500 with 200 observations",
    grid_search_forecaster,
    ForecasterRecursive(estimator=Ridge(), lags=7),
    y=y, cv=TimeSeriesFold(steps=7, initial_train_size=500), param_grid=param_grid,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case C: typo in a hyperparameter name
run_search(
    "C. grid_search_forecaster: typo in param name ('alphaa')",
    grid_search_forecaster,
    ForecasterRecursive(estimator=Ridge(), lags=7),
    y=y, cv=TimeSeriesFold(steps=7, initial_train_size=150),
    param_grid={"alphaa": [0.1, 1]},
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case D: random_search, OneStepAheadFold, typo
run_search(
    "D. random_search_forecaster + OneStepAheadFold: typo in param name",
    random_search_forecaster,
    ForecasterRecursive(estimator=Ridge(), lags=7),
    y=y, cv=OneStepAheadFold(initial_train_size=150),
    param_distributions={"alphaa": [0.1, 1, 10]}, n_iter=3,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case E: forecaster already fitted by the user before the search
f_prefit = ForecasterRecursive(estimator=Ridge(alpha=1234.0), lags=7, differentiation=1)
f_prefit.fit(y=y)
run_search(
    "E. grid_search_forecaster on an ALREADY FITTED forecaster (alpha=1234), "
    "differentiation mismatch",
    grid_search_forecaster,
    f_prefit,
    y=y, cv=cv_bad_diff, param_grid=param_grid,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case F: multiseries grid search, differentiation mismatch
run_search(
    "F. grid_search_forecaster_multiseries: differentiation mismatch",
    grid_search_forecaster_multiseries,
    ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=7, differentiation=1),
    series=series, cv=cv_bad_diff, param_grid=param_grid,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

# Case G/H: bayesian search with the same configuration error -> raises
def search_space(trial):
    return {"alpha": trial.suggest_float("alpha", 0.01, 10)}


run_search(
    "G. bayesian_search_forecaster: same differentiation mismatch",
    bayesian_search_forecaster,
    ForecasterRecursive(estimator=Ridge(), lags=7, differentiation=1),
    y=y, cv=cv_bad_diff, search_space=search_space, n_trials=4,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)
run_search(
    "H. bayesian_search_forecaster_multiseries: same differentiation mismatch",
    bayesian_search_forecaster_multiseries,
    ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=7, differentiation=1),
    series=series, cv=cv_bad_diff, search_space=search_space, n_trials=4,
    metric="mean_absolute_error", return_best=True, show_progress=False,
)

print(
    "\nEXPECTED: when EVERY combination fails, grid/random search should raise "
    "(e.g. ValueError('All parameter combinations failed ...') from the first "
    "error), like bayesian_search does (G, H), instead of returning an empty "
    "DataFrame with a duplicated 'params' column and an unfitted forecaster "
    "(A-D, F) or a forecaster still fitted with the old parameters (E)."
)
