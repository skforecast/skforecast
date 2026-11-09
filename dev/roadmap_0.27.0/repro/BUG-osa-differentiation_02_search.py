"""
BUG-osa-differentiation: does the candidate selected by a one-step-ahead search
with `differentiation=1` change? Grid searches on skforecast datasets, each
candidate evaluated three ways:

- `now`: `OneStepAheadFold` with the current code.
- `fix`: `OneStepAheadFold` with the prototype of the fix (`_00_fix_prototype.py`),
  equal to a backtesting with `steps=1` and `refit=False` (`_01_scope.py`).
- `horizon`: `TimeSeriesFold` with the horizon of the use case (`steps=h`,
  `refit=False`), the evaluation that the one-step-ahead search approximates.

For each search: the candidate selected by MAE and by MASE with `now` and `fix`,
the Kendall tau between the MAE rankings of `now` and `fix` and of each one with
`horizon`, and the MAE at the horizon of the selected candidates, relative to
the best candidate at the horizon.

Run with PYTHONPATH pointing to the code to test (0.27.x or a branch).
"""
import importlib.util
import pathlib
import platform
import warnings
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import skforecast
from scipy.stats import kendalltau
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, grid_search_forecaster,
    grid_search_forecaster_multiseries,
)

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location(
    "fix_prototype", pathlib.Path(__file__).with_name("BUG-osa-differentiation_00_fix_prototype.py")
)
fix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix)

print(
    f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
    f"scikit-learn {sklearn.__version__}, lightgbm {lightgbm.__version__}, "
    f"skforecast from {skforecast.__file__}"
)

METRICS = ["mean_absolute_error", "mean_absolute_scaled_error"]
RIDGE_GRID = {"alpha": [0.01, 1.0, 100.0]}
LGBM_GRID = {"max_depth": [3, 6], "learning_rate": [0.03, 0.1]}


def make_lgbm():
    return LGBMRegressor(n_estimators=200, random_state=123, verbose=-1)


h2o = fetch_dataset("h2o", verbose=False)["x"]
fuel = fetch_dataset("fuel_consumption", verbose=False)["Gasolinas"]
items = fetch_dataset("items_sales", verbose=False)
bike = fetch_dataset("bike_sharing", verbose=False)["users"].iloc[-24 * 120:]

CASES = [
    # name, data, estimator factory, param_grid, lags_grid, initial_train_size, horizon
    ("h2o, Ridge", h2o, lambda: Ridge(), RIDGE_GRID, [3, 6, 12, 13, 24, 36], 144, 12),
    ("h2o, LightGBM", h2o, make_lgbm, LGBM_GRID, [3, 6, 12, 24], 144, 12),
    ("fuel Gasolinas, Ridge", fuel, lambda: Ridge(), RIDGE_GRID, [3, 6, 12, 13, 24, 36], 524, 12),
    ("fuel Gasolinas, LightGBM", fuel, make_lgbm, LGBM_GRID, [3, 6, 12, 24], 524, 12),
    ("items_sales item_1, Ridge", items["item_1"], lambda: Ridge(), RIDGE_GRID, [7, 14, 21, 28, 56], 912, 7),
    ("bike_sharing users (120 days), Ridge", bike, lambda: Ridge(), RIDGE_GRID, [24, 48, 72, 168], 24 * 90, 24),
    ("bike_sharing users (120 days), LightGBM", bike, make_lgbm, LGBM_GRID, [24, 48, 168], 24 * 90, 24),
    ("items_sales, 3 series, Ridge", items, lambda: Ridge(), RIDGE_GRID, [7, 14, 21, 28], 912, 7),
]


def search(data, estimator, param_grid, lags_grid, cv):
    if isinstance(data, pd.DataFrame):
        results = grid_search_forecaster_multiseries(
            forecaster=ForecasterRecursiveMultiSeries(estimator, lags=1, differentiation=1, encoding="ordinal"),
            series=data, cv=cv, param_grid=param_grid, lags_grid=lags_grid, metric=METRICS,
            aggregate_metric=["weighted_average"], return_best=False, verbose=False,
            show_progress=False, suppress_warnings=True,
        )
        results = results.rename(columns={f"{m}__weighted_average": m for m in METRICS})
    else:
        results = grid_search_forecaster(
            forecaster=ForecasterRecursive(estimator, lags=1, differentiation=1),
            y=data, cv=cv, param_grid=param_grid, lags_grid=lags_grid, metric=METRICS,
            return_best=False, verbose=False, show_progress=False, suppress_warnings=True,
        )
    results.index = [f"{len(lags)} lags, {params}" for lags, params in zip(results["lags"], results["params"])]
    return results[METRICS]


rows = []
for name, data, estimator, grid, lags_grid, n_train, h in CASES:
    now = search(data, estimator(), grid, lags_grid, OneStepAheadFold(initial_train_size=n_train))
    with fix.enabled():
        fixed = search(data, estimator(), grid, lags_grid, OneStepAheadFold(initial_train_size=n_train))
    horizon = search(
        data, estimator(), grid, lags_grid,
        TimeSeriesFold(steps=h, initial_train_size=n_train, refit=False),
    )
    keys = horizon.index
    now, fixed = now.loc[keys], fixed.loc[keys]
    mae_h = horizon["mean_absolute_error"]
    best_h = mae_h.min()
    pick = {
        "now MAE": now["mean_absolute_error"].idxmin(),
        "fix MAE": fixed["mean_absolute_error"].idxmin(),
        "now MASE": now["mean_absolute_scaled_error"].idxmin(),
        "fix MASE": fixed["mean_absolute_scaled_error"].idxmin(),
        "horizon": mae_h.idxmin(),
    }
    rows.append({
        "case": name,
        "candidates": len(keys),
        "pick now": pick["now MAE"],
        "pick fix": pick["fix MAE"],
        "pick horizon": pick["horizon"],
        "same pick now/fix": pick["now MAE"] == pick["fix MAE"],
        "MASE pick = MAE pick (now, fix)": f"{pick['now MASE'] == pick['now MAE']}, {pick['fix MASE'] == pick['fix MAE']}",
        "tau now~fix": kendalltau(now["mean_absolute_error"], fixed["mean_absolute_error"])[0],
        "tau now~horizon": kendalltau(now["mean_absolute_error"], mae_h)[0],
        "tau fix~horizon": kendalltau(fixed["mean_absolute_error"], mae_h)[0],
        "horizon MAE loss, now pick": mae_h[pick["now MAE"]] / best_h - 1,
        "horizon MAE loss, fix pick": mae_h[pick["fix MAE"]] / best_h - 1,
        "OSA MAE now/fix (best)": now["mean_absolute_error"].min() / fixed["mean_absolute_error"].min(),
    })

table = pd.DataFrame(rows)
for col in ["tau now~fix", "tau now~horizon", "tau fix~horizon", "OSA MAE now/fix (best)"]:
    table[col] = table[col].map(lambda x: f"{x:.2f}")
for col in ["horizon MAE loss, now pick", "horizon MAE loss, fix pick"]:
    table[col] = table[col].map(lambda x: f"{100 * x:.1f}%")
print()
print("| " + " | ".join(table.columns) + " |")
print("|" + "---|" * len(table.columns))
for row in table.itertuples(index=False):
    print("| " + " | ".join(str(v) for v in row) + " |")
