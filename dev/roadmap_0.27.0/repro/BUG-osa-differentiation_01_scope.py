"""
BUG-osa-differentiation: scope. One-step-ahead metrics (`OneStepAheadFold`) of
the four forecasters with `differentiation` (None, 1, 2) and without or with a
transformer (StandardScaler, log), compared with a backtesting with `steps=1`
and `refit=False`, which predicts each point one step ahead from the observed
values with the same fitted estimator. Without differentiation, both are equal
(the existing equivalence tests). Columns:

- `now`: OSA with the current code.
- `fix`: OSA with the prototype of the fix (`_00_fix_prototype.py`).
- `bt`: backtesting with `steps=1`, `refit=False`.

The direct forecasters have `steps=1` in the first table. With `steps=3`, the
one-step-ahead search and the backtesting do not use the same rows even without
differentiation (a separate, pre-existing difference, out of the scope of this
bug), so the second table only compares `now` with `fix`.

Run with PYTHONPATH pointing to the code to test (0.27.x or a branch).
"""
import importlib.util
import pathlib
import platform
import warnings
import numpy as np
import pandas as pd
import sklearn
import skforecast
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skforecast.datasets import fetch_dataset
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, backtesting_forecaster,
    backtesting_forecaster_multiseries, grid_search_forecaster,
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
    f"scikit-learn {sklearn.__version__}, skforecast from {skforecast.__file__}"
)

METRICS = [
    "mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error",
    "mean_absolute_scaled_error", "root_mean_squared_scaled_error",
]
SHORT = ["MAE", "MSE", "MAPE", "MASE", "RMSSE"]
TRANSFORMERS = {
    "none": None,
    "StandardScaler": StandardScaler(),
    "log": FunctionTransformer(func=np.log, inverse_func=np.exp),
}

y = fetch_dataset("h2o", verbose=False)["x"].rename("y")
series = fetch_dataset("items_sales", verbose=False)
N_Y, N_SERIES = 144, 912  # last 60 months, last 185 days


def osa_single(forecaster):
    results = grid_search_forecaster(
        forecaster=forecaster, y=y, cv=OneStepAheadFold(initial_train_size=N_Y),
        param_grid={"alpha": [1.0]}, metric=METRICS, return_best=False,
        verbose=False, show_progress=False,
    )
    return results[METRICS].iloc[0].to_numpy(dtype=float)


def bt_single(forecaster):
    metric, _ = backtesting_forecaster(
        forecaster=forecaster, y=y,
        cv=TimeSeriesFold(steps=1, initial_train_size=N_Y, refit=False),
        metric=METRICS, show_progress=False,
    )
    return metric[METRICS].iloc[0].to_numpy(dtype=float)


def osa_multi(forecaster):
    results = grid_search_forecaster_multiseries(
        forecaster=forecaster, series=series, cv=OneStepAheadFold(initial_train_size=N_SERIES),
        param_grid={"alpha": [1.0]}, metric=METRICS, aggregate_metric=["average"],
        return_best=False, verbose=False, show_progress=False,
    )
    columns = [f"{m}__average" if f"{m}__average" in results else m for m in METRICS]
    return results[columns].iloc[0].to_numpy(dtype=float)


def bt_multi(forecaster):
    metric, _ = backtesting_forecaster_multiseries(
        forecaster=forecaster, series=series,
        cv=TimeSeriesFold(steps=1, initial_train_size=N_SERIES, refit=False),
        metric=METRICS, add_aggregated_metric=True, show_progress=False,
    )
    metric = metric.set_index("levels")
    row = "average" if "average" in metric.index else metric.index[0]
    return metric.loc[row, METRICS].to_numpy(dtype=float)


def make(kind, diff, transformer, steps=1):
    t = None if transformer is None else sklearn.clone(transformer)
    if kind == "ForecasterRecursive":
        return ForecasterRecursive(Ridge(), lags=12, differentiation=diff, transformer_y=t)
    if kind == "ForecasterDirect":
        return ForecasterDirect(Ridge(), lags=12, steps=steps, differentiation=diff, transformer_y=t)
    if kind == "ForecasterRecursiveMultiSeries":
        return ForecasterRecursiveMultiSeries(
            Ridge(), lags=14, differentiation=diff, transformer_series=t, encoding="ordinal"
        )
    return ForecasterDirectMultiVariate(
        Ridge(), level="item_1", lags=14, steps=steps, differentiation=diff, transformer_series=t
    )


print("\n| Forecaster | diff | transformer | MAE now | MAE fix | MAE bt | MAE now/bt "
      "| MASE now | MASE fix | MASE bt | RMSSE now/bt | MAPE now/bt | fix == bt (5 metrics) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for kind in ["ForecasterRecursive", "ForecasterDirect",
             "ForecasterRecursiveMultiSeries", "ForecasterDirectMultiVariate"]:
    single = kind in ("ForecasterRecursive", "ForecasterDirect")
    osa, bt = (osa_single, bt_single) if single else (osa_multi, bt_multi)
    for diff in [None, 1, 2]:
        for tname, transformer in TRANSFORMERS.items():
            now = osa(make(kind, diff, transformer))
            with fix.enabled():
                fixed = osa(make(kind, diff, transformer))
            ref = bt(make(kind, diff, transformer))
            ok = np.allclose(fixed, ref, rtol=1e-8)
            print(
                f"| {kind} | {diff} | {tname} | {now[0]:.4f} | {fixed[0]:.4f} | {ref[0]:.4f} | "
                f"{now[0] / ref[0]:.2f} | {now[3]:.4f} | {fixed[3]:.4f} | {ref[3]:.4f} | "
                f"{now[4] / ref[4]:.2f} | {now[2] / ref[2]:.2f} | {'yes' if ok else 'NO'} |"
            )

print("\nDirect forecasters with `steps=3`, no transformer\n")
print("| Forecaster | diff | MAE now | MAE fix | MAE now/fix | MASE now | MASE fix |")
print("|---|---|---|---|---|---|---|")
for kind in ["ForecasterDirect", "ForecasterDirectMultiVariate"]:
    osa = osa_single if kind == "ForecasterDirect" else osa_multi
    for diff in [1, 2]:
        now = osa(make(kind, diff, None, steps=3))
        with fix.enabled():
            fixed = osa(make(kind, diff, None, steps=3))
        print(f"| {kind} | {diff} | {now[0]:.4f} | {fixed[0]:.4f} | {now[0] / fixed[0]:.2f} | "
              f"{now[3]:.4f} | {fixed[3]:.4f} |")
