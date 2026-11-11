"""
PERF-03: cost of the sklearn metrics used for the metrics passed as strings in
backtesting and hyperparameter search.

Prototype in memory (no change in the package): replaces the sklearn MAE and
MSE used by `_get_metric` with numpy versions, without and with the checks
that sklearn does today in the 1D case used by skforecast (float and 1D
conversion, `ValueError` with inconsistent lengths, no samples, NaN or
infinity). It prints:

- the cost per call of each version (pandas Series of 100 values),
- that the checked version gives the same results and errors as sklearn,
- the wall time of the full call (median of 3) with 1000 series, MAE and MSE
  and 8 candidates: one-step-ahead grid search with Ridge and with LightGBM,
  and backtesting with Ridge.

Expected: same metrics (max diff 0); one-step-ahead grid search about 2.6x
faster with Ridge and 1.4x with LightGBM; backtesting about 1.15x. The checks
add about 1 us per call.
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import skforecast.metrics.metrics as skm
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, grid_search_forecaster_multiseries,
    backtesting_forecaster_multiseries
)

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)

SK_MAE, SK_MSE = skm.mean_absolute_error, skm.mean_squared_error


def _check(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"{[len(y_true), len(y_pred)]}"
        )
    if y_true.size == 0:
        raise ValueError("Found array with 0 sample(s)")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Input contains NaN or infinity.")
    return y_true, y_pred


def mae_bare(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse_bare(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae_checked(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse_checked(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


for func in (mae_bare, mae_checked):
    func.__name__ = "mean_absolute_error"
for func in (mse_bare, mse_checked):
    func.__name__ = "mean_squared_error"

VERSIONS = {
    "sklearn": (SK_MAE, SK_MSE),
    "numpy without checks": (mae_bare, mse_bare),
    "numpy with checks": (mae_checked, mse_checked),
}


def use(name):
    skm.mean_absolute_error, skm.mean_squared_error = VERSIONS[name]


def timed(f, n=3):
    times, out = [], None
    for _ in range(n):
        s = time.perf_counter()
        out = f()
        times.append(time.perf_counter() - s)
    return np.median(times), out


# Cost per call
ys = pd.Series(rng.normal(size=100))
yp = pd.Series(rng.normal(size=100))
for name, (mae, _) in VERSIONS.items():
    n = 3000 if name == "sklearn" else 20000
    s = time.perf_counter()
    for _ in range(n):
        mae(ys, yp)
    print(f"per call {name:22s} {(time.perf_counter() - s) / n * 1e6:7.1f} us")

# Same results and same errors as sklearn
assert mae_checked(ys, yp) == SK_MAE(ys, yp)
assert mse_checked(ys, yp) == SK_MSE(ys, yp)
for bad in ([1.0, np.nan], [1.0], [1.0, np.inf]):
    for func in (SK_MAE, mae_checked):
        try:
            func(np.array([1.0, 2.0]), np.array(bad))
            raise AssertionError(f"{func.__module__}: no error with {bad}")
        except ValueError:
            pass
print("checked version: same results and same errors as sklearn")

# Wall time of the full call
n_obs, n_series = 400, 1000
idx = pd.date_range("2020-01-01", periods=n_obs, freq="D")
t = np.arange(n_obs)
series = pd.DataFrame(
    10 + 3 * np.sin(2 * np.pi * t / 7)[:, None] + rng.normal(0, 1, (n_obs, n_series)),
    index=idx, columns=[f"s{i}" for i in range(n_series)]
)
metrics = ["mean_absolute_error", "mean_squared_error"]
cases = {
    "grid OneStepAhead Ridge": lambda: grid_search_forecaster_multiseries(
        ForecasterRecursiveMultiSeries(Ridge(), lags=7, encoding="ordinal"),
        series=series, cv=OneStepAheadFold(initial_train_size=300),
        param_grid={"alpha": [0.01, 0.1, 1, 10]}, lags_grid=[7, 14],
        metric=metrics, return_best=False, show_progress=False, verbose=False),
    "grid OneStepAhead LightGBM": lambda: grid_search_forecaster_multiseries(
        ForecasterRecursiveMultiSeries(
            LGBMRegressor(verbose=-1, random_state=1), lags=7, encoding="ordinal"
        ),
        series=series, cv=OneStepAheadFold(initial_train_size=300),
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 6]}, lags_grid=[7, 14],
        metric=metrics, return_best=False, show_progress=False, verbose=False),
    "backtesting Ridge": lambda: backtesting_forecaster_multiseries(
        ForecasterRecursiveMultiSeries(Ridge(), lags=7, encoding="ordinal"),
        series=series, cv=TimeSeriesFold(steps=7, initial_train_size=300, refit=False),
        metric=metrics, show_progress=False, n_jobs=1)[0],
}
for label, f in cases.items():
    results = {}
    for name in VERSIONS:
        use(name)
        results[name] = timed(f)
    base, base_out = results["sklearn"]
    num = base_out.select_dtypes("number").columns
    diff = np.nanmax(np.abs(
        base_out[num].to_numpy(float)
        - results["numpy with checks"][1][num].to_numpy(float)
    ))
    print(
        f"{n_series} series | {label:27s} | "
        + " | ".join(f"{n} {v[0]:5.2f} s ({base / v[0]:4.2f}x)" for n, v in results.items())
        + f" | max diff {diff:.1e}"
    )
use("sklearn")
