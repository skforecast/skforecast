"""
Backtesting with intervals, out-of-sample residuals and refit: time and
predictions of the installed skforecast (base or branch, chosen with
PYTHONPATH). Each scenario runs once; the driver alternates base and branch.

The predictions of each run are saved as pickles in <output_dir>, to compare
base and branch with `pandas.testing.assert_frame_equal`. n_jobs=1.

git worktree add --detach <base_dir> origin/0.28.x
PYTHONPATH=<base_dir> python dev/preprocessing_speedups/t3_backtesting_bench.py base <output_dir>
python dev/preprocessing_speedups/t3_backtesting_bench.py branch <output_dir>
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import skforecast
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
)
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

warnings.filterwarnings("ignore")


def lgbm(n_estimators=100):
    return LGBMRegressor(n_estimators=n_estimators, verbose=-1, random_state=123)


def hgbr():
    return HistGradientBoostingRegressor(max_iter=100, random_state=123)


def single_series(n):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    season = np.sin(2 * np.pi * t / 24)
    return pd.Series(
        50 + 5 * season + 0.1 * np.cumsum(rng.normal(0, 0.3, n))
        + (0.3 + 1.5 * ((season + 1) / 2) ** 2) * rng.normal(0, 1, n),
        index=pd.date_range("2000-01-01", periods=n, freq="h"), name="y",
    )


def multi_series(n_series=100):
    rng = np.random.default_rng(0)
    lengths = rng.integers(2000, 5001, size=n_series)
    full_len = int(lengths.max())
    index = pd.date_range(end="2025-01-01", periods=full_len, freq="h")
    season = np.sin(2 * np.pi * np.arange(full_len) / 24)
    sigma = 0.3 + 1.5 * ((season + 1) / 2) ** 2
    series = {}
    for i, n in enumerate(lengths):
        y = (
            rng.uniform(10, 100) + rng.uniform(3, 8) * season
            + 0.1 * np.cumsum(rng.normal(0, 0.3, full_len))
            + sigma * rng.normal(0, 1, full_len)
        )
        series[f"series_{i:03d}"] = pd.Series(y[-n:], index=index[-n:])
    return pd.DataFrame(series)


def run_single(forecaster, y, n_train, n_cal, steps, binned):
    forecaster.fit(y=y.iloc[:n_train])
    cv_cal = TimeSeriesFold(steps=steps, initial_train_size=None, refit=False)
    _, pred_cal = backtesting_forecaster(
        forecaster, y=y.iloc[n_train - forecaster.window_size: n_train + n_cal],
        cv=cv_cal, metric="mean_absolute_error", show_progress=False, n_jobs=1,
    )
    forecaster.set_out_sample_residuals(
        y_true=y.loc[pred_cal.index], y_pred=pred_cal["pred"]
    )
    cv = TimeSeriesFold(steps=steps, initial_train_size=n_train + n_cal, refit=True)
    t0 = time.perf_counter()
    _, pred = backtesting_forecaster(
        forecaster, y=y, cv=cv, metric="mean_absolute_error",
        interval=[0.1, 0.9], interval_method="conformal",
        use_in_sample_residuals=False, use_binned_residuals=binned,
        show_progress=False, n_jobs=1,
    )
    return time.perf_counter() - t0, pred


def run_multi(forecaster, series, n_test, n_cal, steps):
    index = series.index
    end_train = index[-(n_test + n_cal) - 1]
    forecaster.fit(series=series.loc[:end_train])
    ws = forecaster.window_size
    cal = series.iloc[-(n_test + n_cal + ws): -n_test]
    cv_cal = TimeSeriesFold(steps=steps, initial_train_size=None, refit=False)
    _, pred_cal = backtesting_forecaster_multiseries(
        forecaster, series=cal, cv=cv_cal, metric="mean_absolute_error",
        show_progress=False, n_jobs=1,
    )
    forecaster.set_out_sample_residuals(
        y_true={k: g["y_true"] for k, g in pred_cal.groupby("level")},
        y_pred={k: g["pred"] for k, g in pred_cal.groupby("level")},
    )
    cv = TimeSeriesFold(steps=steps, initial_train_size=index[-n_test - 1], refit=True)
    t0 = time.perf_counter()
    _, pred = backtesting_forecaster_multiseries(
        forecaster, series=series, cv=cv, metric="mean_absolute_error",
        interval=[0.1, 0.9], interval_method="conformal",
        use_in_sample_residuals=False, use_binned_residuals=True,
        show_progress=False, n_jobs=1,
    )
    return time.perf_counter() - t0, pred


def window_features():
    return RollingFeatures(stats=["mean", "std", "min"], window_sizes=24)


def main():
    label, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    y_long = single_series(60_000)
    y_short = single_series(10_000)
    series = multi_series()
    scenarios = {
        "ForecasterRecursive, LightGBM, 60,000 obs., 10 fits, binned": lambda: run_single(
            ForecasterRecursive(estimator=lgbm(), lags=48, window_features=window_features()),
            y_long, 40_000, 5_000, 1_500, True,
        ),
        "ForecasterRecursive, LightGBM, 60,000 obs., 10 fits, not binned": lambda: run_single(
            ForecasterRecursive(estimator=lgbm(), lags=48, window_features=window_features()),
            y_long, 40_000, 5_000, 1_500, False,
        ),
        "ForecasterRecursive, HGBR, 60,000 obs., 10 fits, binned": lambda: run_single(
            ForecasterRecursive(estimator=hgbr(), lags=48, window_features=window_features()),
            y_long, 40_000, 5_000, 1_500, True,
        ),
        "ForecasterDirect, LightGBM (50 trees), 10,000 obs., 24 steps, 5 fits, binned": lambda: run_single(
            ForecasterDirect(estimator=lgbm(50), lags=48, steps=24, window_features=window_features()),
            y_short, 9_000, 880, 24, True,
        ),
        "ForecasterRecursiveMultiSeries, LightGBM, 100 series, 7 fits, binned": lambda: run_multi(
            ForecasterRecursiveMultiSeries(
                estimator=lgbm(), lags=48, encoding="ordinal",
                transformer_series=StandardScaler(), window_features=window_features(),
            ),
            series, 336, 336, 48,
        ),
        "ForecasterRecursiveMultiSeries, HGBR, 100 series, 7 fits, binned": lambda: run_multi(
            ForecasterRecursiveMultiSeries(
                estimator=hgbr(), lags=48, encoding="ordinal",
                transformer_series=StandardScaler(), window_features=window_features(),
            ),
            series, 336, 336, 48,
        ),
    }
    times = {}
    for i, (name, func) in enumerate(scenarios.items()):
        seconds, pred = func()
        times[name] = seconds
        with open(out_dir / f"{label}_{i}.pkl", "wb") as f:
            pickle.dump(pred, f)
    print(json.dumps({"label": label, "skforecast": skforecast.__file__, "times": times}))


if __name__ == "__main__":
    main()
