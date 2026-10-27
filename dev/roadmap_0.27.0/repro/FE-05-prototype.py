"""
FE-05 prototype: compare several forecasters of any family in one call.

Design checked here (not package code):
1. Dispatch each forecaster to its existing backtesting function, unchanged.
2. Normalize every output to long format: forecaster, level, fold, pred,
   lower_bound, upper_bound (foundation q_* columns are mapped).
3. Recompute every metric centrally, on the points that all forecasters
   predict, with one y_train for the scaled metrics (MASE), so that all models
   are measured with the same ruler.

Also measures the two problems that motivate step 3: the MASE scale changes
with each forecaster's window_size, and the differentiation check of the cv.

Run: python dev/roadmap_0.27.0/repro/FE-05-prototype.py
"""

import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from skforecast.datasets import load_demo_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation
from sklearn.metrics import mean_absolute_error as _mae
from skforecast.metrics import add_y_train_argument, mean_absolute_scaled_error

mean_absolute_error = add_y_train_argument(_mae)
mean_absolute_error.__name__ = "mean_absolute_error"
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    backtesting_foundation,
    backtesting_stats,
)
from skforecast.recursive import (
    ForecasterEquivalentDate,
    ForecasterRecursive,
    ForecasterRecursiveMultiSeries,
    ForecasterStats,
)
from skforecast.stats import Arima, Ets

warnings.filterwarnings("ignore")

SINGLE = {"ForecasterRecursive", "ForecasterDirect", "ForecasterEquivalentDate"}
GLOBAL = {"ForecasterRecursiveMultiSeries", "ForecasterDirectMultiVariate", "ForecasterRnn"}


def _run_one(forecaster, series, cv, interval, exog, kwargs):
    """Call the backtesting function of the forecaster family; long output."""
    name = type(forecaster).__name__
    cv = deepcopy(cv)
    # Workaround until QW-03: a single cv cannot serve forecasters with a
    # different differentiation, so set it per forecaster.
    if hasattr(forecaster, "differentiation_max"):
        cv.set_params({"differentiation": forecaster.differentiation_max})
    common = dict(cv=cv, metric="mean_absolute_error", exog=exog,
                  show_progress=False, suppress_warnings=True)
    if name in SINGLE or name == "ForecasterStats":
        y = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
        if name == "ForecasterStats":
            _, p = backtesting_stats(forecaster, y=y, interval=interval, **common, **kwargs)
        else:
            if name == "ForecasterEquivalentDate" and interval is not None:
                kwargs = {"interval_method": "conformal", **kwargs}  # BUG-11
            _, p = backtesting_forecaster(forecaster, y=y, interval=interval, **common, **kwargs)
        p = p.assign(level=y.name)
    elif name in GLOBAL:
        frame = series.to_frame() if isinstance(series, pd.Series) else series
        _, p = backtesting_forecaster_multiseries(forecaster, series=frame, interval=interval, **common, **kwargs)
    elif name == "ForecasterFoundation":
        quantiles = None if interval is None else [interval[0], 0.5, interval[1]]
        _, p = backtesting_foundation(forecaster, series=series, quantiles=quantiles, **common, **kwargs)
        if quantiles is not None:
            # BUG-06: with quantiles the output has no `pred`; q_0.5 is the point
            p = p.rename(columns={f"q_{interval[0]}": "lower_bound", "q_0.5": "pred",
                                  f"q_{interval[1]}": "upper_bound"})
    else:
        raise TypeError(f"{name} is not supported")
    p = p.rename_axis("idx").reset_index()
    if "estimator_id" in p.columns:  # ForecasterStats with several estimators
        p = p.rename(columns={"estimator_id": "sub_id"})
    return p


def compare_forecasters(forecasters, series, cv, metrics, interval=None, exog=None,
                        backtesting_kwargs=None, common_points=True):
    backtesting_kwargs = backtesting_kwargs or {}
    preds, elapsed = [], {}
    for fname, forecaster in forecasters.items():
        start = time.perf_counter()
        p = _run_one(forecaster, series, cv, interval, exog, backtesting_kwargs.get(fname, {}))
        elapsed[fname] = time.perf_counter() - start
        p["base_name"] = fname
        if "sub_id" in p.columns:
            p["forecaster"] = fname + " (" + p.pop("sub_id").str.replace("skforecast.", "") + ")"
        else:
            p["forecaster"] = fname
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    # Overlapping folds: keep the last prediction of each date, as backtesting does
    preds = preds.drop_duplicates(subset=["forecaster", "level", "idx"], keep="last")

    keys = preds.groupby("forecaster")[["level", "idx"]].apply(lambda d: set(map(tuple, d.to_numpy())))
    common = set.intersection(*keys)
    dropped = {f: len(k - common) for f, k in keys.items() if len(k - common)}
    if dropped and common_points:
        print(f"  WARNING: metrics on {len(common)} common points; dropped per forecaster: {dropped}")
        mask = [k in common for k in map(tuple, preds[["level", "idx"]].to_numpy())]
        preds = preds.loc[mask]

    # One ruler: y_train for scaled metrics is every observation before the first
    # test point, the same for all forecasters (window_size is not usable: in
    # foundation models it is the context length, longer than the series)
    frame = series.to_frame() if isinstance(series, pd.Series) else series
    rows = []
    for (fname, level), g in preds.groupby(["forecaster", "level"], sort=False):
        y_true = frame[level].loc[g["idx"]].to_numpy()
        y_train = frame[level].iloc[:cv.initial_train_size].dropna().to_numpy()
        row = {"forecaster": fname, "level": level, "n_predictions": len(g)}
        for m in metrics:
            row[m.__name__] = m(y_true=y_true, y_pred=g["pred"].to_numpy(), y_train=y_train)
        if "lower_bound" in g and g["lower_bound"].notna().all():
            row["coverage"] = np.mean((y_true >= g["lower_bound"]) & (y_true <= g["upper_bound"]))
            row["interval_width"] = np.mean(g["upper_bound"] - g["lower_bound"])
        row["elapsed_time"] = elapsed[g["base_name"].iloc[0]]
        rows.append(row)
    return pd.DataFrame(rows), preds


def main():
    y = load_demo_dataset()
    y.name = "y"
    n_test = 36
    cv = TimeSeriesFold(steps=12, initial_train_size=len(y) - n_test)

    print("1) MASE scale depends on window_size (same predictions, y_train with and without the first 24 obs)")
    _, p = backtesting_forecaster(ForecasterRecursive(Ridge(), lags=24), y=y, cv=cv,
                                  metric="mean_absolute_error", show_progress=False)
    y_true = y.loc[p.index].to_numpy()
    for ws in [1, 12, 24]:
        y_train = y.iloc[ws:len(y) - n_test].to_numpy()
        print(f"   y_train from obs {ws:>2}: MASE = {mean_absolute_scaled_error(y_true, p['pred'].to_numpy(), y_train):.4f}")

    print("\n2) Prototype on h2o (single series), interval [0.1, 0.9]")
    forecasters = {
        "Seasonal naive": ForecasterEquivalentDate(offset=12),
        "Ridge": ForecasterRecursive(Ridge(), lags=12, transformer_y=StandardScaler()),
        "LightGBM (diff=1)": ForecasterRecursive(
            LGBMRegressor(n_estimators=200, max_depth=4, verbose=-1, random_state=123),
            lags=24, differentiation=1),
        "Stats": ForecasterStats([Arima(order=None, seasonal_order=None, m=12), Ets(m=12)]),
        "Chronos-2": ForecasterFoundation(FoundationModel("autogluon/chronos-2-small", device_map="cpu")),
    }
    metrics, preds = compare_forecasters(
        forecasters, y, cv, [mean_absolute_error, mean_absolute_scaled_error], interval=[0.1, 0.9],
        backtesting_kwargs={"Ridge": {"interval_method": "conformal"}},
    )
    pd.set_option("display.width", 200)
    print(metrics.round(4).to_string(index=False))
    print("\n   predictions columns:", list(preds.columns), "| rows:", len(preds))

    print("\n   window_size of Chronos-2:", forecasters["Chronos-2"].window_size, "| len(y):", len(y))
    print("\n3) Same comparison as a multi-series panel (2 series, one shorter): global and foundation")
    panel = pd.DataFrame({"a": y, "b": y.where(y.index >= y.index[110]) * 1.3})
    metrics, _ = compare_forecasters(
        {"Ridge global": ForecasterRecursiveMultiSeries(Ridge(), lags=12),
         "Ridge global lags=72": ForecasterRecursiveMultiSeries(Ridge(), lags=72),
         "Chronos-2": ForecasterFoundation(FoundationModel("autogluon/chronos-2-small", device_map="cpu"))},
        panel, cv, [mean_absolute_error, mean_absolute_scaled_error],
    )
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
