"""
Benchmark behind the forecaster finder (decision tree of the
`choosing-a-forecaster` skill).

Each case answers one question of the tree with real data: the same backtesting
folds for every model, a fixed and reasonable configuration per model (no
tuning on the test period) and the same metric for all of them.

Metric: MASE, the MAE of each series divided by the MAE of the one-step naive
forecast over all the observations before the test period (the full history,
also when a model is trained on a truncated history, so the scale does not
change between history lengths). Averaged across series. Classification:
accuracy and balanced accuracy.

Time: seconds of the whole backtest (fit + predict) on CPU.

Usage:
    python benchmark.py            # all cases
    python benchmark.py vic h2o    # cases whose name contains any of the words
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from skforecast.datasets import fetch_dataset
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    backtesting_foundation,
    backtesting_stats,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import (
    ForecasterEquivalentDate,
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
    ForecasterStats,
)
from skforecast.stats import Arima, Ets

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / os.environ.get("BENCH_OUT", "results")
OUT.mkdir(exist_ok=True)
CHRONOS_ID = "autogluon/chronos-2-small"

FAMILY = {
    "Seasonal naive": "Baseline",
    "LightGBM": "ML",
    "Ridge": "ML",
    "LightGBM (diff)": "ML",
    "Ridge (diff)": "ML",
    "LightGBM (no exog)": "ML",
    "LightGBM drivers (DirectMultiVariate)": "ML",
    "Chronos-2": "Foundation",
    "Chronos-2 (no exog)": "Foundation",
    "Chronos-2 (past-only covariates)": "Foundation",
    "AutoETS": "Statistical",
    "AutoARIMA": "Statistical",
}


@dataclass
class Case:
    name: str
    question: str            # Question of the tree that the case tests
    loader: callable          # Returns (series DataFrame, exog DataFrame | None)
    steps: int
    n_folds: int
    season: int
    lags: int | list
    window: int | None
    calendar: list | None
    exog_mode: str = "none"  # 'future', 'past' or 'none'
    histories: list = field(default_factory=lambda: [None])  # None = full
    stats: list = field(default_factory=list)  # 'ets', 'arima' (single series)


# ==============================================================================
# Data loaders: each returns a wide DataFrame (one column per series, the
# target first) with a frequency, and the exogenous variables or None.
# ==============================================================================
def load_vic_daily():
    d = fetch_dataset("vic_electricity", verbose=False)
    d = d.resample("D").agg(
        {"Demand": "sum", "Temperature": ["max", "mean"], "Holiday": "max"}
    )
    d.columns = ["demand", "temp_max", "temp_mean", "holiday"]
    d = d.iloc[1:-1].asfreq("D")
    d["holiday"] = d["holiday"].astype(float)
    return d[["demand"]], d[["temp_max", "temp_mean", "holiday"]]


def load_bike_daily():
    d = fetch_dataset("bike_sharing", verbose=False)
    d = d.resample("D").agg(
        {"users": "sum", "temp": "mean", "hum": "mean", "windspeed": "mean",
         "holiday": "max", "workingday": "max"}
    ).asfreq("D")
    return d[["users"]], d[["temp", "hum", "windspeed", "holiday", "workingday"]].astype(float)


def load_bike_hourly():
    d = fetch_dataset("bike_sharing", verbose=False).asfreq("h")
    d = d.iloc[-24 * 400:]
    return d[["users"]], d[["temp", "hum", "windspeed"]].astype(float)


def load_transport_madrid():
    d = fetch_dataset("public_transport_madrid", verbose=False).asfreq("D")
    return d[["total"]], None


def _monthly(name, col, end="2019-12-01"):
    d = fetch_dataset(name, verbose=False)
    d = d.loc[:end, [col]].astype(float)
    return d.asfreq("MS"), None


def load_h2o():
    d = fetch_dataset("h2o", verbose=False).astype(float)
    d.columns = ["h2o"]
    return d.asfreq("MS"), None


def load_fuel():
    return _monthly("fuel_consumption", "Gasolinas")


def load_expenditures():
    return _monthly("expenditures_australia", "turnover")


def load_ett_hourly():
    d = fetch_dataset("ett_m2", verbose=False).resample("h").mean().asfreq("h")
    d = d.iloc[-24 * 400:]
    return d[["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]], None


def load_store_sales():
    d = fetch_dataset("store_sales", verbose=False)
    d = d.loc[d["store"] == 1]
    s = d.pivot_table(index=d.index, columns="item", values="sales").asfreq("D")
    s.columns = [f"item_{c}" for c in s.columns]
    s.index.name = None
    s.columns.name = None
    return s.astype(float), None


def load_bdg2():
    d = fetch_dataset("bdg2_daily", verbose=False)
    site = "Fox"  # 128 buildings with complete daily data in 2016-2017
    d = d.loc[d["site_id"] == site]
    s = d.pivot_table(index=d.index, columns="building_id", values="meter_reading")
    s = s.asfreq("D")
    # Buildings with complete, non constant data
    s = s.loc[:, (s.isna().sum() == 0) & (s.std() > 0) & ((s == 0).mean() < 0.02)]
    s = s.iloc[:, :40]
    temp = d.groupby(level=0)["airTemperature"].mean().asfreq("D").interpolate()
    s.index.name = None
    s.columns.name = None
    return s.astype(float), temp.to_frame("temp").astype(float)


def load_tourism():
    d = fetch_dataset("australia_tourism", verbose=False)
    d["series"] = d["Region"] + " | " + d["Purpose"]
    s = d.pivot_table(index=d.index, columns="series", values="Trips").asfreq("QS")
    s = s.loc[:, s.isna().sum() == 0]
    s.index.name = None
    s.columns.name = None
    return s.astype(float), None


def load_bicimad_weekly():
    d = fetch_dataset("bicimad", verbose=False)
    d = d.resample("W-SUN").sum().iloc[1:-1].asfreq("W-SUN")
    return d[["users"]].astype(float), None


def load_m4_hourly():
    d = fetch_dataset("m4_hourly", verbose=False)
    # M4 timestamps are artificial and do not overlap: align the 50 first
    # series with 1008 hourly values on a common index.
    lengths = d.groupby("series_id").size()
    ids = sorted(lengths[lengths == 1008].index)[:50]
    idx = pd.date_range("2020-01-06", periods=1008, freq="h")
    s = pd.DataFrame({i: d.loc[d["series_id"] == i, "value"].to_numpy() for i in ids}, index=idx)
    return s.astype(float), None


CASES = [
    # Frequencies not covered by the cases below
    Case("bicimad_weekly", "1 series, weekly, no extra info",
         load_bicimad_weekly, 8, 6, 52, [1, 2, 3, 4, 52], 4, ["month"], "none",
         [104, 208, None], ["ets"]),
    Case("bike_sharing_hourly", "1 series, hourly, weather known in advance",
         load_bike_hourly, 24, 28, 24, [1, 2, 3, 24, 48, 168], 24,
         ["hour", "day_of_week"], "future", [24 * 90, None]),
    Case("m4_hourly_50_series", "50 series, hourly, no extra info",
         load_m4_hourly, 48, 4, 24, [1, 2, 3, 24, 48, 168], 24,
         None, "none"),
    # Q4 How much history (single series, known future info) + Q5 priority
    Case("vic_electricity_daily", "1 series, info known in advance (temperature, holidays)",
         load_vic_daily, 28, 6, 7, 7, 7, ["day_of_week", "month"], "future",
         [90, 182, 365, 730, None], ["ets"]),
    Case("bike_sharing_daily", "1 series, info known in advance (weather, holidays)",
         load_bike_daily, 28, 6, 7, 7, 7, ["day_of_week", "month"], "future",
         [90, 182, 365, None], ["ets"]),
    Case("public_transport_madrid", "1 series, no extra info, daily",
         load_transport_madrid, 28, 6, 7, 14, 7, ["day_of_week", "month"], "none",
         [90, 182, 365, None], ["ets"]),
    # Q4 Short history, monthly, no extra info
    Case("h2o_monthly", "1 series, short monthly history",
         load_h2o, 12, 3, 12, 12, None, ["month"], "none",
         [36, 60, 120, None], ["ets", "arima"]),
    Case("fuel_gasoline_monthly", "1 series, monthly history",
         load_fuel, 12, 5, 12, 12, None, ["month"], "none",
         [36, 60, 120, None], ["ets", "arima"]),
    Case("expenditures_australia_monthly", "1 series, monthly history",
         load_expenditures, 12, 5, 12, 12, None, ["month"], "none",
         [36, 60, 120, None], ["ets", "arima"]),
    # Q3 Info known only up to today (past-only covariates)
    Case("vic_electricity_daily_past_only", "1 series, temperature known only up to today",
         load_vic_daily, 28, 6, 7, 7, 7, ["day_of_week", "month"], "past"),
    Case("bike_sharing_hourly_past_only", "1 series, weather readings known only up to today",
         load_bike_hourly, 24, 28, 24, [1, 2, 3, 24, 48, 168], 24,
         ["hour", "day_of_week"], "past"),
    Case("ett_hourly_past_only", "1 series, other sensor series known only up to today",
         load_ett_hourly, 24, 28, 24, [1, 2, 3, 24, 48, 168], 24,
         ["hour", "day_of_week"], "past"),
    # Q2 Several series
    Case("store_sales_50_items", "50 series, daily, no extra info",
         load_store_sales, 28, 8, 7, 14, 7, ["day_of_week", "month"], "none",
         [182, 365, None]),
    Case("bdg2_buildings_energy", "40 buildings, daily, temperature known in advance",
         load_bdg2, 28, 6, 7, 7, 7, ["day_of_week", "month"], "future",
         [90, 182, None]),
    Case("australia_tourism_quarterly", "Many short quarterly series",
         load_tourism, 4, 3, 4, 4, None, ["quarter"], "none"),
]


# ==============================================================================
# Models
# ==============================================================================
def lgbm(n_train):
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=5 if n_train < 500 else 20, random_state=123, verbose=-1,
    )


def ml_single(case, estimator, use_exog, diff=None):
    return ForecasterRecursive(
        estimator=estimator,
        lags=case.lags,
        differentiation=diff,
        window_features=RollingFeatures("mean", case.window) if case.window else None,
        calendar_features=CalendarFeatures(features=case.calendar) if case.calendar else None,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler() if use_exog else None,
    )


def ml_multi(case, estimator, use_exog, ridge=False, diff=None):
    return ForecasterRecursiveMultiSeries(
        estimator=estimator,
        lags=case.lags,
        differentiation=diff,
        window_features=RollingFeatures("mean", case.window) if case.window else None,
        calendar_features=(
            CalendarFeatures(features=case.calendar, encoding="onehot" if ridge else "cyclical")
            if case.calendar else None
        ),
        encoding=None if ridge else "ordinal",
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler() if use_exog else None,
    )


def chronos():
    return ForecasterFoundation(estimator=FoundationModel(CHRONOS_ID))


def to_long(pred, level=None):
    """Normalize backtesting predictions to columns ['level', 'pred']."""
    pred = pred.copy()
    if "level" not in pred.columns:
        pred["level"] = level
    return pred[["level", "pred"]]


def run_models(case, series, exog, n_train):
    """Run every model of the case. Returns {model: (pred_long, seconds)}."""
    cv = TimeSeriesFold(steps=case.steps, initial_train_size=n_train, refit=False, verbose=False)
    single = series.shape[1] == 1 or case.exog_mode == "past"
    target = series.columns[0]
    y = series[target]
    use_exog = case.exog_mode == "future" and exog is not None
    results = {}

    def timed(name, fn):
        start = time.perf_counter()
        results[name] = (fn(), time.perf_counter() - start)
        print(f"    {name:<40} {results[name][1]:7.1f} s", flush=True)

    # Baseline: seasonal naive (one per series)
    def naive():
        preds = []
        cols = [target] if case.exog_mode == "past" else series.columns
        for col in cols:
            _, p = backtesting_forecaster(
                ForecasterEquivalentDate(offset=case.season), y=series[col], cv=cv,
                metric="mean_absolute_error", show_progress=False,
            )
            preds.append(to_long(p, col))
        return pd.concat(preds)
    timed("Seasonal naive", naive)

    if single:
        X = exog if use_exog else None
        variants = [("LightGBM", lgbm(n_train), None), ("Ridge", Ridge(), None),
                    ("LightGBM (diff)", lgbm(n_train), 1), ("Ridge (diff)", Ridge(), 1)]
        if case.exog_mode == "past":
            variants = [("LightGBM (no exog)", lgbm(n_train), None)]
        for label, est, diff in variants:
            timed(label, lambda est=est, diff=diff: to_long(backtesting_forecaster(
                ml_single(case, est, X is not None, diff), y=y, cv=cv, exog=X,
                metric="mean_absolute_error", show_progress=False)[1], target))
        if case.exog_mode == "past":
            # Drivers: lags of every other series as predictors of the target
            drivers = (series if exog is None else pd.concat([series, exog], axis=1))
            timed("LightGBM drivers (DirectMultiVariate)", lambda: to_long(
                backtesting_forecaster_multiseries(
                    ForecasterDirectMultiVariate(
                        estimator=lgbm(n_train), level=target, steps=case.steps,
                        lags=case.lags,
                        window_features=RollingFeatures("mean", case.window) if case.window else None,
                        calendar_features=CalendarFeatures(features=case.calendar),
                        transformer_series=StandardScaler(),
                    ),
                    series=drivers, cv=cv, metric="mean_absolute_error",
                    show_progress=False)[1]))
    else:
        X = exog if use_exog else None
        timed("LightGBM", lambda: backtesting_forecaster_multiseries(
            ml_multi(case, lgbm(n_train), X is not None), series=series, cv=cv, exog=X,
            metric="mean_absolute_error", show_progress=False)[1][["level", "pred"]])
        timed("LightGBM (diff)", lambda: backtesting_forecaster_multiseries(
            ml_multi(case, lgbm(n_train), X is not None, diff=1), series=series, cv=cv,
            exog=X, metric="mean_absolute_error", show_progress=False)[1][["level", "pred"]])
        timed("Ridge", lambda: backtesting_forecaster_multiseries(
            ml_multi(case, Ridge(), X is not None, ridge=True), series=series, cv=cv,
            exog=X, metric="mean_absolute_error", show_progress=False)[1][["level", "pred"]])

    # Foundation: Chronos-2 zero-shot
    if case.exog_mode == "past":
        timed("Chronos-2 (no exog)", lambda: to_long(backtesting_foundation(
            chronos(), series=y, cv=cv, metric="mean_absolute_error",
            show_progress=False)[1]).assign(level=target))
        covariates = series.drop(columns=target) if exog is None else exog

        def chronos_past_only():
            fc = chronos()
            preds = []
            for origin in range(n_train, len(y), case.steps):
                fc.fit(series=y.iloc[:origin], exog=covariates.iloc[:origin])
                p = fc.predict(steps=min(case.steps, len(y) - origin))  # no future exog
                preds.append(to_long(p).assign(level=target))
            return pd.concat(preds)
        timed("Chronos-2 (past-only covariates)", chronos_past_only)
    else:
        series_in = y if single else series
        timed("Chronos-2", lambda: to_long(backtesting_foundation(
            chronos(), series=series_in, cv=cv, exog=exog if use_exog else None,
            metric="mean_absolute_error", show_progress=False)[1]).assign(
                level=lambda d: d["level"] if not single else target))

    # Statistical benchmark (single series only)
    for stat in case.stats:
        est = (Ets(m=case.season) if stat == "ets"
               else Arima(order=None, seasonal_order=None, m=case.season))
        label = "AutoETS" if stat == "ets" else "AutoARIMA"
        timed(label, lambda est=est: to_long(backtesting_stats(
            ForecasterStats(estimator=est), y=y, cv=cv, metric="mean_absolute_error",
            show_progress=False)[1], target))

    return results


def mase_table(results, series_full, test_start, levels):
    """MASE per model, averaged across series, scaled with the full history."""
    rows = []
    hist = series_full.iloc[:test_start]
    scale = hist.diff().abs().mean()
    for model, (pred, secs) in results.items():
        per_series = []
        for level in levels:
            p = pred.loc[pred["level"] == level, "pred"]
            if p.empty:
                continue
            y_true = series_full.loc[p.index, level]
            per_series.append(np.mean(np.abs(y_true.values - p.values)) / scale[level])
        rows.append({"model": model, "family": FAMILY[model],
                     "MASE": float(np.mean(per_series)), "seconds": secs})
    return pd.DataFrame(rows)


def run_case(case):
    print(f"\n=== {case.name}: {case.question}", flush=True)
    series, exog = case.loader()
    n_test = case.steps * case.n_folds
    test_start = len(series) - n_test
    levels = [series.columns[0]] if case.exog_mode == "past" else list(series.columns)
    tables = []
    for history in case.histories:
        start = 0 if history is None else test_start - history
        n_train = test_start - start
        print(f"  history: {n_train} obs per series", flush=True)
        s = series.iloc[start:]
        x = exog.iloc[start:] if exog is not None else None
        res = run_models(case, s, x, n_train)
        t = mase_table(res, series, test_start, levels)
        t.insert(0, "history_obs", n_train)
        t.insert(0, "history", "full" if history is None else str(history))
        t.insert(0, "case", case.name)
        tables.append(t)
        print(t[["model", "MASE", "seconds"]].round(3).to_string(index=False), flush=True)
    out = pd.concat(tables)
    out["question"] = case.question
    out["n_series"] = len(levels)
    out.to_csv(OUT / f"{case.name}.csv", index=False)
    return out


def run_classification():
    """Q1: a category instead of a number."""
    print("\n=== vic_electricity_classification: forecast a category", flush=True)
    d = fetch_dataset("vic_electricity_classification", verbose=False).asfreq("h")
    y = d["Demand"]
    X = d[["Temperature", "Holiday", "Hour_of_day"]].astype(float)
    X["day_of_week"] = X.index.dayofweek.astype(float)  # Classifier has no calendar_features
    steps, n_folds = 24, 28
    n_train = len(y) - steps * n_folds
    cv = TimeSeriesFold(steps=steps, initial_train_size=n_train, refit=False, verbose=False)
    rows = []
    start = time.perf_counter()
    f = ForecasterRecursiveClassifier(
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       random_state=123, verbose=-1),
        lags=[1, 2, 3, 24, 168],
    )
    m, _ = backtesting_forecaster(f, y=y, cv=cv, exog=X, show_progress=False,
                                  metric=["accuracy_score", "balanced_accuracy_score"])
    rows.append({"model": "LightGBM classifier", "accuracy": m["accuracy_score"].iloc[0],
                 "balanced_accuracy": m["balanced_accuracy_score"].iloc[0],
                 "seconds": time.perf_counter() - start})
    # Baselines computed by hand: no classification baseline exists in skforecast
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    test = y.iloc[n_train:]
    for name, lag in [("Same hour yesterday", 24), ("Same hour last week", 168)]:
        pred = []
        for origin in range(n_train, len(y), steps):
            for h in range(min(steps, len(y) - origin)):
                k = int(np.ceil((h + 1) / lag)) * lag  # last value known at the origin
                pred.append(y.iloc[origin + h - k])
        rows.append({"model": name, "accuracy": accuracy_score(test, pred),
                     "balanced_accuracy": balanced_accuracy_score(test, pred), "seconds": 0.0})
    mode = y.iloc[:n_train].mode().iloc[0]
    rows.append({"model": "Most frequent class", "accuracy": accuracy_score(test, [mode] * len(test)),
                 "balanced_accuracy": balanced_accuracy_score(test, [mode] * len(test)), "seconds": 0.0})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv(OUT / "vic_electricity_classification.csv", index=False)
    return out


if __name__ == "__main__":
    words = sys.argv[1:]
    selected = [c for c in CASES if not words or any(w in c.name for w in words)]
    env = {p: __import__(p).__version__ for p in
           ["skforecast", "numpy", "pandas", "sklearn", "lightgbm", "torch"]}
    env["python"] = sys.version.split()[0]
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))
    print(env)
    for case in selected:
        run_case(case)
    if not words or any("class" in w for w in words):
        run_classification()
