"""
Control experiments of the review of the forecaster finder study.

Same cases, folds and metric as cases.py. It adds the checks that decide
whether the comparison between families is fair:

- ML refitted in every fold (expanding window), next to the frozen ML of the
  original study. Chronos-2 and the statistical models already use everything
  before each fold (`backtesting_stats` forces `refit=True` for Ets and Arima).
- A stronger LightGBM ("LightGBM+": more lags, two rolling windows, mild
  regularization), the LightGBM of the finder snippet (library defaults, no
  scaler) and the ML variant selected on a validation period taken from the
  training data, instead of "the best variant on the test period".
- Chronos-2 small, base (`amazon/chronos-2`) and synth
  (`autogluon/chronos-2-synth`, trained only on synthetic data: it cannot have
  seen any public dataset).
- AutoARIMA in the weekly case (Ets drops the seasonality when m > 24) and
  AutoETS per series in the quarterly multi-series case.
- Per-series and per-fold errors, to aggregate in other ways.

Usage:
    python -I study.py              # all cases
    python -I study.py vic h2o      # cases whose name contains any of the words
    python -I study.py --extra      # bdg2 sites, m4 subsets, classification
    python -I study.py --horizon    # same test period with a horizon 4 times shorter
    python -I study.py --stats      # only the statistical models, replaced in the CSV files
    python -I study.py --snippet    # the ML block of the finder snippets, as written
    python -I study.py --d4         # the cases of cases_d4.py (more datasets per type of data)
    python -I study.py --arima      # AutoARIMA (also with exog) in daily, monthly and quarterly series
    python -I study.py --tune       # LightGBM with 20 trials of search on validation
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("BENCH_OUT", "results")
sys.path.insert(0, str(Path(__file__).parent))  # `python -I` does not add it
# Source of the finder page: the settings per frequency of the snippets
FINDER = Path(__file__).resolve().parents[2] / "tools" / "docs" / "forecaster_finder"

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import cases as B  # noqa: E402
from skforecast.datasets import fetch_dataset
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
OUT = B.OUT
CHRONOS = {
    "Chronos-2": "autogluon/chronos-2-small",
    "Chronos-2 base": "amazon/chronos-2",
    "Chronos-2 synth": "autogluon/chronos-2-synth",
}
_PIPELINES = {}


def family(model):
    if model.startswith("Chronos"):
        return "Foundation"
    if model.startswith("Auto"):
        return "Statistical"
    if model.startswith("Seasonal"):
        return "Baseline"
    return "ML"


# ==============================================================================
# Models
# ==============================================================================
def strong_settings(case, n_train):
    """Lags and rolling windows of LightGBM+, by seasonal period of the case."""
    s = case.season
    if s == 24:
        lags = list(range(1, 25)) + [48, 72, 96, 120, 144, 168, 336]
        windows = [24, 168]
    elif s == 7:
        lags = list(range(1, 29))
        if n_train >= 2 * 364 + 28:
            lags.append(364)
        windows = [7, 28]
    elif s == 52:
        lags = list(range(1, 9)) + [52]
        windows = [4, 13]
    elif s == 12:
        lags = list(range(1, 13)) + ([24] if n_train >= 72 else [])
        windows = [3, 12]
    else:
        lags = list(range(1, 5)) + ([8] if n_train >= 40 else [])
        windows = [4]
    return lags, windows


def lgbm_strong(n_rows):
    return LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=31 if n_rows >= 1000 else 15,
        min_child_samples=5 if n_rows < 500 else 20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=123, verbose=-1,
    )


def calendar(case, onehot=False):
    if not case.calendar:
        return None
    return CalendarFeatures(features=case.calendar, encoding="onehot" if onehot else "cyclical")


def make_ml(case, kind, diff, use_exog, multi, n_train, n_series):
    """Forecaster of a variant: 'lgbm', 'ridge', 'strong' or 'snippet'."""
    if kind == "strong":
        lags, windows = strong_settings(case, n_train)
        roll = RollingFeatures(["mean"] * len(windows), windows)
        est = lgbm_strong(n_train * n_series)
    else:
        lags = case.lags
        roll = RollingFeatures("mean", case.window) if case.window else None
        est = {"lgbm": B.lgbm(n_train), "ridge": Ridge(),
               "snippet": LGBMRegressor(random_state=123, verbose=-1)}[kind]
    scale = kind != "snippet"
    if multi:
        ridge = kind == "ridge"
        return ForecasterRecursiveMultiSeries(
            estimator=est, lags=lags, differentiation=diff, window_features=roll,
            calendar_features=calendar(case, onehot=ridge) if kind != "snippet" else (
                CalendarFeatures(features=case.calendar) if case.calendar else None),
            encoding=None if ridge else "ordinal",
            transformer_series=StandardScaler() if scale else None,
            transformer_exog=StandardScaler() if (use_exog and scale) else None,
        )
    return ForecasterRecursive(
        estimator=est, lags=lags, differentiation=diff, window_features=roll,
        calendar_features=CalendarFeatures(features=case.calendar) if case.calendar else None,
        transformer_y=StandardScaler() if scale else None,
        transformer_exog=StandardScaler() if (use_exog and scale) else None,
    )


def backtest_ml(forecaster, data, exog, cv, multi, target):
    if multi:
        return backtesting_forecaster_multiseries(
            forecaster, series=data, cv=cv, exog=exog, metric="mean_absolute_error",
            show_progress=False)[1][["level", "pred"]]
    return B.to_long(backtesting_forecaster(
        forecaster, y=data, cv=cv, exog=exog, metric="mean_absolute_error",
        show_progress=False)[1], target)


def chronos(name):
    return ForecasterFoundation(estimator=FoundationModel(CHRONOS[name]))


VARIANTS = [  # label, kind, differentiation
    ("LightGBM", "lgbm", None), ("Ridge", "ridge", None),
    ("LightGBM (diff)", "lgbm", 1), ("Ridge (diff)", "ridge", 1),
    ("LightGBM+", "strong", None), ("LightGBM+ (diff)", "strong", 1),
    ("LightGBM snippet", "snippet", None),
]
ORIGINAL = ["LightGBM", "Ridge", "LightGBM (diff)", "Ridge (diff)"]
CANDIDATES = ORIGINAL + ["LightGBM+", "LightGBM+ (diff)"]


def validation_scores(case, data, exog, n_train, multi, target, use_exog, n_series, labels):
    """
    MAE of each variant on a validation period taken from the end of the
    training data (frozen model, up to 3 folds). Returns ({label: score}, n_val).
    """
    train = data.iloc[:n_train]
    x = exog.iloc[:n_train] if use_exog else None
    scores = {}
    for folds in (min(case.n_folds, 3), 2, 1):
        n_val = folds * case.steps
        if n_train - n_val < 3 * case.season:
            continue
        cv = TimeSeriesFold(steps=case.steps, initial_train_size=n_train - n_val,
                            refit=False, verbose=False)
        for label, kind, diff in VARIANTS:
            if label not in labels:
                continue
            try:
                p = backtest_ml(make_ml(case, kind, diff, use_exog, multi, n_train - n_val,
                                        n_series), train, x, cv, multi, target)
                if multi:
                    err = [np.abs(train.loc[g.index, lev] - g["pred"]).mean() /
                           train[lev].iloc[:n_train - n_val].diff().abs().mean()
                           for lev, g in p.groupby("level")]
                    scores[label] = float(np.mean(err))
                else:
                    scores[label] = float(np.abs(train.loc[p.index] - p["pred"]).mean())
            except Exception:
                continue
        if scores:
            return scores, n_val
    return scores, 0


def run_models(case, series, exog, n_train, full):
    """Every model of the case. Returns {model: (pred_long, seconds, note)}."""
    def fold_cv(refit):
        return TimeSeriesFold(steps=case.steps, initial_train_size=n_train, refit=refit,
                              fixed_train_size=False, verbose=False)
    past = case.exog_mode == "past"
    multi = series.shape[1] > 1 and not past
    target = series.columns[0]
    y = series[target]
    data = series if multi else y
    n_series = series.shape[1] if multi else 1
    use_exog = case.exog_mode == "future" and exog is not None
    X = exog if use_exog else None
    results = {}

    def timed(name, fn, note=""):
        start = time.perf_counter()
        try:
            results[name] = (fn(), time.perf_counter() - start, note)
        except Exception as error:
            print(f"    {name:<40} FAILED: {type(error).__name__}: {str(error)[:120]}", flush=True)
            return
        print(f"    {name:<40} {results[name][1]:7.1f} s", flush=True)

    def naive():
        cols = [target] if past else series.columns
        return pd.concat([
            B.to_long(backtesting_forecaster(
                ForecasterEquivalentDate(offset=case.season), y=series[col], cv=fold_cv(False),
                metric="mean_absolute_error", show_progress=False)[1], col)
            for col in cols])
    timed("Seasonal naive", naive)

    if past:
        drivers = series if exog is None else pd.concat([series, exog], axis=1)
        for refit in (False, True):
            tag = " [refit]" if refit else ""
            timed("LightGBM (no exog)" + tag, lambda: backtest_ml(
                make_ml(case, "lgbm", None, False, False, n_train, 1), y, None,
                fold_cv(refit), False, target))
            timed("LightGBM drivers (DirectMultiVariate)" + tag, lambda: B.to_long(
                backtesting_forecaster_multiseries(
                    B.ForecasterDirectMultiVariate(
                        estimator=B.lgbm(n_train), level=target, steps=case.steps,
                        lags=case.lags,
                        window_features=RollingFeatures("mean", case.window) if case.window else None,
                        calendar_features=CalendarFeatures(features=case.calendar),
                        transformer_series=StandardScaler(),
                    ),
                    series=drivers, cv=fold_cv(refit), metric="mean_absolute_error",
                    show_progress=False)[1]))
        covariates = series.drop(columns=target) if exog is None else exog

        def chronos_loop(name, with_covariates):
            fc = chronos(name)
            preds = []
            for origin in range(n_train, len(y), case.steps):
                fc.fit(series=y.iloc[:origin],
                       exog=covariates.iloc[:origin] if with_covariates else None)
                p = fc.predict(steps=min(case.steps, len(y) - origin))  # No future exog
                preds.append(B.to_long(p).assign(level=target))
            return pd.concat(preds)
        for name in CHRONOS:
            timed(f"{name} (no exog)", lambda: B.to_long(backtesting_foundation(
                chronos(name), series=y, cv=fold_cv(False), metric="mean_absolute_error",
                show_progress=False)[1]).assign(level=target))
            timed(f"{name} (no exog, own loop)", lambda: chronos_loop(name, False))
            timed(f"{name} (past-only covariates)", lambda: chronos_loop(name, True))
        return results

    variants = [v for v in VARIANTS if not (multi and v[0] == "Ridge (diff)")]
    for label, kind, diff in variants:
        for refit in (False, True):
            if kind in ("strong",) and not refit:
                continue
            timed(label + (" [refit]" if refit else ""), lambda: backtest_ml(
                make_ml(case, kind, diff, use_exog, multi, n_train, n_series), data, X,
                fold_cv(refit), multi, target))

    # ML variant selected on validation, then refitted in every fold of the test
    labels = [v[0] for v in variants]
    for name, pool in [("ML selected on validation (4)", [c for c in ORIGINAL if c in labels]),
                       ("ML selected on validation (6)", [c for c in CANDIDATES if c in labels])]:
        start = time.perf_counter()
        scores, n_val = validation_scores(case, data, exog, n_train, multi, target, use_exog,
                                          n_series, pool)
        scores = {k: v for k, v in scores.items() if f"{k} [refit]" in results}
        if not scores:
            print(f"    {name:<40} no validation period available", flush=True)
            continue
        pick = min(scores, key=scores.get)
        secs = time.perf_counter() - start + results[f"{pick} [refit]"][1]
        results[name] = (results[f"{pick} [refit]"][0], secs, f"{pick}; n_val={n_val}")
        print(f"    {name:<40} {secs:7.1f} s  -> {pick}", flush=True)

    for name in (CHRONOS if full else ["Chronos-2"]):
        timed(name, lambda: B.to_long(backtesting_foundation(
            chronos(name), series=data, cv=fold_cv(False), exog=X,
            metric="mean_absolute_error", show_progress=False)[1]).assign(
                level=lambda d: d["level"] if multi else target))

    stats_models(case, series, fold_cv(False), multi, full, timed)
    return results


def stats_models(case, series, cv, multi, full, timed):
    """AutoETS and AutoARIMA of the case (`backtesting_stats` always refits them)."""
    target = series.columns[0]
    y = series[target]
    stats = list(case.stats)
    if case.season == 52 and "arima" not in stats and full:
        stats.append("arima")  # Ets drops the seasonality when m > 24
    for stat in stats:
        est = (Ets(m=case.season) if stat == "ets"
               else Arima(order=None, seasonal_order=None, m=case.season))
        label = "AutoETS" if stat == "ets" else "AutoARIMA"
        timed(label, lambda est=est: B.to_long(backtesting_stats(
            ForecasterStats(estimator=est), y=y, cv=cv,
            metric="mean_absolute_error", show_progress=False)[1], target))
    if multi and case.season == 4:
        timed("AutoETS", lambda: pd.concat([
            B.to_long(backtesting_stats(
                ForecasterStats(estimator=Ets(m=case.season)), y=series[col],
                cv=cv, metric="mean_absolute_error", show_progress=False,
                n_jobs=1)[1], col)
            for col in series.columns]), note="one model per series")


def run_snippet():
    """
    Measure the ML block of the finder snippets as finder.py writes it:
    `LGBMRegressor(min_child_samples=5)`, the lags, rolling mean and calendar
    features of the frequency, series scaled only in the multi-series
    forecaster, refitted in every fold. Adds the model 'LightGBM snippet v2
    [refit]' to the CSV files of the main run.
    """
    import ast

    sys.path.insert(0, str(FINDER))
    import finder as tree
    freq_of = {24: "hourly", 7: "daily", 52: "weekly", 12: "monthly", 4: "quarterly"}
    label = "LightGBM snippet v2 [refit]"
    for case in B.CASES:
        if case.exog_mode == "past":
            continue
        f = tree.FREQ[freq_of[case.season]]
        print(f"\n=== snippet {case.name}", flush=True)
        series, exog = case.loader()
        test_start = len(series) - case.steps * case.n_folds
        multi = series.shape[1] > 1
        target = series.columns[0]
        levels = list(series.columns)
        use_exog = case.exog_mode == "future" and exog is not None
        summary, details = [], []
        for history in case.histories:
            start = 0 if history is None else test_start - history
            n_train = test_start - start
            s = series.iloc[start:]
            x = exog.iloc[start:] if use_exog else None
            cv = TimeSeriesFold(steps=case.steps, initial_train_size=n_train, refit=True,
                                fixed_train_size=False, verbose=False)
            kwargs = dict(
                estimator=LGBMRegressor(min_child_samples=5, random_state=123, verbose=-1),
                lags=ast.literal_eval(f["lags"]),
                window_features=RollingFeatures("mean", f["window"]) if f["window"] else None,
                # m4_hourly has artificial dates: no calendar, as in the main run
                calendar_features=(CalendarFeatures(features=ast.literal_eval(f["calendar"]))
                                   if case.calendar else None),
            )
            forecaster = (ForecasterRecursiveMultiSeries(transformer_series=StandardScaler(), **kwargs)
                          if multi else ForecasterRecursive(**kwargs))
            res = {}
            t0 = time.perf_counter()
            res["Seasonal naive"] = (pd.concat([
                B.to_long(backtesting_forecaster(
                    ForecasterEquivalentDate(offset=case.season), y=s[col],
                    cv=TimeSeriesFold(steps=case.steps, initial_train_size=n_train,
                                      refit=False, verbose=False),
                    metric="mean_absolute_error", show_progress=False)[1], col)
                for col in s.columns]), 0.0, "")
            t0 = time.perf_counter()
            pred = backtest_ml(forecaster, s if multi else s[target], x, cv, multi, target)
            res[label] = (pred, time.perf_counter() - t0, "as written by finder.py")
            t, d = tables(res, series, test_start, levels, case.steps)
            for frame in (t, d):
                frame.insert(0, "history_obs", n_train)
                frame.insert(0, "history", "full" if history is None else str(history))
                frame.insert(0, "case", case.name)
            t["rel"] = t["MASE"] / t.loc[t.model == "Seasonal naive", "MASE"].iloc[0]
            if multi:
                d = d.groupby(["case", "history", "history_obs", "model", "level"],
                              as_index=False).agg(mae=("mae", "mean"), scale=("scale", "first"))
            summary.append(t[t.model == label])
            details.append(d[d.model == label])
            print(f"  history {n_train}: rel {t.loc[t.model == label, 'rel'].iloc[0]:.3f}", flush=True)
        new = pd.concat(summary).assign(n_series=len(levels))
        for path, rows in [(OUT / f"{case.name}.csv", new),
                           (OUT / "detail" / f"{case.name}.csv", pd.concat(details).round(6))]:
            old = pd.read_csv(path)
            pd.concat([old[old["model"] != label], rows[old.columns]]).to_csv(path, index=False)


def snippet_forecaster(case, multi):
    """ML forecaster of the finder snippets, as finder.py writes it."""
    import ast

    sys.path.insert(0, str(FINDER))
    import finder as tree
    f = tree.FREQ[{24: "hourly", 7: "daily", 52: "weekly", 12: "monthly", 4: "quarterly"}[case.season]]
    kwargs = dict(
        estimator=LGBMRegressor(min_child_samples=5, random_state=123, verbose=-1),
        lags=ast.literal_eval(f["lags"]),
        window_features=RollingFeatures("mean", f["window"]) if f["window"] else None,
        # Series with artificial dates have no calendar features, as in the main run
        calendar_features=(CalendarFeatures(features=ast.literal_eval(f["calendar"]))
                           if case.calendar else None),
    )
    if multi:
        return ForecasterRecursiveMultiSeries(transformer_series=StandardScaler(), **kwargs)
    return ForecasterRecursive(**kwargs)


def run_d4(words):
    """
    Cases of cases_d4.py (decision D4: six or more datasets per type of data),
    with the models that the rules use: seasonal naive, LightGBM refitted,
    LightGBM+, the LightGBM of the snippets, Chronos-2 small, and AutoETS and
    AutoARIMA in the series of one season of 24 or less. Full history only.
    Writes d4_<case>.csv, with the dataset and the kind of known variables.
    """
    from cases_d4 import D4_CASES
    for case, dataset, exog_kind in D4_CASES:
        if words and not any(w in case.name for w in words):
            continue
        print(f"\n=== d4 {case.name}: {case.question}", flush=True)
        series, exog = case.loader()
        test_start = len(series) - case.steps * case.n_folds
        multi = series.shape[1] > 1
        target = series.columns[0]
        levels = list(series.columns)
        use_exog = case.exog_mode == "future"
        X = exog if use_exog else None
        data = series if multi else series[target]

        def cv(refit):
            return TimeSeriesFold(steps=case.steps, initial_train_size=test_start, refit=refit,
                                  fixed_train_size=False, verbose=False)
        res = {}

        def timed(name, fn, note=""):
            t0 = time.perf_counter()
            try:
                res[name] = (fn(), time.perf_counter() - t0, note)
            except Exception as error:
                print(f"    {name:<32} FAILED: {type(error).__name__}: {str(error)[:120]}", flush=True)
                return
            print(f"    {name:<32} {res[name][1]:7.1f} s", flush=True)
        timed("Seasonal naive", lambda: pd.concat([
            B.to_long(backtesting_forecaster(
                ForecasterEquivalentDate(offset=case.season), y=series[col], cv=cv(False),
                metric="mean_absolute_error", show_progress=False)[1], col)
            for col in series.columns]))
        for label, kind in [("LightGBM [refit]", "lgbm"), ("LightGBM+ [refit]", "strong")]:
            timed(label, lambda kind=kind: backtest_ml(
                make_ml(case, kind, None, use_exog, multi, test_start, len(levels)), data, X,
                cv(True), multi, target))
        timed("LightGBM snippet v2 [refit]", lambda: backtest_ml(
            snippet_forecaster(case, multi), data, X, cv(True), multi, target))
        timed("Chronos-2", lambda: B.to_long(backtesting_foundation(
            chronos("Chronos-2"), series=data, cv=cv(False), exog=X,
            metric="mean_absolute_error", show_progress=False)[1]).assign(
                level=lambda d: d["level"] if multi else target))
        if not multi:
            stats_models(case, series, cv(False), multi, True, timed)
        t, _ = tables(res, series, test_start, levels, case.steps)
        t.insert(0, "history_obs", test_start)
        t.insert(0, "history", "full")
        t.insert(0, "case", case.name)
        t["rel"] = t["MASE"] / t.loc[t.model == "Seasonal naive", "MASE"].iloc[0]
        t["n_series"] = len(levels)
        t["dataset"] = dataset
        t["exog_kind"] = exog_kind
        t["cell"] = ("several series" if multi else
                     "one series, known in advance" if use_exog else
                     "one series, nothing known in advance")
        t.to_csv(OUT / f"d4_{case.name}.csv", index=False)
        print(t[["model", "rel", "seconds"]].round(3).to_string(index=False), flush=True)

    # AutoARIMA per series in the quarterly multi-series case of the main run
    if not words or any(w in "tourism_arima" for w in words):
        case = next(c for c in B.CASES if c.name == "australia_tourism_quarterly")
        print("\n=== d4 australia_tourism_quarterly: AutoARIMA per series", flush=True)
        series, _ = case.loader()
        test_start = len(series) - case.steps * case.n_folds
        fold = TimeSeriesFold(steps=case.steps, initial_train_size=test_start, refit=False,
                              fixed_train_size=False, verbose=False)
        t0 = time.perf_counter()
        pred = pd.concat([
            B.to_long(backtesting_stats(
                ForecasterStats(estimator=Arima(order=None, seasonal_order=None, m=case.season)),
                y=series[col], cv=fold, metric="mean_absolute_error", show_progress=False,
                n_jobs=1)[1], col)
            for col in series.columns])
        res = {"AutoARIMA": (pred, time.perf_counter() - t0, "one model per series")}
        t, d = tables(res, series, test_start, list(series.columns), case.steps)
        for frame in (t, d):
            frame.insert(0, "history_obs", test_start)
            frame.insert(0, "history", "full")
            frame.insert(0, "case", case.name)
        d = d.groupby(["case", "history", "history_obs", "model", "level"],
                      as_index=False).agg(mae=("mae", "mean"), scale=("scale", "first"))
        old = pd.read_csv(OUT / f"{case.name}.csv")
        t["rel"] = t["MASE"] / old.loc[old.model == "Seasonal naive", "MASE"].iloc[0]
        t["n_series"] = series.shape[1]
        pd.concat([old[old.model != "AutoARIMA"], t[old.columns]]).to_csv(
            OUT / f"{case.name}.csv", index=False)
        old_d = pd.read_csv(OUT / "detail" / f"{case.name}.csv")
        pd.concat([old_d[old_d.model != "AutoARIMA"], d.round(6)[old_d.columns]]).to_csv(
            OUT / "detail" / f"{case.name}.csv", index=False)
        print(t[["model", "rel", "seconds"]].round(3).to_string(index=False), flush=True)


def run_arima():
    """
    AutoARIMA in every single-series case with a season of 12 or less (daily,
    monthly, quarterly), full history: without exog ('AutoARIMA') where it is
    not measured yet, and with the variables known in advance where there are
    ('AutoARIMA (exog)'), as on the home page of the documentation. To decide
    which statistical model the finder shows, and with which frequencies.
    """
    from cases_d4 import D4_CASES
    todo = [(c, OUT / f"{c.name}.csv") for c in B.CASES if c.exog_mode != "past"]
    todo += [(c, OUT / f"d4_{c.name}.csv") for c, _, _ in D4_CASES]
    for case, path in todo:
        if case.season > 12:
            continue
        series, exog = case.loader()
        if series.shape[1] > 1:
            continue
        print(f"\n=== arima {case.name}", flush=True)
        test_start = len(series) - case.steps * case.n_folds
        target = series.columns[0]
        y = series[target]
        fold = TimeSeriesFold(steps=case.steps, initial_train_size=test_start, refit=False,
                              fixed_train_size=False, verbose=False)
        old = pd.read_csv(path)
        full = old[old.history == "full"]
        res = {}
        variants = []
        if not (full.model == "AutoARIMA").any():
            variants.append(("AutoARIMA", None))
        if case.exog_mode == "future" and exog is not None:
            variants.append(("AutoARIMA (exog)", exog))
        for label, x in variants:
            t0 = time.perf_counter()
            try:
                pred = B.to_long(backtesting_stats(
                    ForecasterStats(estimator=Arima(order=None, seasonal_order=None, m=case.season)),
                    y=y, exog=x, cv=fold, metric="mean_absolute_error", show_progress=False)[1],
                    target)
            except Exception as error:
                print(f"    {label:<20} FAILED: {type(error).__name__}: {str(error)[:120]}", flush=True)
                continue
            res[label] = (pred, time.perf_counter() - t0, "")
            print(f"    {label:<20} {res[label][1]:7.1f} s", flush=True)
        if not res:
            continue
        t, _ = tables(res, series, test_start, [target], case.steps)
        naive = full.loc[full.model == "Seasonal naive"].iloc[0]
        t.insert(0, "history_obs", test_start)
        t.insert(0, "history", "full")
        t.insert(0, "case", case.name)
        t["rel"] = t["MASE"] / naive["MASE"]
        for c in old.columns:
            if c not in t:
                t[c] = naive[c]
        pd.concat([old[~((old.history == "full") & old.model.isin(res))], t[old.columns]]).to_csv(
            path, index=False)
        print(t[["model", "rel", "seconds"]].round(3).to_string(index=False), flush=True)


def run_tune(words):
    """
    LightGBM with a small, fixed tuning budget, to answer how often the ML
    forecaster reaches Chronos-2 once it is worked on: 20 trials of Bayesian
    search (10 without differentiation and 10 with `differentiation=1`) over
    the lags and six hyperparameters, scored on a validation period taken from
    the end of the training data (never on the test period). The best
    configuration is then backtested on the test period, refitted in every
    fold. Full history of every case. Adds the model 'LightGBM tuned [refit]'.
    """
    from cases_d4 import D4_CASES
    from skforecast.model_selection import (
        bayesian_search_forecaster,
        bayesian_search_forecaster_multiseries,
    )
    todo = [(c, OUT / f"{c.name}.csv") for c in B.CASES]
    todo += [(c, OUT / f"d4_{c.name}.csv") for c, _, _ in D4_CASES]
    label = "LightGBM tuned [refit]"
    for case, path in todo:
        if words and not any(w in case.name for w in words):
            continue
        print(f"\n=== tune {case.name}", flush=True)
        series, exog = case.loader()
        past = case.exog_mode == "past"
        if past:
            series = series[[series.columns[0]]]  # Only the target, no extra variables
        test_start = len(series) - case.steps * case.n_folds
        multi = series.shape[1] > 1
        target = series.columns[0]
        levels = list(series.columns)
        use_exog = case.exog_mode == "future" and exog is not None
        X = exog if use_exog else None
        data = series if multi else series[target]
        n_val = 0
        for folds in (min(case.n_folds, 3), 2, 1):
            if test_start - folds * case.steps >= 3 * case.season:
                n_val = folds * case.steps
                break
        if not n_val:
            print("    no validation period available", flush=True)
            continue
        strong_lags, _ = strong_settings(case, test_start - n_val)

        def search_space(trial):
            return {
                "lags": trial.suggest_categorical("lags", [case.lags, strong_lags]),
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
        cv_val = TimeSeriesFold(steps=case.steps, initial_train_size=test_start - n_val,
                                refit=False, verbose=False)
        t0 = time.perf_counter()
        best = None
        for diff in (None, 1):
            forecaster = make_ml(case, "lgbm", diff, use_exog, multi, test_start, len(levels))
            try:
                if multi:
                    res, _ = bayesian_search_forecaster_multiseries(
                        forecaster, series=data.iloc[:test_start],
                        exog=X.iloc[:test_start] if use_exog else None, cv=cv_val,
                        search_space=search_space, metric="mean_absolute_scaled_error",
                        aggregate_metric="average", n_trials=10, random_state=123,
                        return_best=False, show_progress=False, suppress_warnings=True)
                else:
                    res, _ = bayesian_search_forecaster(
                        forecaster, y=data.iloc[:test_start],
                        exog=X.iloc[:test_start] if use_exog else None, cv=cv_val,
                        search_space=search_space, metric="mean_absolute_scaled_error",
                        n_trials=10, random_state=123, return_best=False,
                        show_progress=False, suppress_warnings=True)
            except Exception as error:
                print(f"    diff={diff} FAILED: {type(error).__name__}: {str(error)[:120]}", flush=True)
                continue
            row = res.iloc[0]
            score = float(row[[c for c in res.columns if "mean_absolute_scaled_error" in c][0]])
            if best is None or score < best[0]:
                best = (score, diff, row["lags"], dict(row["params"]))
        if best is None:
            continue
        score, diff, lags, params = best
        forecaster = make_ml(case, "lgbm", diff, use_exog, multi, test_start, len(levels))
        forecaster.set_params(params)
        forecaster.set_lags(lags)
        search_seconds = time.perf_counter() - t0
        cv = TimeSeriesFold(steps=case.steps, initial_train_size=test_start, refit=True,
                            fixed_train_size=False, verbose=False)
        pred = backtest_ml(forecaster, data, X, cv, multi, target)
        note = (f"diff={diff}; lags={'strong' if list(np.atleast_1d(lags)) == list(strong_lags) else 'case'}; "
                f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()} }; "
                f"n_val={n_val}; validation MASE {score:.3f}")
        res = {label: (pred, time.perf_counter() - t0, note)}
        t, _ = tables(res, series, test_start, levels, case.steps)
        old = pd.read_csv(path)
        full = old[old.history == "full"]
        naive = full.loc[full.model == "Seasonal naive"].iloc[0]
        t.insert(0, "history_obs", test_start)
        t.insert(0, "history", "full")
        t.insert(0, "case", case.name)
        t["rel"] = t["MASE"] / naive["MASE"]
        for c in old.columns:
            if c not in t:
                t[c] = naive[c]
        pd.concat([old[old.model != label], t[old.columns]]).to_csv(path, index=False)
        ref = {m: float(full.loc[full.model == m, "rel"].iloc[0]) for m in
               ("LightGBM [refit]", "LightGBM (no exog) [refit]", "Chronos-2", "Chronos-2 (no exog)")
               if (full.model == m).any()}
        print(f"    tuned rel {t['rel'].iloc[0]:.3f} ({search_seconds:.0f} s of search) | {ref} | {note[:110]}",
              flush=True)


def run_stats():
    """
    Run again only the statistical models and replace their rows in the CSV
    files of the main run (the rest of the models are not affected by a change
    in `skforecast.stats`). The previous values are kept in
    `stats_previous.csv` the first time.
    """
    previous = []
    for case in B.CASES:
        multi_ets = case.season == 4 and case.exog_mode != "past"
        if not case.stats and case.season != 52 and not multi_ets:
            continue
        print(f"\n=== stats {case.name}", flush=True)
        series, exog = case.loader()
        test_start = len(series) - case.steps * case.n_folds
        multi = series.shape[1] > 1
        levels = list(series.columns)
        summary, details = [], []
        for history in case.histories:
            start = 0 if history is None else test_start - history
            n_train = test_start - start
            s = series.iloc[start:]
            cv = TimeSeriesFold(steps=case.steps, initial_train_size=n_train, refit=False,
                                fixed_train_size=False, verbose=False)
            res = {}

            def timed(name, fn, note=""):
                t0 = time.perf_counter()
                res[name] = (fn(), time.perf_counter() - t0, note)
                print(f"    {name:<40} {res[name][1]:7.1f} s", flush=True)
            timed("Seasonal naive", lambda: pd.concat([
                B.to_long(backtesting_forecaster(
                    ForecasterEquivalentDate(offset=case.season), y=s[col], cv=cv,
                    metric="mean_absolute_error", show_progress=False)[1], col)
                for col in s.columns]))
            stats_models(case, s, cv, multi, history is None, timed)
            t, d = tables(res, series, test_start, levels, case.steps)
            for frame in (t, d):
                frame.insert(0, "history_obs", n_train)
                frame.insert(0, "history", "full" if history is None else str(history))
                frame.insert(0, "case", case.name)
            t["rel"] = t["MASE"] / t.loc[t.model == "Seasonal naive", "MASE"].iloc[0]
            if multi:
                d = d.groupby(["case", "history", "history_obs", "model", "level"],
                              as_index=False).agg(mae=("mae", "mean"), scale=("scale", "first"))
            summary.append(t[t.family == "Statistical"])
            details.append(d[d.model.str.startswith("Auto")])
        new = pd.concat(summary).assign(n_series=len(levels))
        for path, rows in [(OUT / f"{case.name}.csv", new),
                           (OUT / "detail" / f"{case.name}.csv", pd.concat(details).round(6))]:
            old = pd.read_csv(path)
            is_stat = old["model"].str.startswith("Auto")
            if path.parent == OUT:
                previous.append(old[is_stat])
            pd.concat([old[~is_stat], rows[old.columns]]).to_csv(path, index=False)
        print(new[["history_obs", "model", "rel", "seconds"]].round(3).to_string(index=False), flush=True)
    if not (OUT / "stats_previous.csv").exists():
        pd.concat(previous).to_csv(OUT / "stats_previous.csv", index=False)


def tables(results, series_full, test_start, levels, steps):
    """Summary (MASE per model) and detail (MAE per model, series and fold)."""
    scale = series_full.iloc[:test_start].diff().abs().mean()
    pos = pd.Series(np.arange(len(series_full)), index=series_full.index)
    rows, detail = [], []
    for model, (pred, secs, note) in results.items():
        per_series = []
        for level in levels:
            p = pred.loc[pred["level"] == level, "pred"]
            if p.empty:
                continue
            err = np.abs(series_full.loc[p.index, level].to_numpy() - p.to_numpy())
            per_series.append(err.mean() / scale[level])
            fold = ((pos.loc[p.index].to_numpy() - test_start) // steps)
            for f in np.unique(fold):
                detail.append({"model": model, "level": level, "fold": int(f),
                               "mae": float(err[fold == f].mean()),
                               "scale": float(scale[level])})
        rows.append({"model": model, "family": family(model),
                     "MASE": float(np.mean(per_series)), "seconds": secs, "note": note})
    return pd.DataFrame(rows), pd.DataFrame(detail)


def run_case(case, tag=None, loader=None):
    name = tag or case.name
    print(f"\n=== {name}: {case.question}", flush=True)
    series, exog = (loader or case.loader)()
    test_start = len(series) - case.steps * case.n_folds
    levels = [series.columns[0]] if case.exog_mode == "past" else list(series.columns)
    summary, details = [], []
    for history in case.histories:
        start = 0 if history is None else test_start - history
        n_train = test_start - start
        print(f"  history: {n_train} obs per series", flush=True)
        res = run_models(case, series.iloc[start:], exog.iloc[start:] if exog is not None else None,
                         n_train, full=history is None)
        t, d = tables(res, series, test_start, levels, case.steps)
        for frame in (t, d):
            frame.insert(0, "history_obs", n_train)
            frame.insert(0, "history", "full" if history is None else str(history))
            frame.insert(0, "case", name)
        naive = t.loc[t.model == "Seasonal naive", "MASE"].iloc[0]
        t["rel"] = t["MASE"] / naive
        summary.append(t)
        if len(levels) > 1:  # Several series: one row per series, folds averaged
            d = d.groupby(["case", "history", "history_obs", "model", "level"],
                          as_index=False).agg(mae=("mae", "mean"), scale=("scale", "first"))
        details.append(d)
        print(t[["model", "rel", "seconds", "note"]].round(3).to_string(index=False), flush=True)
    out = pd.concat(summary)
    out["n_series"] = len(levels)
    out.to_csv(OUT / f"{name}.csv", index=False)
    if tag is None:  # The extra checks keep only the summary
        (OUT / "detail").mkdir(exist_ok=True)
        pd.concat(details).round(6).to_csv(OUT / "detail" / f"{name}.csv", index=False)
    return out


# ==============================================================================
# Extra checks: selection of series in bdg2 and m4_hourly, classification
# ==============================================================================
def bdg2_loader(site, n=40, seed=None):
    def load():
        d = fetch_dataset("bdg2_daily", verbose=False)
        d = d.loc[d["site_id"] == site]
        s = d.pivot_table(index=d.index, columns="building_id", values="meter_reading").asfreq("D")
        s = s.loc[:, (s.isna().sum() == 0) & (s.std() > 0) & ((s == 0).mean() < 0.02)]
        cols = list(s.columns)
        if seed is not None:
            cols = list(np.random.default_rng(seed).permutation(cols))
        s = s[cols[:n]]
        temp = d.groupby(level=0)["airTemperature"].mean().asfreq("D").interpolate().bfill().ffill()
        s.index.name = None
        s.columns.name = None
        return s.astype(float), temp.to_frame("temp").astype(float)
    return load


def m4_loader(length, seed):
    def load():
        d = fetch_dataset("m4_hourly", verbose=False)
        lengths = d.groupby("series_id").size()
        ids = sorted(lengths[lengths == length].index)
        ids = list(np.random.default_rng(seed).permutation(ids))[:50]
        idx = pd.date_range("2020-01-06", periods=length, freq="h")
        s = pd.DataFrame({i: d.loc[d["series_id"] == i, "value"].to_numpy() for i in ids}, index=idx)
        return s.astype(float), None
    return load


def run_extra():
    from dataclasses import replace
    bdg2 = replace(next(c for c in B.CASES if c.name == "bdg2_buildings_energy"), histories=[None])
    d = fetch_dataset("bdg2_daily", verbose=False)
    sizes = {}
    for site in sorted(d["site_id"].unique()):
        s = d.loc[d["site_id"] == site].pivot_table(
            index=d.loc[d["site_id"] == site].index, columns="building_id", values="meter_reading").asfreq("D")
        sizes[site] = int(((s.isna().sum() == 0) & (s.std() > 0) & ((s == 0).mean() < 0.02)).sum())
    print("bdg2 buildings with complete data per site:", sizes, flush=True)
    for site in [s for s, n in sizes.items() if n >= 40]:
        run_case(bdg2, tag=f"extra_bdg2_{site}_first40", loader=bdg2_loader(site))
    run_case(bdg2, tag="extra_bdg2_Fox_random40", loader=bdg2_loader("Fox", seed=1))
    m4 = next(c for c in B.CASES if c.name == "m4_hourly_50_series")
    run_case(m4, tag="extra_m4_hourly_1008_random50", loader=m4_loader(1008, 1))
    run_case(m4, tag="extra_m4_hourly_748_random50", loader=m4_loader(748, 1))
    run_classification()


def run_horizon():
    """
    Same test period with a horizon four times shorter (and four times more
    folds): does the order between families depend on how far ahead the
    forecast goes? ML is refitted every 4 folds, the same dates as in the main
    run. Full history, one LightGBM, LightGBM+ and Chronos-2 small.
    """
    from dataclasses import replace
    rows = []
    for case in B.CASES:
        if case.exog_mode == "past" or case.steps < 4:
            continue
        short = replace(case, steps=case.steps // 4, n_folds=case.n_folds * 4, histories=[None])
        print(f"\n=== horizon {case.name}: {case.steps} -> {short.steps} steps", flush=True)
        series, exog = case.loader()
        n_train = len(series) - case.steps * case.n_folds
        multi = series.shape[1] > 1
        target = series.columns[0]
        data = series if multi else series[target]
        use_exog = case.exog_mode == "future" and exog is not None
        X = exog if use_exog else None
        res = {}
        for label, c, refit in [("long", case, True), ("short", short, 4)]:
            def cv(r):
                return TimeSeriesFold(steps=c.steps, initial_train_size=n_train, refit=r,
                                      fixed_train_size=False, verbose=False)
            res[f"Seasonal naive|{label}"] = (pd.concat([
                B.to_long(backtesting_forecaster(
                    ForecasterEquivalentDate(offset=c.season), y=series[col], cv=cv(False),
                    metric="mean_absolute_error", show_progress=False)[1], col)
                for col in series.columns]), 0.0, "")
            for name, kind in [("LightGBM [refit]", "lgbm"), ("LightGBM+ [refit]", "strong")]:
                res[f"{name}|{label}"] = (backtest_ml(
                    make_ml(c, kind, None, use_exog, multi, n_train, series.shape[1]), data, X,
                    cv(refit), multi, target), 0.0, "")
            res[f"Chronos-2|{label}"] = (B.to_long(backtesting_foundation(
                chronos("Chronos-2"), series=data, cv=cv(False), exog=X,
                metric="mean_absolute_error", show_progress=False)[1]).assign(
                    level=lambda d: d["level"] if multi else target), 0.0, "")
        t, _ = tables(res, series, n_train, list(series.columns), case.steps)
        t[["model", "horizon"]] = t["model"].str.split("|", expand=True)
        t["steps"] = t["horizon"].map({"long": case.steps, "short": short.steps})
        naive = t.loc[t.model == "Seasonal naive"].set_index("horizon")["MASE"]
        t["rel"] = t["MASE"] / t["horizon"].map(naive)
        t.insert(0, "case", case.name)
        print(t[["model", "steps", "MASE", "rel"]].round(3).to_string(index=False), flush=True)
        rows.append(t.drop(columns=["seconds", "note", "family"]))
    pd.concat(rows).to_csv(OUT / "extra_horizon.csv", index=False)


def run_classification():
    """Classifier frozen and refitted, against the rule 'same hour yesterday'."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    print("\n=== extra_classification", flush=True)
    d = fetch_dataset("vic_electricity_classification", verbose=False).asfreq("h")
    y = d["Demand"]
    X = d[["Temperature", "Holiday", "Hour_of_day"]].astype(float)
    X["day_of_week"] = X.index.dayofweek.astype(float)
    steps, n_folds = 24, 28
    n_train = len(y) - steps * n_folds
    test = y.iloc[n_train:]
    rows = []
    for label, refit, exog in [("LightGBM classifier", False, X), ("LightGBM classifier [refit]", True, X),
                               ("LightGBM classifier, no exog", False, None)]:
        cv = TimeSeriesFold(steps=steps, initial_train_size=n_train, refit=refit,
                            fixed_train_size=False, verbose=False)
        f = ForecasterRecursiveClassifier(
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           random_state=123, verbose=-1), lags=[1, 2, 3, 24, 168])
        _, p = backtesting_forecaster(f, y=y, cv=cv, exog=exog, show_progress=False,
                                      metric="accuracy_score")
        rows.append({"model": label, "accuracy": accuracy_score(test, p["pred"]),
                     "balanced_accuracy": balanced_accuracy_score(test, p["pred"])})
    for name, lag in [("Same hour yesterday", 24), ("Same hour last week", 168)]:
        pred = [y.iloc[o + h - int(np.ceil((h + 1) / lag)) * lag]
                for o in range(n_train, len(y), steps) for h in range(min(steps, len(y) - o))]
        rows.append({"model": name, "accuracy": accuracy_score(test, pred),
                     "balanced_accuracy": balanced_accuracy_score(test, pred)})
    out = pd.DataFrame(rows)
    out["class_share"] = json.dumps(test.value_counts(normalize=True).round(3).to_dict())
    print(out.round(3).to_string(index=False), flush=True)
    out.to_csv(OUT / "extra_classification.csv", index=False)


if __name__ == "__main__":
    import importlib.metadata as md
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    env = {p: __import__(p).__version__ for p in
           ["skforecast", "numpy", "pandas", "sklearn", "lightgbm", "torch", "numba", "scipy"]}
    env["chronos-forecasting"] = md.version("chronos-forecasting")
    env["python"] = sys.version.split()[0]
    env["chronos_models"] = CHRONOS
    env["git_commit"] = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
    name = ("environment_stats.json" if "--stats" in sys.argv else
            "environment_d4.json" if "--d4" in sys.argv else "environment.json")
    partial = any(flag in sys.argv for flag in ("--extra", "--horizon", "--snippet", "--arima", "--tune"))
    if name != "environment.json" or not partial:
        (OUT / name).write_text(json.dumps(env, indent=2))
    print(env, flush=True)
    if "--extra" in sys.argv:
        run_extra()
    elif "--horizon" in sys.argv:
        run_horizon()
    elif "--stats" in sys.argv:
        run_stats()
    elif "--snippet" in sys.argv:
        run_snippet()
    elif "--d4" in sys.argv:
        run_d4(args)
    elif "--arima" in sys.argv:
        run_arima()
    elif "--tune" in sys.argv:
        run_tune(args)
    else:
        for case in B.CASES:
            if not args or any(w in case.name for w in args):
                run_case(case)
