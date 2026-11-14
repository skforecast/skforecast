"""
Benchmarks of the code paths that depend on pandas internals (INFORME.md,
sections 3.6 and 4), plus the `fit` of the four forecasters that build their
training matrix in numpy.

This script measures one skforecast tree (the one on PYTHONPATH) in the
current process. To compare two trees, use `run_ab.py`, which alternates
processes of each tree.

Usage:
    python bench_hotspots.py --out results.json [--bench all|control|name1,name2]
        [--mode time|memory] [--warmup 2] [--repeat 5] [--expect-tree <root>]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
)
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
)
from skforecast.utils import check_predict_input, expand_index

import common

LAGS = 24


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="h")


def _y(n: int, seed: int = 0, name: str = "y") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(size=n)), index=_index(n), name=name)


def _exog_float(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 3)), index=_index(n),
                        columns=["f1", "f2", "f3"])


def _exog_mixed(n: int, seed: int = 1) -> pd.DataFrame:
    """
    3 float + 1 int + 1 category.
    """
    rng = np.random.default_rng(seed)
    exog = _exog_float(n, seed)
    exog["i1"] = rng.integers(0, 100, size=n)
    exog["c1"] = pd.Categorical(rng.choice(list("abcde"), size=n))
    return exog


def _exog_fit(n: int, seed: int = 1) -> pd.DataFrame:
    """
    3 float + 1 category (the `fit` measures of section 3.6).
    """
    return _exog_mixed(n, seed).drop(columns="i1")


def _exog_int(n: int, k: int = 10, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.integers(0, 100, size=(n, k)), index=_index(n),
                        columns=[f"i{i}" for i in range(k)])


def _exog_wide(n: int, seed: int = 4) -> pd.DataFrame:
    """
    29 float + 1 category: a wide `exog`, whose copy is large.
    """
    rng = np.random.default_rng(seed)
    exog = pd.DataFrame(rng.normal(size=(n, 29)), index=_index(n),
                        columns=[f"f{i}" for i in range(29)])
    exog["c1"] = pd.Categorical(rng.choice(list("abcde"), size=n))
    return exog


def _series_wide(n_series: int, n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = 100 + np.cumsum(rng.normal(size=(n, n_series)), axis=0)
    return pd.DataFrame(values, index=_index(n),
                        columns=[f"s{i}" for i in range(n_series)])


def _future(exog: pd.DataFrame, steps: int, seed: int = 9) -> pd.DataFrame:
    """
    Exogenous values for the `steps` periods after the end of `exog`.
    """
    idx = pd.date_range(exog.index[-1], periods=steps + 1, freq=exog.index.freq)[1:]
    future = exog.iloc[:steps].copy()
    future.index = idx
    return future


# ---------------------------------------------------------------------------
# Benchmarks: each function prepares the data (not timed) and returns the
# callable that is timed. The callable returns an object whose fingerprint
# checks that both trees compute the same thing (None for `fit`).
# ---------------------------------------------------------------------------

def ms_create_train_X_y_block():
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(series=series, exog=exog)


def ms_create_train_X_y_concat():
    series = _series_wide(100, 2000)
    exog = _exog_int(2000)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(series=series, exog=exog)


def ms_create_train_X_y_onehot_dict():
    wide = _series_wide(50, 2000)
    series = {c: wide[c] for c in wide.columns}
    f = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=LAGS, encoding="onehot",
        window_features=RollingFeatures(stats=["mean", "std"], window_sizes=LAGS),
    )
    return lambda: f.create_train_X_y(series=series)


def _ms_fit(encoding):
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000)
    f = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=LAGS, encoding=encoding
    )

    def run():
        f.fit(series=series, exog=exog)
        return None

    return run


def ms_fit_linear():
    return _ms_fit("ordinal")


def ms_fit_linear_encoding_none():
    return _ms_fit(None)


def ms_get_level_row_slices():
    # Private method: if a PR changes it, this benchmark reports an error for
    # one of the two trees.
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=LAGS)
    f.fit(series=series, exog=exog)
    X_train = f.create_train_X_y(series=series, exog=exog)[0]
    return lambda: f._get_level_row_slices(X_train)


def ms_predict_levels():
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=LAGS)
    f.fit(series=series, exog=exog)
    exog_future = _future(exog, 24)
    return lambda: f.predict(steps=24, exog=exog_future)


def recursive_create_train_X_y():
    y, exog = _y(200_000), _exog_mixed(200_000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def recursive_create_train_X_y_float():
    y, exog = _y(200_000), _exog_float(200_000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def direct_create_train_X_y_float():
    y, exog = _y(50_000), _exog_float(50_000)
    f = ForecasterDirect(estimator=LinearRegression(), steps=12, lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def direct_create_train_X_y():
    y, exog = _y(50_000), _exog_mixed(50_000)
    f = ForecasterDirect(estimator=LinearRegression(), steps=12, lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def recursive_fit_linear():
    y, exog = _y(200_000), _exog_float(200_000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)

    def run():
        f.fit(y=y, exog=exog)
        return None

    return run


def direct_multivariate_create_train_X_y():
    series, exog = _series_wide(5, 30_000), _exog_mixed(30_000)
    f = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level="s0", steps=6, lags=LAGS
    )
    return lambda: f.create_train_X_y(series=series, exog=exog)


def _y_class(n: int) -> pd.Series:
    diff = np.diff(_y(n + 1).to_numpy())
    labels = np.where(diff > 0.5, "up", np.where(diff < -0.5, "down", "flat"))
    return pd.Series(list(labels), index=_index(n), name="y")


def classifier_create_train_X_y():
    y, exog = _y_class(100_000), _exog_mixed(100_000)
    f = ForecasterRecursiveClassifier(estimator=LogisticRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def predict_loop():
    y, exog = _y(3000), _exog_float(3000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    f.fit(y=y, exog=exog)
    exog_future = _future(exog, 500)
    exog_one = exog_future.iloc[:1]

    def run():
        for _ in range(200):
            f.predict(steps=1, exog=exog_one)
        return f.predict(steps=500, exog=exog_future)

    return run


def expand_index_check_predict_input():
    y, exog = _y(3000), _exog_float(3000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    f.fit(y=y, exog=exog)
    exog_future = _future(exog, 10)
    last_window = f.last_window_

    def run():
        for _ in range(2000):
            expand_index(index=last_window.index, steps=10)
            check_predict_input(
                forecaster_name=type(f).__name__,
                steps=10,
                is_fitted=True,
                exog_in_=True,
                index_type_=f.index_type_,
                index_freq_=f.index_freq_,
                window_size=f.window_size,
                last_window=last_window,
                exog=exog_future,
                exog_names_in_=f.exog_names_in_,
            )
        return expand_index(index=last_window.index, steps=10)

    return run


def _backtesting(n, initial, steps, refit):
    y = _y(n)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    cv = TimeSeriesFold(steps=steps, initial_train_size=initial, refit=refit)
    return lambda: backtesting_forecaster(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error",
        n_jobs=1, show_progress=False,
    )


def backtesting_recursive():
    return _backtesting(3000, 2000, 1, False)    # 1000 folds of 1 step


def backtesting_recursive_refit():
    return _backtesting(600, 300, 5, True)       # 60 folds with refit


def backtesting_multiseries():
    series = _series_wide(20, 1500)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=LAGS)
    cv = TimeSeriesFold(steps=5, initial_train_size=1000, refit=False)  # 100 folds
    return lambda: backtesting_forecaster_multiseries(
        forecaster=f, series=series, cv=cv, metric="mean_absolute_error",
        n_jobs=1, show_progress=False,
    )


def window_features():
    y = _y(300_000)
    wf = RollingFeatures(stats=["mean", "std", "min", "max"], window_sizes=LAGS)
    return lambda: wf.transform_batch(y)


def _fit(make_forecaster, data):
    f = make_forecaster()

    def run():
        f.fit(**data)
        return None

    return run


def fit_recursive():
    return _fit(lambda: ForecasterRecursive(estimator=LinearRegression(), lags=LAGS),
                {"y": _y(100_000), "exog": _exog_fit(100_000)})


def fit_direct():
    return _fit(lambda: ForecasterDirect(estimator=LinearRegression(), steps=3,
                                         lags=LAGS, n_jobs=1),
                {"y": _y(100_000), "exog": _exog_fit(100_000)})


def fit_direct_multivariate():
    return _fit(lambda: ForecasterDirectMultiVariate(
                    estimator=LinearRegression(), level="s0", steps=3, lags=LAGS,
                    n_jobs=1),
                {"series": _series_wide(3, 100_000), "exog": _exog_fit(100_000)})


def fit_classifier():
    return _fit(lambda: ForecasterRecursiveClassifier(
                    estimator=LogisticRegression(max_iter=50), lags=LAGS),
                {"y": _y_class(100_000), "exog": _exog_fit(100_000)})


def recursive_create_train_X_y_noexog():
    """
    Only lags, with an estimator that does not need a C-contiguous matrix:
    `X_train` wraps the view of the lags (D13 of the INFORME).
    """
    y = _y(200_000)
    f = ForecasterRecursive(estimator=HistGradientBoostingRegressor(), lags=LAGS)
    return lambda: f.create_train_X_y(y=y)


def _predict_loop(f, y, exog):
    f.fit(y=y, exog=exog)
    exog_future = _future(exog, 500)
    exog_one = exog_future.iloc[:1]

    def run():
        for _ in range(200):
            f.predict(steps=1, exog=exog_one)
        return f.predict(steps=500, exog=exog_future)

    return run


def predict_loop_categorical():
    """
    `predict_loop` with a categorical exog encoded without `transformer_exog`
    (the shallow copy of `exog` before the encoding).
    """
    return _predict_loop(
        ForecasterRecursive(estimator=LinearRegression(), lags=LAGS),
        _y(3000), _exog_fit(3000)
    )


def predict_loop_scaler():
    """
    `predict_loop` with scalers that copy their input (`copy=True`).
    """
    return _predict_loop(
        ForecasterRecursive(estimator=LinearRegression(), lags=LAGS,
                            transformer_y=StandardScaler(),
                            transformer_exog=StandardScaler()),
        _y(3000), _exog_float(3000)
    )


def predict_loop_pipeline():
    """
    `predict_loop` with Pipelines as transformers (no `copy` attribute).
    """
    return _predict_loop(
        ForecasterRecursive(estimator=LinearRegression(), lags=LAGS,
                            transformer_y=make_pipeline(StandardScaler()),
                            transformer_exog=make_pipeline(StandardScaler())),
        _y(3000), _exog_float(3000)
    )


def fit_recursive_transformers():
    return _fit(lambda: ForecasterRecursive(estimator=LinearRegression(), lags=LAGS,
                                            transformer_y=StandardScaler(),
                                            transformer_exog=StandardScaler()),
                {"y": _y(100_000), "exog": _exog_float(100_000)})


def recursive_create_train_X_y_wide_exog():
    """
    A wide `exog` with a category encoded without `transformer_exog`: the
    copy of `exog` before the encoding is large.
    """
    y, exog = _y(200_000), _exog_wide(200_000)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=LAGS)
    return lambda: f.create_train_X_y(y=y, exog=exog)


def fit_recursive_wide_exog():
    return _fit(lambda: ForecasterRecursive(estimator=LinearRegression(), lags=LAGS),
                {"y": _y(200_000), "exog": _exog_wide(200_000)})


BENCHMARKS = {
    fn.__name__: fn for fn in [
        ms_create_train_X_y_block, ms_create_train_X_y_concat,
        ms_create_train_X_y_onehot_dict, ms_fit_linear, ms_fit_linear_encoding_none,
        ms_get_level_row_slices, ms_predict_levels, recursive_create_train_X_y,
        recursive_create_train_X_y_float, direct_create_train_X_y_float,
        recursive_fit_linear, direct_create_train_X_y,
        direct_multivariate_create_train_X_y, classifier_create_train_X_y,
        predict_loop, expand_index_check_predict_input, backtesting_recursive,
        backtesting_recursive_refit, backtesting_multiseries, window_features,
        fit_recursive, fit_direct, fit_direct_multivariate, fit_classifier,
        recursive_create_train_X_y_noexog, predict_loop_categorical,
        predict_loop_scaler, predict_loop_pipeline, fit_recursive_transformers,
        recursive_create_train_X_y_wide_exog, fit_recursive_wide_exog,
    ]
}

# PRs that should not change performance (section 7): matrices, predict,
# backtesting and the four `fit`.
CONTROL = [
    "ms_create_train_X_y_block", "recursive_create_train_X_y",
    "direct_create_train_X_y", "direct_multivariate_create_train_X_y",
    "classifier_create_train_X_y", "predict_loop", "backtesting_recursive",
    "fit_recursive", "fit_direct", "fit_direct_multivariate", "fit_classifier",
]


def select(spec: str) -> list[str]:
    if spec == "all":
        return list(BENCHMARKS)
    if spec == "control":
        return list(CONTROL)
    names = spec.split(",")
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise SystemExit(f"Unknown benchmarks: {unknown}. Available: {list(BENCHMARKS)}")
    return names


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def measure_time(run, warmup: int, repeat: int) -> tuple[list[float], object]:
    out = None
    for _ in range(warmup):
        out = run()
    times = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        out = run()
        times.append((time.perf_counter() - start) * 1000)
    return times, out


def measure_memory(run, repeat: int = 2) -> float:
    """
    Peak of traced memory (MiB) above the memory in use before the call,
    minimum of `repeat` calls. tracemalloc counts numpy allocations.
    """
    run()  # warm-up: lazy imports, numba compilation
    peaks = []
    for _ in range(repeat):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        run()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        peaks.append((peak - before) / 2**20)
    return min(peaks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", required=True)
    parser.add_argument("--bench", default="all")
    parser.add_argument("--mode", choices=["time", "memory"], default="time")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--expect-tree", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    common.ensure_hash_seed()
    common.check_tree(args.expect_tree)
    warnings.simplefilter("ignore")

    results = {}
    for name in select(args.bench):
        cpu_before = psutil.cpu_percent(interval=0.5)
        entry = {"cpu_percent_before": cpu_before}
        run = None
        try:
            run = BENCHMARKS[name]()
            if args.mode == "time":
                times, out = measure_time(run, args.warmup, args.repeat)
                entry.update({
                    "times_ms": times,
                    "min_ms": min(times),
                    "median_ms": statistics.median(times),
                    "output": None if out is None else common.fingerprint(out, False),
                })
                shown = f"min {min(times):9.1f} ms  median {statistics.median(times):9.1f} ms"
            else:
                peak = measure_memory(run)
                entry["peak_mib"] = peak
                shown = f"peak {peak:8.1f} MiB"
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            shown = f"ERROR {entry['error']}"
        entry["cpu_percent_after"] = psutil.cpu_percent(interval=None)
        results[name] = entry
        print(f"{name:40s} {shown}  (cpu {cpu_before:.0f}%)", flush=True)
        del run
        gc.collect()

    payload = {
        "label": args.label,
        "mode": args.mode,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "environment": common.environment(),
        "cpu_count": psutil.cpu_count(),
        "benchmarks": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
