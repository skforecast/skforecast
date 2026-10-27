"""
Benchmarks of the skforecast code paths whose cost depends on pandas internals
(block layout, copies, views): matrix assembly of every forecaster, the single
F block of ForecasterRecursiveMultiSeries, _get_level_row_slices,
DataFrame.insert, pd.concat, expand_index / check_predict_input in the
predict loop, the backtesting folds and the window features.

Protocol (common.py): per benchmark, the variants ("current" and, with
--patch, "patched:<name>") run interleaved after warm-up; median and minimum
of `repeats` wall-clock times; peak memory (tracemalloc and RSS high-water
mark) in a separate pass. Versions are recorded next to the numbers. Run one
process at a time, with the machine otherwise idle:

    /root/.venvs/skforecast/bin/python dev/pandas3_migration/bench_hotspots.py --repeats 7
    PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/bench_hotspots.py --repeats 7
    ... --patch no_copy_keyword          # adds the variant patched:no_copy_keyword
    ... --only ms_create_train_X_y_block predict_loop

A prototype is a module in prototypes/ exposing apply() and restore(), which
monkeypatch skforecast in place (the library is never modified). Results go
to results/bench_pandas<version>[_<patch>].json.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "prototypes"))
from common import ab_interleaved, env_info, fingerprint, peak_memory, save_json  # noqa: E402

warnings.simplefilter("ignore")

BENCHMARKS: dict[str, callable] = {}


def bench(name):
    def deco(fn):
        BENCHMARKS[name] = fn
        return fn
    return deco


# ----------------------------------------------------------------------------
# Data builders
# ----------------------------------------------------------------------------
def _series_wide(n_series, n_obs, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_obs, freq="h")
    return pd.DataFrame(rng.normal(size=(n_obs, n_series)).cumsum(axis=0) + 100.0,
                        index=idx, columns=[f"s{i}" for i in range(n_series)])


def _exog_mixed(n_obs, n_float=3, n_int=1, n_cat=1, n_bool=0, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_obs, freq="h")
    cols = {}
    for i in range(n_float):
        cols[f"f{i}"] = rng.normal(size=n_obs)
    for i in range(n_int):
        cols[f"i{i}"] = rng.integers(0, 10, n_obs)
    for i in range(n_cat):
        cols[f"c{i}"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], n_obs))
    for i in range(n_bool):
        cols[f"b{i}"] = rng.integers(0, 2, n_obs).astype(bool)
    return pd.DataFrame(cols, index=idx)


def _y(n_obs, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_obs, freq="h")
    return pd.Series(rng.normal(size=n_obs).cumsum() + 100.0, index=idx, name="y")


# ----------------------------------------------------------------------------
# Benchmarks: each returns (callable, description); the callable returns an
# object whose fingerprint must be identical across variants and versions.
# ----------------------------------------------------------------------------
@bench("ms_create_train_X_y_block")
def _():
    """ForecasterRecursiveMultiSeries, single-block path (ordinal encoding,
    float + int + category exog -> 2 inserted columns), 100 series x 2000 obs."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000, n_float=3, n_int=1, n_cat=1)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="ordinal")
    return lambda: f.create_train_X_y(series=series, exog=exog), "100 series x 2000 obs, lags 24, exog 3 float + 1 int + 1 category"


@bench("ms_create_train_X_y_concat")
def _():
    """ForecasterRecursiveMultiSeries, pd.concat path (more inserted columns
    than block columns: lags 3, 10 int exog), 100 series x 2000 obs."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000, n_float=0, n_int=10, n_cat=0)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=3, encoding="ordinal")
    return lambda: f.create_train_X_y(series=series, exog=exog), "100 series x 2000 obs, lags 3, exog 10 int (concat path)"


@bench("ms_create_train_X_y_onehot_dict")
def _():
    """ForecasterRecursiveMultiSeries with series and exog as dicts, onehot
    encoding, window features, 50 series of different lengths."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.preprocessing import RollingFeatures
    wide = _series_wide(50, 3000)
    exog = _exog_mixed(3000, n_float=2, n_int=1, n_cat=1)
    series = {c: wide[c].iloc[i * 10:] for i, c in enumerate(wide.columns)}
    exog_d = {c: exog.iloc[i * 10:] for i, c in enumerate(wide.columns)}
    f = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=12, encoding="onehot",
        window_features=RollingFeatures(stats=["mean", "std"], window_sizes=24),
    )
    return lambda: f.create_train_X_y(series=series, exog=exog_d), "50 series (dict, uneven) x <=3000 obs, onehot, window features"


@bench("ms_fit_linear")
def _():
    """ForecasterRecursiveMultiSeries.fit with LinearRegression (includes the
    estimator's own conversion of X_train to an array)."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series = _series_wide(100, 2000)
    exog = _exog_mixed(2000, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="ordinal")

    def run():
        f.fit(series=series, exog=exog, store_in_sample_residuals=True)
        return f.estimator.coef_
    return run, "fit 100 series x 2000 obs, lags 24, LinearRegression, in-sample residuals"


@bench("ms_get_level_row_slices")
def _():
    """_get_level_row_slices on the X_train of 200 series (ordinal and onehot)."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series = _series_wide(200, 1000)
    f_ord = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="ordinal")
    f_oh = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="onehot")
    X_ord, _ = f_ord.create_train_X_y(series=series)
    X_oh, _ = f_oh.create_train_X_y(series=series)

    def run():
        a = f_ord._get_level_row_slices(X_ord)
        b = f_oh._get_level_row_slices(X_oh)
        return {k: (v.start, v.stop) for k, v in a.items()}, {k: (v.start, v.stop) for k, v in b.items()}
    return run, "200 series x 1000 obs: ordinal + onehot slices"


@bench("ms_predict_levels")
def _():
    """ForecasterRecursiveMultiSeries.predict, 100 levels, 24 steps, exog."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series = _series_wide(100, 1000)
    exog = _exog_mixed(1100, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="ordinal")
    f.fit(series=series, exog=exog.iloc[:1000])
    exog_pred = exog.iloc[1000:1024]
    return lambda: f.predict(steps=24, exog=exog_pred), "predict 100 levels x 24 steps with exog"


@bench("recursive_create_train_X_y")
def _():
    """ForecasterRecursive.create_train_X_y, 200k obs, lags 24, window
    features, exog 3 float + 1 category (ndarray -> DataFrame + astype)."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import RollingFeatures
    y = _y(200_000)
    exog = _exog_mixed(200_000, n_float=3, n_int=0, n_cat=1)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=24,
                            window_features=RollingFeatures(stats=["mean", "max"], window_sizes=24))
    return lambda: f.create_train_X_y(y=y, exog=exog), "200k obs, lags 24, 2 window features, exog 3 float + 1 category"


@bench("recursive_fit_linear")
def _():
    """ForecasterRecursive.fit with LinearRegression, 200k obs."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    y = _y(200_000)
    exog = _exog_mixed(200_000, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=24)

    def run():
        f.fit(y=y, exog=exog, store_in_sample_residuals=True)
        return f.estimator.coef_
    return run, "fit 200k obs, lags 24, exog 3 float + 1 int, LinearRegression"


@bench("direct_create_train_X_y")
def _():
    """ForecasterDirect.create_train_X_y, steps 12, lags 24, 50k obs, exog."""
    from sklearn.linear_model import LinearRegression
    from skforecast.direct import ForecasterDirect
    y = _y(50_000)
    exog = _exog_mixed(50_000, n_float=3, n_int=1, n_cat=0)
    f = ForecasterDirect(estimator=LinearRegression(), lags=24, steps=12)
    return lambda: f.create_train_X_y(y=y, exog=exog), "50k obs, steps 12, lags 24, exog 3 float + 1 int"


@bench("direct_multivariate_create_train_X_y")
def _():
    """ForecasterDirectMultiVariate.create_train_X_y, 5 series, 30k obs."""
    from sklearn.linear_model import LinearRegression
    from skforecast.direct import ForecasterDirectMultiVariate
    series = _series_wide(5, 30_000)
    exog = _exog_mixed(30_000, n_float=2, n_int=1, n_cat=0)
    f = ForecasterDirectMultiVariate(estimator=LinearRegression(), level="s0", lags=24, steps=6)
    return lambda: f.create_train_X_y(series=series, exog=exog), "5 series x 30k obs, steps 6, lags 24, exog 2 float + 1 int"


@bench("classifier_create_train_X_y")
def _():
    """ForecasterRecursiveClassifier.create_train_X_y, 100k obs."""
    from sklearn.tree import DecisionTreeClassifier
    from skforecast.recursive import ForecasterRecursiveClassifier
    y = (_y(100_000) > 100).astype(int)
    exog = _exog_mixed(100_000, n_float=3, n_int=1, n_cat=1)
    f = ForecasterRecursiveClassifier(estimator=DecisionTreeClassifier(), lags=24)
    return lambda: f.create_train_X_y(y=y, exog=exog), "100k obs, lags 24, exog 3 float + 1 int + 1 category"


@bench("predict_loop")
def _():
    """ForecasterRecursive.predict(steps=1) called 200 times (per-call cost of
    check_predict_input, expand_index, inputs preparation) plus one
    predict(steps=500)."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    y = _y(2_500)
    exog = _exog_mixed(2_500, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=24)
    f.fit(y=y.iloc[:2000], exog=exog.iloc[:2000])
    exog_1 = exog.iloc[2000:2001]
    exog_500 = exog.iloc[2000:2500]

    def run():
        out = [f.predict(steps=1, exog=exog_1) for _ in range(200)]
        out.append(f.predict(steps=500, exog=exog_500))
        return out
    return run, "200 x predict(steps=1, exog) + 1 x predict(steps=500, exog)"


@bench("expand_index_check_predict_input")
def _():
    """expand_index (DatetimeIndex with freq, and RangeIndex) 5000 times each
    and check_predict_input 5000 times."""
    from skforecast.utils import expand_index, check_predict_input
    idx = pd.date_range("2020-01-01", periods=1000, freq="h")
    ridx = pd.RangeIndex(1000)
    lw = pd.Series(np.arange(30.0), index=idx[-30:])
    exog = pd.DataFrame({"a": np.arange(10.0)}, index=expand_index(idx, 10))

    def run():
        a = [expand_index(idx, 10) for _ in range(5000)]
        b = [expand_index(ridx, 10) for _ in range(5000)]
        for _ in range(5000):
            check_predict_input(
                forecaster_name="ForecasterRecursive", steps=10, is_fitted=True, exog_in_=True,
                index_type_=pd.DatetimeIndex, index_freq_="h", window_size=30, last_window=lw,
                exog=exog, exog_names_in_=["a"],
            )
        return a[-1], b[-1]
    return run, "5000 x expand_index (DatetimeIndex) + 5000 x expand_index (RangeIndex) + 5000 x check_predict_input"


@bench("backtesting_recursive")
def _():
    """backtesting_forecaster, 3000 obs, steps 1, refit False -> 1000 folds."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
    y = _y(3_000)
    exog = _exog_mixed(3_000, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=24)
    cv = TimeSeriesFold(steps=1, initial_train_size=2000, refit=False)

    def run():
        return backtesting_forecaster(forecaster=f, y=y, exog=exog, cv=cv, metric="mean_absolute_error",
                                      n_jobs=1, show_progress=False, verbose=False)
    return run, "3000 obs, steps 1, 1000 folds, refit False"


@bench("backtesting_recursive_refit")
def _():
    """backtesting_forecaster with refit True, 60 folds of 10 steps."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
    y = _y(20_600)
    exog = _exog_mixed(20_600, n_float=3, n_int=1, n_cat=0)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=24)
    cv = TimeSeriesFold(steps=10, initial_train_size=20_000, refit=True, fixed_train_size=False)

    def run():
        return backtesting_forecaster(forecaster=f, y=y, exog=exog, cv=cv, metric="mean_absolute_error",
                                      n_jobs=1, show_progress=False, verbose=False)
    return run, "20600 obs, steps 10, 60 folds, refit True"


@bench("backtesting_multiseries")
def _():
    """backtesting_forecaster_multiseries, 20 series x 1500 obs, steps 5,
    refit False (100 folds), exog dict."""
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold
    wide = _series_wide(20, 1500)
    exog = _exog_mixed(1500, n_float=3, n_int=1, n_cat=0)
    series = {c: wide[c] for c in wide.columns}
    exog_d = {c: exog for c in wide.columns}
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=24, encoding="ordinal")
    cv = TimeSeriesFold(steps=5, initial_train_size=1000, refit=False)

    def run():
        return backtesting_forecaster_multiseries(forecaster=f, series=series, exog=exog_d, cv=cv,
                                                  metric="mean_absolute_error", n_jobs=1,
                                                  show_progress=False, verbose=False)
    return run, "20 series x 1500 obs (dicts), steps 5, 100 folds, refit False"


@bench("window_features")
def _():
    """RollingFeatures.transform_batch on 300k obs and transform on a window
    2000 times."""
    from skforecast.preprocessing import RollingFeatures
    y = _y(300_000)
    rf = RollingFeatures(stats=["mean", "std", "min", "max"], window_sizes=[24, 24, 48, 48])
    window = y.iloc[-48:].to_numpy()

    def run():
        a = rf.transform_batch(y)
        b = [rf.transform(window) for _ in range(2000)]
        return a, b[-1]
    return run, "transform_batch 300k obs (4 stats) + 2000 x transform"


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mem-repeats", type=int, default=3)
    ap.add_argument("--patch", default=None, help="module in prototypes/ with apply()/restore()")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    patch = importlib.import_module(a.patch) if a.patch else None
    results = {"env": env_info(), "patch": a.patch, "repeats": a.repeats, "warmup": a.warmup,
               "benchmarks": {}}
    for name, builder in BENCHMARKS.items():
        if a.only and name not in a.only:
            continue
        print(f"== {name}", flush=True)
        if patch:
            patch.restore()
        fn, desc = builder()
        variants = {"current": fn}
        if patch:
            def patched(fn=fn):
                patch.apply()
                try:
                    return fn()
                finally:
                    patch.restore()
            variants[f"patched:{a.patch}"] = patched
        fps = {k: fingerprint(v(), strict=False) for k, v in variants.items()}
        times = ab_interleaved(variants, repeats=a.repeats, warmup=a.warmup)
        rec = {"description": desc, "fingerprints": fps, "identical_outputs": len(set(fps.values())) == 1, "time": times}
        if not a.no_memory:
            rec["memory"] = {k: peak_memory(v, repeats=a.mem_repeats) for k, v in variants.items()}
        results["benchmarks"][name] = rec
        for k in variants:
            mem = rec.get("memory", {}).get(k, {})
            print(f"   {k:28s} median {times[k]['median_s']*1000:9.1f} ms  min {times[k]['min_s']*1000:9.1f} ms"
                  f"  peak {mem.get('tracemalloc_peak_mib', float('nan')):8.1f} MiB (tracemalloc)"
                  f"  rss+{mem.get('rss_hwm_increase_mib') if mem.get('rss_hwm_increase_mib') is not None else float('nan'):8.1f} MiB",
                  flush=True)
        if not rec["identical_outputs"]:
            print("   !!! outputs differ between variants", flush=True)
    tag = f"pandas{pd.__version__}" + (f"_{a.patch}" if a.patch else "") + (f"_{a.tag}" if a.tag else "")
    path = os.path.join(HERE, "results", f"bench_{tag}.json")
    save_json(path, results)
    print("saved", path)


if __name__ == "__main__":
    main()
