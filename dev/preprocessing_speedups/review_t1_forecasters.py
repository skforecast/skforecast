"""
Review of T1: end-to-end equivalence in every forecaster that calls
`transform_numpy`, beyond the cases of `check_equivalence.py`.

Each configuration is built and run from scratch with the `transform_numpy`
of before T1 and with each candidate (`review_t1_common.py`). PACKAGE is the
function that is now in the package: once T1 is implemented, its row is the
acceptance check (identical outputs, and the shortcut in every eligible call). The outputs
of every workflow (predictions, intervals, quantiles, out-of-sample residuals,
backtesting, one-step-ahead search and errors) are compared by bytes, and the
calls in which the shortcut applies are counted.

python dev/preprocessing_speedups/review_t1_forecasters.py
~/.venvs/skforecast-min/bin/python dev/preprocessing_speedups/review_t1_forecasters.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from prototypes import fast_transform_numpy
from review_t1_common import (
    ORIG_TRANSFORM_NUMPY,
    PACKAGE_TRANSFORM_NUMPY,
    Counted,
    env_line,
    import_modules,
    patched,
    plan_literal,
    proposed,
    proposed_isinf_first,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    grid_search_forecaster,
    grid_search_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

warnings.filterwarnings("ignore")
os.environ.setdefault("KERAS_BACKEND", "torch")

CANDIDATES = {
    "PACKAGE": PACKAGE_TRANSFORM_NUMPY,
    "PLAN": plan_literal,
    "PROTO": fast_transform_numpy,
    "PROPOSED": proposed,
    "PROPOSED-B": proposed_isinf_first,
}

rng = np.random.default_rng(0)
N = 300
idx = pd.date_range("2024-01-01", periods=N, freq="h")
y = pd.Series(50 + rng.normal(0, 3, N).cumsum(), index=idx, name="y")
series = pd.DataFrame(
    {f"s{i}": 20 * (i + 1) + rng.normal(0, 2, N).cumsum() for i in range(3)}, index=idx
)
# Series of different length: the shorter ones start with NaN
series_nan = series.copy()
series_nan.iloc[:60, 1] = np.nan
series_nan.iloc[:120, 2] = np.nan
y_tr, y_te = y.iloc[:-24], y.iloc[-24:]
s_tr, s_te = series.iloc[:-24], series.iloc[-24:]
sn_tr = series_nan.iloc[:-24]


def as_bytes(obj):
    """Numeric content of a result as bytes (floats in float64)."""
    if isinstance(obj, tuple):
        return b"|".join(as_bytes(o) for o in obj)
    if isinstance(obj, dict):
        return b"|".join(str(k).encode() + as_bytes(v) for k, v in sorted(obj.items(), key=str))
    if obj is None:
        return b"None"
    if isinstance(obj, pd.DataFrame):
        obj = obj.select_dtypes("number")
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.to_numpy()
    arr = np.ascontiguousarray(np.asarray(obj, dtype=float))
    return str(arr.shape).encode() + arr.tobytes()


def collect(build, workflows, func):
    out = {}
    with patched(func):
        try:
            forecaster = build()
        except Exception as exc:
            return {"build": ("raise", type(exc).__name__, str(exc))}
        for wf in workflows:
            try:
                out[wf.__name__] = ("ok", as_bytes(wf(forecaster)))
            except Exception as exc:
                out[wf.__name__] = ("raise", type(exc).__name__, str(exc))
    return out


def run(title, build, workflows):
    ref = collect(build, workflows, ORIG_TRANSFORM_NUMPY)
    errors = {k: f"{v[1]}: {v[2][:90]}" for k, v in ref.items() if v[0] == "raise"}
    print(f"\n##### {title} ({len(ref)} workflows"
          + (f", {len(errors)} of them raise with the current path" if errors else "") + ")")
    for k, v in errors.items():
        print(f"   current, {k}: {v}")
    for name, func in CANDIDATES.items():
        counted = Counted(func)
        got = collect(build, workflows, counted)
        diff = [k for k in ref if got.get(k) != ref[k]]
        print(f"   {name:10s} {'identical' if not diff else 'DIFFERENT in ' + ', '.join(diff):28s} "
              f"shortcut in {counted.hits:4d} of {counted.calls:4d} calls with a transformer")
        for k in diff:
            g = got.get(k)
            print(f"      {k}: {g[1] + ': ' + g[2][:120] if g and g[0] == 'raise' else 'other values'}")


# ------------------------------------------------------------ single series


def fit(f):
    f.fit(y_tr, store_in_sample_residuals=True)
    return f.in_sample_residuals_


def predict(f):
    return f.predict(steps=24)


def bootstrapping(f):
    return f.predict_interval(steps=24, method="bootstrapping", n_boot=25)


def conformal(f):
    return f.predict_interval(steps=24, method="conformal")


def quantiles(f):
    return f.predict_quantiles(steps=24, n_boot=25)


def out_sample(f):
    f.set_out_sample_residuals(y_true=y_te, y_pred=f.predict(steps=24))
    return f.out_sample_residuals_, f.out_sample_residuals_by_bin_


def backtest(f):
    cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=False, verbose=False)
    return backtesting_forecaster(f, y=y, cv=cv, metric="mean_absolute_error",
                                  show_progress=False, n_jobs=1)


def backtest_interval(f):
    cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=True, verbose=False)
    return backtesting_forecaster(f, y=y, cv=cv, metric="mean_absolute_error",
                                  interval=[0.1, 0.9], interval_method="conformal",
                                  show_progress=False, n_jobs=1)


def one_step_ahead(f):
    cv = OneStepAheadFold(initial_train_size=200)
    return grid_search_forecaster(
        f, y=y, cv=cv, param_grid={"alpha": [0.1, 1.0]}, lags_grid=[5],
        metric="mean_absolute_error", return_best=False, show_progress=False, verbose=False)


def inf_last_window(f):
    lw = y_tr.iloc[-f.window_size:].copy()
    lw.iloc[-2] = np.inf
    return f.predict(steps=5, last_window=lw)


SINGLE = [fit, predict, bootstrapping, conformal, quantiles, out_sample, backtest,
          backtest_interval, one_step_ahead, inf_last_window]

# ------------------------------------------------------------- multi series


def multi(data_train, data_full, direct=False):
    kw = {} if direct else {"levels": None}

    def fit(f):
        f.fit(data_train, store_in_sample_residuals=True)
        return f.in_sample_residuals_

    def predict(f):
        return f.predict(steps=24, **kw)

    def bootstrapping(f):
        return f.predict_interval(steps=24, method="bootstrapping", n_boot=25, **kw)

    def conformal(f):
        return f.predict_interval(steps=24, method="conformal", **kw)

    def quantiles(f):
        return f.predict_quantiles(steps=24, n_boot=25, **kw)

    def out_sample(f):
        pred = f.predict(steps=24, **kw)
        levels = [f.level] if direct else list(s_te.columns)
        y_pred = {k: pred.loc[pred["level"] == k, "pred"] for k in levels}
        f.set_out_sample_residuals(y_true={k: s_te[k] for k in levels}, y_pred=y_pred)
        return f.out_sample_residuals_, f.out_sample_residuals_by_bin_

    def unknown_level(f):
        if direct or f.encoding == "ordinal":
            return None
        lw = data_train.iloc[-f.window_size:][["s0"]].rename(columns={"s0": "new_series"})
        return f.predict(steps=5, levels=["new_series"], last_window=lw)

    def backtest(f):
        cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=False, verbose=False)
        return backtesting_forecaster_multiseries(
            f, series=data_full, cv=cv, metric="mean_absolute_error",
            show_progress=False, n_jobs=1, suppress_warnings=True)

    def backtest_interval(f):
        cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=True, verbose=False)
        return backtesting_forecaster_multiseries(
            f, series=data_full, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9],
            interval_method="bootstrapping", n_boot=25,
            show_progress=False, n_jobs=1, suppress_warnings=True)

    def one_step_ahead(f):
        cv = OneStepAheadFold(initial_train_size=200)
        return grid_search_forecaster_multiseries(
            f, series=data_full, cv=cv, param_grid={"alpha": [0.1, 1.0]}, lags_grid=[5],
            metric="mean_absolute_error", return_best=False, show_progress=False,
            verbose=False, suppress_warnings=True)

    def inf_last_window(f):
        lw = data_train.iloc[-f.window_size:].copy()
        lw.iloc[-2, 0] = np.inf
        return f.predict(steps=5, last_window=lw, **kw)

    return [fit, predict, bootstrapping, conformal, quantiles, out_sample, unknown_level,
            backtest, backtest_interval, one_step_ahead, inf_last_window]


def main():
    print(env_line())
    import_modules()

    run("ForecasterRecursive",
        lambda: ForecasterRecursive(Ridge(), lags=5, transformer_y=StandardScaler()), SINGLE)
    run("ForecasterRecursive, differentiation=1",
        lambda: ForecasterRecursive(Ridge(), lags=5, transformer_y=StandardScaler(),
                                    differentiation=1), SINGLE)
    run("ForecasterRecursive, StandardScaler(with_std=False, copy=False)",
        lambda: ForecasterRecursive(
            Ridge(), lags=5, transformer_y=StandardScaler(with_std=False, copy=False)), SINGLE)
    run("ForecasterDirect",
        lambda: ForecasterDirect(Ridge(), lags=5, steps=24, transformer_y=StandardScaler()),
        SINGLE)
    run("ForecasterDirect, differentiation=1",
        lambda: ForecasterDirect(Ridge(), lags=5, steps=24, transformer_y=StandardScaler(),
                                 differentiation=1), SINGLE)

    run("ForecasterRecursiveMultiSeries",
        lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5,
                                               transformer_series=StandardScaler()),
        multi(s_tr, series))
    run("ForecasterRecursiveMultiSeries, series of different length (NaN at the start)",
        lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5,
                                               transformer_series=StandardScaler()),
        multi(sn_tr, series_nan))
    run("ForecasterRecursiveMultiSeries, dict of transformers (one series without), onehot",
        lambda: ForecasterRecursiveMultiSeries(
            Ridge(), lags=5, encoding="onehot",
            transformer_series={"s0": StandardScaler(), "s1": StandardScaler(with_mean=False),
                                "s2": None, "_unknown_level": StandardScaler()}),
        multi(s_tr, series))
    run("ForecasterRecursiveMultiSeries, encoding=None, differentiation=1",
        lambda: ForecasterRecursiveMultiSeries(
            Ridge(), lags=5, encoding=None, differentiation=1,
            transformer_series=StandardScaler()),
        multi(s_tr, series))
    run("ForecasterDirectMultiVariate",
        lambda: ForecasterDirectMultiVariate(Ridge(), level="s0", lags=5, steps=24,
                                             transformer_series=StandardScaler()),
        multi(s_tr, series, direct=True))
    run("ForecasterDirectMultiVariate, differentiation=1",
        lambda: ForecasterDirectMultiVariate(Ridge(), level="s0", lags=5, steps=24,
                                             transformer_series=StandardScaler(),
                                             differentiation=1),
        multi(s_tr, series, direct=True))

    try:
        from skforecast.recursive import ForecasterStats
        from skforecast.stats import Arima, Ets, Sarimax

        def fit(f):
            f.fit(y_tr)

        def predict(f):
            return f.predict(steps=24)

        def interval(f):
            return f.predict_interval(steps=24, alpha=0.05)

        def last_window(f):
            return f.predict(steps=12, last_window=y_te)

        run("ForecasterStats (Arima)",
            lambda: ForecasterStats(estimator=Arima(order=(1, 0, 0)),
                                    transformer_y=StandardScaler()),
            [fit, predict, interval])
        run("ForecasterStats (Sarimax), with last_window",
            lambda: ForecasterStats(estimator=Sarimax(order=(1, 0, 0)),
                                    transformer_y=StandardScaler()),
            [fit, predict, interval, last_window])
        run("ForecasterStats (Arima + Ets)",
            lambda: ForecasterStats(estimator=[Arima(order=(1, 0, 0)), Ets()],
                                    transformer_y=StandardScaler()),
            [fit, predict, interval])
    except Exception as exc:
        print(f"\n[skip] ForecasterStats: {type(exc).__name__}: {exc}")

    try:
        import keras

        from skforecast.deep_learning import ForecasterRnn, create_and_compile_model

        def build():
            keras.utils.set_random_seed(123)
            model = create_and_compile_model(
                series=s_tr, lags=5, steps=4, levels=["s0", "s1"],
                recurrent_units=4, dense_units=4)
            return ForecasterRnn(
                estimator=model, levels=["s0", "s1"], lags=5,
                transformer_series=StandardScaler(),
                fit_kwargs={"epochs": 1, "batch_size": 32, "verbose": 0})

        def fit(f):
            f.fit(s_tr, store_in_sample_residuals=True)

        def predict(f):
            return f.predict(steps=4)

        def conformal(f):
            return f.predict_interval(steps=4, method="conformal")

        def out_sample(f):
            pred = f.predict(steps=4)
            y_pred = {k: pred.loc[pred["level"] == k, "pred"] for k in ["s0", "s1"]}
            f.set_out_sample_residuals(
                y_true={k: s_te[k].iloc[:4] for k in ["s0", "s1"]}, y_pred=y_pred)
            return f.out_sample_residuals_

        run("ForecasterRnn", build, [fit, predict, conformal, out_sample])
    except ImportError:
        print("\n[skip] ForecasterRnn: keras is not installed in this environment")


if __name__ == "__main__":
    main()
