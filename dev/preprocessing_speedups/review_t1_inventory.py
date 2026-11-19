"""
Review of T1, point 1: runtime inventory of the calls to `transform_numpy`.

Every forecaster that calls `transform_numpy` is run through its public
workflows (fit, predict, intervals, quantiles, out-of-sample residuals,
backtesting and one-step-ahead search) with a `StandardScaler`. A tracer
records, per call site (file:line), the arguments, the input array and whether
the conditions of the shortcut hold, as written in PLAN.md (`PLAN=`) and as
coded in the prototype (`PROTOTYPE=`).

python dev/preprocessing_speedups/review_t1_inventory.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from review_t1_common import Tracer, env_line, import_modules, patched
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

rng = np.random.default_rng(0)
N = 300
idx = pd.date_range("2024-01-01", periods=N, freq="h")
y = pd.Series(50 + rng.normal(0, 3, N).cumsum(), index=idx, name="y")
series = pd.DataFrame(
    {f"s{i}": 20 * (i + 1) + rng.normal(0, 2, N).cumsum() for i in range(3)}, index=idx
)
y_tr, y_te = y.iloc[:-24], y.iloc[-24:]
s_tr, s_te = series.iloc[:-24], series.iloc[-24:]


def state(name, transformer, user_obj=None):
    print(
        f"  {name}: type={type(transformer).__name__}, "
        f"n_features_in_={getattr(transformer, 'n_features_in_', None)}, "
        f"feature_names_in_={getattr(transformer, 'feature_names_in_', None)}, "
        f"mean_.dtype={getattr(getattr(transformer, 'mean_', None), 'dtype', None)}"
        + (f", same object as the user's={transformer is user_obj}" if user_obj is not None else "")
    )


def run(title, build, workflows, user_transformers=()):
    tracer = Tracer(user_transformers)
    with patched(tracer):
        forecaster = build()
        for wf in workflows:
            try:
                wf(forecaster)
            except Exception as exc:  # report, keep the rest of the inventory
                print(f"  [workflow failed] {wf.__name__}: {type(exc).__name__}: {exc}")
    tracer.report(title)
    return forecaster


def single_workflows(fit_kwargs=None, backtest=True, osa=True, quantiles=True):
    def fit(f):
        f.fit(y_tr, store_in_sample_residuals=True)

    def predict(f):
        f.predict(steps=24)

    def boot(f):
        f.predict_interval(steps=24, method="bootstrapping", n_boot=25)

    def conformal(f):
        f.predict_interval(steps=24, method="conformal")

    def quant(f):
        f.predict_quantiles(steps=24, n_boot=25)

    def out_sample(f):
        f.set_out_sample_residuals(y_true=y_te, y_pred=f.predict(steps=24))

    def bt(f):
        cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=False, verbose=False)
        backtesting_forecaster(f, y=y, cv=cv, metric="mean_absolute_error",
                               show_progress=False, n_jobs=1)

    def one_step(f):
        cv = OneStepAheadFold(initial_train_size=200)
        grid_search_forecaster(f, y=y, cv=cv, param_grid={"alpha": [0.1, 1.0]},
                               lags_grid=[5], metric="mean_absolute_error",
                               return_best=False, show_progress=False, verbose=False)

    wfs = [fit, predict, boot, conformal]
    if quantiles:
        wfs.append(quant)
    wfs.append(out_sample)
    if backtest:
        wfs.append(bt)
    if osa:
        wfs.append(one_step)
    return wfs


def multi_workflows(level_kw=None, direct=False):
    kw = {} if direct else {"levels": None}

    def fit(f):
        f.fit(s_tr, store_in_sample_residuals=True)

    def predict(f):
        f.predict(steps=24, **kw)

    def boot(f):
        f.predict_interval(steps=24, method="bootstrapping", n_boot=25, **kw)

    def conformal(f):
        f.predict_interval(steps=24, method="conformal", **kw)

    def quant(f):
        f.predict_quantiles(steps=24, n_boot=25, **kw)

    def out_sample(f):
        if direct:
            lvl = f.level
            pred = f.predict(steps=24)["pred"]
            f.set_out_sample_residuals(y_true={lvl: s_te[lvl]}, y_pred={lvl: pred})
        else:
            pred = f.predict(steps=24)
            y_pred = {k: pred.loc[pred["level"] == k, "pred"] for k in s_te.columns}
            f.set_out_sample_residuals(y_true={k: s_te[k] for k in s_te.columns}, y_pred=y_pred)

    def unknown_level(f):
        # NOTE: A series not seen in training uses the transformer of
        # '_unknown_level'. With an ordinal encoding its code is NaN, which
        # Ridge does not accept, so this is only run with `encoding=None`.
        if not direct and f.encoding is None:
            lw = s_tr.iloc[-f.window_size:].rename(columns={"s0": "new_series"})[["new_series"]]
            f.predict(steps=5, levels=["new_series"], last_window=lw)

    def bt(f):
        cv = TimeSeriesFold(steps=12, initial_train_size=200, refit=False, verbose=False)
        backtesting_forecaster_multiseries(
            f, series=series, cv=cv, metric="mean_absolute_error",
            show_progress=False, n_jobs=1, suppress_warnings=True)

    def one_step(f):
        cv = OneStepAheadFold(initial_train_size=200)
        grid_search_forecaster_multiseries(
            f, series=series, cv=cv, param_grid={"alpha": [0.1, 1.0]},
            lags_grid=[5], metric="mean_absolute_error", return_best=False,
            show_progress=False, verbose=False, suppress_warnings=True)

    return [fit, predict, boot, conformal, quant, out_sample, unknown_level, bt, one_step]


def main():
    print(env_line())
    import_modules()

    # --- ForecasterRecursive
    sc = StandardScaler()
    f = run("ForecasterRecursive, transformer_y=StandardScaler()",
            lambda: ForecasterRecursive(Ridge(), lags=5, transformer_y=sc),
            single_workflows(), [sc])
    state("transformer_y", f.transformer_y, sc)

    sc = StandardScaler()
    f = run("ForecasterRecursive, differentiation=1",
            lambda: ForecasterRecursive(Ridge(), lags=5, transformer_y=sc, differentiation=1),
            single_workflows(), [sc])

    # --- ForecasterDirect
    sc = StandardScaler()
    f = run("ForecasterDirect, transformer_y=StandardScaler()",
            lambda: ForecasterDirect(Ridge(), lags=5, steps=24, transformer_y=sc),
            single_workflows(), [sc])
    state("transformer_y", f.transformer_y, sc)

    # --- ForecasterRecursiveMultiSeries
    sc = StandardScaler()
    f = run("ForecasterRecursiveMultiSeries, transformer_series=StandardScaler()",
            lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5, transformer_series=sc),
            multi_workflows(), [sc])
    for k, t in f.transformer_series_.items():
        state(f"transformer_series_[{k!r}]", t, sc)

    scs = {"s0": StandardScaler(), "s1": StandardScaler(), "s2": None,
           "_unknown_level": StandardScaler()}
    f = run("ForecasterRecursiveMultiSeries, transformer_series=dict (s2 without transformer)",
            lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5, transformer_series=scs),
            multi_workflows(), [v for v in scs.values() if v is not None])
    for k, t in f.transformer_series_.items():
        state(f"transformer_series_[{k!r}]", t, scs.get(k))

    sc = StandardScaler()
    f = run("ForecasterRecursiveMultiSeries, encoding=None, differentiation=1",
            lambda: ForecasterRecursiveMultiSeries(
                Ridge(), lags=5, transformer_series=sc, encoding=None, differentiation=1),
            multi_workflows(), [sc])
    for k, t in f.transformer_series_.items():
        state(f"transformer_series_[{k!r}]", t, sc)

    # --- ForecasterDirectMultiVariate
    sc = StandardScaler()
    f = run("ForecasterDirectMultiVariate, transformer_series=StandardScaler()",
            lambda: ForecasterDirectMultiVariate(
                Ridge(), level="s0", lags=5, steps=24, transformer_series=sc),
            multi_workflows(direct=True), [sc])
    for k, t in f.transformer_series_.items():
        state(f"transformer_series_[{k!r}]", t, sc)

    # --- ForecasterStats
    try:
        from skforecast.recursive import ForecasterStats
        from skforecast.stats import Arima, Ets, Sarimax

        def stats_fit(f):
            f.fit(y_tr)

        def stats_predict(f):
            f.predict(steps=24)

        def stats_interval(f):
            f.predict_interval(steps=24, alpha=0.05)

        def stats_last_window(f):
            f.predict(steps=24, last_window=y_te)

        sc = StandardScaler()
        f = run("ForecasterStats (Arima), transformer_y=StandardScaler()",
                lambda: ForecasterStats(estimator=Arima(order=(1, 0, 0)), transformer_y=sc),
                [stats_fit, stats_predict, stats_interval], [sc])
        state("transformer_y", f.transformer_y, sc)

        sc = StandardScaler()
        f = run("ForecasterStats (Sarimax), predict with last_window",
                lambda: ForecasterStats(estimator=Sarimax(order=(1, 0, 0)), transformer_y=sc),
                [stats_fit, stats_last_window], [sc])

        sc = StandardScaler()
        f = run("ForecasterStats (Arima + Ets), transformer_y=StandardScaler()",
                lambda: ForecasterStats(
                    estimator=[Arima(order=(1, 0, 0)), Ets()], transformer_y=sc),
                [stats_fit, stats_predict, stats_interval], [sc])
    except Exception as exc:
        print(f"[skip] ForecasterStats: {type(exc).__name__}: {exc}")

    # --- ForecasterRnn (needs keras and torch)
    try:
        import os

        os.environ.setdefault("KERAS_BACKEND", "torch")
        from skforecast.deep_learning import ForecasterRnn, create_and_compile_model

        def build_rnn():
            model = create_and_compile_model(
                series=s_tr, lags=5, steps=4, levels=["s0", "s1"],
                recurrent_units=4, dense_units=4)
            return ForecasterRnn(
                estimator=model, levels=["s0", "s1"], lags=5,
                transformer_series=sc,
                fit_kwargs={"epochs": 1, "batch_size": 32, "verbose": 0})

        def rnn_fit(f):
            f.fit(s_tr, store_in_sample_residuals=True)

        def rnn_predict(f):
            p = f.predict(steps=4)
            print(f"  ForecasterRnn: dtype of estimator.predict: "
                  f"{f.estimator.predict(np.zeros((1, 5, 3)), verbose=0).dtype}, "
                  f"dtype of pred: {p['pred'].dtype}")

        def rnn_conformal(f):
            f.predict_interval(steps=4, method="conformal")

        def rnn_out_sample(f):
            pred = f.predict(steps=4)
            y_pred = {k: pred.loc[pred["level"] == k, "pred"] for k in ["s0", "s1"]}
            f.set_out_sample_residuals(
                y_true={k: s_te[k].iloc[:4] for k in ["s0", "s1"]}, y_pred=y_pred)

        sc = StandardScaler()
        f = run("ForecasterRnn, transformer_series=StandardScaler()", build_rnn,
                [rnn_fit, rnn_predict, rnn_conformal, rnn_out_sample], [sc])
        for k, t in f.transformer_series_.items():
            state(f"transformer_series_[{k!r}]", t, sc)
    except ImportError as exc:
        print(f"\n[skip] ForecasterRnn: {exc}")


if __name__ == "__main__":
    main()
