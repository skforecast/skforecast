"""
Identity cases of the pandas 3 migration (INFORME.md, section 5).

Runs every case against the skforecast tree on PYTHONPATH and writes a JSON
with, per case:

- the strict and normalized fingerprint of each output;
- `inputs_modified`: inputs whose strict fingerprint changed during the case
  (the user's data must never be modified);
- `repeat_mismatch`: methods that returned different results when called
  twice in a row with the same arguments;
- `shared_memory`: pairs `output <-> input` whose numpy buffers overlap;
- `warnings`: deprecation warnings attributed to the skforecast package;
- `error`: the exception, if the case failed.

Compare two JSON files with `identity_compare.py`.

Usage:
    python identity_cases.py --out results.json [--index-unit native|ns]
        [--cases name1,name2] [--expect-tree <root of the skforecast tree>]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.drift_detection import PopulationDriftDetector, RangeDriftDetector
from skforecast.feature_selection import select_features
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    backtesting_stats,
    grid_search_forecaster,
    grid_search_forecaster_multiseries,
)
from skforecast.preprocessing import (
    CalendarFeatures,
    RollingFeatures,
    reshape_exog_long_to_dict,
    reshape_series_long_to_dict,
)
from skforecast.recursive import (
    ForecasterEquivalentDate,
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
    ForecasterStats,
)
from skforecast.stats import Arar, Arima, Ets, Sarimax
from skforecast.utils import load_forecaster, save_forecaster

import common

N = 300
N_TRAIN = 250
STEPS = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_data(index_unit: str) -> dict:
    """
    Deterministic data. With `index_unit='native'` the index is what
    `pd.date_range` returns (`[ns]` in pandas 2, `[us]` in pandas 3); with
    `'ns'` it is forced to nanoseconds, to compare like with like.
    """
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=N, freq="D")
    if index_unit == "ns":
        idx = idx.as_unit("ns")

    y = pd.Series(10 + np.cumsum(rng.normal(0, 1, N)), index=idx, name="y")
    exog = pd.DataFrame(
        {
            "exog_1": rng.normal(size=N),
            "exog_2": rng.uniform(size=N),
            "exog_int": rng.integers(0, 10, size=N),
        },
        index=idx,
    )
    # Category with an unused level ('d') on purpose.
    exog_cat = exog.assign(
        exog_cat=pd.Categorical(
            rng.choice(["a", "b", "c"], size=N), categories=["a", "b", "c", "d"]
        )
    )
    # Plain Python strings: `object` in pandas 2, `str` in pandas 3.
    exog_text = exog[["exog_1"]].assign(
        exog_text=list(rng.choice(["low", "mid", "high"], size=N))
    )
    series = pd.DataFrame(
        {
            "s1": 10 + np.cumsum(rng.normal(0, 1, N)),
            "s2": 20 + np.cumsum(rng.normal(0, 1, N)),
            "s3": 30 + np.cumsum(rng.normal(0, 1, N)),
        },
        index=idx,
    )
    diff = np.diff(y.to_numpy(), prepend=y.iloc[0])
    y_class = pd.Series(
        list(np.where(diff > 0.5, "up", np.where(diff < -0.5, "down", "flat"))),
        index=idx,
        name="y",
    )
    return {
        "idx": idx,
        "y": y,
        "exog": exog,
        "exog_cat": exog_cat,
        "exog_text": exog_text,
        "series": series,
        "y_class": y_class,
    }


def train(obj):
    return obj.iloc[:N_TRAIN]


def test(obj):
    return obj.iloc[N_TRAIN:N_TRAIN + STEPS]


def lgbm():
    return LGBMRegressor(n_estimators=20, random_state=123, verbose=-1)


# ---------------------------------------------------------------------------
# Case context
# ---------------------------------------------------------------------------

class Ctx:
    """
    Collects the outputs of a case and runs the checks on its inputs.
    """

    def __init__(self):
        self.outputs: dict = {}
        self.inputs: dict = {}
        self._input_fps: dict = {}
        self.repeat_mismatch: list[str] = []

    def inp(self, name: str, obj):
        """
        Register an input passed by the user; returns it unchanged.
        """
        self.inputs[name] = obj
        self._input_fps[name] = common.fingerprint(obj, True)
        return obj

    def out(self, name: str, obj) -> None:
        self.outputs[name] = obj

    def twice(self, name: str, fn):
        """
        Call `fn` twice, store the first result and flag a mismatch.
        """
        first = fn()
        second = fn()
        if common.fingerprint(first, True) != common.fingerprint(second, True):
            self.repeat_mismatch.append(name)
        self.out(name, first)
        return first

    def modified_inputs(self) -> list[str]:
        return [
            name for name, obj in self.inputs.items()
            if common.fingerprint(obj, True) != self._input_fps[name]
        ]


def fit_outputs(ctx: Ctx, forecaster, prefix: str = "") -> None:
    """
    Attributes stored by `fit` that the migration can change.
    """
    for attr in ["exog_dtypes_in_", "exog_dtypes_out_", "last_window_",
                 "in_sample_residuals_", "in_sample_residuals_by_bin_",
                 "X_train_features_names_out_", "series_names_in_",
                 "training_range_", "index_freq_"]:
        if hasattr(forecaster, attr):
            ctx.out(f"{prefix}{attr}", getattr(forecaster, attr))


# ---------------------------------------------------------------------------
# Cases of the study (21)
# ---------------------------------------------------------------------------

def case_recursive_exog_numeric(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog"]))
    exog_test = ctx.inp("exog_test", test(d["exog"]))
    f = ForecasterRecursive(
        estimator=LinearRegression(), lags=5,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
    )
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_interval_bootstrapping", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, n_boot=50, use_binned_residuals=False))
    ctx.twice("predict_interval_binned", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, n_boot=50, use_binned_residuals=True))


def case_recursive_calendar(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    f = ForecasterRecursive(
        estimator=Ridge(), lags=5,
        calendar_features=CalendarFeatures(
            features=["month", "day_of_week", "day_of_year"]
        ),
    )
    ctx.out("create_train_X_y", f.create_train_X_y(y=y))
    f.fit(y=y)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS))


def case_recursive_categorical_lgbm(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog_cat"]))
    exog_test = ctx.inp("exog_test", test(d["exog_cat"]))
    f = ForecasterRecursive(estimator=lgbm(), lags=5, categorical_features="auto")
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))


def case_recursive_text_exog_lgbm(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog_text"]))
    exog_test = ctx.inp("exog_test", test(d["exog_text"]))
    f = ForecasterRecursive(estimator=lgbm(), lags=5, categorical_features="auto")
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))


def case_recursive_text_no_categorical_features(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog_text"]))
    exog_test = ctx.inp("exog_test", test(d["exog_text"]))
    transformer = make_column_transformer(
        (OneHotEncoder(sparse_output=False), ["exog_text"]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    f = ForecasterRecursive(
        estimator=Ridge(), lags=5, transformer_exog=transformer,
        categorical_features=None,
    )
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))


def case_direct(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog"]))
    exog_test = ctx.inp("exog_test", test(d["exog"]))
    f = ForecasterDirect(
        estimator=Ridge(), steps=STEPS, lags=5,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
    )
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_interval", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, n_boot=50))


def _multiseries_wide(ctx, d, encoding):
    series = ctx.inp("series", train(d["series"]))
    exog = ctx.inp("exog", train(d["exog"]))
    exog_test = ctx.inp("exog_test", test(d["exog"]))
    f = ForecasterRecursiveMultiSeries(
        estimator=Ridge(), lags=5, encoding=encoding,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
    )
    ctx.out("create_train_X_y", f.create_train_X_y(series=series, exog=exog))
    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_interval_bootstrapping", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, method="bootstrapping", n_boot=30,
        use_binned_residuals=False))
    ctx.twice("predict_interval_binned", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, method="bootstrapping", n_boot=30,
        use_binned_residuals=True))


def case_multiseries_wide_ordinal(ctx, d):
    _multiseries_wide(ctx, d, "ordinal")


def case_multiseries_wide_ordinal_category(ctx, d):
    _multiseries_wide(ctx, d, "ordinal_category")


def case_multiseries_wide_onehot(ctx, d):
    _multiseries_wide(ctx, d, "onehot")


def _series_dict(d):
    s = d["series"]
    return {
        "s1": s["s1"].iloc[:N_TRAIN],
        "s2": s["s2"].iloc[30:N_TRAIN],      # starts later
        "s3": s["s3"].iloc[:N_TRAIN - 20],   # ends earlier
    }


def _exog_dict(d, part):
    e = part(d["exog_cat"])
    return {
        "s1": e[["exog_1", "exog_cat"]],
        "s2": e[["exog_1", "exog_2", "exog_cat"]],
        # s3 without exogenous variables
    }


def case_multiseries_dict(ctx, d):
    series = ctx.inp("series", _series_dict(d))
    exog = ctx.inp("exog", _exog_dict(d, train))
    f = ForecasterRecursiveMultiSeries(estimator=lgbm(), lags=5, encoding="ordinal")
    ctx.out("create_train_X_y", f.create_train_X_y(series=series, exog=exog))
    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    exog_test = ctx.inp("exog_test", _exog_dict(d, test))
    ctx.twice("predict", lambda: f.predict(steps=STEPS, levels=["s1", "s2"],
                                           exog=exog_test))


def case_multiseries_long(ctx, d):
    s = train(d["series"])
    long_series = ctx.inp("long_series", (
        s.rename_axis("datetime").reset_index()
        .melt(id_vars="datetime", var_name="series_id", value_name="value")
    ))
    e = train(d["exog_cat"])[["exog_1", "exog_cat"]]
    long_exog = ctx.inp("long_exog", pd.concat(
        [e.rename_axis("datetime").reset_index().assign(series_id=name)
         for name in ["s1", "s2", "s3"]],
        ignore_index=True,
    ))
    series_dict = reshape_series_long_to_dict(
        data=long_series, series_id="series_id", index="datetime",
        values="value", freq="D",
    )
    exog_dict = reshape_exog_long_to_dict(
        data=long_exog, series_id="series_id", index="datetime", freq="D",
    )
    ctx.out("series_dict", series_dict)
    ctx.out("exog_dict", exog_dict)
    f = ForecasterRecursiveMultiSeries(estimator=lgbm(), lags=5)
    f.fit(series=series_dict, exog=exog_dict)
    fit_outputs(ctx, f)
    e_test = test(d["exog_cat"])[["exog_1", "exog_cat"]]
    exog_test = ctx.inp("exog_test", {k: e_test for k in ["s1", "s2", "s3"]})
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))


def case_direct_multivariate(ctx, d):
    series = ctx.inp("series", train(d["series"]))
    exog = ctx.inp("exog", train(d["exog"]))
    exog_test = ctx.inp("exog_test", test(d["exog"]))
    f = ForecasterDirectMultiVariate(estimator=Ridge(), level="s1", steps=STEPS, lags=5)
    ctx.out("create_train_X_y", f.create_train_X_y(series=series, exog=exog))
    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_interval", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, method="bootstrapping", n_boot=50))


def case_classifier(ctx, d):
    y = ctx.inp("y", train(d["y_class"]))
    exog = ctx.inp("exog", train(d["exog_cat"]))
    exog_test = ctx.inp("exog_test", test(d["exog_cat"]))
    f = ForecasterRecursiveClassifier(
        estimator=DecisionTreeClassifier(random_state=123), lags=5
    )
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_proba", lambda: f.predict_proba(steps=STEPS, exog=exog_test))


def _stats(ctx, d, estimator):
    y = ctx.inp("y", train(d["y"]))
    f = ForecasterStats(estimator=estimator)
    f.fit(y=y)
    ctx.twice("predict", lambda: f.predict(steps=STEPS))
    ctx.twice("predict_interval", lambda: f.predict_interval(
        steps=STEPS, interval=[0.05, 0.95]))


def case_stats_arima(ctx, d):
    _stats(ctx, d, Arima(order=(1, 0, 1)))


def case_equivalent_date(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    for name, offset in [("offset_dateoffset", pd.DateOffset(days=7)),
                         ("offset_int", 7)]:
        f = ForecasterEquivalentDate(offset=offset, n_offsets=2)
        f.fit(y=y, store_in_sample_residuals=True)
        ctx.twice(f"{name}.predict", lambda: f.predict(steps=STEPS))
        ctx.twice(f"{name}.predict_interval", lambda: f.predict_interval(
            steps=STEPS, method="conformal"))


def case_backtesting_conformal(ctx, d):
    y = ctx.inp("y", d["y"])
    exog = ctx.inp("exog", d["exog"])
    f = ForecasterRecursive(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, refit=False)
    metric, preds = backtesting_forecaster(
        forecaster=f, y=y, exog=exog, cv=cv, metric="mean_absolute_error",
        interval=[0.1, 0.9], interval_method="conformal", show_progress=False,
    )
    ctx.out("metric", metric)
    ctx.out("predictions", preds)


def case_backtesting_multiseries_bootstrapping(ctx, d):
    series = ctx.inp("series", d["series"])
    exog = ctx.inp("exog", d["exog"])
    f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, refit=False)
    metric, preds = backtesting_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv,
        metric=["mean_absolute_error", "mean_absolute_scaled_error"],
        interval=[0.1, 0.9], interval_method="bootstrapping", n_boot=30,
        show_progress=False,
    )
    ctx.out("metric", metric)
    ctx.out("predictions", preds)


def case_backtesting_stats(ctx, d):
    y = ctx.inp("y", d["y"])
    f = ForecasterStats(estimator=Arima(order=(1, 0, 1)))
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, refit=False)
    metric, preds = backtesting_stats(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error",
        interval=[0.05, 0.95], show_progress=False,
    )
    ctx.out("metric", metric)
    ctx.out("predictions", preds)


def case_grid_search(ctx, d):
    y = ctx.inp("y", d["y"])
    exog = ctx.inp("exog", d["exog"])
    f = ForecasterRecursive(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, refit=False)
    results = grid_search_forecaster(
        forecaster=f, y=y, exog=exog, cv=cv, param_grid={"alpha": [0.1, 1.0]},
        lags_grid=[3, 5], metric="mean_absolute_error", return_best=False,
        show_progress=False,
    )
    ctx.out("results", results)


def case_grid_search_multiseries_one_step(ctx, d):
    series = ctx.inp("series", d["series"])
    exog = ctx.inp("exog", d["exog"])
    f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=5)
    cv = OneStepAheadFold(initial_train_size=N_TRAIN)
    results = grid_search_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv,
        param_grid={"alpha": [0.1, 1.0]}, lags_grid=[3, 5],
        metric="mean_absolute_error", return_best=False, show_progress=False,
    )
    ctx.out("results", results)


def case_save_load(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog_text"]))
    exog_test = ctx.inp("exog_test", test(d["exog_text"]))
    f = ForecasterRecursive(estimator=lgbm(), lags=5)
    f.fit(y=y, exog=exog)
    ctx.out("predict_before", f.predict(steps=STEPS, exog=exog_test))
    with tempfile.TemporaryDirectory() as tmp:
        for backend in ["joblib", "skops"]:
            path = str(Path(tmp) / f"forecaster_{backend}")
            save_forecaster(f, file_name=path, backend=backend, verbose=False)
            ext = {"joblib": ".joblib", "skops": ".skops"}[backend]
            loaded = load_forecaster(path + ext, trusted=True, verbose=False)
            ctx.out(f"{backend}.predict", loaded.predict(steps=STEPS, exog=exog_test))
            ctx.out(f"{backend}.exog_dtypes_in_", loaded.exog_dtypes_in_)


# ---------------------------------------------------------------------------
# Extension (section 5, "Límites de la comprobación")
# ---------------------------------------------------------------------------

def case_recursive_differentiation(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog"]))
    exog_test = ctx.inp("exog_test", test(d["exog"]))
    f = ForecasterRecursive(estimator=Ridge(), lags=5, differentiation=1)
    ctx.out("create_train_X_y", f.create_train_X_y(y=y, exog=exog))
    f.fit(y=y, exog=exog, store_in_sample_residuals=True)
    fit_outputs(ctx, f)
    ctx.twice("predict", lambda: f.predict(steps=STEPS, exog=exog_test))
    ctx.twice("predict_interval", lambda: f.predict_interval(
        steps=STEPS, exog=exog_test, n_boot=50))


def case_recursive_quantiles_dist_out_sample(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    f = ForecasterRecursive(estimator=Ridge(), lags=5)
    f.fit(y=y, store_in_sample_residuals=True)
    ctx.twice("predict_quantiles", lambda: f.predict_quantiles(
        steps=STEPS, quantiles=[0.1, 0.5, 0.9], n_boot=50))
    ctx.twice("predict_dist", lambda: f.predict_dist(
        steps=STEPS, distribution=norm, n_boot=50))
    # Out-of-sample residuals from a backtest on the last part of the data.
    y_true = ctx.inp("y_true", d["y"].iloc[N_TRAIN:N_TRAIN + 40])
    y_pred = ctx.inp("y_pred", y_true.shift(1).bfill())
    f.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    ctx.out("out_sample_residuals_", f.out_sample_residuals_)
    ctx.out("out_sample_residuals_by_bin_", f.out_sample_residuals_by_bin_)
    ctx.twice("predict_interval_out_sample", lambda: f.predict_interval(
        steps=STEPS, n_boot=50, use_in_sample_residuals=False))


def case_multiseries_encoding_none(ctx, d):
    _multiseries_wide(ctx, d, None)


def case_drift_range(ctx, d):
    series = ctx.inp("series", train(d["series"]))
    exog = ctx.inp("exog", train(d["exog_cat"]))
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)
    ctx.out("series_values_range_", detector.series_values_range_)
    ctx.out("exog_values_range_", detector.exog_values_range_)
    last_window = ctx.inp("last_window", test(d["series"]) * 3)
    exog_new = ctx.inp("exog_new", test(d["exog_cat"]).assign(exog_1=100.0))
    ctx.twice("predict", lambda: detector.predict(
        last_window=last_window, exog=exog_new, verbose=False))


def case_drift_population(ctx, d):
    # Without the unused category: `PopulationDriftDetector.fit` raises a
    # `ValueError` from `chi2_contingency` when a categorical column has
    # unused categories (pre-existing, found while building the harness).
    data = d["exog_cat"].assign(
        exog_cat=d["exog_cat"]["exog_cat"].cat.remove_unused_categories()
    )
    ref = ctx.inp("ref", train(data).assign(y=train(d["y"])))
    new = ctx.inp("new", data.iloc[N_TRAIN:].assign(y=d["y"].iloc[N_TRAIN:] * 2))
    detector = PopulationDriftDetector(chunk_size=25)
    detector.fit(ref)
    ctx.twice("predict", lambda: detector.predict(new))


def case_select_features(ctx, d):
    y = ctx.inp("y", train(d["y"]))
    exog = ctx.inp("exog", train(d["exog"]))
    f = ForecasterRecursive(
        estimator=Ridge(), lags=10,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
    )
    selected = select_features(
        forecaster=f, selector=SelectFromModel(Ridge()), y=y, exog=exog,
        subsample=0.5, random_state=123, verbose=False,
    )
    ctx.out("selected", selected)


def case_stats_sarimax(ctx, d):
    _stats(ctx, d, Sarimax(order=(1, 0, 1), maxiter=200, disp=False))


def case_stats_ets(ctx, d):
    _stats(ctx, d, Ets(model="AAN"))


def case_stats_arar(ctx, d):
    _stats(ctx, d, Arar())


CASES = {
    name[len("case_"):]: fn
    for name, fn in list(globals().items())
    if name.startswith("case_") and callable(fn)
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(name: str, data: dict) -> dict:
    ctx = Ctx()
    start = time.perf_counter()
    error = None
    with common.capture_package_warnings() as warns:
        try:
            CASES[name](ctx, data)
        except Exception as exc:  # recorded and compared, never raised
            error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc(limit=-3)
            error += "\n" + tb
    return {
        "error": error,
        "seconds": round(time.perf_counter() - start, 3),
        "outputs": {k: common.fingerprints(v) for k, v in ctx.outputs.items()},
        "inputs_modified": ctx.modified_inputs(),
        "repeat_mismatch": ctx.repeat_mismatch,
        "shared_memory": common.shared_memory(ctx.outputs, ctx.inputs),
        "warnings": warns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", required=True, help="Output JSON file.")
    parser.add_argument("--index-unit", choices=["native", "ns"], default="native")
    parser.add_argument("--cases", default=None,
                        help="Comma separated subset of cases (default: all).")
    parser.add_argument("--expect-tree", default=None,
                        help="Abort unless skforecast is imported from this root.")
    parser.add_argument("--label", default=None, help="Free text stored in the JSON.")
    args = parser.parse_args()

    common.ensure_hash_seed()
    common.check_tree(args.expect_tree)
    names = args.cases.split(",") if args.cases else list(CASES)
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}. Available: {list(CASES)}")

    data = make_data(args.index_unit)
    results = {}
    for name in names:
        results[name] = run_case(name, data)
        status = "ERROR" if results[name]["error"] else "ok"
        print(f"{name:45s} {status:5s} {results[name]['seconds']:7.2f} s", flush=True)

    payload = {
        "label": args.label,
        "index_unit": args.index_unit,
        "environment": common.environment(),
        "cases": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    n_out = sum(len(r["outputs"]) for r in results.values())
    n_err = sum(1 for r in results.values() if r["error"])
    print(f"{len(results)} cases, {n_out} outputs, {n_err} errors -> {args.out}")


if __name__ == "__main__":
    main()
