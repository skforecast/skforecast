"""
Identity check: fingerprints (SHA-256) of the outputs of skforecast across the
five autoregressive forecasters, ForecasterStats, ForecasterEquivalentDate,
numeric, categorical and text exogenous variables, exog as dict and in long
format, window and calendar features, intervals (bootstrapping, conformal,
binned), backtesting, hyperparameter search and joblib/skops persistence.

Run it once per environment (one process at a time):

    /root/.venvs/skforecast/bin/python dev/pandas3_migration/identity_cases.py
    PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/identity_cases.py

Each run writes results/identity_pandas<version>.json with, per case, the
strict fingerprint (dtypes, datetime units and freq included), the normalised
fingerprint (values only), a description (shapes, dtypes, index dtype) and the
pandas deprecation warnings raised. Compare two runs with identity_compare.py.

Optional: --index-unit ns forces the unit of the input index (pandas 3 infers
[us] from strings, pandas 2 always [ns]); --tag <name> appends a suffix to the output file (used to compare
"before" and "after" a prototype patch, see prototypes/README.md), and
--patch <module> imports a module from prototypes/ before running the cases
(the module monkeypatches skforecast).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import describe, env_info, fingerprint, result_path, save_json  # noqa: E402

CASES: dict[str, callable] = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


# ----------------------------------------------------------------------------
# Deterministic data
# ----------------------------------------------------------------------------
def make_data(index_unit=None):
    rng = np.random.default_rng(123)
    n = 300
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    if index_unit:
        # pandas 3 infers datetime64[us] from strings; force a unit to compare
        # like with like (pandas 2 always gives [ns]).
        index = index.as_unit(index_unit)
    y = pd.Series(
        10 + np.sin(np.arange(n) * 2 * np.pi / 7) * 3 + rng.normal(0, 0.5, n),
        index=index, name="y",
    )
    exog = pd.DataFrame(
        {
            "exog_float": rng.normal(0, 1, n),
            "exog_float2": np.cos(np.arange(n) / 10.0),
            "exog_int": rng.integers(0, 5, n),
        },
        index=index,
    )
    cats = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    exog_cat = exog.copy()
    exog_cat["exog_cat"] = pd.Categorical(cats, categories=["a", "b", "c"])
    exog_text = exog.copy()
    exog_text["exog_text"] = cats  # object in pandas 2, str in pandas 3
    series = pd.DataFrame(
        {
            "s1": y.to_numpy(),
            "s2": y.to_numpy() * 1.5 + rng.normal(0, 0.3, n),
            "s3": y.to_numpy() ** 0.5 + rng.normal(0, 0.1, n),
        },
        index=index,
    )
    # Dict of series with different lengths and dict of exog
    series_dict = {
        "s1": series["s1"],
        "s2": series["s2"].iloc[30:],
        "s3": series["s3"].iloc[:-20],
    }
    exog_dict = {
        "s1": exog_cat,
        "s2": exog_cat.iloc[30:],
        "s3": exog_cat.iloc[:-20],
    }
    # Long format (series and exog)
    series_long = (
        series.stack().rename("value").reset_index()
        .rename(columns={"level_0": "date", "level_1": "series_id"})
    )
    series_long.columns = ["date", "series_id", "value"]
    exog_long = pd.concat(
        [exog_cat.assign(series_id=k) for k in ["s1", "s2", "s3"]]
    ).reset_index().rename(columns={"index": "date"})
    # Classification target
    y_class = pd.Series((y > y.rolling(7).mean().bfill()).astype(int).to_numpy(),
                        index=index, name="y")
    return {
        "y": y, "exog": exog, "exog_cat": exog_cat, "exog_text": exog_text,
        "series": series, "series_dict": series_dict, "exog_dict": exog_dict,
        "series_long": series_long, "exog_long": exog_long, "y_class": y_class,
    }


STEPS = 10
TRAIN_END = 250  # observations used to fit; the rest is kept for predict exog


def split(d):
    return {
        "y_train": d["y"].iloc[:TRAIN_END],
        "exog_train": d["exog"].iloc[:TRAIN_END],
        "exog_test": d["exog"].iloc[TRAIN_END:TRAIN_END + STEPS],
        "exog_cat_train": d["exog_cat"].iloc[:TRAIN_END],
        "exog_cat_test": d["exog_cat"].iloc[TRAIN_END:TRAIN_END + STEPS],
        "exog_text_train": d["exog_text"].iloc[:TRAIN_END],
        "exog_text_test": d["exog_text"].iloc[TRAIN_END:TRAIN_END + STEPS],
        "series_train": d["series"].iloc[:TRAIN_END],
        "series_dict_train": {k: v.loc[: d["y"].index[TRAIN_END - 1]] for k, v in d["series_dict"].items()},
        "exog_dict_train": {k: v.loc[: d["y"].index[TRAIN_END - 1]] for k, v in d["exog_dict"].items()},
        "exog_dict_test": {k: v.loc[d["y"].index[TRAIN_END]: d["y"].index[TRAIN_END + STEPS - 1]] for k, v in d["exog_dict"].items()},
        "y_class_train": d["y_class"].iloc[:TRAIN_END],
    }


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------
@case("recursive_numeric_exog")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import RollingFeatures
    f = ForecasterRecursive(
        estimator=LinearRegression(), lags=7,
        window_features=RollingFeatures(stats=["mean", "std"], window_sizes=7),
    )
    X, y = f.create_train_X_y(y=s["y_train"], exog=s["exog_train"])
    f.fit(y=s["y_train"], exog=s["exog_train"], store_in_sample_residuals=True)
    out = {
        "X_train": X, "y_train": y,
        "exog_dtypes_in_": f.exog_dtypes_in_, "exog_dtypes_out_": f.exog_dtypes_out_,
        "predict": f.predict(steps=STEPS, exog=s["exog_test"]),
        "predict_interval_bootstrapping": f.predict_interval(
            steps=STEPS, exog=s["exog_test"], method="bootstrapping", n_boot=50,
            use_binned_residuals=False, random_state=123),
        "predict_interval_binned": f.predict_interval(
            steps=STEPS, exog=s["exog_test"], method="bootstrapping", n_boot=50,
            use_binned_residuals=True, random_state=123),
        "predict_interval_conformal": f.predict_interval(
            steps=STEPS, exog=s["exog_test"], method="conformal", interval=[0.05, 0.95]),
        "in_sample_residuals_": f.in_sample_residuals_,
        "in_sample_residuals_by_bin_": f.in_sample_residuals_by_bin_,
        "create_predict_X": f.create_predict_X(steps=STEPS, exog=s["exog_test"]),
    }
    return out


@case("recursive_calendar_features")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import CalendarFeatures
    f = ForecasterRecursive(
        estimator=LinearRegression(), lags=7,
        calendar_features=CalendarFeatures(features=["day_of_week", "month"], encoding=None),
    )
    X, y = f.create_train_X_y(y=s["y_train"])
    f.fit(y=s["y_train"])
    return {"X_train": X, "predict": f.predict(steps=STEPS)}


@case("recursive_categorical_exog_lgbm")
def _(d, s):
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive
    f = ForecasterRecursive(
        estimator=LGBMRegressor(n_estimators=20, random_state=123, verbose=-1, n_jobs=1),
        lags=7, categorical_features="auto",
    )
    X, y = f.create_train_X_y(y=s["y_train"], exog=s["exog_cat_train"])
    f.fit(y=s["y_train"], exog=s["exog_cat_train"])
    return {
        "X_train": X, "exog_dtypes_in_": f.exog_dtypes_in_, "exog_dtypes_out_": f.exog_dtypes_out_,
        "predict": f.predict(steps=STEPS, exog=s["exog_cat_test"]),
    }


@case("recursive_text_exog_lgbm")
def _(d, s):
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive
    f = ForecasterRecursive(
        estimator=LGBMRegressor(n_estimators=20, random_state=123, verbose=-1, n_jobs=1),
        lags=7, categorical_features="auto",
    )
    X, y = f.create_train_X_y(y=s["y_train"], exog=s["exog_text_train"])
    f.fit(y=s["y_train"], exog=s["exog_text_train"])
    return {
        "X_train": X, "exog_dtypes_in_": f.exog_dtypes_in_, "exog_dtypes_out_": f.exog_dtypes_out_,
        "predict": f.predict(steps=STEPS, exog=s["exog_text_test"]),
    }


@case("recursive_text_exog_linear_no_categorical")
def _(d, s):
    # Text column with categorical_features=None: must raise the same error
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    f = ForecasterRecursive(estimator=LinearRegression(), lags=7, categorical_features=None)
    try:
        f.fit(y=s["y_train"], exog=s["exog_text_train"])
        return {"result": "fitted"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


@case("direct")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.direct import ForecasterDirect
    f = ForecasterDirect(estimator=LinearRegression(), lags=7, steps=5)
    X, y = f.create_train_X_y(y=s["y_train"], exog=s["exog_train"])
    f.fit(y=s["y_train"], exog=s["exog_train"], store_in_sample_residuals=True)
    return {
        "X_train": X, "y_train": y,
        "predict": f.predict(steps=5, exog=s["exog_test"]),
        "predict_interval_conformal": f.predict_interval(
            steps=5, exog=s["exog_test"], method="conformal", interval=[0.05, 0.95]),
        "predict_interval_bootstrapping": f.predict_interval(
            steps=5, exog=s["exog_test"], method="bootstrapping", n_boot=50, random_state=123),
    }


def _multiseries(encoding, series, exog, exog_test, levels=None):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.preprocessing import RollingFeatures
    f = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=7, encoding=encoding,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
        dropna_from_series=False,
    )
    X, y = f.create_train_X_y(series=series, exog=exog)
    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    out = {
        "X_train": X, "y_train": y,
        "exog_dtypes_in_": f.exog_dtypes_in_, "exog_dtypes_out_": f.exog_dtypes_out_,
        "series_names_in_": f.series_names_in_,
        "predict": f.predict(steps=STEPS, exog=exog_test, levels=levels),
        "predict_interval_bootstrapping": f.predict_interval(
            steps=STEPS, exog=exog_test, levels=levels, method="bootstrapping", n_boot=30,
            use_binned_residuals=False, random_state=123),
        "predict_interval_conformal": f.predict_interval(
            steps=STEPS, exog=exog_test, levels=levels, method="conformal", interval=[0.05, 0.95]),
        "in_sample_residuals_": f.in_sample_residuals_,
        "level_row_slices": {k: (v.start, v.stop) for k, v in f._get_level_row_slices(X).items()},
    }
    return out


@case("multiseries_wide_ordinal")
def _(d, s):
    return _multiseries("ordinal", s["series_train"], s["exog_cat_train"], s["exog_cat_test"])


@case("multiseries_wide_ordinal_category")
def _(d, s):
    return _multiseries("ordinal_category", s["series_train"], s["exog_cat_train"], s["exog_cat_test"])


@case("multiseries_wide_onehot")
def _(d, s):
    return _multiseries("onehot", s["series_train"], s["exog_cat_train"], s["exog_cat_test"])


@case("multiseries_dict_ordinal")
def _(d, s):
    return _multiseries("ordinal", s["series_dict_train"], s["exog_dict_train"], s["exog_dict_test"])


@case("multiseries_long_format")
def _(d, s):
    from skforecast.preprocessing import reshape_series_long_to_dict, reshape_exog_long_to_dict
    sd = reshape_series_long_to_dict(d["series_long"], freq="D", series_id="series_id", index="date", values="value")
    ed = reshape_exog_long_to_dict(d["exog_long"], freq="D", series_id="series_id", index="date")
    end = d["y"].index[TRAIN_END - 1]
    sd_train = {k: v.loc[:end] for k, v in sd.items()}
    ed_train = {k: v.loc[:end] for k, v in ed.items()}
    ed_test = {k: v.loc[d["y"].index[TRAIN_END]: d["y"].index[TRAIN_END + STEPS - 1]] for k, v in ed.items()}
    out = _multiseries("ordinal", sd_train, ed_train, ed_test)
    out["reshaped_series"] = sd
    out["reshaped_exog"] = ed
    return out


@case("direct_multivariate")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.direct import ForecasterDirectMultiVariate
    f = ForecasterDirectMultiVariate(estimator=LinearRegression(), level="s1", lags=7, steps=3)
    X, y = f.create_train_X_y(series=s["series_train"], exog=s["exog_train"])
    f.fit(series=s["series_train"], exog=s["exog_train"], store_in_sample_residuals=True)
    return {
        "X_train": X, "y_train": y,
        "predict": f.predict(steps=3, exog=s["exog_test"]),
        "predict_interval_conformal": f.predict_interval(
            steps=3, exog=s["exog_test"], method="conformal", interval=[0.05, 0.95]),
    }


@case("recursive_classifier")
def _(d, s):
    from sklearn.tree import DecisionTreeClassifier
    from skforecast.recursive import ForecasterRecursiveClassifier
    f = ForecasterRecursiveClassifier(estimator=DecisionTreeClassifier(random_state=123, max_depth=3), lags=7)
    X, y = f.create_train_X_y(y=s["y_class_train"], exog=s["exog_cat_train"])
    f.fit(y=s["y_class_train"], exog=s["exog_cat_train"])
    return {
        "X_train": X, "y_train": y,
        "predict": f.predict(steps=STEPS, exog=s["exog_cat_test"]),
        "predict_proba": f.predict_proba(steps=STEPS, exog=s["exog_cat_test"]),
    }


@case("stats_arima")
def _(d, s):
    from skforecast.recursive import ForecasterStats
    from skforecast.stats import Arima
    f = ForecasterStats(estimator=Arima(order=(1, 0, 1)))
    f.fit(y=s["y_train"], exog=s["exog_train"][["exog_float"]])
    return {
        "predict": f.predict(steps=STEPS, exog=s["exog_test"][["exog_float"]]),
        "predict_interval": f.predict_interval(steps=STEPS, exog=s["exog_test"][["exog_float"]], interval=[0.05, 0.95]),
    }


@case("equivalent_date")
def _(d, s):
    from skforecast.recursive import ForecasterEquivalentDate
    f = ForecasterEquivalentDate(offset=pd.offsets.Day(7), n_offsets=2)
    f.fit(y=s["y_train"])
    return {"predict": f.predict(steps=STEPS)}


@case("backtesting_recursive")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
    f = ForecasterRecursive(estimator=LinearRegression(), lags=7)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=200, refit=True, fixed_train_size=False)
    metric, preds = backtesting_forecaster(
        forecaster=f, y=d["y"], exog=d["exog"], cv=cv, metric=["mean_absolute_error", "mean_squared_error"],
        interval=[0.1, 0.9], interval_method="conformal", n_jobs=1, show_progress=False, verbose=False,
    )
    return {"metric": metric, "predictions": preds}


@case("backtesting_multiseries")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold
    f = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=7, encoding="ordinal")
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=200, refit=False)
    metric, preds = backtesting_forecaster_multiseries(
        forecaster=f, series=d["series_dict"], exog=d["exog_dict"], cv=cv, metric="mean_absolute_error",
        interval=[0.1, 0.9], interval_method="bootstrapping", n_boot=30, n_jobs=1,
        show_progress=False, verbose=False,
    )
    return {"metric": metric, "predictions": preds}


@case("backtesting_stats")
def _(d, s):
    from skforecast.recursive import ForecasterStats
    from skforecast.stats import Arima
    from skforecast.model_selection import backtesting_stats, TimeSeriesFold
    f = ForecasterStats(estimator=Arima(order=(1, 0, 1)))
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=250, refit=False)
    metric, preds = backtesting_stats(
        forecaster=f, y=d["y"], cv=cv, metric="mean_absolute_error", interval=[0.05, 0.95],
        n_jobs=1, show_progress=False, verbose=False,
    )
    return {"metric": metric, "predictions": preds}


@case("grid_search_recursive")
def _(d, s):
    from sklearn.linear_model import Ridge
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
    f = ForecasterRecursive(estimator=Ridge(), lags=7)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=220, refit=False)
    results = grid_search_forecaster(
        forecaster=f, y=d["y"], exog=d["exog"], cv=cv, param_grid={"alpha": [0.1, 1.0]},
        lags_grid=[3, 7], metric="mean_absolute_error", return_best=False, n_jobs=1,
        show_progress=False, verbose=False,
    )
    # The params column holds dicts; hash its repr.
    results = results.copy()
    results["params"] = results["params"].astype(str)
    results["lags"] = results["lags"].astype(str)
    return {"results": results}


@case("grid_search_multiseries")
def _(d, s):
    from sklearn.linear_model import Ridge
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.model_selection import grid_search_forecaster_multiseries, OneStepAheadFold
    f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=7, encoding="ordinal_category")
    cv = OneStepAheadFold(initial_train_size=220)
    results = grid_search_forecaster_multiseries(
        forecaster=f, series=d["series"], exog=d["exog_cat"], cv=cv, param_grid={"alpha": [0.1, 1.0]},
        lags_grid=[3, 7], metric="mean_absolute_error", return_best=False, n_jobs=1,
        show_progress=False, verbose=False,
    )
    results = results.copy()
    for col in ("params", "lags", "levels"):
        if col in results:
            results[col] = results[col].astype(str)
    return {"results": results}


@case("save_load_joblib_skops")
def _(d, s):
    from sklearn.linear_model import LinearRegression
    from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
    from skforecast.utils import save_forecaster, load_forecaster
    out = {}
    f = ForecasterRecursive(estimator=LinearRegression(), lags=7, categorical_features="auto")
    f.fit(y=s["y_train"], exog=s["exog_text_train"])
    fm = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=7, encoding="ordinal_category")
    fm.fit(series=s["series_dict_train"], exog=s["exog_dict_train"])
    persist_dir = os.path.join(HERE, "results", "persisted", f"pandas{pd.__version__}")
    os.makedirs(persist_dir, exist_ok=True)
    for backend, ext in (("joblib", "joblib"), ("skops", "skops")):
        for name, fc, kw in (
            ("recursive", f, {"steps": STEPS, "exog": s["exog_text_test"]}),
            ("multiseries", fm, {"steps": STEPS, "exog": s["exog_dict_test"]}),
        ):
            path = os.path.join(persist_dir, f"{name}.{ext}")
            save_forecaster(fc, file_name=path, backend=backend, verbose=False)
            loaded = load_forecaster(path, backend=backend, trusted=True, verbose=False)
            out[f"{backend}_{name}_predict_after_load"] = loaded.predict(**kw)
            out[f"{backend}_{name}_exog_dtypes_in_after_load"] = loaded.exog_dtypes_in_
    return out


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def run(selected=None, tag="", patch=None, index_unit=None):
    if patch:
        import importlib
        sys.path.insert(0, os.path.join(HERE, "prototypes"))
        importlib.import_module(patch)
    d = make_data(index_unit)
    s = split(d)
    results = {"env": env_info(), "tag": tag, "patch": patch, "index_unit": index_unit,
               "input_index_dtype": str(d["y"].index.dtype), "cases": {}}
    pd4 = getattr(pd.errors, "Pandas4Warning", None)
    for name, fn in CASES.items():
        if selected and name not in selected:
            continue
        rec = {"outputs": {}, "pandas_warnings": [], "error": None}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                outputs = fn(d, s)
                for k, v in outputs.items():
                    rec["outputs"][k] = {
                        "strict": fingerprint(v, strict=True),
                        "normalized": fingerprint(v, strict=False),
                        "describe": describe(v),
                    }
            except Exception:  # noqa: BLE001
                rec["error"] = traceback.format_exc()[-2000:]
        for x in w:
            if issubclass(x.category, (DeprecationWarning, FutureWarning)) or (pd4 and issubclass(x.category, pd4)):
                rec["pandas_warnings"].append(f"{x.category.__name__}: {str(x.message)[:120]} ({os.path.basename(x.filename)}:{x.lineno})")
        rec["pandas_warnings"] = sorted(set(rec["pandas_warnings"]))
        results["cases"][name] = rec
        status = "ERROR" if rec["error"] else f"{len(rec['outputs'])} outputs"
        print(f"{name:45s} {status}  warnings={len(rec['pandas_warnings'])}", flush=True)
        if rec["error"]:
            print(rec["error"].splitlines()[-1])
    path = result_path("identity", tag)
    save_json(path, results)
    print("saved", path)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--index-unit", default=None, help="force the unit of the input DatetimeIndex, e.g. ns")
    a = ap.parse_args()
    run(a.cases, a.tag, a.patch, a.index_unit)
