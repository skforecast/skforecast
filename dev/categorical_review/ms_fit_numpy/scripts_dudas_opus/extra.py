"""
Extra scenarios for ForecasterRecursiveMultiSeries. The tree is chosen with
PYTHONPATH. Usage: python extra.py [catboost]
"""
import sys
import json
import warnings

import numpy as np
import pandas as pd

import skforecast
print("skforecast from:", skforecast.__file__, flush=True)

from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, grid_search_forecaster_multiseries,
    backtesting_forecaster_multiseries
)
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

with_catboost = len(sys.argv) > 1 and sys.argv[1] == "catboost"

rng = np.random.default_rng(1)
n, steps = 240, 12
idx = pd.date_range("2020-01-01", periods=n + steps, freq="D")
levels = ["a", "b", "c"]
offsets = {"a": 0.0, "b": 100.0, "c": 200.0}
effect = {10: 0.0, 20: 40.0, 30: -40.0}
cat_all = {lv: rng.choice([10, 20, 30], size=n + steps) for lv in levels}
truth = {lv: offsets[lv] + np.array([effect[c] for c in cat_all[lv]]) for lv in levels}
y_all = {lv: truth[lv] + rng.normal(size=n + steps) for lv in levels}
series = pd.DataFrame({lv: y_all[lv][:n] for lv in levels}, index=idx[:n])


def exog_for(sl, kind="category", per_series_categories=False, name="cat"):
    out = {}
    for lv in levels:
        c = cat_all[lv][sl]
        if kind == "category":
            cats = [10, 20, 30]
            if per_series_categories:
                cats = sorted(set(cat_all[lv][:n]))  # observed only
                cats = {"a": [10, 20, 30], "b": [10, 20, 30, 40], "c": [5, 10, 20, 30]}[lv]
            col = pd.Categorical(c, categories=cats)
        elif kind == "int":
            col = c.astype("int64")
        elif kind == "pyarrow":
            col = pd.array(c.astype(float), dtype="float64[pyarrow]")
        out[lv] = pd.DataFrame({"num": rng.normal(size=len(c)), name: col}, index=idx[sl])
    return out


def mae_of(p):
    p = p.pivot_table(index=p.index, columns="level", values="pred")
    return round(float(np.mean([
        np.abs(p[lv].to_numpy() - truth[lv][n:n + steps]).mean() for lv in levels
    ])), 3)


def run(name, make_f, exog_kind="category", per_series=False, col_name="cat", series_obj=None):
    rec = {"case": name}
    try:
        f = make_f()
        ex_tr = exog_for(slice(0, n), exog_kind, per_series, col_name)
        ex_te = exog_for(slice(n, n + steps), exog_kind, per_series, col_name)
        # keep the same values: re-create with the same cats (rng only affects 'num')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.fit(series=series if series_obj is None else series_obj, exog=ex_tr)
            p = f.predict(steps=steps, exog=ex_te, suppress_warnings=True)
        rec.update(status="ok", mae=mae_of(p),
                   warnings=sorted({x.category.__name__ for x in w
                                    if x.category.__module__.startswith("skforecast")
                                    and x.category.__name__ != "InputTypeWarning"}))
        try:
            rec["exog_dtypes_out_"] = {k: str(v) for k, v in f.exog_dtypes_out_.items()}
        except Exception:
            pass
    except Exception as e:
        rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:220]}")
    print(json.dumps(rec), flush=True)
    return rec


lgbm = lambda **kw: LGBMRegressor(n_estimators=60, verbose=-1, random_state=0, **kw)

if not with_catboost:
    # (a) integer (non-category) exog and user configuration by name
    run("int exog, LGBM fit_kwargs by name, cf=None",
        lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3, categorical_features=None,
                                               fit_kwargs={"categorical_feature": ["cat"]}),
        exog_kind="int")
    run("int exog, LGBM constructor name:cat, cf=None",
        lambda: ForecasterRecursiveMultiSeries(lgbm(categorical_feature="name:cat"), lags=3,
                                               categorical_features=None),
        exog_kind="int")
    run("int exog, HGB constructor by name, cf=None",
        lambda: ForecasterRecursiveMultiSeries(
            HistGradientBoostingRegressor(max_iter=60, categorical_features=["cat"]),
            lags=3, categorical_features=None),
        exog_kind="int")
    run("int exog, LGBM fit_kwargs by name, cf=None, ordinal_category",
        lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3, categorical_features=None,
                                               encoding="ordinal_category",
                                               fit_kwargs={"categorical_feature": ["cat"]}),
        exog_kind="int")
    run("int exog, XGB feature_types, cf=None, ordinal_category",
        lambda: ForecasterRecursiveMultiSeries(
            XGBRegressor(n_estimators=60, enable_categorical=True,
                         feature_types=["q", "q", "q", "q", "q", "c"]),
            lags=3, categorical_features=None, encoding="ordinal_category"),
        exog_kind="int")
    # (c) category with different categories per series (dict exog)
    for cf in ["auto", None]:
        for est_name, est in [("LGBM", lambda: lgbm()), ("HGB", lambda: HistGradientBoostingRegressor(max_iter=60))]:
            run(f"per-series categories, {est_name}, cf={cf}",
                lambda est=est, cf=cf: ForecasterRecursiveMultiSeries(est(), lags=3, categorical_features=cf),
                per_series=True)
    # (d) pyarrow exog
    run("pyarrow float exog, LGBM, cf=auto",
        lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3), exog_kind="pyarrow")
    # (e) names with forbidden characters
    run("exog name with [ ], LGBM",
        lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3), col_name="cat[1]")
    run("exog name with [ ], XGB",
        lambda: ForecasterRecursiveMultiSeries(XGBRegressor(n_estimators=60), lags=3), col_name="cat<1>")
    run("series name with [ ], onehot, LGBM",
        lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3, encoding="onehot"),
        series_obj=series.rename(columns={"a": "a[1]"}))
    # (f) set_in_sample_residuals consistency with category exog not managed
    for est_name, est in [("LGBM", lambda: lgbm()), ("XGB", lambda: XGBRegressor(n_estimators=60))]:
        rec = {"case": f"set_in_sample_residuals == fit residuals, {est_name}, cf=None"}
        try:
            f = ForecasterRecursiveMultiSeries(est(), lags=3, categorical_features=None)
            ex = exog_for(slice(0, n))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(series=series, exog=ex, store_in_sample_residuals=True)
                r1 = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
                f.set_in_sample_residuals(series=series, exog=ex)
                r2 = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
            rec.update(status="ok", equal=all(np.allclose(r1[k], r2[k]) for k in r1),
                       mean_abs_resid=round(float(np.mean(np.abs(r1["a"]))), 3))
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:220]}")
        print(json.dumps(rec), flush=True)
    # (g) OneStepAheadFold metric vs backtesting with refit=False, steps=1
    series_full = pd.DataFrame({lv: y_all[lv] for lv in levels}, index=idx)
    ex_full = exog_for(slice(0, n + steps))
    for est_name, est in [("LGBM", lambda: lgbm()), ("XGB", lambda: XGBRegressor(n_estimators=60))]:
        rec = {"case": f"OneStepAhead vs backtesting steps=1, {est_name}, cf=None"}
        try:
            f = ForecasterRecursiveMultiSeries(est(), lags=3, categorical_features=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grid_search_forecaster_multiseries(
                    forecaster=f, series=series_full, exog=ex_full,
                    param_grid={"n_estimators": [60]}, lags_grid=[3],
                    cv=OneStepAheadFold(initial_train_size=n), metric="mean_absolute_error",
                    return_best=False, verbose=False, show_progress=False,
                    aggregate_metric=["average"])
                m_osa = float(res["mean_absolute_error__average"].iloc[0])
                f2 = ForecasterRecursiveMultiSeries(est(), lags=3, categorical_features=None)
                m_bt, _ = backtesting_forecaster_multiseries(
                    forecaster=f2, series=series_full, exog=ex_full,
                    cv=TimeSeriesFold(steps=1, initial_train_size=n, refit=False),
                    metric="mean_absolute_error", show_progress=False)
                m_bt = float(m_bt.loc[m_bt["levels"] == "average", "mean_absolute_error"].iloc[0])
            rec.update(status="ok", osa=round(m_osa, 3), backtesting=round(m_bt, 3))
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:220]}")
        print(json.dumps(rec), flush=True)
else:
    from catboost import CatBoostRegressor
    cb = lambda **kw: CatBoostRegressor(iterations=60, verbose=0, allow_writing_files=False,
                                        random_seed=0, thread_count=1, **kw)
    run("int exog, CatBoost fit_kwargs by name, cf=None",
        lambda: ForecasterRecursiveMultiSeries(cb(), lags=3, categorical_features=None,
                                               fit_kwargs={"cat_features": ["cat"]}),
        exog_kind="int")
    run("int exog, CatBoost fit_kwargs by position, cf=None",
        lambda: ForecasterRecursiveMultiSeries(cb(), lags=3, categorical_features=None,
                                               fit_kwargs={"cat_features": [5]}),
        exog_kind="int")
    run("int exog, CatBoost constructor by position, cf=None",
        lambda: ForecasterRecursiveMultiSeries(cb(cat_features=[5]), lags=3, categorical_features=None),
        exog_kind="int")
    run("CatBoost, ordinal_category, cf=auto",
        lambda: ForecasterRecursiveMultiSeries(cb(), lags=3, encoding="ordinal_category"))
    run("CatBoost, ordinal_category, cf=None, int exog",
        lambda: ForecasterRecursiveMultiSeries(cb(), lags=3, encoding="ordinal_category",
                                               categorical_features=None),
        exog_kind="int")
