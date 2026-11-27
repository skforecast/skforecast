# Specific cases: fit failures, user-visible behaviors and holes.
# Usage: python cases.py <out.json>   (tree selected with PYTHONPATH)
import sys
import json
import warnings
import numpy as np
import pandas as pd
import skforecast

print("skforecast file:", skforecast.__file__, flush=True)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)

N = 200
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=N, freq="D")
cat_vals = rng.choice([10, 20, 30], size=N)
num = rng.normal(size=N)
effect = {10: 5.0, 20: -5.0, 30: 12.0}
base = np.array([effect[v] for v in cat_vals]) + num
series = pd.DataFrame(
    {s: base + i * 2 + rng.normal(scale=0.1, size=N) for i, s in enumerate(["s_a", "s_b", "s_c"])},
    index=idx,
)
exog = pd.DataFrame(
    {"exog_num": num, "exog_cat": pd.Categorical(cat_vals, categories=[10, 20, 30])},
    index=idx,
)
LGB = dict(n_estimators=20, verbose=-1, min_child_samples=5, min_data_per_group=1,
           cat_smooth=0, random_state=1)

results = {}


def case(name):
    def deco(fun):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                out = fun()
                res = {"status": "ok", "out": out}
            except Exception as e:
                res = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:260]}"}
        res["warnings"] = sorted({
            f"{x.category.__name__}: {str(x.message)[:120]}" for x in w
            if x.category.__name__ not in ("FutureWarning", "DeprecationWarning",
                                           "InputTypeWarning")
        })
        results[name] = res
        print(name, "->", json.dumps(res, default=str)[:600], flush=True)
        return fun
    return deco


def fit_predict(est, series=series, exog=exog, steps=3, **kw):
    f = ForecasterRecursiveMultiSeries(estimator=est, lags=3, **kw)
    if isinstance(series, pd.DataFrame):
        s_tr, e_tr, e_te = series.iloc[:-steps], exog.iloc[:-steps], exog.iloc[-steps:]
    else:
        s_tr = {k: v.iloc[:-steps] for k, v in series.items()}
        e_tr = {k: (None if v is None else v.iloc[:-steps]) for k, v in exog.items()}
        e_te = {k: (None if v is None else v.iloc[-steps:]) for k, v in exog.items()}
    f.fit(series=s_tr, exog=e_tr)
    p = f.predict(steps=steps, exog=e_te)
    return f, p


# A. pyarrow exog -------------------------------------------------------------
for est_name, est in [("lgbm", LGBMRegressor(**LGB)), ("xgb", XGBRegressor(n_estimators=10)),
                      ("hgb", HistGradientBoostingRegressor(max_iter=10)),
                      ("ridge", Ridge())]:
    @case(f"A_pyarrow_exog_{est_name}")
    def _():
        ex = pd.DataFrame({"exog_num": pd.array(num, dtype="double[pyarrow]"),
                           "exog_int": pd.array(cat_vals, dtype="int64[pyarrow]")}, index=idx)
        f, p = fit_predict(est, exog=ex)
        return {"pred_mean": float(p["pred"].mean())}

# B. names with special characters ---------------------------------------------
for est_name, est in [("lgbm", LGBMRegressor(**LGB)), ("xgb", XGBRegressor(n_estimators=10))]:
    @case(f"B_series_name_brackets_onehot_{est_name}")
    def _():
        s = series.rename(columns={"s_a": "s[a]", "s_b": "s<b>"})
        f, p = fit_predict(est, series=s, encoding="onehot")
        return {"pred_mean": float(p["pred"].mean())}

    @case(f"B_exog_name_special_{est_name}")
    def _():
        ex = exog.rename(columns={"exog_num": "exog[num]<1>"})
        f, p = fit_predict(est, exog=ex)
        return {"pred_mean": float(p["pred"].mean())}

    @case(f"B_exog_name_json_chars_{est_name}")
    def _():
        ex = exog.rename(columns={"exog_num": 'exog "num": {x}'})
        f, p = fit_predict(est, exog=ex)
        return {"pred_mean": float(p["pred"].mean())}

# C. nullable and bool exog ----------------------------------------------------
for est_name, est in [("lgbm", LGBMRegressor(**LGB)),
                      ("hgb", HistGradientBoostingRegressor(max_iter=10)),
                      ("xgb", XGBRegressor(n_estimators=10))]:
    @case(f"C_nullable_bool_exog_{est_name}")
    def _():
        v = pd.array(cat_vals, dtype="Int64")
        v[::7] = pd.NA
        ex = pd.DataFrame({"exog_num": num, "exog_Int64": v,
                           "exog_bool": cat_vals > 15}, index=idx)
        f, p = fit_predict(est, exog=ex)
        return {"pred_mean": float(p["pred"].mean())}

# D. feature names on the estimator -------------------------------------------
@case("D_feature_names_on_estimator")
def _():
    out = {}
    f, _p = fit_predict(LinearRegression(), categorical_features="auto")
    out["sklearn_feature_names_in_"] = hasattr(f.estimator, "feature_names_in_")
    X, y = f.create_train_X_y(series=series, exog=exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f.estimator.predict(X)
        out["sklearn_predict_with_X_train_df_warnings"] = [str(x.message)[:90] for x in w]
    f, _p = fit_predict(LGBMRegressor(**LGB))
    out["lgbm_feature_name_"] = list(f.estimator.feature_name_)
    X, y = f.create_train_X_y(series=series, exog=exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f.estimator.predict(X)
        out["lgbm_predict_with_X_train_df_warnings"] = [str(x.message)[:90] for x in w]
    f, _p = fit_predict(XGBRegressor(n_estimators=10))
    out["xgb_feature_names"] = f.estimator.get_booster().feature_names
    out["importances_cols"] = f.get_feature_importances()["feature"].tolist()
    return out

# E. Pipeline that selects columns by name ------------------------------------
@case("E_pipeline_selects_columns_by_name")
def _():
    pipe = make_pipeline(
        ColumnTransformer([("sc", StandardScaler(), ["lag_1", "exog_num"])],
                          remainder="passthrough"),
        Ridge(),
    )
    f, p = fit_predict(pipe, exog=exog[["exog_num"]])
    return {"pred_mean": float(p["pred"].mean())}

# F. in-place estimator with mixed dtypes ---------------------------------------
@case("F_copy_X_False_int_exog")
def _():
    ex = pd.DataFrame({"exog_num": num, "exog_int": cat_vals}, index=idx)
    f, p = fit_predict(LinearRegression(copy_X=False), exog=ex)
    return {"pred_mean": float(p["pred"].mean())}

@case("F_copy_X_False_float_exog")
def _():
    f, p = fit_predict(LinearRegression(copy_X=False), exog=exog[["exog_num"]])
    return {"pred_mean": float(p["pred"].mean())}

# G. unmanaged non numeric exog ---------------------------------------------------
for cf_name, cf in [("None", None), ("list", ["exog_cat"])]:
    @case(f"G_string_category_unmanaged_cf_{cf_name}_lgbm")
    def _():
        ex = exog.assign(exog_txt=pd.Categorical(np.where(cat_vals > 15, "a", "b")))
        f, p = fit_predict(LGBMRegressor(**LGB), exog=ex, categorical_features=cf)
        return {"pred_mean": float(p["pred"].mean())}

    @case(f"G_object_str_unmanaged_cf_{cf_name}_lgbm")
    def _():
        ex = exog.assign(exog_txt=np.where(cat_vals > 15, "a", "b"))
        f, p = fit_predict(LGBMRegressor(**LGB), exog=ex, categorical_features=cf)
        return {"pred_mean": float(p["pred"].mean())}

# I. set_in_sample_residuals equals the residuals of fit ----------------------------
for est_name, mk in [("lgbm", lambda: LGBMRegressor(**LGB)),
                     ("xgb_enable", lambda: XGBRegressor(n_estimators=10, enable_categorical=True)),
                     ("hgb", lambda: HistGradientBoostingRegressor(max_iter=10))]:
    for cf_name, cf in [("None", None), ("auto", "auto")]:
        @case(f"I_set_in_sample_residuals_{est_name}_cf_{cf_name}")
        def _():
            f = ForecasterRecursiveMultiSeries(estimator=mk(), lags=3, categorical_features=cf)
            f.fit(series=series, exog=exog, store_in_sample_residuals=True)
            r_fit = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
            f.set_in_sample_residuals(series=series, exog=exog)
            r_set = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
            return {"max_abs_diff": float(max(np.abs(r_fit[k] - r_set[k]).max() for k in r_fit))}

# J. one-step-ahead search metric equals backtesting with steps=1 --------------------
for est_name, mk, grid in [
    ("lgbm", lambda: LGBMRegressor(**LGB), {"n_estimators": [20]}),
    ("xgb_enable", lambda: XGBRegressor(n_estimators=10, enable_categorical=True), {"n_estimators": [10]}),
    ("xgb_feature_types", lambda: XGBRegressor(
        n_estimators=10, enable_categorical=True,
        feature_types=["q", "q", "q", "q", "q", "c"]), {"n_estimators": [10]}),
    ("hgb", lambda: HistGradientBoostingRegressor(max_iter=10), {"max_iter": [10]}),
    ("catboost_fit_pos", lambda: CatBoostRegressor(
        iterations=20, verbose=0, allow_writing_files=False, random_seed=1,
        thread_count=1), {"iterations": [20]}),
]:
    for cf_name, cf in [("None", None), ("auto", "auto")]:
        @case(f"J_one_step_vs_backtesting_{est_name}_cf_{cf_name}")
        def _():
            fk = {"cat_features": [5]} if est_name.startswith("catboost") and cf is None else None
            if est_name == "xgb_feature_types" and cf is not None:
                return "skipped"
            f = ForecasterRecursiveMultiSeries(
                estimator=mk(), lags=3, categorical_features=cf, fit_kwargs=fk
            )
            cv_one = OneStepAheadFold(initial_train_size=150)
            res = grid_search_forecaster_multiseries(
                forecaster=f, series=series, exog=exog, cv=cv_one, param_grid=grid,
                lags_grid=[3], metric="mean_absolute_error", return_best=False,
                verbose=False, show_progress=False,
            )
            m_one = float(res["mean_absolute_error__weighted_average"].iloc[0])
            f2 = ForecasterRecursiveMultiSeries(
                estimator=mk(), lags=3, categorical_features=cf, fit_kwargs=fk
            )
            cv_bt = TimeSeriesFold(steps=1, initial_train_size=150, refit=False)
            m, _p = backtesting_forecaster_multiseries(
                forecaster=f2, series=series, exog=exog, cv=cv_bt,
                metric="mean_absolute_error", verbose=False, show_progress=False,
                add_aggregated_metric=True, n_jobs=1,
            )
            m_bt = float(m.loc[m["levels"] == "weighted_average", "mean_absolute_error"].iloc[0])
            return {"one_step": m_one, "backtesting": m_bt, "equal": bool(abs(m_one - m_bt) < 1e-8)}

# K. dict series of different lengths, exog missing for one series ---------------------
for est_name, mk in [("lgbm", lambda: LGBMRegressor(**LGB)),
                     ("hgb", lambda: HistGradientBoostingRegressor(max_iter=10))]:
    for cf_name, cf in [("None", None), ("auto", "auto")]:
        @case(f"K_dict_series_exog_missing_{est_name}_cf_{cf_name}")
        def _():
            s = {"s_a": series["s_a"], "s_b": series["s_b"].iloc[40:], "s_c": series["s_c"].iloc[80:]}
            e = {"s_a": exog, "s_b": exog.iloc[40:], "s_c": None}
            f, p = fit_predict(mk(), series=s, exog=e, categorical_features=cf)
            return {"pred": [round(float(v), 4) for v in p["pred"].to_numpy()[:6]]}

        @case(f"K_dict_exog_column_missing_in_one_series_{est_name}_cf_{cf_name}")
        def _():
            s = {"s_a": series["s_a"], "s_b": series["s_b"].iloc[40:], "s_c": series["s_c"].iloc[80:]}
            e = {"s_a": exog, "s_b": exog.iloc[40:], "s_c": exog[["exog_num"]].iloc[80:]}
            f, p = fit_predict(mk(), series=s, exog=e, categorical_features=cf)
            return {"pred": [round(float(v), 4) for v in p["pred"].to_numpy()[:6]]}

# M. intervals after fit ---------------------------------------------------------------
@case("M_predict_interval_cf_None_lgbm")
def _():
    f = ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(**LGB), lags=3,
                                       categorical_features=None)
    f.fit(series=series.iloc[:-3], exog=exog.iloc[:-3], store_in_sample_residuals=True)
    p = f.predict_interval(steps=3, exog=exog.iloc[-3:], method="conformal")
    return {"cols": p.columns.tolist(), "pred_mean": float(p["pred"].mean())}

# P. accuracy of the cf=None category case (real values at the end) ---------------------
for est_name, mk in [("lgbm", lambda: LGBMRegressor(**LGB)),
                     ("xgb_enable", lambda: XGBRegressor(n_estimators=20, enable_categorical=True)),
                     ("hgb", lambda: HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=5))]:
    @case(f"P_mae_cf_None_category_10_20_30_{est_name}")
    def _():
        f, p = fit_predict(mk(), steps=20, categorical_features=None)
        true = series.iloc[-20:].melt(ignore_index=False)
        mae = float(np.mean(np.abs(
            p.reset_index().sort_values(["level", "index"])["pred"].to_numpy()
            - true.reset_index().sort_values(["variable", "index"])["value"].to_numpy()
        )))
        return {"mae_20_steps": round(mae, 4)}

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, default=str)
print("done", len(results))
