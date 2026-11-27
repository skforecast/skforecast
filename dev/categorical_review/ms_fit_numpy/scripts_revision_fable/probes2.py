"""Second set of probes. Prints one block per probe; run on base and branch and diff."""
import copy
import sys
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
import common as C


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out = f"ERR {type(e).__name__}: {str(e)[:200]}"
    return out, w


def fmt(p):
    return p if isinstance(p, str) else np.round(p["pred"].to_numpy()[:6], 4).tolist()


print("\n===== P11: guide 'training and prediction matrices': estimator.predict(create_train_X_y)")
series, exog, exog_fut, *_ = C.make_data("float", n=120)
for name, est in {"LGBMRegressor": LGBMRegressor(n_estimators=10, verbose=-1, random_state=1),
                  "Ridge": Ridge(), "HGB": HistGradientBoostingRegressor(max_iter=10),
                  "XGBRegressor": XGBRegressor(n_estimators=10)}.items():
    for enc in ["ordinal", "ordinal_category", "onehot"]:
        f = ForecasterRecursiveMultiSeries(est, lags=3, encoding=enc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit(series=series, exog=exog)
            X_train, y_train = f.create_train_X_y(series=series, exog=exog)
            X_pred = f.create_predict_X(steps=2, exog=exog_fut)
        p1, w1 = run(lambda: f.estimator.predict(X_train))
        p2, w2 = run(lambda: f.estimator.predict(X_pred.drop(columns="level")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = f.predict(steps=2, exog=exog_fut)["pred"].to_numpy()
        same = "-" if isinstance(p2, str) else bool(np.allclose(np.sort(p2), np.sort(ref)))
        print(f"  {name:14} enc={enc:17} predict(X_train): {'ok' if not isinstance(p1, str) else p1[:90]} warnings={sorted({str(x.message)[:75] for x in w1})} | "
              f"predict(X_predict) equals forecaster.predict: {same} {'' if not isinstance(p2, str) else p2[:90]} warnings={sorted({str(x.message)[:60] for x in w2})}")

print("\n===== P13: backtesting with refit, n_jobs=1 vs n_jobs=2, with categorical_features=None")
series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=150)
for est_name in ["lgbm", "xgb", "hgb"]:
    for cf in [None, "auto"]:
        out = {}
        for n_jobs in (1, 2):
            f = ForecasterRecursiveMultiSeries(C.make_estimator(est_name), lags=3, encoding="ordinal_category", categorical_features=cf)
            r, w = run(lambda: backtesting_forecaster_multiseries(
                forecaster=f, series=series, exog=exog,
                cv=TimeSeriesFold(steps=5, initial_train_size=100, refit=True),
                metric="mean_absolute_error", n_jobs=n_jobs, show_progress=False))
            n_dt = sum("category" in str(x.message) and "DataTypeWarning" in x.category.__name__ for x in w)
            out[n_jobs] = (r if isinstance(r, str) else r[1]["pred"].to_numpy(), n_dt)
        same = (not isinstance(out[1][0], str)) and (not isinstance(out[2][0], str)) and np.array_equal(out[1][0], out[2][0])
        print(f"  {est_name:5} cf={str(cf):5} n_jobs 1 vs 2 same predictions: {same} | DataTypeWarning (category) emitted: n_jobs=1 -> {out[1][1]}, n_jobs=2 -> {out[2][1]}"
              f"{'' if not isinstance(out[1][0], str) else ' ' + out[1][0][:100]}")

print("\n===== P14: predict with exog of another dtype than in fit (category in fit)")
series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=150)
variants = {
    "same (category 10,20,30)": exog_fut,
    "int64": exog_fut.astype({"x_c": "int64"}),
    "category, other order": exog_fut.assign(x_c=pd.Categorical(np.asarray(exog_fut["x_c"]), categories=[30, 10, 20])),
    "category, fewer categories": exog_fut.assign(x_c=pd.Categorical(np.asarray(exog_fut["x_c"]))),
    "float": exog_fut.astype({"x_c": "int64"}).astype({"x_c": "float64"}),
}
for cf in [None, "auto"]:
    f = ForecasterRecursiveMultiSeries(C.make_estimator("lgbm"), lags=3, categorical_features=cf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.fit(series=series, exog=exog)
    for vname, ex in variants.items():
        p, w = run(lambda: f.predict(steps=3, exog=ex))
        print(f"  cf={str(cf):5} exog in predict: {vname:28} -> {fmt(p)}")

print("\n===== P18: deepcopy and sklearn clone state, XGBoost with None and ordinal_category")
f = ForecasterRecursiveMultiSeries(XGBRegressor(n_estimators=10, n_jobs=1), lags=3, encoding="ordinal_category", categorical_features=None)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f.fit(series=series, exog=exog)
    p0 = f.predict(steps=3, exog=exog_fut)["pred"].to_numpy()
    f2 = copy.deepcopy(f)
    p1 = f2.predict(steps=3, exog=exog_fut)["pred"].to_numpy()
    X_tr, _ = f.create_train_X_y(series=series, exog=exog)
    direct, w = run(lambda: f.estimator.predict(X_tr.to_numpy(dtype=float)))
print(f"  deepcopy predicts the same: {np.array_equal(p0, p1)} | estimator.feature_types={f.estimator.feature_types} "
      f"booster.feature_types={f.estimator.get_booster().feature_types} | estimator.predict(numpy): {'ok' if not isinstance(direct, str) else direct[:120]}")

print("\n===== P19: set_in_sample_residuals equals the residuals of fit, categorical_features=None")
for est_name in ["lgbm", "xgb", "hgb", "catboost", "ridge"]:
    for enc in ["ordinal", "ordinal_category", "onehot", None]:
        f = ForecasterRecursiveMultiSeries(C.make_estimator(est_name), lags=3, encoding=enc, categorical_features=None)
        def go():
            f.fit(series=series, exog=exog, store_in_sample_residuals=True)
            r0 = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
            f.set_in_sample_residuals(series=series, exog=exog)
            r1 = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
            return max(float(np.abs(r0[k] - r1[k]).max()) for k in r0)
        r, w = run(go)
        print(f"  {est_name:8} enc={str(enc):17} max |residual fit - residual set_in_sample_residuals| = {r if isinstance(r, str) else round(r, 6)}")
