"""
P1, items 2 and 4. Other things that depend on the DataFrame given to the
estimator in ForecasterRecursiveMultiSeries.fit:
  A. forecasters saved before the fix (estimator fitted with a DataFrame);
  B. `fit_kwargs` of the user that name columns (LightGBM `categorical_feature`);
  C. exog dtypes other than float64 (int, bool, nullable): fit against predict,
     today and with the prototype;
  D. CatBoost with a plain float matrix.
"""
import _env
import os
import tempfile
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.utils import save_forecaster, load_forecaster
from p1_common import make_data, numpy_fit, spy_fit, attempt, r

lgbm = lambda: LGBMRegressor(n_estimators=30, verbose=-1, n_jobs=1, random_state=0)
EST = {"LGBM": lgbm, "XGB": lambda: XGBRegressor(n_estimators=30, n_jobs=1, random_state=0),
       "HGB": lambda: HistGradientBoostingRegressor(max_iter=30, random_state=0), "Ridge": lambda: Ridge()}

print("# A. Forecaster fitted today (DataFrame), saved and loaded")
series, exog, exog_p = make_data(tuple("stuvwxyz"), n_series=6)
for name, mk in EST.items():
    for enc in ("ordinal", "ordinal_category"):
        f = ForecasterRecursiveMultiSeries(mk(), lags=2, encoding=enc)
        f.fit(series=series, exog=exog, store_in_sample_residuals=True)
        p0 = f.predict(8, exog=exog_p)["pred"].to_numpy()
        with tempfile.TemporaryDirectory() as d:
            save_forecaster(f, os.path.join(d, "f.joblib"), verbose=False)
            g = load_forecaster(os.path.join(d, "f.joblib"), verbose=False)
        X, y = g.create_train_X_y(series=series, exog=exog)
        # What a numpy `set_in_sample_residuals` (:4695) would compute with the old estimator.
        res_numpy = y.to_numpy() - g.estimator.predict(X.to_numpy(dtype=float))
        res_df = y.to_numpy() - g.estimator.predict(X)
        print(f"  {name:5} {enc:16} predict after load identical: {np.array_equal(p0, g.predict(8, exog=exog_p)['pred'].to_numpy())}"
              f" | residuals with numpy == with DataFrame: {np.allclose(res_numpy, res_df)}")

print("\n# B. LightGBM, categorical_features=None, fit_kwargs of the user (category ints 0..7)")
series, exog, exog_p = make_data(tuple(range(8)), n_series=6)
for fk in ({"categorical_feature": ["c"]}, {"categorical_feature": [3]}, {"categorical_feature": "auto"},
           {"feature_name": ["l1", "lev", "x", "c"]}):
    out = []
    for numpy in (False, True):
        f = ForecasterRecursiveMultiSeries(lgbm(), lags=1, categorical_features=None, fit_kwargs=fk)
        if numpy:
            with numpy_fit():
                res = attempt(lambda: f.fit(series=series, exog=exog))
        else:
            res = attempt(lambda: f.fit(series=series, exog=exog))
        out.append("fit -> " + res if isinstance(res, str) else str(r(attempt(
            lambda: f.predict(8, levels=["a"], exog=exog_p)["pred"].to_numpy()), 0)))
    fr = ForecasterRecursive(lgbm(), lags=1, categorical_features=None, fit_kwargs=fk)
    res = attempt(lambda: fr.fit(series["a"], exog=exog))
    print(f"  fit_kwargs={fk}\n    today: {out[0]}\n    fix  : {out[1]}\n    ForecasterRecursive today: {'ok' if not isinstance(res, str) else res}")

print("\n# C. exog dtypes (2 series, lags=1): what the estimator gets, fit and predict, today and fix")
n = 300
idx = pd.date_range("2020", periods=n, freq="D")
fidx = pd.date_range(idx[-1] + idx.freq, periods=4, freq="D")
rng = np.random.default_rng(1)
k = rng.integers(0, 5, n)
y = pd.Series(3.0 * k + rng.normal(size=n), index=idx)
series2 = {"a": y, "b": y + 5}
kp = np.array([0, 1, 3, 4])
COLS = {
    "int64": (k, kp),
    "bool": (k > 2, kp > 2),
    "Int64": (pd.array(k, dtype="Int64"), pd.array(kp, dtype="Int64")),
    "Int64 with NA (fit only)": (pd.array(np.where(k == 0, None, k), dtype="Int64"), pd.array(kp, dtype="Int64")),
    "Float64 with NA (fit only)": (pd.array(np.where(k == 0, None, k), dtype="Float64"), pd.array(kp, dtype="Float64")),
    "boolean": (pd.array(k > 2, dtype="boolean"), pd.array(kp > 2, dtype="boolean")),
}
for col, (v, vp) in COLS.items():
    ex = pd.DataFrame({"x": rng.normal(size=n), "e": v}, index=idx)
    exp = pd.DataFrame({"x": np.zeros(4), "e": vp}, index=fidx)
    print(f"  exog dtype {col}")
    for name, mk in EST.items():
        row = []
        for numpy in (False, True):
            f = ForecasterRecursiveMultiSeries(mk(), lags=1)
            log = []
            with spy_fit(type(f.estimator), log):
                if numpy:
                    with numpy_fit():
                        res = attempt(lambda: f.fit(series=series2, exog=ex))
                else:
                    res = attempt(lambda: f.fit(series=series2, exog=ex))
            if isinstance(res, str):
                row.append(("fit -> " + res[:90], None))
                continue
            got = log[0]["dtypes"]["e"] if isinstance(log[0]["dtypes"], dict) else log[0]["dtypes"]
            p = attempt(lambda: f.predict(4, levels=["a"], exog=exp)["pred"].to_numpy())
            row.append((f"fit gets {got}, " + ("predict -> " + p[:70] if isinstance(p, str) else "predict ok"), p))
        (sa, pa), (sb, pb) = row
        same = "" if isinstance(pa, (str, type(None))) or isinstance(pb, (str, type(None))) else f" | identical: {np.array_equal(pa, pb)}"
        print(f"    {name:5} today: {sa} | fix: {sb}{same}")

print("\n# D. CatBoost, cat_features and a plain float matrix (no cast of the categorical columns to int)")
series, exog, _ = make_data(tuple("stuvwxyz"), n_series=2)
f = ForecasterRecursiveMultiSeries(CatBoostRegressor(verbose=0, iterations=10, thread_count=1), lags=1)
X, y = f.create_train_X_y(series=series, exog=exog)
print("  ", attempt(lambda: CatBoostRegressor(verbose=0, iterations=10, thread_count=1).fit(
    X.to_numpy(dtype=float), y, cat_features=[3]) and "fit ok"))
print(_env.versions())
