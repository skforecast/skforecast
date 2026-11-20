"""
P1, item 1. Mechanism of the wrong predictions of ForecasterRecursiveMultiSeries
with LightGBM and a `category` exog: what `estimator.fit` receives, what the
booster stores and what `predict` gives, for several `categorical_features`
and category layouts. Expected prediction of series 'a': -50 / 0 / +50.
"""
import _env
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from p1_common import make_data, spy_fit, attempt, r

mk = lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0)
print("default categorical_features:", ForecasterRecursiveMultiSeries(mk(), lags=1).categorical_features,
      "| default encoding:", ForecasterRecursiveMultiSeries(mk(), lags=1).encoding)


def run(label, categories, cf, n_cat_cols=1):
    series, exog, exog_p = make_data(categories, n_cat_cols=n_cat_cols)
    print(f"\n## {label}: categories={list(categories)}, categorical_features={cf!r}")
    log = []
    fm = ForecasterRecursiveMultiSeries(mk(), lags=1, encoding="ordinal", categorical_features=cf)
    with spy_fit(LGBMRegressor, log):
        res = attempt(lambda: fm.fit(series=series, exog=exog, store_in_sample_residuals=False))
    if isinstance(res, str):
        print("  MultiSeries fit ->", res)
    else:
        d = log[0]
        print("  fit X:", d["type"], {k: v for k, v in d["dtypes"].items() if k != "lag_1"}, "kwargs:", d["kwargs"])
        b = fm.estimator.booster_
        print("  booster pandas_categorical:", b.pandas_categorical,
              "| categorical splits:", '"decision_type": "=="' in str(b.dump_model()).replace("'", '"'))
        print("  MultiSeries predict            :", r(attempt(
            lambda: fm.predict(3, levels=["a"], exog=exog_p)["pred"])))
        print("  estimator.predict(DataFrame Xp):", r(attempt(lambda: fm.estimator.predict(
            fm.create_predict_X(3, levels=["a"], exog=exog_p).drop(columns="level")))))
    fr = ForecasterRecursive(mk(), lags=1, categorical_features=cf)
    log = []
    with spy_fit(LGBMRegressor, log):
        res = attempt(lambda: fr.fit(series["a"], exog=exog))
    if isinstance(res, str):
        print("  Recursive fit ->", res)
    else:
        print("  Recursive: fit X", log[0]["type"], log[0]["dtypes"], "kwargs:", log[0]["kwargs"],
              "| predict:", r(attempt(lambda: fr.predict(3, exog=exog_p))))


run("ints not 0..n-1", (10, 20, 30), None)
run("ints 0..n-1 sorted", (0, 1, 2), None)
run("ints 0..n-1, categories in another order", (2, 0, 1), None)
run("strings", ("u", "v", "w"), None)
run("ints not 0..n-1", (10, 20, 30), "auto")
run("ints 0..n-1, another order", (2, 0, 1), "auto")
run("strings", ("u", "v", "w"), "auto")
run("list that omits c2 (ints)", (10, 20, 30), ["c"], n_cat_cols=2)
run("list that omits c2 (strings)", ("u", "v", "w"), ["c"], n_cat_cols=2)

print("\n## Level column, per encoding (LightGBM, no exog)")
series, _, _ = make_data()
for enc in ("ordinal", "ordinal_category", "onehot", None):
    for cf in ("auto", None):
        log = []
        fm = ForecasterRecursiveMultiSeries(mk(), lags=1, encoding=enc, categorical_features=cf)
        with spy_fit(LGBMRegressor, log):
            fm.fit(series=series, store_in_sample_residuals=False)
        Xp = fm.create_predict_X(2)
        d = log[0]
        cat_split = '"decision_type": "=="' in str(fm.estimator.booster_.dump_model()).replace("'", '"')
        est_pred = attempt(lambda: fm.estimator.predict(Xp[fm.X_train_features_names_out_]))
        same = est_pred if isinstance(est_pred, str) else np.allclose(
            fm.predict(2)["pred"].to_numpy(), est_pred)
        print(f"  encoding={enc!s:16} cf={cf!s:5} fit dtypes="
              f"{ {k: v for k, v in d['dtypes'].items() if k != 'lag_1'} } kwargs={d['kwargs']} "
              f"categorical splits={cat_split} | predict==estimator.predict(create_predict_X): {same}")
print(_env.versions())
