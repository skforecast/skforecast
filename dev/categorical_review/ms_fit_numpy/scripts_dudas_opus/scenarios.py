"""
Scenario matrix for the categorical handling of ForecasterRecursiveMultiSeries.

Usage: python scenarios.py <family> <out_json>
The tree is chosen with PYTHONPATH. Each scenario records whether `fit` and
`predict` succeed, the skforecast warnings, and the MAE of the predictions
against the noise-free truth (an inconsistency between fit and predict shows
as a large MAE).
"""
import sys
import json
import warnings
import traceback

import numpy as np
import pandas as pd

import skforecast
print("skforecast from:", skforecast.__file__, flush=True)
print("pandas", pd.__version__, flush=True)

from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

family = sys.argv[1]
out_path = sys.argv[2]

rng = np.random.default_rng(0)
n = 240
steps = 12
idx = pd.date_range("2020-01-01", periods=n + steps, freq="D")
levels = ["a", "b", "c"]
offsets = {"a": 0.0, "b": 100.0, "c": 200.0}
effect = {10: 0.0, 20: 40.0, 30: -40.0}

cat_all = {lv: rng.choice([10, 20, 30], size=n + steps) for lv in levels}
num_all = {lv: rng.normal(size=n + steps) for lv in levels}
truth = {
    lv: offsets[lv] + np.array([effect[c] for c in cat_all[lv]])
    for lv in levels
}
y_all = {lv: truth[lv] + rng.normal(scale=1.0, size=n + steps) for lv in levels}

series_wide = pd.DataFrame({lv: y_all[lv][:n] for lv in levels}, index=idx[:n])
series_dict = {lv: series_wide[lv] for lv in levels}


def make_exog(lv, sl, cat_dtype=True):
    c = cat_all[lv][sl]
    return pd.DataFrame(
        {
            "num": num_all[lv][sl],
            "cat": pd.Categorical(c, categories=[10, 20, 30]) if cat_dtype else c.astype(float),
        },
        index=idx[sl],
    )


exog_train = {lv: make_exog(lv, slice(0, n)) for lv in levels}
exog_test = {lv: make_exog(lv, slice(n, n + steps)) for lv in levels}


def lgbm():
    return LGBMRegressor(n_estimators=60, verbose=-1, random_state=0)


def xgb():
    return XGBRegressor(n_estimators=60, random_state=0)


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=60, random_state=0, **kw)


def cb(**kw):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=60, verbose=0, allow_writing_files=False, random_seed=0,
        thread_count=1, **kw
    )


# feature names for lags=3: lag_1..3, [level], num, cat
def scenarios():
    encs = ["ordinal", "ordinal_category", "onehot", None]
    cfs = ["auto", None, ["num_missing"]]  # list without 'cat' handled below
    out = []
    fams = {
        "lgbm": lgbm,
        "xgb": xgb,
        "hgb": hgb,
        "ridge": lambda: Ridge(),
        "pipe_lgbm": lambda: make_pipeline(StandardScaler(), lgbm()),
        "ttr_lgbm": lambda: TransformedTargetRegressor(regressor=lgbm()),
        "catboost": cb,
    }
    for name, factory in fams.items():
        if family != "all" and name != family:
            continue
        for enc in encs:
            for cf in ["auto", None, "list_other"]:
                out.append(dict(est=name, factory=factory, enc=enc, cf=cf, fk=None, user=None))
        # user configuration with categorical_features=None
        if name == "lgbm":
            for enc in encs:
                out.append(dict(est=name, factory=factory, enc=enc, cf=None,
                                fk={"categorical_feature": ["cat"]}, user="fit_kwargs by name"))
                out.append(dict(est=name, factory=lambda: LGBMRegressor(
                    n_estimators=60, verbose=-1, random_state=0, categorical_feature="name:cat"),
                    enc=enc, cf=None, fk=None, user="constructor by name"))
            # by position: position of 'cat' depends on the encoding
            for enc, pos in [("ordinal", 5), ("ordinal_category", 5), (None, 4)]:
                out.append(dict(est=name, factory=factory, enc=enc, cf=None,
                                fk={"categorical_feature": [pos]}, user="fit_kwargs by position"))
        if name == "catboost":
            for enc in encs:
                out.append(dict(est=name, factory=factory, enc=enc, cf=None,
                                fk={"cat_features": ["cat"]}, user="fit_kwargs by name"))
            for enc, pos in [("ordinal", 5), (None, 4)]:
                out.append(dict(est=name, factory=factory, enc=enc, cf=None,
                                fk={"cat_features": [pos]}, user="fit_kwargs by position"))
        if name == "hgb":
            for enc in encs:
                out.append(dict(est=name, factory=lambda: hgb(categorical_features=["cat"]),
                                enc=enc, cf=None, fk=None, user="constructor by name"))
            for enc, pos in [("ordinal", 5), (None, 4)]:
                out.append(dict(est=name, factory=lambda pos=pos: hgb(categorical_features=[pos]),
                                enc=enc, cf=None, fk=None, user="constructor by position"))
        if name == "xgb":
            for enc, nfeat, pos in [("ordinal", 6, 5), (None, 5, 4)]:
                ft = ["q"] * nfeat
                ft[pos] = "c"
                out.append(dict(est=name, factory=lambda ft=ft: XGBRegressor(
                    n_estimators=60, random_state=0, enable_categorical=True, feature_types=ft),
                    enc=enc, cf=None, fk=None, user="constructor feature_types"))
    return out


results = []
for sc in scenarios():
    for series_type in ["wide", "dict"]:
        rec = {k: v for k, v in sc.items() if k != "factory"}
        rec["series"] = series_type
        cf = sc["cf"]
        if cf == "list_other":
            cf_arg = ["num"]  # a list that omits 'cat'
        else:
            cf_arg = cf
        try:
            f = ForecasterRecursiveMultiSeries(
                estimator=sc["factory"](), lags=3, encoding=sc["enc"],
                categorical_features=cf_arg, fit_kwargs=sc["fk"],
            )
            s = series_wide if series_type == "wide" else series_dict
            with warnings.catch_warnings(record=True) as rec_w:
                warnings.simplefilter("always")
                f.fit(series=s, exog=exog_train, suppress_warnings=False)
                p = f.predict(steps=steps, exog=exog_test, suppress_warnings=True)
            ws = sorted({
                w.category.__name__ for w in rec_w
                if w.category.__module__.startswith("skforecast")
            })
            p = p.pivot_table(index=p.index, columns="level", values="pred") if "level" in p.columns else p
            mae = float(np.mean([
                np.abs(p[lv].to_numpy() - truth[lv][n:n + steps]).mean() for lv in levels
            ]))
            rec.update(status="ok", mae=round(mae, 3), warnings=ws)
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:160]}")
        results.append(rec)
        print(json.dumps(rec, default=str), flush=True)

with open(out_path, "w") as fh:
    json.dump(results, fh, default=str, indent=1)
