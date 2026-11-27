"""
CatBoost with `cat_features` in the constructor and a category exog not in
`categorical_features`. The tree is chosen with PYTHONPATH.
"""
import json
import warnings

import numpy as np
import pandas as pd

import skforecast
print("skforecast from:", skforecast.__file__, flush=True)

from skforecast.recursive import ForecasterRecursiveMultiSeries
from catboost import CatBoostRegressor

rng = np.random.default_rng(2)
n, steps = 240, 12
idx = pd.date_range("2020-01-01", periods=n + steps, freq="D")
levels = ["a", "b", "c"]
offsets = {"a": 0.0, "b": 100.0, "c": 200.0}
effect = {10: 0.0, 20: 40.0, 30: -40.0}
cat_all = {lv: rng.choice([10, 20, 30], size=n + steps) for lv in levels}
truth = {lv: offsets[lv] + np.array([effect[c] for c in cat_all[lv]]) for lv in levels}
y_all = {lv: truth[lv] + rng.normal(size=n + steps) for lv in levels}
series = pd.DataFrame({lv: y_all[lv][:n] for lv in levels}, index=idx[:n])
num_all = {lv: rng.normal(size=n + steps) for lv in levels}


def exog_for(sl, kind):
    out = {}
    for lv in levels:
        c = cat_all[lv][sl]
        col = pd.Categorical(c, categories=[10, 20, 30]) if kind == "category" else c.astype("int64")
        out[lv] = pd.DataFrame({"num": num_all[lv][sl], "cat": col}, index=idx[sl])
    return out


for kind in ["category", "int"]:
    for enc, cat_features in [("ordinal", (5,)), (None, (4,)), ("ordinal_category", (3, 5))]:
        rec = {"kind": kind, "enc": enc, "cat_features": cat_features}
        try:
            f = ForecasterRecursiveMultiSeries(
                CatBoostRegressor(iterations=60, verbose=0, allow_writing_files=False,
                                  random_seed=0, thread_count=1, cat_features=cat_features),
                lags=3, encoding=enc, categorical_features=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(series=series, exog=exog_for(slice(0, n), kind))
                p = f.predict(steps=steps, exog=exog_for(slice(n, n + steps), kind))
            p = p.pivot_table(index=p.index, columns="level", values="pred")
            mae = float(np.mean([np.abs(p[lv].to_numpy() - truth[lv][n:]).mean() for lv in levels]))
            rec.update(status="ok", mae=round(mae, 3))
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:160]}")
        print(json.dumps(rec), flush=True)
