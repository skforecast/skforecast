# What the other forecasters support for the same categorical situations.
# Usage: python others.py <out.json>   (tree selected with PYTHONPATH)
import sys
import json
import warnings
import numpy as np
import pandas as pd
import skforecast

print("skforecast file:", skforecast.__file__, flush=True)

from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

N = 200
LAGS = 3
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=N, freq="D")
cat_vals = rng.choice([10, 20, 30], size=N)
num = rng.normal(size=N)
effect = {10: 5.0, 20: -5.0, 30: 12.0}
y = pd.Series(np.array([effect[v] for v in cat_vals]) + num, index=idx, name="y")
series = pd.DataFrame({"s_a": y, "s_b": y + 2}, index=idx)
exog = pd.DataFrame(
    {"exog_num": num, "exog_cat": pd.Categorical(cat_vals, categories=[10, 20, 30])},
    index=idx,
)
LGB = dict(n_estimators=20, verbose=-1, min_child_samples=5, min_data_per_group=1,
           cat_smooth=0, random_state=1)


def estimators(pos, n_features):
    ft = ["q"] * n_features
    ft[pos] = "c"
    return {
        "lgbm|none": (LGBMRegressor(**LGB), None),
        "lgbm|init_name": (LGBMRegressor(categorical_feature=["exog_cat"], **LGB), None),
        "lgbm|init_pos": (LGBMRegressor(categorical_feature=[pos], **LGB), None),
        "lgbm|fit_name": (LGBMRegressor(**LGB), {"categorical_feature": ["exog_cat"]}),
        "lgbm|fit_pos": (LGBMRegressor(**LGB), {"categorical_feature": [pos]}),
        "xgb|none": (XGBRegressor(n_estimators=10), None),
        "xgb|init_enable": (XGBRegressor(n_estimators=10, enable_categorical=True), None),
        "xgb|init_pos": (XGBRegressor(n_estimators=10, enable_categorical=True,
                                      feature_types=ft), None),
        "hgb|none": (HistGradientBoostingRegressor(max_iter=10), None),
        "hgb|init_name": (HistGradientBoostingRegressor(
            max_iter=10, categorical_features=["exog_cat"]), None),
        "hgb|init_pos": (HistGradientBoostingRegressor(
            max_iter=10, categorical_features=[pos]), None),
        "catboost|fit_name": (CatBoostRegressor(
            iterations=10, verbose=0, allow_writing_files=False, thread_count=1),
            {"cat_features": ["exog_cat"]}),
        "catboost|fit_pos": (CatBoostRegressor(
            iterations=10, verbose=0, allow_writing_files=False, thread_count=1),
            {"cat_features": [pos]}),
        "catboost|init_pos_tuple": (CatBoostRegressor(
            iterations=10, verbose=0, allow_writing_files=False, thread_count=1,
            cat_features=(pos,)), None),
    }


def effective_categorical(est, names):
    mod = type(est).__module__.split(".")[0]
    try:
        if mod == "lightgbm":
            txt = est.booster_.model_to_string()
            line = [l for l in txt.splitlines() if l.startswith("feature_infos=")][0]
            infos = line[len("feature_infos="):].split(" ")
            return [names[i] for i, v in enumerate(infos)
                    if not v.startswith("[") and v != "none"]
        if mod == "xgboost":
            ft = est.get_booster().feature_types
            return [] if ft is None else [names[i] for i, t in enumerate(ft) if t == "c"]
        if mod == "catboost":
            return [names[i] for i in est.get_cat_feature_indices()]
        if type(est).__name__.startswith("HistGradientBoosting"):
            m = est.is_categorical_
            return [] if m is None else [names[i] for i in np.flatnonzero(m)]
    except Exception as e:
        return f"?? {type(e).__name__}: {e}"
    return []


results = {}
for fname in ["ForecasterRecursive", "ForecasterDirect", "ForecasterDirectMultiVariate"]:
    if fname == "ForecasterDirectMultiVariate":
        names = [f"s_a_lag_{i}" for i in range(1, 4)] + [f"s_b_lag_{i}" for i in range(1, 4)]
        names += ["exog_num", "exog_cat"]
    else:
        names = [f"lag_{i}" for i in range(1, 4)] + ["exog_num", "exog_cat"]
    pos = names.index("exog_cat")
    for cf_name, cf in [("auto", "auto"), ("None", None)]:
        for key, (est, fk) in estimators(pos, len(names)).items():
            k = f"{fname}|{key}|cf={cf_name}"
            res = {}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    if fname == "ForecasterRecursive":
                        f = ForecasterRecursive(estimator=est, lags=LAGS,
                                                categorical_features=cf, fit_kwargs=fk)
                        f.fit(y=y.iloc[:-5], exog=exog.iloc[:-5])
                        fitted = f.estimator
                    elif fname == "ForecasterDirect":
                        f = ForecasterDirect(estimator=est, lags=LAGS, steps=1,
                                             categorical_features=cf, fit_kwargs=fk)
                        f.fit(y=y.iloc[:-5], exog=exog.iloc[:-5])
                        fitted = f.estimators_[1]
                    else:
                        f = ForecasterDirectMultiVariate(
                            estimator=est, level="s_a", lags=LAGS, steps=1,
                            categorical_features=cf, fit_kwargs=fk)
                        f.fit(series=series.iloc[:-5], exog=exog.iloc[:-5])
                        fitted = f.estimators_[1]
                    p = f.predict(steps=1, exog=exog.iloc[-5:])
                    res["status"] = "ok"
                    res["effective_cat"] = effective_categorical(
                        fitted, f.X_train_features_names_out_
                        if fname != "ForecasterDirect" else names
                    )
                    res["names"] = list(getattr(f, "X_train_features_names_out_", []))
                except Exception as e:
                    res["status"] = "error"
                    res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            res["warnings"] = sorted({
                f"{x.category.__name__}: {str(x.message)[:100]}" for x in w
                if x.category.__name__ not in ("FutureWarning", "DeprecationWarning")
            })
            results[k] = res
            print(k, "->", res["status"], res.get("effective_cat"), res.get("error", "")[:140],
                  "| W:", [x.split(":")[0] for x in res["warnings"]], flush=True)

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1)
