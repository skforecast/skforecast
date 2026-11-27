# Scenario matrix of the categorical handling of ForecasterRecursiveMultiSeries.
# Usage: python scenarios.py <family> <out.json>
# The tree under test is selected with PYTHONPATH.
import sys
import json
import warnings
import itertools
import numpy as np
import pandas as pd
import skforecast

print("skforecast file:", skforecast.__file__, flush=True)

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

family = sys.argv[1]
out_path = sys.argv[2]
data_kind = sys.argv[3] if len(sys.argv) > 3 else "wide"

LAGS = 3
N = 200
SERIES = ["s_a", "s_b", "s_c"]


def make_data(kind):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=N, freq="D")
    cat_vals = rng.choice([10, 20, 30], size=N)
    cat2_vals = rng.choice([5, 7, 9], size=N)
    num = rng.normal(size=N)
    effect = {10: 5.0, 20: -5.0, 30: 12.0}
    effect2 = {5: 0.0, 7: 3.0, 9: -3.0}
    base = (
        np.array([effect[v] for v in cat_vals])
        + np.array([effect2[v] for v in cat2_vals])
        + num
    )
    series = pd.DataFrame(
        {s: base + i * 2 + rng.normal(scale=0.1, size=N) for i, s in enumerate(SERIES)},
        index=idx,
    )
    exog = pd.DataFrame(
        {
            "exog_num": num,
            "exog_cat": pd.Categorical(cat_vals, categories=[10, 20, 30]),
            "exog_cat2": pd.Categorical(cat2_vals, categories=[5, 7, 9]),
        },
        index=idx,
    )
    if kind == "wide_int":
        exog = exog.astype({"exog_cat": "int64", "exog_cat2": "int64"})
        return series, exog
    if kind == "wide":
        return series, exog
    # dict: different lengths and one exog per series
    starts = {"s_a": 0, "s_b": 30, "s_c": 60}
    series_d = {s: series[s].iloc[starts[s]:].copy() for s in SERIES}
    exog_d = {s: exog.iloc[starts[s]:].copy() for s in SERIES}
    return series_d, exog_d


def feature_names(encoding):
    names = [f"lag_{i}" for i in range(1, LAGS + 1)]
    if encoding in ("ordinal", "ordinal_category"):
        names.append("_level_skforecast")
    elif encoding == "onehot":
        names.extend(SERIES)
    names.extend(["exog_num", "exog_cat", "exog_cat2"])
    return names


def build(family, user, encoding):
    """Return (estimator, fit_kwargs) or None if the combination does not apply."""
    names = feature_names(encoding)
    pos = names.index("exog_cat")
    fit_kwargs = None
    if family == "lgbm":
        kw = dict(n_estimators=20, verbose=-1, min_child_samples=5,
                  min_data_per_group=1, cat_smooth=0, random_state=1)
        if user == "none":
            est = LGBMRegressor(**kw)
        elif user == "init_name":
            est = LGBMRegressor(categorical_feature=["exog_cat"], **kw)
        elif user == "init_pos":
            est = LGBMRegressor(categorical_feature=[pos], **kw)
        elif user == "fit_name":
            est = LGBMRegressor(**kw)
            fit_kwargs = {"categorical_feature": ["exog_cat"]}
        elif user == "fit_pos":
            est = LGBMRegressor(**kw)
            fit_kwargs = {"categorical_feature": [pos]}
        else:
            return None
    elif family == "xgb":
        kw = dict(n_estimators=20, random_state=1, max_depth=3)
        if user == "none":
            est = XGBRegressor(**kw)
        elif user == "init_enable":
            est = XGBRegressor(enable_categorical=True, **kw)
        elif user == "init_pos":
            ft = ["q"] * len(names)
            ft[pos] = "c"
            est = XGBRegressor(feature_types=ft, enable_categorical=True, **kw)
        else:
            return None
    elif family == "catboost":
        kw = dict(iterations=20, verbose=0, allow_writing_files=False,
                  random_seed=1, thread_count=1)
        if user == "none":
            est = CatBoostRegressor(**kw)
        elif user == "init_name":
            est = CatBoostRegressor(cat_features=("exog_cat",), **kw)
        elif user == "init_pos":
            est = CatBoostRegressor(cat_features=(pos,), **kw)
        elif user == "fit_name":
            est = CatBoostRegressor(**kw)
            fit_kwargs = {"cat_features": ["exog_cat"]}
        elif user == "fit_pos":
            est = CatBoostRegressor(**kw)
            fit_kwargs = {"cat_features": [pos]}
        else:
            return None
    elif family == "hgb":
        kw = dict(max_iter=20, random_state=1, min_samples_leaf=5)
        if user == "none":
            est = HistGradientBoostingRegressor(**kw)
        elif user == "init_name":
            est = HistGradientBoostingRegressor(categorical_features=["exog_cat"], **kw)
        elif user == "init_pos":
            est = HistGradientBoostingRegressor(categorical_features=[pos], **kw)
        elif user == "init_mask":
            mask = [n == "exog_cat" for n in names]
            est = HistGradientBoostingRegressor(categorical_features=mask, **kw)
        else:
            return None
    elif family == "ridge":
        if user != "none":
            return None
        est = Ridge()
    elif family == "pipe_lgbm":
        if user != "none":
            return None
        est = Pipeline([
            ("id", FunctionTransformer()),
            ("m", LGBMRegressor(n_estimators=20, verbose=-1, min_child_samples=5,
                                min_data_per_group=1, cat_smooth=0, random_state=1)),
        ])
    elif family == "ttr_lgbm":
        kw = dict(n_estimators=20, verbose=-1, min_child_samples=5,
                  min_data_per_group=1, cat_smooth=0, random_state=1)
        if user == "none":
            est = TransformedTargetRegressor(regressor=LGBMRegressor(**kw))
        elif user == "init_pos":
            est = TransformedTargetRegressor(
                regressor=LGBMRegressor(categorical_feature=[pos], **kw)
            )
        else:
            return None
    elif family == "ttr_xgb":
        if user != "none":
            return None
        est = TransformedTargetRegressor(
            regressor=XGBRegressor(n_estimators=20, random_state=1, max_depth=3,
                                   enable_categorical=True)
        )
    else:
        raise ValueError(family)
    return est, fit_kwargs


def inner(est):
    if isinstance(est, Pipeline):
        return est[-1]
    if isinstance(est, TransformedTargetRegressor):
        return est.regressor_
    return est


def effective_categorical(est, names):
    """Names of the features the fitted model treats as categorical."""
    est = inner(est)
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
            if ft is None:
                return []
            return [names[i] for i, t in enumerate(ft) if t == "c"]
        if mod == "catboost":
            return [names[i] for i in est.get_cat_feature_indices()]
        if type(est).__name__.startswith("HistGradientBoosting"):
            m = est.is_categorical_
            if m is None:
                return []
            return [names[i] for i in np.flatnonzero(m)]
    except Exception as e:
        return f"?? {type(e).__name__}: {e}"
    return []


CAT_PARAMS = ("categorical_feature", "feature_types", "enable_categorical",
              "cat_features", "categorical_features")


def cat_params(est):
    try:
        p = est.get_params(deep=True)
    except Exception:
        return {}
    return {k: repr(v) for k, v in p.items() if k.split("__")[-1] in CAT_PARAMS}


def run_one(family, user, cf, encoding, kind):
    built = build(family, user, encoding)
    if built is None:
        return None
    est, fit_kwargs = built
    series, exog = make_data(kind)
    res = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            forecaster = ForecasterRecursiveMultiSeries(
                estimator=est, lags=LAGS, encoding=encoding,
                categorical_features=cf, fit_kwargs=fit_kwargs,
            )
            params_before = cat_params(forecaster.estimator)
            captured = {}
            target = forecaster.estimator
            orig_fit = target.fit

            def spy(X, y, **kw):
                captured["X"] = X.copy()
                captured["kw"] = {k: repr(v) for k, v in kw.items() if k != "sample_weight"}
                return orig_fit(X, y, **kw)

            target.fit = spy
            forecaster.fit(series=series, exog=exog)
            est_fitted = forecaster.estimator
            X = captured["X"]
            res["fit_X_type"] = type(X).__name__
            if isinstance(X, pd.DataFrame):
                res["fit_X_dtypes"] = sorted({str(d) for d in X.dtypes})
            else:
                res["fit_X_dtypes"] = [str(X.dtype)]
            res["fit_kwargs_sent"] = captured["kw"]
            res["params_before"] = params_before
            res["params_after"] = cat_params(est_fitted)
            names = forecaster.X_train_features_names_out_
            res["n_features"] = len(names)
            res["effective_cat"] = effective_categorical(est_fitted, names)
            # Consistency between fit and predict: last 12 training rows of s_a.
            n_rows = 12
            diffs = []
            for j in range(n_rows):
                # Row with target at position `pos` of the series s_a.
                if kind in ("wide", "wide_int"):
                    s_a, e_a = series["s_a"], exog
                else:
                    s_a, e_a = series["s_a"], exog["s_a"]
                pos = len(s_a) - 1 - j
                n_a = len(s_a) - LAGS
                lw = s_a.iloc[pos - LAGS:pos].to_frame()
                ex = e_a.iloc[[pos]]
                if kind not in ("wide", "wide_int"):
                    ex = {"s_a": ex}
                r = n_a - 1 - j
                row = X.iloc[[r]] if isinstance(X, pd.DataFrame) else X[[r]]
                p_est = float(np.ravel(est_fitted.predict(row))[0])
                p_fc = float(
                    forecaster.predict(steps=1, levels=["s_a"], last_window=lw, exog=ex)
                    ["pred"].iloc[0]
                )
                diffs.append(abs(p_est - p_fc))
            res["max_abs_diff"] = float(max(diffs))
            res["n_rows_differ"] = int(sum(d > 1e-6 for d in diffs))
            res["consistent"] = bool(max(diffs) < 1e-6)
            res["status"] = "ok"
        except Exception as e:
            res["status"] = "error"
            res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["warnings"] = sorted({
        f"{x.category.__name__}: {str(x.message)[:110]}" for x in w
        if x.category.__name__ not in ("FutureWarning", "DeprecationWarning")
    })
    return res


USERS = ["none", "init_name", "init_pos", "init_mask", "init_enable", "fit_name", "fit_pos"]
CFS = {"auto": "auto", "list_all": ["exog_cat", "exog_cat2"],
       "list_partial": ["exog_cat2"], "None": None}
ENCODINGS = ["ordinal", "ordinal_category", "onehot", None]
import os
if os.environ.get("SCEN_CFS"):
    CFS = {k: v for k, v in CFS.items() if k in os.environ["SCEN_CFS"].split(",")}
if os.environ.get("SCEN_ENCS"):
    ENCODINGS = [e for e in ENCODINGS if str(e) in os.environ["SCEN_ENCS"].split(",")]

results = {}
for user, (cf_name, cf), encoding in itertools.product(USERS, CFS.items(), ENCODINGS):
    key = f"{family}|user={user}|cf={cf_name}|enc={encoding}|{data_kind}"
    r = run_one(family, user, cf, encoding, data_kind)
    if r is None:
        continue
    results[key] = r
    print(key, "->", r["status"], r.get("consistent"), r.get("effective_cat"),
          r.get("error", "")[:120], flush=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print("done", len(results))
