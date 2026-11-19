"""
P2, items 6 and 7: `transformer_exog` outputs and categorical columns.

For each case: dtypes of exog after `transformer_exog` and encoding
(`exog_dtypes_out_`), dtype of `exog.to_numpy()` today, whether the proposal's
condition applies, and fit/predict today vs proposal per estimator.
"""
import _env
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
import p2_proto as P

n, steps = 60, 3
y = P.make_y(n + steps)
idx = y.index
k = np.arange(n + steps)
base = pd.DataFrame({"a": np.linspace(0, 1, n + steps), "h": k % 4}, index=idx)
words = np.array(["lo", "mid", "hi", "top"], dtype=object)[k % 4]
with_str = base.assign(s=words)
with_cat = base.assign(c=pd.Categorical(k % 4), b=k % 3 == 0)
unseen = with_str.copy()
unseen.iloc[n + 1, unseen.columns.get_loc("s")] = "NEW"          # unknown at predict
nan_cat = with_str.copy()
nan_cat.iloc[[10, n + 1], nan_cat.columns.get_loc("s")] = np.nan  # missing category


def ohe(**kw):
    return ColumnTransformer(
        [("oh", OneHotEncoder(**kw), ["s"])], remainder="passthrough",
        verbose_feature_names_out=False)


cases = {
    # name: (exog, transformer_exog, forecaster kwargs)
    "FunctionTransformer adds a bool column (pandas out)": (
        base, FunctionTransformer(lambda d: d.assign(flag=d["a"] > 0.5)), {}),
    "FunctionTransformer convert_dtypes (Float64 + Int64)": (
        base, FunctionTransformer(lambda d: d.convert_dtypes()), {}),
    "FunctionTransformer -> Int64 + float (pandas out)": (
        base, FunctionTransformer(lambda d: d.astype({"h": "Int64"})), {}),
    "StandardScaler set_output(pandas) on float + bool": (
        base.assign(b=k % 3 == 0),
        ColumnTransformer([("sc", StandardScaler(), ["a"])], remainder="passthrough",
                          verbose_feature_names_out=False).set_output(transform="pandas"), {}),
    "OneHot sparse (default)": (with_str, ohe(), {}),
    "OneHot dense dtype=bool, numpy out": (with_str, ohe(sparse_output=False, dtype=bool), {}),
    "OneHot dense dtype=bool, set_output(pandas)": (
        with_str, ohe(sparse_output=False, dtype=bool).set_output(transform="pandas"), {}),
    "ColumnTransformer scales `a`, passes text through (numpy out: all object)": (
        with_str, ColumnTransformer([("sc", StandardScaler(), ["a"])], remainder="passthrough"), {}),
    "same, set_output(pandas), categorical_features='auto'": (
        with_str, ColumnTransformer([("sc", StandardScaler(), ["a"])], remainder="passthrough",
                                    verbose_feature_names_out=False).set_output(transform="pandas"), {}),
    "same, set_output(pandas), categorical_features=None (text stays)": (
        with_str, ColumnTransformer([("sc", StandardScaler(), ["a"])], remainder="passthrough",
                                    verbose_feature_names_out=False).set_output(transform="pandas"),
        {"categorical_features": None}),
    "text column, categorical_features=['h'] (text not declared)": (
        with_str, None, {"categorical_features": ["h"]}),
    "category(int) + bool, categorical_features=None": (
        with_cat, None, {"categorical_features": None}),
    "category(int) + bool, categorical_features='auto'": (with_cat, None, {}),
    "category(str), categorical_features=None": (
        base.assign(c=pd.Categorical(words)), None, {"categorical_features": None}),
    "text + bool, 'auto', unknown category at predict": (
        unseen.assign(b=k % 3 == 0), None, {}),
    "text + bool, 'auto', NaN category in fit and predict": (
        nan_cat.assign(b=k % 3 == 0), None, {}),
}
estimators = {
    "Ridge": lambda: Ridge(),
    "LGBM": lambda: LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=5, verbose=0, thread_count=1, allow_writing_files=False),
}
forecasters = {
    "Rec": lambda e, **kw: ForecasterRecursive(e, lags=3, **kw),
    "Dir": lambda e, **kw: ForecasterDirect(e, lags=3, steps=steps, n_jobs=1, **kw),
}


def experiment(fname, ename, exog, transformer, kw):
    f = forecasters[fname](estimators[ename](), transformer_exog=transformer, **kw)
    with P.capture_fit(f.estimator) as seen:
        _, fit_st, fit_w = P.run(lambda: f.fit(y.iloc[:n], exog=exog.iloc[:n]))
    pred, pred_st, pred_w = (None, "-", [])
    if fit_st == "ok":
        pred, pred_st, pred_w = P.run(lambda: f.predict(steps=steps, exog=exog.iloc[n:]))
    out = getattr(f, "exog_dtypes_out_", None)
    return dict(x=seen.get("dtype", "not reached"), fit=fit_st, pred=pred_st,
                w=(fit_w, pred_w), values=pred,
                out=None if out is None else {c: str(d) for c, d in out.items()})


def arrow(a, b):
    return f"{a} -> =" if a == b else f"{a} -> {b}"


for cname, (exog, transformer, kw) in cases.items():
    print(f"\n### {cname}")
    groups, shown = {}, False
    for fname in forecasters:
        for ename in estimators:
            today = experiment(fname, ename, exog, transformer, kw)
            P.calls.clear()
            with P.proposal():
                new = experiment(fname, ename, exog, transformer, kw)
            if not shown and today["out"] is not None:
                print(f"  exog_dtypes_out_ = {today['out']}; proposal applied: {bool(P.calls)}")
                shown = True
            sig = (f"X {today['x']} -> {new['x']} | fit {arrow(today['fit'], new['fit'])}"
                   f" | predict {arrow(today['pred'], new['pred'])}"
                   f" | eq {P.same(today['values'], new['values'])}"
                   f" | warn (fit, predict) {arrow(today['w'], new['w'])}")
            groups.setdefault(sig, {}).setdefault(fname, []).append(ename)
    for sig, who in groups.items():
        label = "; ".join(f"{f}: {','.join(e)}" for f, e in who.items())
        print(f"  [{label}]\n      {sig}")
print(_env.versions())
