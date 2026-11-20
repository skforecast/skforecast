"""
Option (c), item 6 and the limits found on the way.

A. `categorical_features='auto'` (default): the prototype of (c) must not
   change anything. Backtesting predictions of today against the prototype
   (which for 'auto' is only the numpy fit), big dataset.
B. What `categorical_features=None` means in each forecaster today: what the
   estimator gets for a `category` exog (10..80) and whether LightGBM uses
   categorical splits. One series, so the forecasters are comparable.
C. Estimators that are not a bare LightGBM/XGBoost/HGB/CatBoost: a subclass, a
   TransformedTargetRegressor and a Pipeline. Today they read the dtype of the
   DataFrame; the index cannot be given to them.
D. Ordered `category` (pandas `ordered=True`).
E. The documented pattern `estimator.predict(create_predict_X(...))`.
F. Exog as a dict whose `category` dtypes differ between series.
G. `'auto'` without categorical columns resets the user's estimator
   configuration without a warning (utils.py:654-662).
"""
import _env
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.utils.utils import _build_predict_function
from p1_common import make_data, numpy_fit, spy_fit, EFFECT8
from p1c_common import make_big, option_c, head, backtest, test_rows, to_matrix, attempt, fmt

lgbm = lambda **kw: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0, **kw)
xgb = lambda **kw: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0, **kw)
hgb = lambda **kw: HistGradientBoostingRegressor(max_iter=40, random_state=0, **kw)
cat = lambda: CatBoostRegressor(iterations=40, verbose=0, thread_count=1, random_state=0, allow_writing_files=False)


class MyLGBM(LGBMRegressor):
    pass


def cat_splits(est):
    """Number of categorical splits of a LightGBM model (inside wrappers too)."""
    for attr in ("regressor_", "_final_estimator"):
        est = getattr(est, attr, est)
    df = est.booster_.trees_to_dataframe()
    return int((df["decision_type"] == "==").sum())


print("# A. 'auto' (default): today against the prototype of (c), 30 series, c40, c3, x")
series, exog, _ = make_big(n=300, unseen_from=250)
for name, mk in {"LightGBM": lgbm, "XGBoost": xgb, "HGB": hgb, "CatBoost": cat}.items():
    out = []
    for enc in ("ordinal", "ordinal_category"):
        new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=3, encoding=enc)
        p0 = backtest(new(), series, exog, 200)["pred"]
        with option_c("codes"):
            p1 = backtest(new(), series, exog, 200)["pred"]
        out.append(f"encoding={enc}: {fmt(np.abs(p1 - p0).max())}")
    print(f"  {name:8} {' | '.join(out)}")

print("\n# B. categorical_features=None in each forecaster today (LightGBM, one series, c = 10..80)")
series, exog, exog_p = make_data(tuple(range(10, 90, 10)), n_series=2)
y = series["a"]
multi = pd.DataFrame({"a": series["a"], "b": series["b"]})
FORECASTERS = {
    "ForecasterRecursive": lambda fk: (ForecasterRecursive(lgbm(), lags=1, categorical_features=None, fit_kwargs=fk),
                                       dict(y=y, exog=exog), 2),
    "ForecasterDirect": lambda fk: (ForecasterDirect(lgbm(), lags=1, steps=1, categorical_features=None, fit_kwargs=fk,
                                                     n_jobs=1), dict(y=y, exog=exog), 2),
    "ForecasterDirectMultiVariate": lambda fk: (ForecasterDirectMultiVariate(
        lgbm(), level="a", lags=1, steps=1, categorical_features=None, fit_kwargs=fk, n_jobs=1),
        dict(series=multi, exog=exog), 3),
    "ForecasterRecursiveMultiSeries": lambda fk: (ForecasterRecursiveMultiSeries(
        lgbm(), lags=1, categorical_features=None, fit_kwargs=fk), dict(series={"a": y}, exog=exog), 3),
}
for name, build in FORECASTERS.items():
    for with_kw in (False, True):
        f, kw, idx = build(None)
        fk = {"categorical_feature": [idx]} if with_kw else None
        f, kw, idx = build(fk)
        log = []
        with spy_fit(LGBMRegressor, log):
            res = attempt(lambda: f.fit(**kw))
        if isinstance(res, str):
            print(f"  {name:31} fit_kwargs={fk}: fit -> {res}")
            continue
        est = f.estimator if hasattr(f, "estimator") and not hasattr(f, "estimators_") else f.estimators_[1]
        d = log[0]
        got = d["type"] + (" c:" + d["dtypes"]["c"] if isinstance(d["dtypes"], dict) else "[" + d["dtypes"] + "]")
        steps = 1 if "Direct" in name else 8
        p = attempt(lambda: f.predict(steps, exog=exog_p))
        if not isinstance(p, str):
            p = p["pred"].to_numpy() if isinstance(p, pd.DataFrame) else p.to_numpy()
            p = f"MAE of predict against the effect {np.abs(p - EFFECT8[:steps]).mean():.1f} ({steps} step)"
        print(f"  {name:31} fit_kwargs={str(fk):30}: estimator gets {got:22} categorical splits {cat_splits(est):3} | {p}")

print("\n# C. Estimators that are not a bare native one, MultiSeries, categorical_features=None, 6 series, c = 10..80")
series, exog, _ = make_data(tuple(range(10, 90, 10)), n_series=6)
s_tr, e_tr = head(series, 200), head(exog, 200)
WRAPPED = {
    "LGBMRegressor": lgbm,
    "subclass of LGBMRegressor": lambda: MyLGBM(n_estimators=40, verbose=-1, n_jobs=1, random_state=0),
    "TransformedTargetRegressor(LGBM)": lambda: TransformedTargetRegressor(regressor=lgbm()),
    "Pipeline('passthrough', LGBM)": lambda: make_pipeline("passthrough", lgbm()),
    "Ridge": lambda: Ridge(),
}
for name, mk in WRAPPED.items():
    for cf in (None, "auto"):
        new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=cf)
        f0 = new()
        res = attempt(lambda: f0.fit(series=s_tr, exog=e_tr))
        if isinstance(res, str):
            print(f"  {name:33} cf={cf!s:4} today: fit -> {res[:120]}")
            continue
        Xte, key = test_rows(f0, series, exog, 200)
        ref = pd.Series(f0.estimator.predict(Xte), index=key)
        bt = backtest(new(), series, exog, 200)["pred"].reindex(key)
        n_cat = "" if name == "Ridge" else f"categorical splits {cat_splits(f0.estimator):3} | "
        out = f"  {name:33} cf={cf!s:4} today: {n_cat}predict vs as fitted: {fmt(np.abs(bt - ref).max())}"
        if cf is None:
            Xtr = f0.create_train_X_y(series=s_tr, exog=e_tr)[0][f0.X_train_features_names_out_]
            ins0 = f0.estimator.predict(Xtr)
            for label, ctx, variant in (("(c) codes", lambda: option_c("codes"), "codes"),
                                        ("values, no index (a')", numpy_fit, "values")):
                with ctx():
                    f1 = new()
                    res = attempt(lambda: f1.fit(series=s_tr, exog=e_tr))
                if isinstance(res, str):
                    out += f"\n  {'':38} {label}: fit -> {res[:110]}"
                    continue
                ins1 = _build_predict_function(f1.estimator)(to_matrix(Xtr, variant)[0])
                out += f"\n  {'':38} {label}: model vs today {fmt(np.abs(ins1 - ins0).max())}"
        print(out)

print("\n# D. Ordered category (c = 10..80, ordered=True), MultiSeries, categorical_features=None")
exog_o = exog.assign(c=exog["c"].cat.as_ordered())
for name, mk in {"LightGBM": lgbm, "XGBoost": xgb, "HGB": hgb}.items():
    new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=None)
    f0 = new()
    f0.fit(series=s_tr, exog=head(exog_o, 200))
    Xte, key = test_rows(f0, series, exog_o, 200)
    ref = pd.Series(f0.estimator.predict(Xte), index=key)
    bt = backtest(new(), series, exog_o, 200)["pred"].reindex(key)
    Xtr = f0.create_train_X_y(series=s_tr, exog=head(exog_o, 200))[0][f0.X_train_features_names_out_]
    ins0 = f0.estimator.predict(Xtr)
    with option_c("codes"):
        f1 = new()
        f1.fit(series=s_tr, exog=head(exog_o, 200))
    ins1 = _build_predict_function(f1.estimator)(to_matrix(Xtr, "codes")[0])
    # Codes as a numeric column, no index: what LightGBM does today with an ordered category.
    est = mk().fit(to_matrix(Xtr, "codes")[0], f0.create_train_X_y(series=s_tr, exog=head(exog_o, 200))[1])
    ins2 = _build_predict_function(est)(to_matrix(Xtr, "codes")[0])
    extra = f"categorical splits today {cat_splits(f0.estimator)} | " if name == "LightGBM" else ""
    print(f"  {name:8} {extra}today predict vs as fitted: {fmt(np.abs(bt - ref).max())} | (c) codes + index: model vs today"
          f" {fmt(np.abs(ins1 - ins0).max())} | codes as numeric, no index: model vs today {fmt(np.abs(ins2 - ins0).max())}")

print("\n# E. estimator.predict(create_predict_X(...)), MultiSeries, c = 10..80, steps=1")
exog_next = exog.iloc[200:201]
for name, mk in {"LightGBM": lgbm, "XGBoost": xgb, "HGB": hgb}.items():
    for cf in (None, "auto"):
        new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=cf)
        for label, ctx in (("today", None), ("(c) codes", lambda: option_c("codes")),
                           ("(c) values", lambda: option_c("values"))):
            if ctx is None:
                f = new(); f.fit(series=s_tr, exog=e_tr)
                Xp = f.create_predict_X(1, exog=exog_next)
                p = f.predict(1, exog=exog_next)["pred"].to_numpy()
            elif cf is None:
                with ctx():
                    f = new(); f.fit(series=s_tr, exog=e_tr)
                    Xp = f.create_predict_X(1, exog=exog_next)
                    p = f.predict(1, exog=exog_next)["pred"].to_numpy()
            else:
                continue
            e = attempt(lambda: f.estimator.predict(Xp[f.X_train_features_names_out_]))
            e2 = attempt(lambda: f.estimator.predict(Xp[f.X_train_features_names_out_].to_numpy(dtype=float)))
            print(f"  {name:8} cf={cf!s:4} {label:10}: create_predict_X c = {Xp['c'].iloc[0]} ({Xp['c'].dtype}) |"
                  f" estimator.predict(DataFrame) vs predict: {e[:60] if isinstance(e, str) else fmt(np.abs(e - p).max())}"
                  f" | estimator.predict(numpy) vs predict: {e2[:60] if isinstance(e2, str) else fmt(np.abs(e2 - p).max())}")

print("\n# F. Exog as a dict, the category dtype of `c` is different in each series (None, LightGBM)")
ex_a = exog.copy()
ex_b = exog.assign(c=exog["c"].cat.reorder_categories(sorted(exog["c"].cat.categories, reverse=True)))
ex_c = exog.assign(c=exog["c"].cat.add_categories([90]))
for label, ex in (("same dtype", {"a": ex_a, "b": ex_a.copy()}), ("other order in b", {"a": ex_a, "b": ex_b}),
                  ("one more category in b", {"a": ex_a, "b": ex_c})):
    f = ForecasterRecursiveMultiSeries(lgbm(), lags=1, categorical_features=None)
    X = attempt(lambda: f.create_train_X_y(series={"a": series["a"], "b": series["b"]}, exog=ex)[0])
    res = attempt(lambda: f.fit(series={"a": series["a"], "b": series["b"]}, exog=ex))
    print(f"  {label:24}: X_train c dtype {X if isinstance(X, str) else X['c'].dtype} | fit {'ok' if not isinstance(res, str) else '-> ' + res[:120]}")

print("\n# G. 'auto', exog without categorical columns (c as int64): estimator configuration of the user after fit")
exog_i = exog.assign(c=exog["c"].astype("int64"))
for name, mk, params in (("XGBoost(feature_types=[q,q,q,c], enable_categorical=True)",
                          lambda: xgb(feature_types=["q", "q", "q", "c"], enable_categorical=True),
                          ("feature_types", "enable_categorical")),
                         ("HGB(categorical_features=[3])", lambda: hgb(categorical_features=[3]),
                          ("categorical_features",))):
    for cf in ("auto", None):
        f = ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=cf)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.fit(series=series, exog=exog_i)
        print(f"  {name:58} cf={cf!s:4}: after fit {({p: f.estimator.get_params()[p] for p in params})},"
              f" warnings about it {len([x for x in w if 'overridden' in str(x.message) or 'been set' in str(x.message)])}")
print(_env.versions())
