"""
Option (c), item 3. The user configures the categorical features of the
estimator (LightGBM `categorical_feature` and CatBoost `cat_features` in
`fit_kwargs`; XGBoost `feature_types` and HistGradientBoosting
`categorical_features` in the constructor) and the forecaster has
`categorical_features=None` (or `'auto'`, to see the existing policy).

Data: 6 series, 300 rows, `c` is a `category` exog with categories 10..80
(p1_common.make_data), lags=1, so the columns of X_train are
[lag_1, _level_skforecast, x, c] (c = index 3, level = index 1).

For each case, today: does `fit` work, warnings, which columns the fitted
model treats as categorical, and whether `predict` (backtesting of 1 step, no
refit) agrees with the estimator evaluated as it was fitted. Then the
prototype of (c) "codes" as written in p1c_common (it calls
`configure_estimator_categorical_features`, that is, the policy of `'auto'`:
override and warn): categorical columns, and whether the model is today's.
"""
import _env
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils.utils import _build_predict_function, cast_catboost_categorical_columns_dataframe
from p1_common import make_data
from p1c_common import option_c, head, backtest, test_rows, to_matrix, attempt, fmt

N_TRAIN = 200
NAMES = ["lag_1", "_level_skforecast", "x", "c"]
lgbm = lambda **kw: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0, **kw)
xgb = lambda **kw: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0, **kw)
hgb = lambda **kw: HistGradientBoostingRegressor(max_iter=40, random_state=0, **kw)
cat = lambda **kw: CatBoostRegressor(iterations=40, verbose=0, thread_count=1, random_state=0,
                                     allow_writing_files=False, **kw)


def categorical_columns(est):
    """Columns the fitted model treats as categorical (LightGBM, XGBoost: with a categorical split)."""
    kind = type(est).__module__.split(".")[0]
    if kind == "lightgbm":
        df = est.booster_.trees_to_dataframe()
        used = sorted(set(df.loc[df["decision_type"] == "==", "split_feature"]))
        return [NAMES[int(u.split("_")[1])] if u.startswith("Column_") else u for u in used]
    if kind == "xgboost":
        df = est.get_booster().trees_to_dataframe()
        used = sorted(set(df.loc[df["Category"].notna(), "Feature"]))
        return [NAMES[int(u[1:])] if u[1:].isdigit() else u for u in used]
    if kind == "catboost":
        return [NAMES[j] for j in est.get_cat_feature_indices()]
    return [] if est.is_categorical_ is None else [NAMES[j] for j in np.flatnonzero(est.is_categorical_)]


def fit(f, series, exog):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = attempt(lambda: f.fit(series=series, exog=exog))
    kinds = sorted({x.category.__name__ + ": " + str(x.message)[:95] for x in w
                    if "overridden" in str(x.message) or "have been set" in str(x.message)
                    or "has been set" in str(x.message)})
    return res, kinds


def as_fitted(f, series, exog):
    X, key = test_rows(f, series, exog, N_TRAIN)
    if type(f.estimator).__module__.startswith("catboost"):
        kw = {"cat_features": list(f.estimator.get_cat_feature_indices())}
        X = cast_catboost_categorical_columns_dataframe(X=X, fit_kwargs=kw, estimator=f.estimator,
                                                        feature_names=list(X.columns))
    return pd.Series(f.estimator.predict(X), index=key)


CASES = [  # (label, estimator factory, fit_kwargs, encoding)
    ("LightGBM, nothing", lgbm, None, "ordinal"),
    ("LightGBM categorical_feature=['c'] (name)", lgbm, {"categorical_feature": ["c"]}, "ordinal"),
    ("LightGBM categorical_feature=[3] (index of c)", lgbm, {"categorical_feature": [3]}, "ordinal"),
    ("LightGBM categorical_feature='auto'", lgbm, {"categorical_feature": "auto"}, "ordinal"),
    ("LightGBM categorical_feature=[1] (level, not c)", lgbm, {"categorical_feature": [1]}, "ordinal"),
    ("LightGBM categorical_feature=[1, 3]", lgbm, {"categorical_feature": [1, 3]}, "ordinal"),
    ("LightGBM categorical_feature=[] ", lgbm, {"categorical_feature": []}, "ordinal"),
    ("CatBoost cat_features=[3]", cat, {"cat_features": [3]}, "ordinal"),
    ("CatBoost cat_features=['c']", cat, {"cat_features": ["c"]}, "ordinal"),
    ("CatBoost(cat_features=[3]) in the constructor", lambda: cat(cat_features=[3]), None, "ordinal"),
    ("XGBoost, nothing", xgb, None, "ordinal"),
    ("XGBoost(enable_categorical=True)", lambda: xgb(enable_categorical=True), None, "ordinal"),
    ("XGBoost(feature_types=[q,q,q,c], enable_categorical=True)",
     lambda: xgb(feature_types=["q", "q", "q", "c"], enable_categorical=True), None, "ordinal"),
    ("XGBoost(feature_types=[q,c,q,q], enable_categorical=True) (level, not c)",
     lambda: xgb(feature_types=["q", "c", "q", "q"], enable_categorical=True), None, "ordinal"),
    ("HGB, nothing (from_dtype)", hgb, None, "ordinal"),
    ("HGB(categorical_features=[3])", lambda: hgb(categorical_features=[3]), None, "ordinal"),
    ("HGB(categorical_features=['c'])", lambda: hgb(categorical_features=["c"]), None, "ordinal"),
    ("HGB(categorical_features=[1]) (level, not c)", lambda: hgb(categorical_features=[1]), None, "ordinal"),
    ("HGB(categorical_features=None)", lambda: hgb(categorical_features=None), None, "ordinal"),
]

series, exog, _ = make_data(tuple(range(10, 90, 10)), n_series=6)
s_tr, e_tr = head(series, N_TRAIN), head(exog, N_TRAIN)
for cf in (None, "auto"):
    print(f"\n# categorical_features={cf!r}")
    for label, mk, fk, enc in CASES:
        new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, encoding=enc, categorical_features=cf,
                                                     fit_kwargs=fk)
        print(f"- {label}")
        f0 = attempt(new)
        if isinstance(f0, str):
            print(f"    forecaster -> {f0}")
            continue
        res, kinds = fit(f0, s_tr, e_tr)
        ins0 = None
        if isinstance(res, str):
            print(f"    today: fit -> {res}")
        else:
            bt = attempt(lambda: backtest(new(), series, exog, N_TRAIN))
            ref = as_fitted(f0, series, exog)
            ok = bt if isinstance(bt, str) else fmt(np.abs(bt["pred"].reindex(ref.index) - ref).max())
            Xtr = f0.create_train_X_y(series=s_tr, exog=e_tr)[0][NAMES]
            if not type(f0.estimator).__module__.startswith("catboost"):
                ins0 = f0.estimator.predict(Xtr)
            print(f"    today: fit ok, categorical in the model {categorical_columns(f0.estimator)}"
                  f" | predict vs as fitted: {ok}")
        for k in kinds:
            print(f"           warning {k}")
        if cf is not None:
            continue
        with option_c("codes"):
            f1 = new()
            res, kinds = fit(f1, s_tr, e_tr)
        if isinstance(res, str):
            print(f"    (c)  : fit -> {res}")
        else:
            out = f"    (c)  : fit ok, categorical in the model {categorical_columns(f1.estimator)}"
            if ins0 is not None:
                Xtr = f1.create_train_X_y(series=s_tr, exog=e_tr)[0][NAMES]
                ins1 = _build_predict_function(f1.estimator)(to_matrix(Xtr, "codes")[0])
                out += f" | model vs today: {fmt(np.abs(ins1 - ins0).max())}"
            print(out)
        for k in kinds:
            print(f"           warning {k}")
print(_env.versions())
