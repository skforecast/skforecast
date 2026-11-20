"""
Option (c), item 4. The level column of `encoding='ordinal_category'` with
`categorical_features=None`: today it is categorical only because of its
dtype (the index is not given to the estimator, :2451-2464). With numpy in
`fit`, (c) has to give its index too.

Datasets: "6 series" (p1_common.make_data, with and without the `category`
exog `c`, categories 10..80) and "30 series" (p1c_common.make_big: c40, c3, x).
For each estimator: columns the model of today treats as categorical, whether
today's `predict` agrees with the estimator as it was fitted, and for the
prototypes of (c) ("codes" and "values") whether the model is today's
(in-sample predictions; LightGBM: tree structure) and whether the backtesting
predictions (1 step, no refit) are the ones of today's estimator as fitted.
`encoding='ordinal'` is included as a control (the level is numeric).
"""
import _env
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils.utils import _build_predict_function
from p1_common import make_data
from p1c_common import (
    make_big, option_c, head, backtest, test_rows, to_matrix, lgbm_structure, same_structure, attempt, fmt,
)

EST = {
    "LightGBM": lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0),
    "XGBoost": lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=40, random_state=0),
    "CatBoost": lambda: CatBoostRegressor(iterations=40, verbose=0, thread_count=1, random_state=0,
                                          allow_writing_files=False),
}


def categorical_columns(est, names):
    kind = type(est).__module__.split(".")[0]
    if kind == "lightgbm":
        df = est.booster_.trees_to_dataframe()
        used = sorted(set(df.loc[df["decision_type"] == "==", "split_feature"]))
        return sorted(names[int(u.split("_")[1])] if u.startswith("Column_") else u for u in used)
    if kind == "xgboost":
        df = est.get_booster().trees_to_dataframe()
        used = sorted(set(df.loc[df["Category"].notna(), "Feature"]))
        return sorted(names[int(u[1:])] if u not in names and u[1:].isdigit() else u for u in used)
    if kind == "catboost":
        return [names[j] for j in est.get_cat_feature_indices()]
    return [] if est.is_categorical_ is None else [names[j] for j in np.flatnonzero(est.is_categorical_)]


s6, e6, _ = make_data(tuple(range(10, 90, 10)), n_series=6)
s30, e30, _ = make_big(n=300)
DATA = {
    "6 series, no exog": (s6, None, 200),
    "6 series, numeric exog x": (s6, e6[["x"]], 200),
    "6 series, x and category c (10..80)": (s6, e6, 200),
    "30 series, x, c40, c3": (s30, e30, 200),
}

for enc in ("ordinal_category", "ordinal"):
    print(f"\n# encoding={enc!r}, categorical_features=None")
    for dlabel, (series, exog, n_train) in DATA.items():
        if enc == "ordinal" and "category" not in dlabel and "c40" not in dlabel:
            continue
        s_tr, e_tr = head(series, n_train), None if exog is None else head(exog, n_train)
        print(f"- {dlabel}")
        for name, mk in EST.items():
            new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=2, encoding=enc, categorical_features=None)
            f0 = new()
            res = attempt(lambda: f0.fit(series=s_tr, exog=e_tr))
            ref = ins0 = None
            if isinstance(res, str):
                print(f"    {name:8} today      : fit -> {res[:110]}")
            else:
                names = f0.X_train_features_names_out_
                Xte, key = test_rows(f0, series, exog, n_train)
                ref = pd.Series(f0.estimator.predict(Xte), index=key)
                Xtr = f0.create_train_X_y(series=s_tr, exog=e_tr)[0][names]
                ins0 = f0.estimator.predict(Xtr)
                bt = backtest(new(), series, exog, n_train)["pred"].reindex(key)
                print(f"    {name:8} today      : categorical {categorical_columns(f0.estimator, names)}"
                      f" | predict vs as fitted: {fmt(np.abs(bt - ref).max())}")
            base = None
            for variant in ("codes", "values"):
                with option_c(variant):
                    f1 = new()
                    res = attempt(lambda: f1.fit(series=s_tr, exog=e_tr))
                    bt = res if isinstance(res, str) else attempt(lambda: backtest(new(), series, exog, n_train))
                    if isinstance(bt, str):
                        print(f"    {name:8} (c) {variant:6} : -> {bt[:110]}")
                        continue
                    names = f1.X_train_features_names_out_
                    Xtr = f1.create_train_X_y(series=s_tr, exog=e_tr)[0][names]
                    ins1 = _build_predict_function(f1.estimator)(to_matrix(Xtr, variant)[0])
                Xte, key = test_rows(f1, series, exog, n_train)
                bt = bt["pred"].reindex(key)
                out = f"    {name:8} (c) {variant:6} : categorical {categorical_columns(f1.estimator, names)}"
                if ins0 is not None:
                    out += f" | model: {fmt(np.abs(ins1 - ins0).max())}"
                    if name == "LightGBM":
                        out += f", trees {same_structure(lgbm_structure(f0.estimator), lgbm_structure(f1.estimator))}"
                    out += f" | predict vs today as fitted: {fmt(np.abs(bt - ref).max())}"
                elif base is None:
                    base = bt
                else:
                    out += f" | predictions vs (c) codes: {fmt(np.abs(bt - base).max())}"
                print(out)
print(_env.versions())
