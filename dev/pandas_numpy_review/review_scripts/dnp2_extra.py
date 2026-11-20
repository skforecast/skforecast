"""
D-NP2: checks that `dnp2_grid.py` does not cover. Run with each tree (first
the current code, which writes the fitted forecasters that section 7 reads):

1. The old recipe (OrdinalEncoder + astype('category') in `transformer_exog`).
2. Categorical features set by the user on the estimator, by index or by name.
3. `encoding='ordinal_category'` with None and a configuration of the user.
4. Wrapped estimators (TransformedTargetRegressor, Pipeline).
5. The names the fitted estimator keeps.
6. Thirteen estimators with a single float block: predictions saved.
7. Forecasters fitted with the current code, pickled and used with each tree.

Usage: SK_TREE=<tree> bash run3.sh dnp2_extra.py OUTDIR   (text to stdout)
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import pickle
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor, make_column_selector, make_column_transformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from p1_common import make_data
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
out = Path(sys.argv[1]) / f"extra_{_env.tag()}"
out.mkdir(parents=True, exist_ok=True)
tree = _env.tree()
LG = lambda **k: LGBMRegressor(verbose=-1, n_estimators=40, min_child_samples=5, random_state=1, n_jobs=1, **k)  # noqa
XG = lambda **k: XGBRegressor(n_estimators=40, random_state=1, n_jobs=1, **k)  # noqa
HG = lambda **k: HistGradientBoostingRegressor(max_iter=40, random_state=1, **k)  # noqa
CB = lambda **k: CatBoostRegressor(iterations=40, verbose=0, random_state=1, thread_count=1, allow_writing_files=False, **k)  # noqa


def metrics(make, series, exog):
    """one-step-ahead MAE and backtesting MAE (1 step, no refit)."""
    res = []
    for name in ("osa", "bt"):
        try:
            f = make()
            if name == "osa":
                r = grid_search_forecaster_multiseries(
                    f, series=series, exog=exog, cv=OneStepAheadFold(initial_train_size=220),
                    param_grid={}, lags_grid=[3], metric="mean_absolute_error", return_best=False,
                    show_progress=False, verbose=False)
                res.append(f"{float(r.filter(like='mean_absolute_error').iloc[0, 0]):6.2f}")
            else:
                m, _ = backtesting_forecaster_multiseries(
                    f, series=series, exog=exog, cv=TimeSeriesFold(initial_train_size=220, steps=1, refit=False),
                    metric="mean_absolute_error", show_progress=False)
                res.append(f"{float(m.loc[m['levels'] == 'weighted_average', 'mean_absolute_error'].iloc[0]):6.2f}")
        except Exception as e:  # noqa
            res.append(f"{type(e).__name__}: {str(e).splitlines()[0][:70]}")
    return f"osa {res[0]} | bt {res[1]}"


series, exog, exog_p = make_data([10, 20, 30, 40, 50, 60, 70, 80], n_series=6)
series = pd.DataFrame(series)
MS = ForecasterRecursiveMultiSeries

print(f"##### tree: {tree}")
print("\n== 1. Legacy recipe: OrdinalEncoder(int) + astype('category') in transformer_exog")
legacy = lambda: make_column_transformer(  # noqa
    (make_pipeline(OrdinalEncoder(dtype=int), FunctionTransformer(lambda x: x.astype("category"), feature_names_out="one-to-one")),
     ["c"]), remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
for name, est in (("lgbm", LG), ("xgb", XG), ("hgb", HG)):
    for cf in (None, "auto"):
        print(f"  {name:5} legacy + categorical_features={cf!s:5}: ",
              metrics(lambda: MS(est(), lags=3, transformer_exog=legacy(), categorical_features=cf), series, exog))

print("\n== 2. Categorical features set by the user on the estimator, categorical_features=None")
ci = 5  # position of 'c': lag_1..3, _level_skforecast, x, c
cases = {
    "lgbm by index": lambda: MS(LG(), lags=3, categorical_features=None, fit_kwargs={"categorical_feature": [ci]}),
    "lgbm by name": lambda: MS(LG(), lags=3, categorical_features=None, fit_kwargs={"categorical_feature": ["c"]}),
    "hgb by index": lambda: MS(HG(categorical_features=[ci]), lags=3, categorical_features=None),
    "hgb by name": lambda: MS(HG(categorical_features=["c"]), lags=3, categorical_features=None),
    "xgb feature_types": lambda: MS(XG(feature_types=["q"] * 5 + ["c"], enable_categorical=True), lags=3, categorical_features=None),
    "catboost by index": lambda: MS(CB(), lags=3, categorical_features=None, fit_kwargs={"cat_features": [ci]}),
    "catboost by name": lambda: MS(CB(), lags=3, categorical_features=None, fit_kwargs={"cat_features": ["c"]}),
}
for k, make in cases.items():
    print(f"  {k:20}", metrics(make, series, exog))
print("  single series (ForecasterRecursive), for reference:")
y = series["a"]
for k, make in {
    "lgbm by index": lambda: ForecasterRecursive(LG(), lags=3, categorical_features=None, fit_kwargs={"categorical_feature": [4]}),
    "lgbm by name": lambda: ForecasterRecursive(LG(), lags=3, categorical_features=None, fit_kwargs={"categorical_feature": ["c"]}),
    "hgb by name": lambda: ForecasterRecursive(HG(categorical_features=["c"]), lags=3, categorical_features=None),
}.items():
    try:
        make().fit(y=y, exog=exog); r = "fit ok"
    except Exception as e:  # noqa
        r = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    print(f"    {k:18}", r)

print("\n== 3. encoding='ordinal_category' + None + user configuration: what the estimator ends with")
for k, make in {
    "lgbm, nothing set": lambda: MS(LG(), lags=3, encoding="ordinal_category", categorical_features=None),
    "lgbm, user index": lambda: MS(LG(), lags=3, encoding="ordinal_category", categorical_features=None, fit_kwargs={"categorical_feature": [ci]}),
    "hgb, user index": lambda: MS(HG(categorical_features=[ci]), lags=3, encoding="ordinal_category", categorical_features=None),
    "xgb, user types": lambda: MS(XG(feature_types=["q"] * 5 + ["c"], enable_categorical=True), lags=3, encoding="ordinal_category", categorical_features=None),
    "catboost, user index": lambda: MS(CB(), lags=3, encoding="ordinal_category", categorical_features=None, fit_kwargs={"cat_features": [ci]}),
}.items():
    try:
        f = make(); f.fit(series=series, exog=exog)
        e = f.estimator
        if isinstance(e, LGBMRegressor):
            s = e.booster_.model_to_string()
            cat_cols = sorted({
                int(x)
                for blk in re.findall(r"split_feature=([^\n]*)\n(?:.*\n)*?decision_type=([^\n]*)", s)
                for x, d in zip(blk[0].split(), blk[1].split()) if int(d) & 1
            })
            info = f"columns with categorical splits: {cat_cols}"
        elif isinstance(e, XGBRegressor):
            info = f"feature_types={e.get_booster().feature_types}"
        elif isinstance(e, HistGradientBoostingRegressor):
            info = f"is_categorical_={np.flatnonzero(e.is_categorical_).tolist() if e.is_categorical_ is not None else None}"
        else:
            info = f"cat features={e.get_cat_feature_indices()}"
        print(f"  {k:22} {info} | {metrics(make, series, exog)}")
    except Exception as ex:  # noqa
        print(f"  {k:22} {type(ex).__name__}: {str(ex).splitlines()[0][:100]}")

print("\n== 4. Wrapped estimators")
ser0, ex0, _ = make_data(list(range(8)), n_series=6); ser0 = pd.DataFrame(ser0)
ex_text = ex0.copy(); ex_text["c"] = ex_text["c"].astype(int).map(dict(enumerate("abcdefgh"))).astype("category")
for k, make, ex in (
    ("TTR(lgbm), 'auto', ordinal, text", lambda: MS(TransformedTargetRegressor(LG()), lags=3, categorical_features="auto"), ex_text),
    ("TTR(lgbm), 'auto', ordinal_category, text", lambda: MS(TransformedTargetRegressor(LG()), lags=3, encoding="ordinal_category", categorical_features="auto"), ex_text),
    ("TTR(lgbm), None, ordinal, category 0..7", lambda: MS(TransformedTargetRegressor(LG()), lags=3, categorical_features=None), ex0),
    ("Pipeline(scaler, lgbm), 'auto', text", lambda: MS(make_pipeline(StandardScaler(), LG()), lags=3, categorical_features="auto"), ex_text),
    ("Pipeline(scaler, lgbm), None, ordinal_category", lambda: MS(make_pipeline(StandardScaler(), LG()), lags=3, encoding="ordinal_category", categorical_features=None), ex0[["x"]]),
    ("Pipeline(scaler, ridge), 'auto', text", lambda: MS(make_pipeline(StandardScaler(), Ridge()), lags=3, categorical_features="auto"), ex_text),
):
    print(f"  {k:46}", metrics(make, ser0, ex))

print("\n== 5. Names the fitted estimator keeps")
for k, make in {"lgbm": lambda: MS(LG(), lags=3), "xgb": lambda: MS(XG(), lags=3), "hgb": lambda: MS(HG(), lags=3),
                "ridge": lambda: MS(Ridge(), lags=3), "catboost": lambda: MS(CB(), lags=3)}.items():
    f = make(); f.fit(series=series, exog=exog[["x"]])
    e = f.estimator
    names = (getattr(e, "feature_name_", None) or getattr(e, "feature_names_", None)
             or (e.get_booster().feature_names if hasattr(e, "get_booster") else None))
    fn = getattr(e, "feature_names_in_", None)
    imp = f.get_feature_importances()
    Xp = f.create_predict_X(steps=2, exog=exog_p[["x"]])
    Xp = Xp.drop(columns="level") if "level" in Xp.columns else Xp
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            p = e.predict(Xp[f.X_train_features_names_out_]); r = "ok"
        except Exception as ex:  # noqa
            r = f"{type(ex).__name__}: {str(ex).splitlines()[0][:60]}"
    ws = sorted({str(x.message)[:50] for x in w})
    print(f"  {k:9} feature_names_in_={None if fn is None else list(fn)[:3]}...; own names={None if names is None else list(names)[:3]}; "
          f"get_feature_importances={imp['feature'].tolist()[:3] if imp is not None else None}; estimator.predict(create_predict_X) {r} {ws}")

print("\n== 6. More estimators, one float block (encoding ordinal / None / onehot), predictions saved")
zoo = {"linear": LinearRegression, "elastic": ElasticNet, "rf": lambda: RandomForestRegressor(n_estimators=10, random_state=1),
       "gbr": lambda: GradientBoostingRegressor(n_estimators=10, random_state=1), "knn": KNeighborsRegressor, "svr": SVR,
       "mlp": lambda: MLPRegressor(max_iter=30, random_state=1), "lgbm": LG, "xgb": XG, "hgb": HG, "catboost": CB,
       "pipe": lambda: make_pipeline(StandardScaler(), Ridge()), "ttr": lambda: TransformedTargetRegressor(Ridge(), transformer=StandardScaler())}
preds = {}
for k, est in zoo.items():
    for enc in ("ordinal", None, "onehot"):
        try:
            f = MS(est(), lags=3, encoding=enc); f.fit(series=series, exog=exog[["x"]])
            preds[(k, enc)] = f.predict(steps=5, exog=exog_p[["x"]])["pred"].to_numpy()
        except Exception as ex:  # noqa
            preds[(k, enc)] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:80]}"
pickle.dump(preds, open(out / f"zoo_{tree}.pkl", "wb"))
print("  errors:", {k: v for k, v in preds.items() if isinstance(v, str)})

print("\n== 7. Forecasters fitted with the current code and saved (only written by 'today', read by both)")
if tree == "today":
    saved = {}
    for k, make, ex in (("lgbm auto text", lambda: MS(LG(), lags=3), ex_text), ("xgb auto ordinal_category", lambda: MS(XG(), lags=3, encoding="ordinal_category"), ex_text),
                        ("hgb auto text", lambda: MS(HG(), lags=3), ex_text), ("catboost auto text", lambda: MS(CB(), lags=3), ex_text),
                        ("lgbm None 0..7", lambda: MS(LG(), lags=3, categorical_features=None), ex0),
                        ("lgbm None 10..80", lambda: MS(LG(), lags=3, categorical_features=None), exog)):
        f = make(); f.fit(series=ser0 if ex is not exog else series, exog=ex, store_in_sample_residuals=True)
        saved[k] = (f, np.concatenate([np.sort(v) for v in f.in_sample_residuals_.values()]))
    pickle.dump(saved, open(out / "saved_today.pkl", "wb"))
saved = pickle.load(open(out / "saved_today.pkl", "rb"))
for k, (f, resid) in saved.items():
    ex = exog if "10..80" in k else (ex0 if "0..7" in k else ex_text)
    se = series if "10..80" in k else ser0
    exp = ex.iloc[:5].copy(); exp.index = pd.date_range(se.index[-1] + se.index.freq, periods=5, freq="D")
    p = f.predict(steps=5, exog=exp)["pred"].to_numpy()
    try:
        f.set_in_sample_residuals(series=se, exog=ex)
        r2 = np.concatenate([np.sort(v) for v in f.in_sample_residuals_.values()])
        same = f"max |diff| with the residuals stored at fit: {np.abs(r2 - resid).max():.3g}"
    except Exception as ex_:  # noqa
        same = f"{type(ex_).__name__}: {str(ex_).splitlines()[0][:80]}"
    print(f"  {k:28} predict sum {p.sum():.6f}; set_in_sample_residuals -> {same}")
print(_env.versions())
