"""
P1, item 2. sklearn Pipelines as estimator of ForecasterRecursiveMultiSeries:
today (DataFrame in `fit`, numpy in `predict`) against the prototype (numpy in
both). ForecasterRecursive is the reference of "numpy in both" in the package.
"""
import _env
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from p1_common import make_data, numpy_fit, attempt, r

series, exog, exog_p = make_data(tuple("stuvwxyz"), n_series=6)
lgbm = lambda: LGBMRegressor(n_estimators=30, verbose=-1, n_jobs=1, random_state=0)


def by_name(X):
    """A step that needs a DataFrame: selects columns by name."""
    return X[["lag_1", "x", "c"]]


PIPES = {
    "StandardScaler + Ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
    "ColumnTransformer by name + Ridge": lambda: make_pipeline(
        ColumnTransformer([("s", StandardScaler(), ["lag_1", "x"])], remainder="passthrough"), Ridge()),
    "ColumnTransformer by position + Ridge": lambda: make_pipeline(
        ColumnTransformer([("s", StandardScaler(), [0, 3])], remainder="passthrough"), Ridge()),
    "FunctionTransformer X[names] + Ridge": lambda: make_pipeline(FunctionTransformer(by_name), Ridge()),
    "passthrough + LGBM (cf='auto')": lambda: Pipeline([("p", "passthrough"), ("m", lgbm())]),
    "StandardScaler + LGBM (cf='auto')": lambda: make_pipeline(StandardScaler(), lgbm()),
}


def run(make, numpy, cf="auto"):
    f = ForecasterRecursiveMultiSeries(make(), lags=2, categorical_features=cf)
    if numpy:
        with numpy_fit():
            res = attempt(lambda: f.fit(series=series, exog=exog))
    else:
        res = attempt(lambda: f.fit(series=series, exog=exog))
    if isinstance(res, str):
        return "fit -> " + res, None
    p = attempt(lambda: f.predict(8, levels=["a"], exog=exog_p)["pred"].to_numpy())
    return ("predict -> " + p, None) if isinstance(p, str) else ("ok", p)


for name, make in PIPES.items():
    (sa, pa), (sb, pb) = run(make, False), run(make, True)
    cmp = "" if pa is None or pb is None else f" | forecasts identical: {np.array_equal(pa, pb)}"
    fr = ForecasterRecursive(make(), lags=2)
    sr = attempt(lambda: fr.fit(series["a"], exog=exog))
    sr = "ok" if not isinstance(sr, str) else "fit -> " + sr
    print(f"\n# {name}\n  multiseries today: {sa}\n  multiseries fix  : {sb}{cmp}\n  ForecasterRecursive today: {sr}")
    if pa is not None and "LGBM" in name:
        print("  forecast today:", r(pa, 0))

print("\n# categorical_features=None, category exog of ints 10..80 (expected 30, -50, 10, 50, -30, 0, 40, -10)")
series, exog, exog_p = make_data(tuple(range(10, 90, 10)), n_series=6)
for name, make in {
    "passthrough + LGBM": lambda: Pipeline([("p", "passthrough"), ("m", lgbm())]),
    "passthrough + HGB": lambda: Pipeline([("p", "passthrough"), ("m", HistGradientBoostingRegressor(max_iter=30, random_state=0))]),
}.items():
    (sa, pa), (sb, pb) = run(make, False, cf=None), run(make, True, cf=None)
    print(f"  {name}: today {sa} {r(pa, 0) if pa is not None else ''}\n{'':{len(name) + 4}}fix   {sb} {r(pb, 0) if pb is not None else ''}")
print(_env.versions())
