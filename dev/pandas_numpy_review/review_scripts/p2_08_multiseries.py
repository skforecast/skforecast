"""
P2, item 11: `ForecasterRecursiveMultiSeries` with a bool / nullable exog column.
Study (I1, I2): in fit the column "stays as a separate block of the DataFrame"
and nullable with NA works in fit and fails in predict.

Per case and estimator: what `estimator.fit` receives, the dtype of the exog
matrix used in the recursive loop of predict (`exog_values_dict[1]`), fit and
predict status, and whether the predictions equal those obtained with the same
exog column cast to float64 beforehand (NA -> NaN), which is what P2 would do.
"""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
import p2_proto as P

n, steps = 60, 3
y = P.make_y(n + steps)
series = pd.DataFrame({"s1": y, "s2": y.to_numpy()[::-1]}, index=y.index)
k = np.arange(n + steps)
a = np.linspace(0, 1, n + steps)


def col(dtype, values, na=None):
    x = pd.array(values, dtype=dtype)
    if na is not None:
        x[na] = pd.NA
    return x


cases = {
    "bool": lambda: k % 3 == 0,
    "boolean": lambda: col("boolean", k % 3 == 0),
    "Int64": lambda: col("Int64", k % 5),
    "Float64": lambda: col("Float64", k / 7),
    "Float64 NA in fit": lambda: col("Float64", k / 7, 30),
    "Float64 NA in predict": lambda: col("Float64", k / 7, n + 1),
    "Int64 NA in predict": lambda: col("Int64", k % 5, n + 1),
    "double[pyarrow]": lambda: col("double[pyarrow]", k / 7),
    "double[pyarrow] null in predict": lambda: col("double[pyarrow]", k / 7, n + 1),
}
estimators = {
    "Ridge": lambda: Ridge(),
    "LGBM": lambda: LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
}


def experiment(est, exog):
    with warnings.catch_warnings():  # advice about scaling with linear models
        warnings.simplefilter("ignore")
        f = ForecasterRecursiveMultiSeries(est, lags=3, encoding="ordinal")
    seen = {}
    cls, orig_fit = type(est), type(est).fit
    orig_rp = ForecasterRecursiveMultiSeries._recursive_predict

    def fit(self, X, y, *args, **kw):
        seen["X"] = (f"DataFrame{sorted({str(d) for d in X.dtypes})}"
                     if isinstance(X, pd.DataFrame) else f"ndarray[{X.dtype}]")
        return orig_fit(self, X, y, *args, **kw)

    def rp(self, *args, **kw):
        ev = kw.get("exog_values_dict")
        seen["predict exog"] = None if ev is None else str(ev[1].dtype)
        return orig_rp(self, *args, **kw)

    cls.fit, ForecasterRecursiveMultiSeries._recursive_predict = fit, rp
    try:
        _, st_fit, w_fit = P.run(lambda: f.fit(series.iloc[:n], exog=exog.iloc[:n]))
        pred, st_pred, w_pred = (None, "-", [])
        if st_fit == "ok":
            pred, st_pred, w_pred = P.run(lambda: f.predict(steps=steps, exog=exog.iloc[n:]))
    finally:
        cls.fit, ForecasterRecursiveMultiSeries._recursive_predict = orig_fit, orig_rp
    return seen, st_fit, st_pred, w_fit, w_pred, pred


for cname, make in cases.items():
    try:
        exog = pd.DataFrame({"a": a, "x": make()}, index=y.index)
    except Exception as e:  # noqa: BLE001
        print(f"\n### {cname}: SKIPPED ({type(e).__name__})")
        continue
    as_float = exog.assign(x=exog["x"].to_numpy(dtype=float, na_value=np.nan))
    print(f"\n### {cname}")
    for ename, make_e in estimators.items():
        seen, st_fit, st_pred, w_fit, w_pred, pred = experiment(make_e(), exog)
        seen_f, sf_fit, sf_pred, wf_fit, wf_pred, pred_f = experiment(make_e(), as_float)
        print(f"  {ename:5s} today   : fit X {seen.get('X')} -> {st_fit[:70]} {w_fit} | "
              f"predict exog matrix {seen.get('predict exog')} -> {st_pred[:75]} {w_pred}")
        print(f"        as float: fit X {seen_f.get('X')} -> {sf_fit[:70]} {wf_fit} | "
              f"predict exog matrix {seen_f.get('predict exog')} -> {sf_pred[:75]} {wf_pred} | "
              f"predictions equal: {P.same(pred, pred_f)}")
print(_env.versions())
