"""
P2, items 1, 2, 3 and 8: for each exog dtype, what the estimator receives and
whether fit/predict results, warnings and errors change with
`exog.to_numpy(dtype=float, na_value=np.nan)` (prototype: see p2_proto.py).

One line per (case, forecaster, group of estimators with the same outcome):
  X: what estimator.fit receives today -> with the proposal [element types
     of the exog column today, if object]; "not reached" = error before it
  fit / predict: status today -> proposal ("=" when the same)
  eq: np.array_equal of the predictions today vs proposal
  warn: warning categories today -> proposal (the `IgnoredArgumentWarning`
     of XGBoost in the recursive forecaster is unrelated and left out)
Usage: p2_01_dtype_matrix.py [substring of the case name]
"""
import _env
import sys
import decimal
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
import p2_proto as P

n, steps = 60, 3
y = P.make_y(n + steps)
idx = y.index
vals = np.arange(n + steps) % 7
NA_FIT, NA_PRED = 30, n + 1  # row with NA inside train / inside the predict horizon


def col(dtype, na=None, values=vals):
    a = pd.array(values, dtype=dtype)
    if na is not None:
        a[na] = pd.NA
    return a


def frame(x, with_float=True):
    d = {"a": np.linspace(0, 1, n + steps)} if with_float else {}
    d["x"] = x
    return pd.DataFrame(d, index=idx)


cases = {}
cases["bool alone"] = lambda: frame(vals > 3, with_float=False)
cases["bool + float"] = lambda: frame(vals > 3)
cases["boolean"] = lambda: frame(col("boolean", values=vals > 3))
cases["boolean NA fit"] = lambda: frame(col("boolean", NA_FIT, vals > 3))
cases["boolean NA pred"] = lambda: frame(col("boolean", NA_PRED, vals > 3))
for dt in ("Int64", "Int8", "UInt64", "Float64", "Float32"):
    cases[dt] = lambda dt=dt: frame(col(dt))
    cases[f"{dt} alone"] = lambda dt=dt: frame(col(dt), with_float=False)
    cases[f"{dt} NA fit"] = lambda dt=dt: frame(col(dt, NA_FIT))
    cases[f"{dt} NA pred"] = lambda dt=dt: frame(col(dt, NA_PRED))
cases["float64 NaN fit (reference)"] = lambda: frame(
    np.where(np.arange(n + steps) == NA_FIT, np.nan, vals.astype(float)))
cases["float64 NaN pred (reference)"] = lambda: frame(
    np.where(np.arange(n + steps) == NA_PRED, np.nan, vals.astype(float)))
for dt in ("int64[pyarrow]", "double[pyarrow]"):
    cases[dt] = lambda dt=dt: frame(col(dt))
    cases[f"{dt} null fit"] = lambda dt=dt: frame(col(dt, NA_FIT))
    cases[f"{dt} null pred"] = lambda dt=dt: frame(col(dt, NA_PRED))
cases["bool[pyarrow]"] = lambda: frame(col("bool[pyarrow]", values=vals > 3))
cases["bool[pyarrow] null fit"] = lambda: frame(col("bool[pyarrow]", NA_FIT, vals > 3))


def _arrow(typ, values):
    import pyarrow as pa
    return pd.array(values, dtype=pd.ArrowDtype(typ(pa)))


cases["decimal128[pyarrow]"] = lambda: frame(_arrow(
    lambda pa: pa.decimal128(10, 2), [decimal.Decimal(int(v)) / 4 for v in vals]))
cases["timestamp[pyarrow]"] = lambda: frame(_arrow(lambda pa: pa.timestamp("ns"), list(idx)))
cases["duration[pyarrow]"] = lambda: frame(_arrow(
    lambda pa: pa.duration("ns"), [pd.Timedelta(days=int(v)) for v in vals]))
cases["string[pyarrow] numeric-looking"] = lambda: frame(col("string[pyarrow]", values=vals.astype(str)))
cases["str/object numeric-looking"] = lambda: frame(vals.astype(str).astype(object))
cases["datetime64"] = lambda: frame(idx.to_numpy())
cases["timedelta64"] = lambda: frame(pd.to_timedelta(vals, unit="D").to_numpy())
cases["complex128"] = lambda: frame(vals + 1j)

cases["Int64 alone NA fit"] = lambda: frame(col("Int64", NA_FIT), with_float=False)
cases["Int64 alone NA pred"] = lambda: frame(col("Int64", NA_PRED), with_float=False)
# The default `categorical_features='auto'` encodes every column that is not
# numeric or bool, so the cases above never leave text or dates in the matrix.
# With `categorical_features=None` they stay (suffix "cat=None").
for name in ("string[pyarrow] numeric-looking", "str/object numeric-looking",
             "datetime64", "timedelta64", "timestamp[pyarrow]", "duration[pyarrow]"):
    cases[name + ", cat=None"] = cases[name]
cases["str words, cat=None"] = lambda: frame(np.array(list("abcdefg"), dtype=object)[vals])
cases["object column of numbers, cat=None"] = lambda: frame(vals.astype(object))

estimators = {
    "Ridge": lambda: Ridge(),
    "LGBM": lambda: LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
    "XGB": lambda: XGBRegressor(n_estimators=5, n_jobs=1),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=5, verbose=0, thread_count=1, allow_writing_files=False),
}
forecasters = {
    "Rec": lambda e, **k: ForecasterRecursive(e, lags=3, **k),
    "Rec dropna": lambda e, **k: ForecasterRecursive(e, lags=3, dropna_from_series=True, **k),
    "Dir": lambda e, **k: ForecasterDirect(e, lags=3, steps=steps, n_jobs=1, **k),
    "DMV": lambda e, **k: ForecasterDirectMultiVariate(
        e, level="y", lags=3, steps=steps, n_jobs=1, **k),
}


def experiment(fname, ename, exog, kw_forecaster):
    est = estimators[ename]()
    f = forecasters[fname](est, **kw_forecaster)
    tr, pr = exog.iloc[:n], exog.iloc[n:]
    target = y.iloc[:n].to_frame() if fname == "DMV" else y.iloc[:n]
    kw = {"series": target} if fname == "DMV" else {"y": target}
    with P.capture_fit(est) as seen:
        _, fit_status, fit_w = P.run(lambda: f.fit(exog=tr, **kw))
    pred, pred_status, pred_w = (None, "-", [])
    if fit_status == "ok":
        pred, pred_status, pred_w = P.run(lambda: f.predict(steps=steps, exog=pr))
    x = seen.get("dtype", "not reached")
    return dict(x=x, elem=seen.get("elem"), fit=fit_status, fit_w=fit_w,
                pred=pred_status, pred_w=pred_w, values=pred)


def arrow(a, b):
    return f"{a} -> =" if a == b else f"{a} -> {b}"


only = sys.argv[1] if len(sys.argv) > 1 else ""
for cname, make in cases.items():
    if only not in cname:
        continue
    try:
        exog = make()
    except Exception as e:  # pyarrow missing
        print(f"\n### {cname}: SKIPPED ({type(e).__name__}: {str(e)[:60]})")
        continue
    print(f"\n### {cname}  dtypes={ {c: str(d) for c, d in exog.dtypes.items()} } "
          f"condition={P.condition(exog)} strict={P.strict_condition(exog)}")
    groups = {}
    kwf = {"categorical_features": None} if "cat=None" in cname else {}
    for fname in forecasters:
        if fname == "Rec dropna" and "fit" not in cname:
            continue
        for ename in estimators:
            today = experiment(fname, ename, exog, kwf)
            with P.proposal():
                new = experiment(fname, ename, exog, kwf)
            sig = (
                f"X {today['x']} -> {new['x']} {today['elem'] or ''}"
                f" | fit {arrow(today['fit'], new['fit'])}"
                f" | predict {arrow(today['pred'], new['pred'])}"
                f" | eq {P.same(today['values'], new['values'])}"
                f" | warn fit {arrow(today['fit_w'], new['fit_w'])}"
                f", predict {arrow(today['pred_w'], new['pred_w'])}"
            )
            groups.setdefault(sig, {}).setdefault(fname, []).append(ename)
    for sig, who in groups.items():
        by_est = {}
        for fname, enames in who.items():
            by_est.setdefault(",".join(enames), []).append(fname)
        label = "; ".join(f"{'/'.join(f)}: {e}" for e, f in by_est.items())
        print(f"  [{label}]\n      {sig}")
print(_env.versions())
