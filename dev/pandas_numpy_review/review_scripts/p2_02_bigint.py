"""
P2, item 4: integers above 2**53. Is the precision "already lost when joined
with the float lags" (study, P2 risk) in every path?

For each exog, today vs proposal: dtype of the matrix the estimator gets, and
whether the big integers come back exact in `create_train_X_y` and
`create_predict_X` (which restore `exog_dtypes_out_`), plus predictions.
"""
import _env
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
import p2_proto as P

n, steps = 40, 2
y = P.make_y(n + steps)
idx = y.index
big = 2**53 + 1 + 2 * np.arange(n + steps, dtype=np.int64)          # odd: not representable
ns = pd.date_range("2024-01-01", periods=n + steps, freq="D").as_unit("ns").asi8 + 1
ubig = (2**64 - 1 - np.arange(n + steps, dtype=np.uint64) * np.uint64(3)).astype(np.uint64)
flag = np.arange(n + steps) % 2 == 0
flt = np.linspace(0, 1, n + steps)

cases = {
    "int64 alone": pd.DataFrame({"id": big}, index=idx),
    "int64 + int64": pd.DataFrame({"id": big, "k": np.arange(n + steps)}, index=idx),
    "int64 + float": pd.DataFrame({"id": big, "a": flt}, index=idx),
    "int64 + bool": pd.DataFrame({"id": big, "b": flag}, index=idx),
    "int64 ns timestamps + bool": pd.DataFrame({"id": ns, "b": flag}, index=idx),
    "Int64 alone": pd.DataFrame({"id": pd.array(big, dtype="Int64")}, index=idx),
    "Int64 + float": pd.DataFrame({"id": pd.array(big, dtype="Int64"), "a": flt}, index=idx),
    "Int64 + bool": pd.DataFrame({"id": pd.array(big, dtype="Int64"), "b": flag}, index=idx),
    "uint64 alone": pd.DataFrame({"id": ubig}, index=idx),
    "uint64 + bool": pd.DataFrame({"id": ubig, "b": flag}, index=idx),
    "UInt64 + float": pd.DataFrame({"id": pd.array(ubig, dtype="UInt64"), "a": flt}, index=idx),
}
try:
    cases["int64[pyarrow] + float"] = pd.DataFrame(
        {"id": pd.array(big, dtype="int64[pyarrow]"), "a": flt}, index=idx)
    import decimal
    import pyarrow as pa
    cases["decimal128(20,0)[pyarrow] + float"] = pd.DataFrame(
        {"id": pd.array([decimal.Decimal(int(v)) for v in big],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 0))), "a": flt}, index=idx)
except Exception as e:  # noqa: BLE001
    print("int64[pyarrow] + float, decimal128 + float: SKIPPED", type(e).__name__)


def describe(res, status, exog):
    """dtype of the returned `id` column and whether every integer in it is
    exactly one of the input integers (the rows differ by forecaster)."""
    if status != "ok":
        return status
    X = res[0] if isinstance(res, tuple) else res
    col = X.filter(like="id").iloc[:, 0]
    valid = {int(v) for v in exog["id"]}
    return f"{col.dtype} exact={all(int(v) in valid for v in col)}"


def fit_predict(f, tr, pr):
    f.fit(y.iloc[:n], exog=tr)
    return f.predict(steps=steps, exog=pr)


def one(make, exog):
    tr, pr = exog.iloc[:n], exog.iloc[n:]
    out = {}
    f = make(Ridge())
    res, st, _ = P.run(lambda: f.create_train_X_y(y.iloc[:n], exog=tr))
    out["ctxy"] = describe(res, st, exog)
    with P.capture_fit(f.estimator) as seen:
        _, st, _ = P.run(lambda: f.fit(y.iloc[:n], exog=tr))
    out["X"] = seen.get("dtype", st)
    res, st, _ = P.run(lambda: f.create_predict_X(steps=steps, exog=pr))
    out["cpX"] = describe(res, st, exog)
    preds = {}
    for name, est in (
        ("Ridge", Ridge()),
        ("LGBM", LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3)),
    ):
        preds[name], out[name], _ = P.run(lambda: fit_predict(make(est), tr, pr))
    return out, preds


makers = {
    "Rec": lambda e: ForecasterRecursive(e, lags=2),
    "Dir": lambda e: ForecasterDirect(e, lags=2, steps=steps, n_jobs=1),
}
for cname, exog in cases.items():
    print(f"\n### {cname}  exog.to_numpy(): {exog.to_numpy().dtype}  condition={P.condition(exog)}")
    for fname, make in makers.items():
        today, p_today = one(make, exog)
        with P.proposal():
            new, p_new = one(make, exog)
        eq = {k: P.same(p_today[k], p_new[k]) for k in p_today}
        print(f"  {fname}: estimator X {today['X']} -> {new['X']}")
        print(f"       create_train_X_y id: {today['ctxy']} -> {new['ctxy']}")
        print(f"       create_predict_X id: {today['cpX']} -> {new['cpX']}")
        print(f"       fit+predict Ridge {today['Ridge']} -> {new['Ridge']}, LGBM {today['LGBM']} -> {new['LGBM']}; predictions equal {eq}")
print(_env.versions())
