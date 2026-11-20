"""
P2, item 5: `create_train_X_y` and `create_predict_X` restore the exog dtypes
from `exog_dtypes_out_` (`compat.astype`). Today the internal matrix is
`object` (or float); with the proposal it is float64. Is the returned
DataFrame the same (dtypes, values, NA vs NaN)?

Per case: dtypes of the exog columns in the output today, and "identical"
(`assert_frame_equal`, exact dtypes) or the difference with the proposal.
"""
import _env
import decimal
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
import p2_proto as P

n, steps = 40, 2
y = P.make_y(n + steps)
idx = y.index
v = np.arange(n + steps) % 5
flt = np.linspace(0, 1, n + steps)


def arr(dtype, values=v, na=None):
    a = pd.array(values, dtype=dtype)
    if na is not None:
        a[na] = pd.NA
    return a


def arrow_decimal():
    import pyarrow as pa
    return pd.array([decimal.Decimal(int(x)) / 8 + decimal.Decimal("0.1") for x in v],
                    dtype=pd.ArrowDtype(pa.decimal128(12, 6)))


NA = [20, n]  # one NA in train and one in the first predicted step
cases = {
    "float + int + bool": lambda: {"a": flt, "h": v, "b": v > 2},
    "bool alone": lambda: {"b": v > 2},
    "float + boolean": lambda: {"a": flt, "b": arr("boolean", v > 2)},
    "float + boolean NA": lambda: {"a": flt, "b": arr("boolean", v > 2, NA)},
    "float + Int64": lambda: {"a": flt, "i": arr("Int64")},
    "float + Int64 NA": lambda: {"a": flt, "i": arr("Int64", na=NA)},
    "Int64 alone NA": lambda: {"i": arr("Int64", na=NA)},
    "float + Int8 + UInt64": lambda: {"a": flt, "i": arr("Int8"), "u": arr("UInt64")},
    "float + Float64 NA": lambda: {"a": flt, "f": arr("Float64", v / 3, NA)},
    "float + Float32": lambda: {"a": flt, "f": arr("Float32", v / 3)},
    "float + float32 + bool": lambda: {"a": flt, "f": (v / 3).astype("float32"), "b": v > 2},
    "float + int64[pyarrow] null": lambda: {"a": flt, "i": arr("int64[pyarrow]", na=NA)},
    "float + double[pyarrow] null": lambda: {"a": flt, "f": arr("double[pyarrow]", v / 3, NA)},
    "float + bool[pyarrow] null": lambda: {"a": flt, "b": arr("bool[pyarrow]", v > 2, NA)},
    "float + decimal128(12,6)[pyarrow]": lambda: {"a": flt, "d": arrow_decimal()},
}
forecasters = {
    "Rec": lambda: ForecasterRecursive(Ridge(), lags=2),
    "Dir": lambda: ForecasterDirect(Ridge(), lags=2, steps=steps, n_jobs=1),
    "DMV": lambda: ForecasterDirectMultiVariate(Ridge(), level="y", lags=2, steps=steps, n_jobs=1),
}


def outputs(fname, exog):
    f = forecasters[fname]()
    tr, pr = exog.iloc[:n], exog.iloc[n:]
    target = y.iloc[:n].to_frame() if fname == "DMV" else y.iloc[:n]
    res, st1, _ = P.run(lambda: f.create_train_X_y(target, exog=tr))
    X_train = res[0] if st1 == "ok" else st1
    # fit is only needed for create_predict_X; LGBM-free: NaN rows are dropped
    f.dropna_from_series = True
    _, st, _ = P.run(lambda: f.fit(target, exog=tr))
    X_pred, st2, _ = P.run(lambda: f.create_predict_X(steps=steps, exog=pr)) if st == "ok" else (None, st, [])
    return X_train, (X_pred if st2 == "ok" else st2)


def compare(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return "identical (same error)" if str(a) == str(b) else f"today: {a if isinstance(a, str) else 'ok'} | proposal: {b if isinstance(b, str) else 'ok'}"
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        return "identical"
    except AssertionError:
        cols = [c for c in a.columns if not a[c].equals(b[c]) or a[c].dtype != b[c].dtype]
        c = cols[0]
        i = next((k for k in range(len(a)) if repr(a[c].iloc[k]) != repr(b[c].iloc[k])), 0)
        return (f"DIFFERENT in {cols}: {c} dtype {a[c].dtype} -> {b[c].dtype}, "
                f"e.g. row {i}: {a[c].iloc[i]!r} -> {b[c].iloc[i]!r}")


def exog_dtypes(X):
    if isinstance(X, str):
        return X
    return {c: str(d) for c, d in X.dtypes.items() if not c.startswith(("lag_", "y_lag"))}


for cname, make in cases.items():
    try:
        exog = pd.DataFrame(make(), index=idx)
    except Exception as e:  # noqa: BLE001
        print(f"\n### {cname}: SKIPPED ({type(e).__name__})")
        continue
    print(f"\n### {cname}")
    for fname in forecasters:
        t_train, t_pred = outputs(fname, exog)
        with P.proposal():
            p_train, p_pred = outputs(fname, exog)
        print(f"  {fname} create_train_X_y today {exog_dtypes(t_train)}\n"
              f"        proposal: {compare(t_train, p_train)}")
        print(f"  {fname} create_predict_X today {exog_dtypes(t_pred)}\n"
              f"        proposal: {compare(t_pred, p_pred)}")
print(_env.versions())
