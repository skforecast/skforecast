"""
P2: the two candidate tests for "all columns numeric or bool" against what
`to_numpy()` gives today. `condition` is the study's wording
(`is_numeric_dtype or is_bool_dtype`); `strict` is `p2_proto.strict_condition`.
A good test is True exactly when today's array is `object` and the float
conversion is lossless and does not raise.
"""
import _env
import decimal
import numpy as np
import pandas as pd
import p2_proto as P

n = 4
f = np.array([0.5, 1.5, 2.5, 3.5])
cols = {
    "float64 only": {"a": f},
    "int64 only": {"i": np.arange(n)},
    "bool only": {"b": f > 1},
    "float64 + int64": {"a": f, "i": np.arange(n)},
    "float64 + float32": {"a": f, "g": f.astype("float32")},
    "float64 + bool": {"a": f, "b": f > 1},
    "Int64 only": {"i": pd.array(np.arange(n), dtype="Int64")},
    "Int64 + Float64": {"i": pd.array(np.arange(n), dtype="Int64"), "f": pd.array(f, dtype="Float64")},
    "float64 + Int64": {"a": f, "i": pd.array(np.arange(n), dtype="Int64")},
    "float64 + boolean": {"a": f, "b": pd.array(f > 1, dtype="boolean")},
    "float64 + complex128": {"a": f, "z": f + 1j},
    "float64 + Sparse[float64]": {"a": f, "s": pd.arrays.SparseArray(f)},
    "float64 + Sparse[bool]": {"a": f, "s": pd.arrays.SparseArray(f > 1)},
    "float64 + category(int)": {"a": f, "c": pd.Categorical([1, 2, 1, 2])},
    "float64 + str": {"a": f, "s": ["1", "2", "3", "4"]},
    "float64 + object numbers": {"a": f, "o": np.array([1, 2, 3, 4], dtype=object)},
    "float64 + datetime64": {"a": f, "d": pd.date_range("2020", periods=n)},
    "float64 + timedelta64": {"a": f, "d": pd.to_timedelta(np.arange(n), unit="D")},
}


def pyarrow_cols():
    import pyarrow as pa
    return {
        "float64 + int64[pyarrow]": {"a": f, "i": pd.array(np.arange(n), dtype="int64[pyarrow]")},
        "float64 + double[pyarrow]": {"a": f, "x": pd.array(f, dtype="double[pyarrow]")},
        "float64 + bool[pyarrow]": {"a": f, "b": pd.array(f > 1, dtype="bool[pyarrow]")},
        "float64 + decimal128[pyarrow]": {"a": f, "d": pd.array(
            [decimal.Decimal(i) for i in range(n)], dtype=pd.ArrowDtype(pa.decimal128(10, 2)))},
        "float64 + timestamp[pyarrow]": {"a": f, "d": pd.array(
            list(pd.date_range("2020", periods=n)), dtype=pd.ArrowDtype(pa.timestamp("ns")))},
        "float64 + string[pyarrow]": {"a": f, "s": pd.array(["1", "2", "3", "4"], dtype="string[pyarrow]")},
    }


try:
    cols.update(pyarrow_cols())
except Exception as e:  # noqa: BLE001
    print("pyarrow cases: SKIPPED", type(e).__name__)

print(f"{'frame':30s} {'to_numpy() today':17s} {'condition':9s} {'strict':6s}  to_numpy(dtype=float, na_value=nan)")
for name, data in cols.items():
    df = pd.DataFrame(data)
    today = str(df.to_numpy().dtype)
    out, status, warns = P.run(lambda: df.to_numpy(dtype=float, na_value=np.nan))
    res = f"ok {warns if warns else ''}" if status == "ok" else status[:60]
    c, s = P.condition(df), P.strict_condition(df)
    flag = ""
    if c and today != "object":
        flag += " <- condition converts a frame that is not object today"
    if c and warns:
        flag += " <- condition accepts a lossy conversion"
    print(f"{name:30s} {today:17s} {c!s:9s} {s!s:6s}  {res}{flag}")
print(_env.versions())
