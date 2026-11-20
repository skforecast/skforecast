"""
P2, item 10: view or copy. `exog.to_numpy()` today vs
`exog.to_numpy(dtype=float)` vs `exog.to_numpy(dtype=float, na_value=np.nan)`:
dtype, whether the result shares memory with the user's DataFrame, and whether
it is writeable. Also what a plain `to_numpy()` gives for each frame (when is
it really `object`?).
No timing here: only whether a new array is allocated.
"""
import _env
import numpy as np
import pandas as pd

n = 1000
rng = np.random.default_rng(0)
f2 = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
two_blocks = f2.copy()
two_blocks["d"] = rng.normal(size=n)           # second float64 block
frames = {
    "float64, one block": f2,
    "float64, two blocks": two_blocks,
    "float64 Series.to_frame()": f2["a"].to_frame(),
    "float32, one block": f2.astype("float32"),
    "int64, one block": pd.DataFrame(rng.integers(0, 9, size=(n, 2)), columns=list("ab")),
    "bool, one block": pd.DataFrame(rng.integers(0, 2, size=(n, 2)).astype(bool), columns=list("ab")),
    "float64 + int64": f2.assign(i=np.arange(n)),
    "float64 + bool": f2.assign(b=np.arange(n) % 2 == 0),
    "int64 + bool": pd.DataFrame({"i": np.arange(n), "b": np.arange(n) % 2 == 0}),
    "Int64 only": pd.DataFrame({"i": pd.array(np.arange(n), dtype="Int64")}),
    "Int64 only, with NA": pd.DataFrame({"i": pd.array([None] + list(range(n - 1)), dtype="Int64")}),
    "Int64 + Float64": pd.DataFrame({"i": pd.array(np.arange(n), dtype="Int64"),
                                     "f": pd.array(rng.normal(size=n), dtype="Float64")}),
    "Float64 only, with NA": pd.DataFrame({"f": pd.array([None] + list(rng.normal(size=n - 1)), dtype="Float64")}),
    "float64 + Int64": f2.assign(i=pd.array(np.arange(n), dtype="Int64")),
    "boolean only": pd.DataFrame({"b": pd.array(np.arange(n) % 2 == 0, dtype="boolean")}),
}
try:
    frames["double[pyarrow] only"] = pd.DataFrame({"f": pd.array(rng.normal(size=n), dtype="double[pyarrow]")})
    frames["float64 + double[pyarrow]"] = f2.assign(f=pd.array(rng.normal(size=n), dtype="double[pyarrow]"))
except Exception as e:  # noqa: BLE001
    print("pyarrow frames: SKIPPED", type(e).__name__)


def describe(df, **kw):
    out = df.to_numpy(**kw)
    shares = any(np.shares_memory(out, df[c].array._ndarray if hasattr(df[c].array, "_ndarray") else
                                  getattr(df[c].array, "_data", np.empty(0)))
                 for c in df.columns if not str(df[c].dtype).endswith("[pyarrow]"))
    return f"{str(out.dtype):8s} {'VIEW' if shares else 'copy'} {'rw' if out.flags.writeable else 'ro'}"


print(f"{'frame':28s} | {'to_numpy()':17s} | {'to_numpy(dtype=float)':21s} | to_numpy(dtype=float, na_value=nan)")
for name, df in frames.items():
    print(f"{name:28s} | {describe(df):17s} | {describe(df, dtype=float):21s} | "
          f"{describe(df, dtype=float, na_value=np.nan)}")
print(_env.versions())
