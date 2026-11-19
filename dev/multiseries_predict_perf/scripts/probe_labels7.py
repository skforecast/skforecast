"""
Change 7 label probe: which cheap construction of the exog columns is
`identical` to `empty_exog.columns` (the DataFrame of empty Series with the
index of the predictions that `_create_predict_inputs` built before change 7),
and is `X.reindex(index=pi, columns=cols)` identical to
`X.reindex_like(empty_exog)`, errors included. `F_implemented` is the one in
the package; `A_Index_keys` is not identical in pandas 3 (consecutive integer
names become a `RangeIndex` in a DataFrame built from a dict). It does not
import skforecast. Results in `MEDIDAS.md`, section 12.

Usage: [NO_PYARROW=1] python probe_labels7.py
"""
import os
import sys

if os.environ.get("NO_PYARROW") == "1":
    class _BlockPyarrow:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
            return None
    sys.meta_path.insert(0, _BlockPyarrow)

import timeit
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
try:
    import pyarrow  # noqa: F401
    HAS_PA = True
except ModuleNotFoundError:
    HAS_PA = False

ts = pd.Timestamp
LABELS = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "int_range0": [0, 1, 2],
    "int_step": [0, 2, 4],
    "int_desc": [3, 2, 1],
    "int_one": [7],
    "int_two": [4, 5],
    "int_gap": [1, 2, 4],
    "np_int": [np.int64(1), np.int64(2), np.int64(3)],
    "uint": [np.uint8(0), np.uint8(1), np.uint8(2)],
    "float": [1.0, 2.0, 3.0],
    "float_nan": [1.0, np.nan, 3.0],
    "bool": [True, False],
    "bool_int": [True, 2, 3],
    "obj_int_str": [1, "b", "c"],
    "obj_bool_str": [True, "b", "c"],
    "obj_none": ["a", None, "c"],
    "np_str": [np.str_("a"), np.str_("b"), np.str_("c")],
    "tz_utc": [ts("2020-01-01", tz="UTC"), ts("2020-01-02", tz="UTC")],
    "naive_ts": [ts("2020-01-01"), ts("2020-01-02")],
    "tuples": [("x", 1), ("x", 2), ("y", 1)],
    "mixed_tuple": [("x", 1), "b"],
}
KINDS = {
    "plain": lambda labels: pd.Index(labels),
    "object": lambda labels: pd.Index(labels, dtype=object),
    "category": lambda labels: pd.CategoricalIndex(labels),
    "string_python": lambda labels: pd.Index(labels, dtype="string[python]"),
    "range": lambda labels: pd.RangeIndex(len(labels)),
}
if HAS_PA:
    KINDS["string_pyarrow"] = lambda labels: pd.Index(labels, dtype="string[pyarrow]")
    KINDS["str_pyarrow"] = lambda labels: pd.Index(
        labels, dtype=pd.StringDtype("pyarrow", na_value=np.nan))

PRED_INDEXES = {
    "date_D": pd.date_range("2021-02-10", periods=3, freq="D"),
    "date_tz_h": pd.date_range("2021-02-10", periods=3, freq="h", tz="Europe/Madrid"),
    "range": pd.RangeIndex(40, 43),
}
EMPTY = np.empty(0)


def today(dtypes, pi):
    return pd.DataFrame(
        data={col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()},
        index=pi,
    )


CANDIDATES = {
    "A_Index_keys": lambda d: pd.Index(list(d)),
    "B_df_lists": lambda d: pd.DataFrame({c: [] for c in d}).columns,
    "C_df_fromkeys_empty": lambda d: pd.DataFrame(dict.fromkeys(d, EMPTY)).columns,
    "D_df_items_empty": lambda d: pd.DataFrame({c: EMPTY for c, _ in d.items()}).columns,
    "E_df_items_empty_nocopy": lambda d: pd.DataFrame(
        {c: EMPTY for c, _ in d.items()}, copy=False).columns,
    "F_implemented": lambda d: pd.DataFrame(
        data={col: [] for col, _ in d.items()}, copy=False).columns,
}


def attrs(ix):
    return (type(ix).__name__, str(ix.dtype), ix.names)


def frames_identical(a, b):
    if type(a) is not type(b):
        return "type"
    if not a.index.identical(b.index):
        return "index"
    if getattr(a.index, "freq", None) != getattr(b.index, "freq", None):
        return "freq"
    if not a.columns.identical(b.columns) or type(a.columns) is not type(b.columns):
        return "columns"
    if (
        not a.dtypes.equals(b.dtypes)
        or [str(x) for x in a.dtypes] != [str(x) for x in b.dtypes]
    ):
        return "dtypes"
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_freq=True,
                                      check_index_type=True, check_column_type=True,
                                      check_flags=True)
    except AssertionError as e:
        return f"values {e}"[:200]
    return None


diffs = {c: [] for c in CANDIDATES}
n_axes = 0
n_reidx = 0
idx_is = []
for lname, labels in LABELS.items():
    for kname, kind in KINDS.items():
        try:
            cols = kind(labels)
        except Exception:
            continue
        rng = np.random.default_rng(0)
        train = pd.DataFrame(rng.normal(size=(5, len(cols))))
        if len(cols) > 1:
            train[1] = pd.Categorical(np.arange(5) % 2)
        train.columns = cols
        dtypes = train.dtypes.to_dict()
        for pname, pi in PRED_INDEXES.items():
            ref = today(dtypes, pi)
            idx_is.append(ref.index is pi)
            for cname, build in CANDIDATES.items():
                n_axes += cname == "A_Index_keys"
                try:
                    c = build(dtypes)
                except Exception as e:
                    diffs[cname].append((lname, kname, pname, f"raise {e!r}"))
                    continue
                if not (c.identical(ref.columns) and attrs(c) == attrs(ref.columns)):
                    diffs[cname].append((lname, kname, pname, "cols",
                                         attrs(c), attrs(ref.columns)))
                    continue
                # Reindex user frames whose columns are of every kind, with
                # the prediction index, a named one, one without freq, one longer.
                for ukname, ukind in KINDS.items():
                    try:
                        ucols = ukind(labels)
                    except Exception:
                        continue
                    for uix in (pi, pi.rename("date"), pd.Index(list(pi)),
                                pi.append(pi[-1:] + (pi[-1] - pi[-2]))):
                        user = pd.DataFrame(rng.normal(size=(len(uix), len(ucols))),
                                            index=uix)
                        user.columns = ucols
                        for uv in (user, user.iloc[:, ::-1], user.iloc[:, :1]):
                            try:
                                exp = uv.reindex_like(ref)
                                exp_err = None
                            except Exception as e:
                                exp, exp_err = None, f"{type(e).__name__}: {e}"
                            try:
                                got = uv.reindex(index=pi, columns=c)
                                got_err = None
                            except Exception as e:
                                got, got_err = None, f"{type(e).__name__}: {e}"
                            n_reidx += cname == "D_df_items_empty"
                            if exp_err or got_err:
                                if exp_err != got_err:
                                    diffs[cname].append((lname, kname, pname, ukname,
                                                         "err", exp_err, got_err))
                                continue
                            r = frames_identical(exp, got)
                            if r:
                                diffs[cname].append((lname, kname, pname, ukname, r))
                            # identical shortcut predicate
                            if uv.columns.identical(ref.columns) != uv.columns.identical(c):
                                diffs[cname].append((lname, kname, pname, ukname, "pred"))

print(f"pandas {pd.__version__} pyarrow={HAS_PA} infer_string="
      f"{pd.get_option('future.infer_string')}")
print(f"axes checked {n_axes}, reindex cases per candidate {n_reidx}, "
      f"empty_exog.index is prediction_index: {all(idx_is)} ({len(idx_is)})")
for cname, d in diffs.items():
    print(f"{cname}: {len(d)} differences")
    seen = set()
    for x in d:
        key = x[:2] + x[4:5]
        if key in seen:
            continue
        seen.add(key)
        if len(seen) <= 6:
            print("   ", x)

# Timings (us per call), three float columns and three mixed (float, int, category).
pi = pd.date_range("2021-02-10", periods=1, freq="D")
float3 = {c: np.dtype(float) for c in ["exog_0", "exog_1", "exog_2"]}
mixed3 = {"exog_0": np.dtype(float), "exog_1": np.dtype("int64"),
          "exog_2": pd.CategoricalDtype([0, 1, 2, 3])}
print("timings (min of 7 x 2000, us):")
for label, d in (("float3", float3), ("mixed3", mixed3)):
    t = min(timeit.repeat(lambda: today(d, pi), number=2000, repeat=7)) / 2000 * 1e6
    row = [f"today={t:.1f}"]
    for cname, build in CANDIDATES.items():
        t = min(timeit.repeat(lambda: build(d), number=2000, repeat=7)) / 2000 * 1e6
        row.append(f"{cname[:1]}={t:.1f}")
    print(f"  {label}: " + "  ".join(row))
