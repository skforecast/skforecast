"""
Change 3 label probe: when can `check_predict_input` skip the selection
`last_window[names]` (names: `levels` or `series_names_in_`) and check the
missing values of the whole window? For window columns and names with many
label types (str in object, category, string and str indexes, int, float,
bool, mixed, None, NaN, pd.NA, NaT, timestamps with and without time zone,
tuples in a MultiIndex, numpy.str_, duplicated columns) and names that are the
columns in other orders, with Python-equal substitutes (bool for int, float
for int, numpy.str_ for str, another time zone, other missing values), with
duplicates, as a subset or as a list of booleans, it compares the outcome of
the selection (missing values found, or the exception) with the outcome of
each candidate condition. The window has a missing value in one column at a
time, and as many rows as columns (a list of booleans of that length selects
rows) or more. Only the names that pass the earlier checks of
`check_predict_input` (every name in the columns, with Python sets) are
counted. It does not import skforecast. Results in `MEDIDAS.md`, section 16.

Candidates:
- `implemented`: the condition in the package (a list, same length, names and
  columns of type str or int, same set).
- `implemented_dup_guard`: the same plus no duplicated names (the plan); it
  is not needed.
- `plan_set`: same length, unique columns and same set, without the type
  guard (differences: booleans and integers, None and pd.NA in pandas 3).
- `ordered_list`: the condition of change 2 (`columns.tolist() == names`),
  also without the type guard.

Usage: [NO_PYARROW=1] python probe_labels3.py
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

import itertools
import warnings
from collections import Counter

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PA = True
except ModuleNotFoundError:
    HAS_PA = False


def outcome(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = ("ok", bool(fn()))
        except Exception as e:  # noqa: BLE001
            res = ("err", type(e).__name__, str(e))
    return res, tuple((x.category.__name__, str(x.message)) for x in w)


def selection(df, names):
    return pd.isna(df[names].to_numpy()).any()


def whole_window(df, names):
    return pd.isna(df.to_numpy()).any()


def implemented(df, names, cols):
    return (
        isinstance(names, list)
        and len(names) == len(cols)
        and set(map(type, names)) <= {str, int}
        and set(map(type, cols)) <= {str, int}
        and set(names) == set(cols)
    )


def implemented_dup_guard(df, names, cols):
    return implemented(df, names, cols) and len(set(names)) == len(names)


def plan_set(df, names, cols):
    return len(names) == len(cols) and df.columns.is_unique and set(names) == set(cols)


def ordered_list(df, names, cols):
    return len(names) == len(cols) and df.columns.is_unique and cols == names


CANDIDATES = {
    "implemented": implemented,
    "implemented_dup_guard": implemented_dup_guard,
    "plan_set": plan_set,
    "ordered_list": ordered_list,
}

ts1 = pd.Timestamp("2020-01-01", tz="UTC")
ts2 = pd.Timestamp("2020-01-02", tz="UTC")
LABELS = {
    "str": ["a", "b", "c"],
    "int": [0, 1, 2],
    "int12": [1, 2, 3],
    "int01": [0, 1],
    "float": [0.0, 1.0, 2.5],
    "float_nan": [0.0, np.nan, 2.5],
    "bool": [True, False],
    "mixed": ["a", 1, 2.5],
    "mixed_bool": ["a", True, 2],
    "none": ["a", None, "c"],
    "pdna": ["a", pd.NA, "c"],
    "nat": ["a", pd.NaT, "c"],
    "nan_obj": ["a", np.nan, "c"],
    "ts_tz": [ts1, ts2],
    "ts_naive": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
    "tuples": [("a", 1), ("b", 2)],
    "npstr": [np.str_("a"), np.str_("b")],
    "dup": ["a", "a", "b"],
    "dup2": ["a", "b", "b"],
    "dup_int": [0, 0, 1],
    "dup_all": ["a", "a"],
    "dup_nan": [np.nan, np.nan, "a"],
}


def index_variants(labels):
    out = {"default": lambda: pd.Index(labels), "object": lambda: pd.Index(labels, dtype=object)}
    if all(isinstance(x, str) for x in labels):
        out["category"] = lambda: pd.CategoricalIndex(labels)
        out["string_python"] = lambda: pd.Index(labels, dtype="string[python]")
        out["str_python"] = lambda: pd.Index(labels, dtype=pd.StringDtype("python", na_value=np.nan))
        if HAS_PA:
            out["string_pyarrow"] = lambda: pd.Index(labels, dtype="string[pyarrow]")
            out["str_pyarrow"] = lambda: pd.Index(labels, dtype=pd.StringDtype("pyarrow", na_value=np.nan))
    if all(isinstance(x, tuple) for x in labels):
        out["multi"] = lambda: pd.MultiIndex.from_tuples(labels)
    if all(isinstance(x, int) and not isinstance(x, bool) for x in labels):
        if labels == list(range(labels[0], labels[-1] + 1)):
            out["range"] = lambda: pd.RangeIndex(labels[0], labels[-1] + 1)
        out["uint"] = lambda: pd.Index(labels, dtype="uint64")
        out["Int64"] = lambda: pd.Index(labels, dtype="Int64")
    if any(isinstance(x, float) for x in labels) and all(isinstance(x, (float, int)) for x in labels):
        out["Float64"] = lambda: pd.Index(labels, dtype="Float64")
    if all(isinstance(x, pd.Timestamp) for x in labels):
        out["datetime"] = lambda: pd.DatetimeIndex(labels)
    return out


def substitutes(x):
    """Python-equal labels, or other missing values, for a label."""
    subs = [x]
    if isinstance(x, (bool, np.bool_)):
        subs += [int(x), float(x), np.bool_(x)]
    elif isinstance(x, (int, np.integer)):
        subs += [np.int64(x), float(x), np.float64(x)]
        if x in (0, 1):
            subs += [bool(x), np.bool_(bool(x))]
    elif isinstance(x, (float, np.floating)):
        if x != x:
            subs += [float("nan"), np.float64("nan"), None, pd.NA, pd.NaT]
        else:
            subs += [np.float64(x)] + ([int(x)] if float(x).is_integer() else [])
    elif isinstance(x, str):
        subs += [np.str_(x)]
    elif x is None:
        subs += [np.nan, pd.NA, pd.NaT]
    elif x is pd.NA:
        subs += [None, np.nan]
    elif x is pd.NaT:
        subs += [None, np.nan, np.datetime64("NaT", "ns")]
    elif isinstance(x, pd.Timestamp):
        if x.tz is not None:
            subs += [x.tz_convert("Europe/Madrid"), x.tz_convert("Asia/Tokyo"),
                     x.to_pydatetime(), x.tz_localize(None)]
        else:
            subs += [x.to_pydatetime(), np.datetime64(x), x.tz_localize("UTC")]
    elif isinstance(x, tuple):
        subs += [tuple(np.str_(y) if isinstance(y, str) else float(y) if isinstance(y, int) else y
                       for y in x)]
    return subs


def name_lists(cols):
    n = len(cols)
    for order in list(itertools.permutations(range(n)))[:6]:
        base = [cols[i] for i in order]
        yield base
        for pos in range(n):
            for s in substitutes(base[pos])[1:]:
                yield base[:pos] + [s] + base[pos + 1:]
        for k in range(1, 6):
            alt = [substitutes(x)[k] for x in base if len(substitutes(x)) > k]
            if len(alt) == n:
                yield alt
    unique = list(dict.fromkeys(cols))
    if len(unique) < n:
        for extra in unique:
            yield unique + [extra] * (n - len(unique))
    if n >= 2:
        yield cols[:-1]
        yield cols[:-1] + [cols[0]]
        yield cols + [cols[0]]
    yield []
    yield [True] * n
    yield [False] * n
    yield [True] + [False] * (n - 1)


def main():
    n_cases = 0
    taken = Counter()
    diffs = {k: [] for k in CANDIDATES}
    for lname, labels in LABELS.items():
        for iname, make_index in index_variants(labels).items():
            try:
                columns = make_index()
            except Exception:  # noqa: BLE001
                continue
            n_cols = len(columns)
            for n_rows in (n_cols, 4):
                for nan_col in range(-1, n_cols):
                    values = np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols)
                    if nan_col >= 0:
                        values[-1, nan_col] = np.nan
                    df = pd.DataFrame(values, columns=columns)
                    cols = df.columns.to_list()
                    for names in name_lists(cols):
                        # check_predict_input raises earlier if a name is not
                        # in the columns (Python sets), in both paths.
                        try:
                            if set(names) - set(cols):
                                continue
                        except Exception:  # noqa: BLE001
                            continue
                        n_cases += 1
                        ref = outcome(lambda: selection(df, names))
                        for cname, cond in CANDIDATES.items():
                            try:
                                take = cond(df, names, df.columns.to_list())
                            except Exception as e:  # noqa: BLE001
                                # The condition itself raises (`list ==`
                                # with pd.NA): also a difference.
                                res = (("err", type(e).__name__, str(e)), ())
                            else:
                                if take:
                                    taken[cname] += 1
                                    res = outcome(lambda: whole_window(df, names))
                                else:
                                    res = ref
                            if res != ref:
                                diffs[cname].append((lname, iname, n_rows, nan_col, names, ref, res))
    print(f"pandas {pd.__version__}, pyarrow {HAS_PA}, {n_cases} cases")
    for cname in CANDIDATES:
        by_labels = Counter((d[0], d[1]) for d in diffs[cname])
        print(f"  {cname:22s} whole window {taken[cname]:6d}  differences {len(diffs[cname]):4d}"
              f"  {dict(by_labels) if by_labels else ''}")
    for cname in CANDIDATES:
        for d in diffs[cname][:3]:
            print(f"  [{cname}] {d[0]}/{d[1]} rows={d[2]} nan_col={d[3]} names={d[4]!r}\n"
                  f"      selection: {d[5]}\n      whole window: {d[6]}")


if __name__ == "__main__":
    main()
