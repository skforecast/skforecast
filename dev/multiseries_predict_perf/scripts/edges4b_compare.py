"""Compare two edges4b.py pickles: outputs, errors and warnings."""
import collections
import pickle
import sys

import pandas as pd


def frame_diff(a, b):
    if type(a) is not type(b):
        return "type"
    if isinstance(a, tuple):
        if len(a) != len(b):
            return "len"
        for x, y in zip(a, b):
            r = frame_diff(x, y)
            if r:
                return r
        return None
    if isinstance(a, (pd.DataFrame, pd.Series)):
        if not (a.index.identical(b.index) and type(a.index) is type(b.index)):
            return "index"
        if getattr(a.index, "freq", None) != getattr(b.index, "freq", None):
            return "freq"
        if isinstance(a, pd.DataFrame):
            if not a.columns.identical(b.columns):
                return "columns"
            if [str(x) for x in a.dtypes] != [str(x) for x in b.dtypes]:
                return "dtypes"
            fn = pd.testing.assert_frame_equal
        else:
            if str(a.dtype) != str(b.dtype) or a.name != b.name:
                return "dtype/name"
            fn = pd.testing.assert_series_equal
        try:
            fn(a, b, check_exact=True)
        except AssertionError:
            return "values"
        return None
    return None if a == b else "value"


base = pickle.load(open(sys.argv[1], "rb"))
new = pickle.load(open(sys.argv[2], "rb"))
print("pandas", base["pandas"], new["pandas"], "cases", len(base["res"]), len(new["res"]))
diffs = collections.Counter()
examples = {}
for key, (out_b, err_b, w_b) in base["res"].items():
    out_n, err_n, w_n = new["res"][key]
    what = None
    if err_b != err_n:
        what = "error"
    elif w_b != w_n:
        what = "warnings"
    elif err_b is None:
        what = frame_diff(out_b, out_n)
    if what:
        diffs[what] += 1
        examples.setdefault(what, (key, err_b, err_n, w_b, w_n))
n_err = sum(1 for v in base["res"].values() if v[1])
n_warn = sum(1 for v in base["res"].values() if v[2])
print(f"base: {n_err} cases with error, {n_warn} with warnings")
print("differences:", sum(diffs.values()), dict(diffs))
for what, ex in examples.items():
    print("  ", what, str(ex)[:400])
