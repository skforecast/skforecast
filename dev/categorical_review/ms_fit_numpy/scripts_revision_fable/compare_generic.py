"""Compare two pickles of {key: {field: result}}. Usage: compare_generic.py a.pkl b.pkl [-v]"""
import sys
import pickle
import collections
import numbers
import numpy as np
import pandas as pd


def is_err(x):
    return isinstance(x, tuple) and len(x) == 3 and x[0] == "ERR"


def flat(x, nums, strs):
    if x is None:
        strs.append("None")
    elif isinstance(x, pd.DataFrame):
        for i, c in enumerate(x.columns):
            strs.append(str(c))
            flat(x.iloc[:, i].tolist(), nums, strs)
    elif isinstance(x, pd.Series):
        flat(x.tolist(), nums, strs)
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            strs.append(str(k))
            flat(x[k], nums, strs)
    elif isinstance(x, np.ndarray):
        if x.dtype == object:
            flat(x.ravel().tolist(), nums, strs)
        else:
            nums.extend(np.asarray(x, dtype=float).ravel().tolist())
    elif isinstance(x, (list, tuple)):
        for v in x:
            flat(v, nums, strs)
    elif isinstance(x, (bool, np.bool_)):
        strs.append(str(bool(x)))
    elif isinstance(x, numbers.Number):
        nums.append(float(x))
    else:
        strs.append(str(x))


def cmp(a, b):
    if is_err(a) and is_err(b):
        return "same(err)" if a[1:] == b[1:] else "err->err'"
    if is_err(a):
        return "err->ok"
    if is_err(b):
        return "ok->err"
    na, sa, nb, sb = [], [], [], []
    flat(a, na, sa)
    flat(b, nb, sb)
    if sa != sb:
        return "DIFF(labels)"
    na, nb = np.array(na), np.array(nb)
    if na.shape != nb.shape:
        return "DIFF(shape)"
    if np.array_equal(na, nb, equal_nan=True):
        return "same"
    if np.allclose(na, nb, rtol=1e-9, atol=1e-12, equal_nan=True):
        return "close"
    return f"DIFF"


A = pickle.load(open(sys.argv[1], "rb"))
B = pickle.load(open(sys.argv[2], "rb"))
verbose = "-v" in sys.argv
print("keys", len(A), len(B), "common", len(set(A) & set(B)))
count = collections.Counter()
details = collections.defaultdict(list)
for key in sorted(set(A) & set(B), key=str):
    for fld in sorted(set(A[key]) | set(B[key])):
        if fld not in A[key] or fld not in B[key]:
            st = "missing-field"
        else:
            st = cmp(A[key][fld], B[key][fld])
        count[(fld, st)] += 1
        if st not in ("same", "same(err)"):
            details[(fld, st)].append(key)
for k, n in sorted(count.items()):
    print(f"{n:5d} {k}")
for k, keys in sorted(details.items()):
    print("\n", k, len(keys))
    for key in keys[: (400 if verbose else 12)]:
        extra = ""
        if verbose:
            a, b = A[key].get(k[0]), B[key].get(k[0])
            extra = f"\n        A: {str(a)[:230]}\n        B: {str(b)[:230]}"
        print("    ", key, extra)
