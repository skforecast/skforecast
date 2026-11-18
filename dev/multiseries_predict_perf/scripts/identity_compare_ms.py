"""
Exact comparison of two result files of identity_ms.py or identity_trims.py:
values bit by bit, dtypes (with the storage of text), index and columns
attributes, warnings (category, text and file they point to) and errors.

Usage: python identity_compare_ms.py <base.pkl> <new.pkl> [--summary]
Run it with the same environment that created the files.
"""
import collections
import pickle
import sys

import numpy as np
import pandas as pd


def dtype_id(dtype):
    return (repr(dtype), getattr(dtype, "storage", None), str(getattr(dtype, "na_value", "")))


def index_attrs(ix):
    attrs = [type(ix).__name__, dtype_id(ix.dtype), ix.names]
    if isinstance(ix, pd.MultiIndex):
        attrs.append([index_attrs(level) for level in ix.levels])
    else:
        attrs += [
            str(getattr(ix, "freq", None)), str(getattr(ix, "tz", None)),
            getattr(ix, "step", None), getattr(ix, "start", None),
        ]
    return attrs


def same(a, b):
    if type(a) is not type(b):
        return f"type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return f"length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            msg = same(x, y)
            if msg:
                return f"[{i}] {msg}"
        return None
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"dict keys {list(a)[:6]} vs {list(b)[:6]}"
        for k in a:
            msg = same(a[k], b[k])
            if msg:
                return f"[{k}] {msg}"
        return None
    if isinstance(a, (pd.DataFrame, pd.Series)):
        if index_attrs(a.index) != index_attrs(b.index):
            return f"index attrs {index_attrs(a.index)} vs {index_attrs(b.index)}"
        try:
            if isinstance(a, pd.DataFrame):
                if index_attrs(a.columns) != index_attrs(b.columns):
                    return f"columns attrs {index_attrs(a.columns)} vs {index_attrs(b.columns)}"
                if [dtype_id(d) for d in a.dtypes] != [dtype_id(d) for d in b.dtypes]:
                    return f"dtypes {list(a.dtypes)} vs {list(b.dtypes)}"
                pd.testing.assert_frame_equal(
                    a, b, check_exact=True, check_freq=True, check_categorical=True
                )
            else:
                if dtype_id(a.dtype) != dtype_id(b.dtype) or a.name != b.name:
                    return f"dtype or name {a.dtype} {a.name} vs {b.dtype} {b.name}"
                pd.testing.assert_series_equal(a, b, check_exact=True, check_freq=True)
        except AssertionError as e:
            return str(e).replace("\n", " ")[:200]
        return None
    if isinstance(a, pd.Index):
        if index_attrs(a) != index_attrs(b):
            return f"index attrs {index_attrs(a)} vs {index_attrs(b)}"
        return None if a.equals(b) else "index values"
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"ndarray {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        if a.dtype == object:
            return None if all(same(x, y) is None for x, y in zip(a.ravel(), b.ravel())) else "ndarray values"
        return None if np.array_equal(a, b, equal_nan=a.dtype.kind in "fc") else "ndarray values"
    if isinstance(a, float) and a != a:
        return None if b != b else f"value {a!r} vs {b!r}"
    try:
        equal = bool(a == b)
    except Exception:  # noqa: BLE001
        equal = repr(a) == repr(b)
    return None if equal else f"value {a!r} vs {b!r}"[:200]


base = pickle.load(open(sys.argv[1], "rb"))
new = pickle.load(open(sys.argv[2], "rb"))
if list(base) != list(new):
    only_base = [k for k in base if k not in new]
    only_new = [k for k in new if k not in base]
    print(f"DIFFERENT CASE LISTS: {len(only_base)} only in base, {len(only_new)} only in new",
          only_base[:3], only_new[:3])
diffs = []
stats = collections.Counter()
errors = collections.Counter()
warns = collections.Counter()
for name in base:
    if name not in new:
        continue
    a, b = base[name], new[name]
    stats["|".join(name.split("|")[:2])] += 1
    if a["err"] is not None:
        errors[(a["err"][0], a["err"][1][:70])] += 1
    for w in a["warn"]:
        warns[(w[0], w[1][:60])] += 1
    msg = None
    if a["err"] != b["err"]:
        msg = f"error {a['err']} vs {b['err']}"
    elif a["warn"] != b["warn"]:
        msg = f"warnings {a['warn']} vs {b['warn']}"
    else:
        msg = same(a["out"], b["out"])
    if msg:
        diffs.append((name, msg))
n_err = sum(errors.values())
n_warn = sum(bool(base[k]["warn"]) for k in base)
print(f"{len(base)} cases, {len(base) - len(diffs)} identical, {len(diffs)} different | "
      f"{n_err} with error ({len(errors)} kinds), {n_warn} with warnings ({len(warns)} kinds) | "
      + ", ".join(f"{k}: {v}" for k, v in stats.items()))
for name, msg in diffs[:40]:
    print("DIFF", name, "->", msg[:300])
if "--summary" in sys.argv:
    print("\nErrors:")
    for (kind, text), count in errors.most_common():
        print(f"{count:6d}  {kind}: {text}")
    print("\nWarnings:")
    for (kind, text), count in warns.most_common():
        print(f"{count:6d}  {kind}: {text}")
