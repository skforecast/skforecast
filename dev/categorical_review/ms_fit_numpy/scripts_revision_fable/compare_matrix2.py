"""Compare the matrix pickles of base and branch. Usage: compare_matrix.py <p2|p3> [detail-filter]"""
import sys
import glob
import pickle
import collections
import numpy as np
import pandas as pd

pdv = sys.argv[1]
TA, TB = sys.argv[2], sys.argv[3]
detail = sys.argv[4] if len(sys.argv) > 4 else None
FIELDS = ["init", "fit", "pred", "resid", "boot", "conf", "imp", "bt", "osa"]


def is_err(x):
    return isinstance(x, tuple) and len(x) == 3 and x[0] == "ERR"


def num(x):
    """Numeric content of a result, as a flat float array."""
    if isinstance(x, dict):
        out = []
        for k in sorted(x, key=str):
            v = x[k]
            if v is None:
                continue
            out.append(np.asarray(v, dtype=float).ravel())
        return np.concatenate(out) if out else np.array([])
    a = np.asarray(x)
    if a.dtype == object:
        vals = []
        for v in a.ravel():
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
        return np.array(vals)
    return a.astype(float).ravel()


def cmp(a, b):
    if a is None and b is None:
        return "same"
    if is_err(a) and is_err(b):
        return "same" if a[1:] == b[1:] else "err->err'"
    if is_err(a):
        return "err->ok"
    if is_err(b):
        return "ok->err"
    if a is None or b is None:
        return "none-mismatch"
    na, nb = num(a), num(b)
    if na.shape != nb.shape:
        return "shape"
    if np.array_equal(na, nb, equal_nan=True):
        return "same"
    if np.allclose(na, nb, rtol=1e-9, atol=1e-12, equal_nan=True):
        return "close"
    return "DIFF"


base, branch = {}, {}
for f in glob.glob(f"out/m_{TA}_{pdv}_*.pkl"):
    base.update(pickle.load(open(f, "rb")))
for f in glob.glob(f"out/m_{TB}_{pdv}_*.pkl"):
    branch.update(pickle.load(open(f, "rb")))
print("configs base", len(base), "branch", len(branch), "common", len(set(base) & set(branch)))

summary = collections.Counter()
groups = collections.defaultdict(list)
osa_bt = {"base": collections.Counter(), "branch": collections.Counter()}
osa_bt_keys = {"base": [], "branch": []}
warn_changes = collections.Counter()
for key in sorted(set(base) & set(branch), key=str):
    a, b = base[key], branch[key]
    est, dtype, cf, enc, as_dict = key
    status = []
    for fld in FIELDS:
        if fld not in a and fld not in b:
            continue
        c = cmp(a.get(fld, ("ERR", "missing", "")), b.get(fld, ("ERR", "missing", "")))
        if fld in ("init", "fit") and c == "same" and is_err(a.get(fld)):
            c = "same(err)"
        if c not in ("same", "same(err)"):
            status.append(f"{fld}:{c}")
    summary[(cf, tuple(status))] += 1
    groups[(cf, tuple(status))].append(key)
    # within-tree equivalence one-step-ahead vs backtesting
    for name, r in (("base", a), ("branch", b)):
        bt, osa = r.get("bt"), r.get("osa")
        if bt is None or osa is None:
            continue
        if is_err(bt) and is_err(osa):
            st = "both err"
        elif is_err(bt):
            st = "bt err, osa ok"
        elif is_err(osa):
            st = "bt ok, osa err"
        else:
            d = max(
                abs(bt["weighted_average"] - osa["mean_absolute_error__weighted_average"]),
                abs(bt["average"] - osa["mean_absolute_error__average"]),
                abs(bt["pooling"] - osa["mean_absolute_error__pooling"]),
            )
            st = "equal" if d < 1e-9 else "DIFFERENT"
        osa_bt[name][(cf, st)] += 1
        if st not in ("equal", "both err"):
            osa_bt_keys[name].append((key, st, bt if is_err(bt) else None, osa if is_err(osa) else None))
    wa = set(x for x in a.get("warn_fit_pred", []))
    wb = set(x for x in b.get("warn_fit_pred", []))
    for x in wb - wa:
        warn_changes[("+", cf, x[0], x[1][:70])] += 1
    for x in wa - wb:
        warn_changes[("-", cf, x[0], x[1][:70])] += 1

print("\n== base -> branch, per categorical_features mode and changed fields")
for (cf, st), n in sorted(summary.items(), key=str):
    print(f"{n:5d}  cf={cf:10}  {'identical' if not st else ' '.join(st)}")
    if st and (detail is None or detail in str((cf, st))):
        c = collections.Counter((k[0], k[1], k[3]) for k in groups[(cf, st)])
        by_est = collections.defaultdict(set)
        for (e, d, enc), _ in c.items():
            by_est[(d, str(enc))].add(e)
        for (d, enc), es in sorted(by_est.items()):
            print(f"           dtype={d:18} enc={enc:17} est={sorted(es)}")

print("\n== one-step-ahead vs backtesting(steps=1, refit=False), within each tree")
for name in ("base", "branch"):
    print(name, dict(sorted(osa_bt[name].items(), key=str)))
    c = collections.Counter((k[0][0], k[0][1], k[0][2], k[1]) for k in osa_bt_keys[name])
    for k, n in sorted(c.items()):
        print("    ", n, k)

print("\n== warnings of fit/predict added (+) or removed (-) by the branch")
for k, n in sorted(warn_changes.items(), key=str):
    print(f"{n:5d} {k}")
