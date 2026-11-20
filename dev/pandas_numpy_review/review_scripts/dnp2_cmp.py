"""
D-NP2: compare the outputs of `dnp2_grid.py` or `dnp2_dtypes.py` obtained
with the current code and with the prototype, in every environment found in
OUTDIR.

Usage: python dnp2_cmp.py grid OUTDIR [-v]
       python dnp2_cmp.py dtypes OUTDIR
"""
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

kind, out = sys.argv[1], Path(sys.argv[2])
verbose = "-v" in sys.argv
KEYS = ("fit", "pred", "resid", "resid2", "interval", "osa", "bt")


def load(name):
    data = pickle.load(open(out / name, "rb"))
    return data["versions"], data["results"]


def same(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return isinstance(x, str) and isinstance(y, str) and x == y
    if x is None or y is None:
        return x is y
    x, y = np.asarray(x, float), np.asarray(y, float)
    return x.shape == y.shape and np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)


def short(v):
    if isinstance(v, str):
        return v[:60]
    if v is None:
        return "-"
    v = np.asarray(v, float)
    return f"{v:.3f}" if v.ndim == 0 else "ok " + str(np.round(v, 3).tolist())[:70]


def is_number(v):
    return isinstance(v, float)


for env in ("p2", "p3", "p3pa"):
    if not (out / f"{kind}_today_{env}.pkl").exists():
        continue
    versions, a = load(f"{kind}_today_{env}.pkl")
    _, b = load(f"{kind}_proto_{env}.pkl")
    print(f"=== {env} {versions.split(', skforecast')[0]}]")
    if kind == "dtypes":
        n_same = 0
        for k in a:
            if same(a[k], b[k]):
                n_same += 1
            else:
                print(f"  {k[0]:22} {k[1]:6} {str(k[2]):5} | today: {short(a[k]):74} | prototype: {short(b[k])}")
        print(f"  same: {n_same} of {len(a)}")
        continue

    summary = defaultdict(Counter)
    detail = []
    for k in a:
        diffs = [key for key in KEYS if not same(a[k].get(key), b[k].get(key))]
        summary[k[2]]["same" if not diffs else "different"] += 1
        if diffs:
            detail.append((k, diffs))
    for cf, c in sorted(summary.items()):
        print(f"  categorical_features={cf:5}: prototype same as today in {c['same']:3} of {c['same'] + c['different']:3}")
    for name, r in (("today", a), ("prototype", b)):
        both = [k for k in r if is_number(r[k].get("osa")) and is_number(r[k].get("bt"))]
        differ = [k for k in both if abs(r[k]["osa"] - r[k]["bt"]) >= 1e-9]
        print(f"  {name}: one-step-ahead MAE != backtesting MAE (1 step, no refit) in {len(differ)} of {len(both)} cases")
    pairs = [k for k in a if k[2] == "None" and k[0] in ("lgbm", "xgb", "hgb") and k[4] == "frame"
             and is_number(a[k].get("osa"))]
    as_auto = [k for k in pairs if abs(a[k]["osa"] - a[(k[0], k[1], "auto", k[3], k[4])]["osa"]) < 1e-9]
    print(f"  today: model fitted with None (one-step-ahead MAE) equal to the one of 'auto' in {len(as_auto)} of {len(pairs)} cases")
    if env == "p3pa" or verbose:
        print("  One-step-ahead MAE | backtesting MAE, series as DataFrame, encoding 'ordinal':")
        print(f"    {'estimator':9} {'categorical_features':21} {'categories':10} | {'today':>33} | prototype")
        for k in a:
            est, enc, cf, cats, series_kind = k
            if enc != "ordinal" or series_kind != "frame" or cf == "auto":
                continue
            cells = []
            for r in (a[k], b[k]):
                if r["fit"] != "ok":
                    cells.append("fit " + r["fit"][:29])
                else:
                    cells.append(" | ".join(f"{r[m]:6.2f}" if is_number(r.get(m)) else str(r.get(m))[:14] for m in ("osa", "bt")))
            print(f"    {est:9} {cf:21} {cats:10} | {cells[0]:>33} | {cells[1]}")
    if verbose:
        for k, diffs in detail:
            print("  ", k, "|", "; ".join(f"{d}: {short(a[k].get(d))} -> {short(b[k].get(d))}" for d in diffs))
