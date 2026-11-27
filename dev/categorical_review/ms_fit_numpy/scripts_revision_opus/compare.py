"""Compare two battery pickles. Usage: python compare.py base.pkl branch.pkl"""
import sys, pickle, collections
import numpy as np

a = pickle.load(open(sys.argv[1], "rb"))
b = pickle.load(open(sys.argv[2], "rb"))

def num(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.ndim == 2:
        return x[:, 1:]
    return x

def eq(x, y):
    x, y = num(x), num(y)
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return x.shape == y.shape and np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
    return x == y

def close(x, y):
    x, y = num(x), num(y)
    try:
        return x.shape == y.shape and np.allclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-9)
    except Exception:
        return False

summary = collections.Counter()
diffs = collections.defaultdict(list)
for k in a:
    ra, rb = a[k], b[k]
    fa, fb = ra["fit"], rb["fit"]
    cf = k[4]
    if fa[0] != fb[0]:
        cat = f"fit status {fa[0]}->{fb[0]}"
        diffs[cat].append((k, fa[1], fb[1]))
        summary[(cf, cat)] += 1
        continue
    if fa[0] == "err":
        same_msg = fa[1].split(":")[0] == fb[1].split(":")[0]
        summary[(cf, "both fit err" + ("" if same_msg else " (diff type)"))] += 1
        if not same_msg:
            diffs["both err diff type"].append((k, fa[1], fb[1]))
        continue
    if fa[2] != fb[2]:
        diffs["fit warnings differ"].append((k, fa[2], fb[2]))
    tags = []
    for part in ["pred", "pi"]:
        pa, pb = ra[part], rb[part]
        if pa[0] != pb[0]:
            tags.append(f"{part} status {pa[0]}->{pb[0]}")
            diffs[f"{part} status"].append((k, pa[1] if pa[0]=="err" else "ok", pb[1] if pb[0]=="err" else "ok"))
        elif pa[0] == "ok":
            if eq(pa[1], pb[1]):
                tags.append(f"{part} identical")
            elif close(pa[1], pb[1]):
                tags.append(f"{part} close")
            else:
                tags.append(f"{part} DIFFERENT")
                diffs[f"{part} different"].append((k, float(np.nanmax(np.abs(num(pa[1]).astype(float)-num(pb[1]).astype(float))))))
        else:
            tags.append(f"{part} both err")
    resa, resb = ra["res"], rb["res"]
    if resa is None or resb is None:
        tags.append("res none" if resa is None and resb is None else "res one none")
    else:
        same = resa.keys() == resb.keys() and all(eq(resa[x], resb[x]) for x in resa)
        tags.append("res identical" if same else "res DIFFERENT")
    if "res2" in ra:
        sa, sb = ra["res2"], rb["res2"]
        if sa[0] != sb[0]:
            tags.append(f"setres {sa[0]}->{sb[0]}")
            diffs["setres status"].append((k, sa[1] if sa[0]=="err" else "ok", sb[1] if sb[0]=="err" else "ok"))
        elif sa[0] == "ok":
            same = sa[1].keys() == sb[1].keys() and all(eq(sa[1][x], sb[1][x]) for x in sa[1])
            tags.append("setres identical" if same else "setres DIFFERENT")
            # consistency within branch: set_in_sample_residuals == fit residuals
            if rb["res"] is not None:
                same_b = rb["res"].keys() == sb[1].keys() and all(eq(rb["res"][x], sb[1][x]) for x in sb[1])
                if not same_b:
                    tags.append("BRANCH setres!=fit res")
            if ra["res"] is not None:
                same_a = ra["res"].keys() == sa[1].keys() and all(eq(ra["res"][x], sa[1][x]) for x in sa[1])
                if not same_a:
                    tags.append("BASE setres!=fit res")
        else:
            tags.append("setres both err")
    if ra["names"] != rb["names"]:
        tags.append("names differ")
    summary[(cf, " | ".join(tags))] += 1

for (cf, t), n in sorted(summary.items(), key=lambda x: (str(x[0][0]), x[0][1])):
    print(f"{cf:11s} {n:5d}  {t}")
print()
for cat, items in diffs.items():
    print("==", cat, len(items))
    for it in items[:400]:
        print("  ", it)
