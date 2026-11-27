# Compare the scenario results of several trees.
# Usage: python summarize.py <kind> <label1> <label2> ...
import sys
import json
import glob
import os
from collections import Counter, defaultdict

kind = sys.argv[1]
labels = sys.argv[2:]
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

data = {}
for label in labels:
    d = {}
    for path in glob.glob(os.path.join(here, f"{label}_*_{kind}.json")):
        with open(path, encoding="utf-8") as f:
            d.update(json.load(f))
    data[label] = d


def cell(r):
    if r is None:
        return "MISSING"
    if r["status"] == "error":
        return "ERR " + r["error"].split(":")[0]
    cats = r["effective_cat"]
    if isinstance(cats, list):
        cats = ",".join(c.replace("_level_skforecast", "LEVEL").replace("exog_", "") for c in cats) or "-"
    tag = "ok" if r["consistent"] else "INCONSISTENT"
    xt = "np" if r["fit_X_type"] == "ndarray" else "df"
    return f"{tag} [{cats}] {xt}"


keys = sorted(set().union(*[set(d) for d in data.values()]))
totals = {label: Counter() for label in labels}
print("| scenario | " + " | ".join(labels) + " |")
print("|---|" + "---|" * len(labels))
for k in keys:
    cells = [cell(data[label].get(k)) for label in labels]
    for label, c in zip(labels, cells):
        totals[label][c.split(" ")[0]] += 1
    mark = "" if len(set(cells)) == 1 else " **"
    print(f"| {k.rsplit('|', 1)[0]}{mark} | " + " | ".join(cells) + " |")
print()
for label in labels:
    print(label, dict(totals[label]), "n =", sum(totals[label].values()))

# Errors detail
print()
for label in labels:
    errs = defaultdict(list)
    for k, r in data[label].items():
        if r["status"] == "error":
            errs[r["error"][:160]].append(k.rsplit("|", 1)[0])
    print(f"== errors {label}")
    for e, ks in errs.items():
        print(f"  [{len(ks)}] {e}")
        for kk in ks[:40]:
            print("       ", kk)
