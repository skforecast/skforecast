"""
Compare the scenario results of several trees. Usage: python compare.py tag1 tag2 ...
"""
import sys
import json
import glob
import os

D = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1:]
fams = ["lgbm", "xgb", "hgb", "ridge", "pipe_lgbm", "ttr_lgbm", "catboost"]


def key(r):
    return (r["est"], str(r["enc"]), str(r["cf"]), str(r["user"]), json.dumps(r["fk"]), r["series"])


data = {}
for t in tags:
    data[t] = {}
    for fam in fams:
        p = os.path.join(D, f"res_{t}_{fam}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            data[t][key(r)] = r


def short(r):
    if r is None:
        return "-"
    if r["status"] == "ok":
        w = [x for x in r["warnings"] if x not in ("InputTypeWarning",)]
        return f"mae={r['mae']}" + (f" W:{','.join(w)}" if w else "")
    return "ERR " + r["err"][:90]


keys = sorted(set().union(*[set(d) for d in data.values()]))
only_diff = "--all" not in os.environ.get("CMP", "")
for k in keys:
    if k[5] != "wide" and os.environ.get("SERIES", "wide") == "wide":
        # dict rows are printed only if they differ from wide
        kw = k[:5] + ("wide",)
        if all(short(data[t].get(k)) == short(data[t].get(kw)) for t in tags):
            continue
    vals = [short(data[t].get(k)) for t in tags]
    print(" | ".join([f"{k[0]:9} enc={k[1]:16} cf={k[2]:10} user={k[3]:24} {k[5]:4}"] + vals))
