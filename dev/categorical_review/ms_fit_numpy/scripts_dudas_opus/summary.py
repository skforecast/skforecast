"""
Counts over the scenario matrix: silent wrong predictions (status ok and MAE
above 10, the true effects are +-40), errors and agreement between trees.
"""
import json
import os
from collections import Counter

D = os.path.dirname(os.path.abspath(__file__))
fams = ["lgbm", "xgb", "hgb", "ridge", "pipe_lgbm", "ttr_lgbm", "catboost"]


def load(tag):
    out = {}
    for fam in fams:
        for r in json.load(open(os.path.join(D, f"res_{tag}_{fam}.json"))):
            k = (r["est"], str(r["enc"]), str(r["cf"]), str(r["user"]), json.dumps(r["fk"]), r["series"])
            out[k] = r


    return out


def cls(r):
    if r["status"] != "ok":
        return "error"
    if r["est"] == "ridge":
        return "ok"  # linear model cannot fit the non-monotone effect
    if r["est"] == "hgb" and r["cf"] == "list_other":
        return "error"
    return "wrong" if r["mae"] > 10 else "ok"


data = {t: load(t) for t in ["base", "branch", "minimal2"]}
keys = sorted(data["base"])
print("scenarios:", len(keys))
for t in data:
    print(t, Counter(cls(data[t][k]) for k in keys))
trans = Counter((cls(data["base"][k]), cls(data["branch"][k]), cls(data["minimal2"][k])) for k in keys)
print("(base, branch, minimal) ->", dict(trans))
same = sum(
    1 for k in keys
    if data["branch"][k]["status"] == data["minimal2"][k]["status"] == "ok"
    and data["branch"][k]["mae"] == data["minimal2"][k]["mae"]
)
both_ok = sum(1 for k in keys if data["branch"][k]["status"] == data["minimal2"][k]["status"] == "ok")
print(f"branch and minimal both ok: {both_ok}, same MAE: {same}")
print("only branch ok:")
for k in keys:
    if data["branch"][k]["status"] == "ok" and data["minimal2"][k]["status"] != "ok":
        print("  ", k[:4], k[5], "|", data["minimal2"][k]["err"][:90])
print("both ok, different MAE:")
for k in keys:
    b, m = data["branch"][k], data["minimal2"][k]
    if b["status"] == m["status"] == "ok" and b["mae"] != m["mae"]:
        print("  ", k[:4], k[5], b["mae"], m["mae"])
