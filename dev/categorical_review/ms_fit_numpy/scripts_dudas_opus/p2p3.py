# Compare the pandas 2 and pandas 3 scenario results (LightGBM and XGBoost).
import json, os
D = os.path.dirname(os.path.abspath(__file__))
n = diff = 0
for t2, t3 in [("p2base", "base"), ("p2branch", "branch"), ("p2minimal2", "minimal2")]:
    for fam in ["lgbm", "xgb"]:
        a = json.load(open(os.path.join(D, f"res_{t2}_{fam}.json")))
        b = json.load(open(os.path.join(D, f"res_{t3}_{fam}.json")))
        for x, y in zip(a, b):
            n += 1
            if (x["status"], x.get("mae"), x.get("err", "")[:40]) != (y["status"], y.get("mae"), y.get("err", "")[:40]):
                diff += 1
                print(t3, fam, x["enc"], x["cf"], x["user"], x.get("mae"), y.get("mae"))
print("compared", n, "different", diff)
