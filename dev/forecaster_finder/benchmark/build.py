"""
Render every distinct snippet of the forecaster finder, run it on synthetic
data with the frequency of the answers and write finder.json.
"""
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finder_tree as tree  # noqa: E402

warnings.filterwarnings("ignore")
LENGTH = {"hourly": 24 * 60, "daily": 730, "weekly": 52 * 5, "monthly": 180, "quarterly": 80}
SEASON = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4}


def synthetic(a):
    f = tree.FREQ[a["freq"]]
    n = LENGTH[a["freq"]]
    idx = pd.date_range("2018-01-01", periods=n, freq=f["alias"])
    rng = np.random.default_rng(123)
    t = np.arange(n)
    exog = pd.DataFrame({"promo": rng.integers(0, 2, n).astype(float),
                         "temp": 20 + 5 * np.sin(2 * np.pi * t / max(SEASON[a["freq"]], 2))}, index=idx)
    def one(level):
        return (level + 0.02 * t + 10 * np.sin(2 * np.pi * t / SEASON[a["freq"]])
                + 3 * exog["promo"].to_numpy() + rng.normal(0, 1, n))
    ns = {"exog": exog}
    if a["target"] == "category":
        v = pd.Series(one(100), index=idx)
        ns["y"] = pd.cut(v, bins=[-np.inf, v.quantile(0.2), v.quantile(0.8), np.inf],
                         labels=["low", "medium", "high"]).astype(str)
    elif a.get("series") == "several":
        ns["series"] = pd.DataFrame({f"item_{i}": one(50 + 10 * i) for i in range(5)}, index=idx)
    else:
        ns["y"] = pd.Series(one(100), index=idx, name="sales")
    return ns


snippets, combos = {}, []
for a in tree.all_answers():
    leaf = tree.decide(a)
    code = tree.render(a, leaf)
    key = snippets.setdefault(code, {"id": f"s{len(snippets)}", "answers": a})["id"]
    combos.append({"answers": a, "leaf": leaf, "snippet": key})

print(f"{len(combos)} answer combinations, {len(snippets)} distinct snippets", flush=True)
results = {}
for code, info in snippets.items():
    start = time.perf_counter()
    ns = synthetic(info["answers"])
    try:
        exec(compile(code, info["id"], "exec"), ns)
        status = "ok"
    except Exception:
        status = traceback.format_exc(limit=2)
    results[info["id"]] = {"code": code, "status": status, "seconds": round(time.perf_counter() - start, 1)}
    print(info["id"], status if status != "ok" else "ok", results[info["id"]]["seconds"], "s",
          json.dumps(info["answers"]), flush=True)

failed = [k for k, v in results.items() if v["status"] != "ok"]
out = {"questions": tree.QUESTIONS, "freq": tree.FREQ, "combos": combos, "snippets": results}
(Path(__file__).parent / "finder.json").write_text(json.dumps(out, indent=1))
print(f"failed: {failed}")
sys.exit(1 if failed else 0)
