"""
Review of the snippets of the forecaster finder (finder_tree.py).

build.py runs every snippet with the warnings silenced and on synthetic series
of a comfortable length. This script answers two questions that build.py does
not: which warnings a user sees when running a snippet as is, and how much
history a snippet needs to run.

Usage:
    python -I check_snippets.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finder_tree as tree  # noqa: E402

OUT = Path(__file__).parent / "results" / "review"
OUT.mkdir(parents=True, exist_ok=True)
SEASON = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4}
LENGTH = {"hourly": 24 * 60, "daily": 730, "weekly": 52 * 5, "monthly": 180, "quarterly": 80}
# Histories that a beginner can plausibly have, per frequency
GRID = {
    "hourly": [24 * d for d in (7, 9, 10, 14, 21, 30, 60)],
    "daily": [90, 120, 150, 180, 270, 365, 730],
    "weekly": [52, 78, 104, 130, 156, 208, 260],
    "monthly": [24, 36, 48, 60, 72, 96, 120],
    "quarterly": [12, 16, 20, 24, 32, 40, 60],
}


def synthetic(a, n):
    f = tree.FREQ[a["freq"]]
    idx = pd.date_range("2018-01-01", periods=n, freq=f["alias"])
    rng = np.random.default_rng(123)
    t = np.arange(n)
    s = SEASON[a["freq"]]
    exog = pd.DataFrame({"promo": rng.integers(0, 2, n).astype(float),
                         "temp": 20 + 5 * np.sin(2 * np.pi * t / max(s, 2))}, index=idx)

    def one(level):
        return (level + 0.02 * t + 10 * np.sin(2 * np.pi * t / s)
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


def run(code, a, n):
    """Run a snippet. Returns (error or None, list of (category, first line))."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(compile(code, "snippet", "exec"), synthetic(a, n))
            error = None
        except Exception as e:
            error = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:150]}"
    seen = sorted({(w.category.__name__, str(w.message).strip().splitlines()[0][:150])
                   for w in caught})
    return error, seen


snippets = {}
for a in tree.all_answers():
    snippets.setdefault(tree.render(a, tree.decide(a)), a)
print(f"{len(snippets)} distinct snippets", flush=True)

# 1. Warnings that the user sees, with the same data as build.py
rows = []
for code, a in snippets.items():
    error, seen = run(code, a, LENGTH[a["freq"]])
    for category, message in seen or [("", "")]:
        rows.append({**{k: a.get(k, "") for k in ("target", "series", "freq", "extra", "priority")},
                     "error": error or "", "warning": category, "message": message})
w = pd.DataFrame(rows)
w.to_csv(OUT / "snippet_warnings.csv", index=False)
n_warn = w.loc[w.warning != ""].drop_duplicates(["target", "series", "freq", "extra", "priority"])
print(f"snippets with an error: {w.loc[w.error != ''].drop_duplicates(['target', 'series', 'freq', 'extra', 'priority']).shape[0]}; "
      f"with at least one warning: {len(n_warn)}")
print(w.loc[w.warning != ""].groupby(["warning", "message"]).agg(
    snippets=("freq", "size"), freqs=("freq", lambda s: ",".join(sorted(set(s))))).to_string())

# 2. Shortest history, from a grid, with which each kind of snippet runs
rows = []
kinds = {}
for code, a in snippets.items():
    if a["extra"] != "none" or a.get("priority", "accuracy") != "accuracy":
        continue
    kinds[(a["target"], a.get("series", ""), a["freq"])] = (code, a)
for (target, series, freq), (code, a) in kinds.items():
    row = {"target": target, "series": series, "freq": freq}
    for n in GRID[freq]:
        error, _ = run(code, a, n)
        row[f"n{GRID[freq].index(n)}"] = f"{n}: {'ok' if error is None else error[:60]}"
        if error is None and "min_ok" not in row:
            row["min_ok"] = n
    rows.append(row)
    print(json.dumps(row), flush=True)
pd.DataFrame(rows).to_csv(OUT / "snippet_min_history.csv", index=False)
