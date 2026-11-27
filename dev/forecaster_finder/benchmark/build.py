"""
Render every distinct snippet of the forecaster finder, run it on synthetic
data of the frequency of the answers and write results/review/finder.json
(rows, results of every combination and snippets): the data of the page.

A snippet fails if it raises an error or any warning: what the user copies
must run clean. The sentence shown with each result is checked against the
results of the study (rules.py), and the figures of the section "How we chose
these rules" are computed from them, not written by hand.

Usage:
    python -I build.py
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

RES = Path(__file__).parent / "results" / "review"
LENGTH = {"hourly": 24 * 60, "daily": 730, "weekly": 52 * 5, "monthly": 180, "quarterly": 80,
          "other": 730}
SEASON = {k: v["season"] for k, v in tree.FREQ.items()}
# Warnings that depend on the data, not on the snippet: scikit-learn raises
# this one when a category that is predicted does not appear in a short test
# period (quarterly data: 8 values).
DATA_WARNINGS = {"UserWarning: y_pred contains classes not in y_true"}


def synthetic(a, n=None):
    """Series (and exog) with the frequency, trend and seasonality of the answers."""
    f = tree.FREQ[a["freq"]]
    n = n or LENGTH[a["freq"]]
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
        ns["series"] = {f"item_{i}": pd.Series(one(50 + 10 * i), index=idx, name=f"item_{i}")
                        for i in range(5)}
    else:
        ns["y"] = pd.Series(one(100), index=idx, name="sales")
    return ns


def run(code, a, n=None):
    """Run a snippet with the warnings recorded. Returns (error, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(compile(code, "snippet", "exec"), synthetic(a, n))
            error = None
        except Exception:
            error = traceback.format_exc(limit=2).strip().splitlines()[-1][:200]
    seen = sorted({f"{w.category.__name__}: {str(w.message).strip().splitlines()[0][:140]}"
                   for w in caught} - DATA_WARNINGS)
    return error, seen


def evidence():
    """
    Figures of the section "How we chose these rules" and the checks of the
    sentences shown with each result (finder_tree.REASON), from the results
    of the study (rules.py). A sentence that the results stop supporting
    makes the build fail: the rule has to be rewritten.
    """
    cells = pd.read_csv(RES / "rules_cells.csv").set_index("cell")
    runs = pd.read_csv(RES / "rules_runs.csv")
    clf = pd.read_csv(RES / "extra_classification.csv").set_index("model")
    low = runs[(runs.n_series == 1) & runs.freq.isin(["monthly", "quarterly"])]
    low = low.groupby("dataset")[["fm", "ml", "arima"]].mean()
    names = {"one series, nothing known in advance": ("one-none", "One series, nothing known in advance"),
             "one series, known in advance": ("one-known", "One series, variables known in advance"),
             "several series": ("several", "Several series")}
    table, checks = [], {}
    for cell, (leaf, label) in names.items():
        c = cells.loc[cell]
        n, ahead = int(c["datasets"]), int(c["Chronos-2 ahead (datasets)"])
        worse = int(c["LightGBM >= naive (datasets)"])
        table.append({"data": label, "datasets": n, "chronos_ahead": ahead,
                      "lightgbm_ahead": n - ahead, "lightgbm_below_baseline": worse,
                      "chronos_below_baseline": int(c["Chronos-2 >= naive (datasets)"])})
        if leaf == "one-none":
            checks[leaf] = ahead >= 0.75 * n and worse >= 0.25 * n
        else:
            checks[leaf] = ahead > 0.5 * n and ahead < n
    checks["stats"] = (low.arima < low.ml).mean() > 0.5
    checks["category"] = (clf.loc["LightGBM classifier [refit]", "balanced_accuracy"]
                          <= clf.loc["Same hour yesterday", "balanced_accuracy"] + 0.02)
    stats = {"datasets": len(low), "arima_ahead_of_lightgbm": int((low.arima < low.ml).sum()),
             "arima_ahead_of_chronos": int((low.arima < low.fm).sum())}
    return {"datasets": int(runs.dataset.nunique()), "table": table, "stats": stats,
            "checks": checks}


def main():
    facts = evidence()
    snippets, combos = {}, []
    for a in tree.all_answers():
        leaf = tree.decide(a)
        code = tree.render(a, leaf)
        key = snippets.setdefault(code, {"id": f"s{len(snippets)}", "answers": a})["id"]
        combos.append({"answers": a, "leaf": leaf["leaf"], "headline": leaf["headline"],
                       "lineup": tree.lineup(a, leaf), "why": tree.reasons(leaf),
                       "prompt": tree.prompt(a, leaf), "snippet": key})

    print(f"{len(combos)} answer combinations, {len(snippets)} distinct snippets", flush=True)
    results = {}
    for code, info in snippets.items():
        start = time.perf_counter()
        error, seen = run(code, info["answers"])
        status = "ok" if error is None and not seen else (error or "; ".join(seen))
        results[info["id"]] = {"code": code, "status": status,
                               "seconds": round(time.perf_counter() - start, 1)}
        print(info["id"], status, results[info["id"]]["seconds"], "s",
              json.dumps(info["answers"]), flush=True)

    failed = [k for k, v in results.items() if v["status"] != "ok"]
    unsupported = [k for k, ok in facts["checks"].items() if not ok]
    print(f"sentences not supported by the study: {unsupported}")
    out = {"rows": tree.ROWS, "default": tree.DEFAULT, "notes": tree.NOTES,
           "intro": tree.INTRO.format(datasets=facts["datasets"]),
           "evidence": {k: facts[k] for k in ("datasets", "table", "stats")},
           "min_history": {k: v["min_history"] for k, v in tree.FREQ.items()},
           "combos": combos, "snippets": results}
    (RES / "finder.json").write_text(json.dumps(out, indent=1))
    print(f"failed (error or warning): {failed}")
    return 1 if failed or unsupported else 0


if __name__ == "__main__":
    sys.exit(main())
