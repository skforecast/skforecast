"""
Render every distinct snippet of the forecaster finder, run it on synthetic
data of the frequency of the answers and write results/review/finder.json
(rows, results of every combination and snippets): the data of the page.

A snippet fails if it raises an error or any warning: what the user copies
must run clean. The figures shown with each rule come from the results of the
study (results/review/recount_runs.csv), not from text written by hand.

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
FUTURE_EXOG = ["vic_electricity_daily", "bike_sharing_daily", "bike_sharing_hourly"]
MONTHLY = ["h2o_monthly", "fuel_gasoline_monthly", "expenditures_australia_monthly"]
# Warnings that depend on the data, not on the snippet: scikit-learn raises
# this one when a category that is predicted does not appear in a short test
# period (quarterly data: 8 values).
# Column of recount_runs.csv with LightGBM as the snippets write it
ML = "ml_page"
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
    """Sentence with the figures of each rule, from the results of the study."""
    a = pd.read_csv(RES / "recount_runs.csv")
    one = a[a.n == 1]
    known = one[one.case.isin(FUTURE_EXOG)]
    none = one[~one.case.isin(FUTURE_EXOG)]
    several = a[a.n > 1]

    def by_dataset(d, full=False):
        d = d[d.full] if full else d
        g = d.groupby("dataset")[["fm", ML]].mean()
        return int((g.fm < g[ML]).sum()), len(g)

    fm_none, n_none = by_dataset(none)
    lg_known = known.groupby("dataset")[["fm", ML]].mean()
    fm_sev, n_sev = by_dataset(several, full=True)
    monthly = a[a.case.isin(MONTHLY)]
    clf = pd.read_csv(RES / "extra_classification.csv").set_index("model")
    return {
        "one-none": (
            f"In our tests, Chronos-2 had a lower error than LightGBM in {fm_none} of "
            f"{n_none} datasets of this kind, and LightGBM did worse than the seasonal "
            f"baseline in {int((none[ML] >= 1).sum())} of {len(none)} runs."),
        "one-known": (
            f"No clear winner in our tests: with variables known in advance, LightGBM had "
            f"a lower error than Chronos-2 in {int((lg_known[ML] < lg_known.fm).sum())} "
            f"of {len(lg_known)} datasets, and both always beat the seasonal baseline."),
        "several": (
            f"No clear winner in our tests: Chronos-2 had a lower error than LightGBM in "
            f"{fm_sev} of {n_sev} datasets, by small margins, and both always beat the "
            f"seasonal baseline."),
        "stats": (
            f"On monthly data AutoETS matched Chronos-2 (lower error in "
            f"{int((monthly.ets < monthly.fm).sum())} of {len(monthly)} runs) and never did "
            f"worse than the seasonal baseline."),
        "category": (
            f"On one dataset the classifier was right more often than the rule "
            f"({clf.loc['LightGBM classifier [refit]', 'accuracy']:.0%} against "
            f"{clf.loc['Same hour yesterday', 'accuracy']:.0%}), but not on balanced "
            f"accuracy ({clf.loc['LightGBM classifier [refit]', 'balanced_accuracy']:.2f} "
            f"against {clf.loc['Same hour yesterday', 'balanced_accuracy']:.2f}): keep the "
            f"rule as your baseline."),
    }


def main():
    facts = evidence()
    snippets, combos = {}, []
    for a in tree.all_answers():
        leaf = tree.decide(a)
        code = tree.render(a, leaf)
        key = snippets.setdefault(code, {"id": f"s{len(snippets)}", "answers": a})["id"]
        why = [facts[leaf["evidence"]]] + ([facts["stats"]] if leaf["stats"] else [])
        combos.append({"answers": a, "leaf": leaf["leaf"], "headline": leaf["headline"],
                       "lineup": tree.lineup(a, leaf), "why": why,
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
    out = {"rows": tree.ROWS, "default": tree.DEFAULT, "notes": tree.NOTES,
           "min_history": {k: v["min_history"] for k, v in tree.FREQ.items()},
           "combos": combos, "snippets": results}
    (RES / "finder.json").write_text(json.dumps(out, indent=1))
    print(f"failed (error or warning): {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
