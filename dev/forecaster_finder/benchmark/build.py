"""
Render every distinct snippet of the forecaster finder, run it on synthetic
data of the frequency of the answers and write results/review/finder.json
(rows, results of every combination and snippets): the data of the page.

Each snippet is run after its sample data block, in an empty namespace, as it
would run pasted into a new notebook. It fails if it raises an error or any
warning: what the user copies must run clean. The sentence shown with each result is checked against the
results of the study (rules.py), and the figures of the section "How we chose
these rules" are computed from them, not written by hand.

Usage:
    python -I build.py
    python -I build.py --ai    # also the lines of skforecast-ai (needs that package)
"""
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finder_tree as tree  # noqa: E402

RES = Path(__file__).parent / "results" / "review"
# Warnings that depend on the data, not on the snippet: scikit-learn raises
# this one when a category that is predicted does not appear in a short test
# period (quarterly data: 8 values).
DATA_WARNINGS = {"UserWarning: y_pred contains classes not in y_true"}


def run(code, a, n=None, after=""):
    """
    Run a snippet after its sample data block (finder_tree.sample_block), in
    an empty namespace, as in a new notebook, followed by `after` (the lines
    of skforecast-ai, which continue the script). Returns (error, warnings).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(compile(tree.sample_block(a, n) + "\n" + code + "\n" + after, "snippet", "exec"), {})
            error = None
        except Exception:
            error = traceback.format_exc(limit=2).strip().splitlines()[-1][:200]
    seen = sorted({f"{w.category.__name__}: {str(w.message).strip().splitlines()[0][:140]}"
                   for w in caught} - DATA_WARNINGS)
    return error, seen


def key(a):
    """Identifier of a set of answers, complete or not (also the hash of the page)."""
    return "-".join(a.get(k) or "_" for k in ("target", "series", "freq", "extra"))


def notebook():
    """Colab notebook with an install cell and a cell to paste the script into."""
    def cell(kind, source):
        base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        return base if kind == "markdown" else {**base, "outputs": [], "execution_count": None}
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": [
            cell("markdown",
                 "# Your first comparison with skforecast\n\n"
                 "1. Run the first cell to install the libraries (about a minute).\n"
                 "2. Paste the script you copied from the skforecast documentation into the "
                 "second cell and run it.\n\n"
                 "The script comes with sample data, so it runs as it is. To use your own "
                 "data, replace the block marked as sample data."),
            cell("code", '%pip install -q "skforecast[stats]" lightgbm chronos-forecasting'),
            cell("code", "# Paste your script here (Ctrl+V or Cmd+V) and run this cell\n"),
        ],
    }


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
    docs = Path(__file__).resolve().parents[3] / "docs"
    missing = [g for g in list(tree.GUIDES.values()) + [tree.EXOG_GUIDE]
               if not any((docs / g).with_suffix(ext).exists() for ext in (".ipynb", ".md"))]
    print(f"user guides that do not exist: {missing}")
    snippets, combos = {}, []
    for a in tree.all_answers():
        leaf = tree.decide(a)
        code = tree.render(a, leaf)
        sid = snippets.setdefault(code, {"id": f"s{len(snippets)}", "answers": a})["id"]
        combos.append({"key": key(a), "answers": a, "leaf": leaf["leaf"],
                       "why": tree.reasons(leaf), "prompt": tree.prompt(a, leaf, code),
                       "sample": tree.sample_block(a), "snippet": sid,
                       "skforecast_ai": tree.ai_snippet(a), "agent": tree.agent_message(a)})
    partial = {key(a): tree.blocks(a) for a in tree.all_partial_answers()}

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

    # The lines of skforecast-ai need that package, which is not a dependency
    # of skforecast: they are run with `--ai`, in an environment that has it.
    if "--ai" in sys.argv:
        import importlib.metadata as md
        ai_failed = []
        for code, info in snippets.items():
            after = tree.ai_snippet(info["answers"])
            if after is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    exec(compile(tree.sample_block(info["answers"]) + "\n" + code + "\n" + after,
                                 "snippet", "exec"), {})
                    status = "ok"
                except Exception:
                    status = traceback.format_exc(limit=2).strip().splitlines()[-1][:200]
                    ai_failed.append(info["id"])
            print("skforecast-ai", info["id"], status, json.dumps(info["answers"]), flush=True)
        versions = {p: md.version(p) for p in ("skforecast-ai", "skforecast", "pandas")}
        (RES / "skforecast_ai_check.json").write_text(json.dumps(
            {"versions": versions, "snippets": sum(tree.ai_snippet(i["answers"]) is not None
                                                   for i in snippets.values()),
             "failed": ai_failed}, indent=1))
        print(f"skforecast-ai lines failed: {ai_failed} ({versions})")
        failed += ai_failed
    unsupported = [k for k, ok in facts["checks"].items() if not ok]
    print(f"sentences not supported by the study: {unsupported}")
    (RES / "first_comparison.ipynb").write_text(json.dumps(notebook(), indent=1))
    out = {"rows": tree.ROWS, "notes": tree.NOTES, "partial": partial,
           "script_steps": tree.SCRIPT_STEPS, "exog_guide": tree.EXOG_GUIDE,
           "intro": tree.INTRO.format(datasets=facts["datasets"]),
           "evidence": {k: facts[k] for k in ("datasets", "table", "stats")},
           "min_history": {k: v["min_history"] for k, v in tree.FREQ.items()},
           "combos": combos, "snippets": results}
    (RES / "finder.json").write_text(json.dumps(out, indent=1))
    print(f"failed (error or warning): {failed}")
    return 1 if failed or unsupported or missing else 0


if __name__ == "__main__":
    sys.exit(main())
