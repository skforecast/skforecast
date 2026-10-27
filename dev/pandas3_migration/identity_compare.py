"""
Compare two identity runs (identity_cases.py) and print, per case and output,
whether the strict and the normalised fingerprints are identical, with the
dtype/index differences when they are not.

    python dev/pandas3_migration/identity_compare.py results/identity_pandas2.3.3.json results/identity_pandas3.1.0.json [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import os


def diff_describe(a: dict, b: dict) -> str:
    parts = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if va != vb:
            if isinstance(va, dict) and isinstance(vb, dict):
                sub = {kk: (va.get(kk), vb.get(kk)) for kk in set(va) | set(vb) if va.get(kk) != vb.get(kk)}
                parts.append(f"{k}: {sub}")
            else:
                parts.append(f"{k}: {va} -> {vb}")
    return "; ".join(parts)


def compare(path_a: str, path_b: str, markdown: str | None = None) -> dict:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    label_a = f"pandas {a['env']['pandas']}{(' ' + a['tag']) if a.get('tag') else ''}"
    label_b = f"pandas {b['env']['pandas']}{(' ' + b['tag']) if b.get('tag') else ''}"
    rows = []
    totals = {"outputs": 0, "strict_equal": 0, "normalized_equal": 0, "different": 0, "errors": 0}
    for case in a["cases"]:
        ca, cb = a["cases"][case], b["cases"].get(case)
        if cb is None:
            rows.append((case, "-", "missing in B", "", ""))
            continue
        if ca["error"] or cb["error"]:
            totals["errors"] += 1
            rows.append((case, "-", "ERROR", (ca["error"] or "").splitlines()[-1][:80], (cb["error"] or "").splitlines()[-1][:80]))
            continue
        for out in ca["outputs"]:
            oa, ob = ca["outputs"][out], cb["outputs"].get(out)
            totals["outputs"] += 1
            if ob is None:
                rows.append((case, out, "missing in B", "", ""))
                totals["different"] += 1
                continue
            if oa["strict"] == ob["strict"]:
                totals["strict_equal"] += 1
                rows.append((case, out, "identical", "", ""))
            elif oa["normalized"] == ob["normalized"]:
                totals["normalized_equal"] += 1
                rows.append((case, out, "values identical, dtype/unit differ", diff_describe(oa["describe"], ob["describe"]), ""))
            else:
                totals["different"] += 1
                rows.append((case, out, "DIFFERENT VALUES", diff_describe(oa["describe"], ob["describe"]), ""))
    warn_rows = []
    for case in a["cases"]:
        wa = set(a["cases"][case]["pandas_warnings"])
        wb = set(b["cases"][case]["pandas_warnings"]) if case in b["cases"] else set()
        for w in sorted(wb - wa):
            warn_rows.append((case, "only " + label_b, w))
        for w in sorted(wa - wb):
            warn_rows.append((case, "only " + label_a, w))
    lines = [f"# Identity check: {label_a} (A) vs {label_b} (B)", "",
             f"Totals: {totals}", "",
             "| case | output | result | dtype/index differences |", "|---|---|---|---|"]
    for case, out, res, d1, d2 in rows:
        lines.append(f"| {case} | {out} | {res} | {d1 or d2} |")
    lines += ["", "## Deprecation warnings present in only one run", "", "| case | run | warning |", "|---|---|---|"]
    for case, where, w in warn_rows:
        lines.append(f"| {case} | {where} | {w} |")
    text = "\n".join(lines)
    print(text)
    if markdown:
        os.makedirs(os.path.dirname(markdown) or ".", exist_ok=True)
        open(markdown, "w").write(text + "\n")
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    compare(args.a, args.b, args.markdown)
