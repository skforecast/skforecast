"""
Markdown table from the JSON files written by bench_hotspots.py.

    python dev/pandas3_migration/bench_report.py [--out results/bench_table.md]

For each benchmark and each pandas version it prints, per variant, the median
and minimum time (ms), the peak memory (tracemalloc, MiB), whether the outputs
of the variants are identical, and the ratio of the patched variant against
"current" in the same process (A/B interleaved) and of pandas 3 against
pandas 2 (separate processes, same machine, run one after the other).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load_all(results_dir: str) -> dict[str, dict]:
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "bench_pandas*.json"))):
        name = os.path.basename(path)[len("bench_"):-len(".json")]
        if name.endswith("_smoke"):
            continue
        runs[name] = json.load(open(path))
    return runs


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main(out: str | None):
    runs = load_all(os.path.join(HERE, "results"))
    if not runs:
        print("no bench_*.json results")
        return
    lines = ["# Medidas de rendimiento (bench_hotspots.py)", ""]
    envs = {}
    for name, r in runs.items():
        e = r["env"]
        envs[name] = f"Python {e['python']}, pandas {e['pandas']}, numpy {e['numpy']}, scikit-learn {e['sklearn']}, numba {e['numba']}"
    lines.append("Entornos:")
    for name, desc in envs.items():
        lines.append(f"- `{name}`: {desc}; repeticiones {runs[name]['repeats']} (+{runs[name]['warmup']} de calentamiento)")
    lines.append("")
    bench_names = []
    for r in runs.values():
        for b in r["benchmarks"]:
            if b not in bench_names:
                bench_names.append(b)
    # Per-version baseline table (current variant only)
    versions = sorted({re.match(r"pandas([\d.]+)", n).group(1) for n in runs})
    base_runs = {v: runs.get(f"pandas{v}") for v in versions}
    lines += ["## pandas 2 frente a pandas 3, código actual", "",
              "| benchmark | " + " | ".join(f"p{v} mediana ms | p{v} mín ms | p{v} pico MiB" for v in versions) + " | ratio tiempo p3/p2 | ratio memoria p3/p2 | salida idéntica p2/p3 |",
              "|---|" + "---|" * (3 * len(versions) + 3)]
    for b in bench_names:
        row = [b]
        vals = {}
        for v in versions:
            r = base_runs.get(v)
            rec = r["benchmarks"].get(b) if r else None
            if rec:
                t = rec["time"]["current"]; m = rec.get("memory", {}).get("current", {})
                vals[v] = (t["median_s"] * 1000, t["min_s"] * 1000, m.get("tracemalloc_peak_mib"), rec["fingerprints"]["current"])
                row += [fmt(vals[v][0]), fmt(vals[v][1]), fmt(vals[v][2])]
            else:
                row += ["-", "-", "-"]
        if len(vals) == 2:
            a, c = vals[versions[0]], vals[versions[1]]
            row += [fmt(c[0] / a[0], 2), fmt(c[2] / a[2], 2) if a[2] and c[2] else "-", "sí" if a[3] == c[3] else "NO"]
        else:
            row += ["-", "-", "-"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # Patched runs
    for name, r in runs.items():
        if not r.get("patch"):
            continue
        lines += [f"## {name}: `current` frente a `patched:{r['patch']}` (A/B intercalado en el mismo proceso)", "",
                  "| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b, rec in r["benchmarks"].items():
            t = rec["time"]; m = rec.get("memory", {})
            cur, pat = t["current"], t[f"patched:{r['patch']}"]
            mc = m.get("current", {}).get("tracemalloc_peak_mib"); mp = m.get(f"patched:{r['patch']}", {}).get("tracemalloc_peak_mib")
            lines.append("| " + " | ".join([
                b, fmt(cur["median_s"] * 1000), fmt(pat["median_s"] * 1000), fmt(pat["median_s"] / cur["median_s"], 3),
                fmt(cur["min_s"] * 1000), fmt(pat["min_s"] * 1000), fmt(mc), fmt(mp),
                fmt(mp / mc, 3) if mc and mp else "-", "sí" if rec["identical_outputs"] else "NO",
            ]) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bench_table.md"))
    a = ap.parse_args()
    main(a.out)
