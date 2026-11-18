"""
Lines whose number of executions grows with the number of series.
Usage: python counts.py <out/tag dir> <case with {n}, e.g. predict_{n}_wide> [env] [n_a n_b]
"""
import json
import os
import sys

d, case = sys.argv[1], sys.argv[2]
env = sys.argv[3] if len(sys.argv) > 3 else "p3"
n_a, n_b = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (20, 100)


def load(n):
    data = json.load(open(os.path.join(d, f"{case.format(n=n)}__{env}.json")))
    return {(r["file"], r["line"]): r for r in data["rows"]}


a, b = load(n_a), load(n_b)
rows = []
for key in sorted(set(a) | set(b)):
    ha = a.get(key, {}).get("hits", 0)
    hb = b.get(key, {}).get("hits", 0)
    per_series = (hb - ha) / (n_b - n_a)
    fixed = ha - per_series * n_a
    rows.append((per_series, fixed, key, (b.get(key) or a.get(key))["code"]))
total_a = sum(r["hits"] for r in a.values())
total_b = sum(r["hits"] for r in b.values())
print(f"{case} [{env}]: lines executed per unit: n={n_a}: {total_a:.0f}, n={n_b}: {total_b:.0f}; "
      f"per series {(total_b - total_a) / (n_b - n_a):.1f}, fixed {total_a - (total_b - total_a) / (n_b - n_a) * n_a:.0f}")
print(f"{'hits/series':>11} {'fixed':>7}  line")
for per_series, fixed, (file, line), code in sorted(rows, key=lambda r: -r[0]):
    if per_series < 0.05:
        continue
    print(f"{per_series:11.2f} {fixed:7.1f}  {file.split('/')[-1]}:{line}  {code[:80]}")
