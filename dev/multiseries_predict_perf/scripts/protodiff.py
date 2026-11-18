"""
Base tree vs prototype tree, line by line (lines matched by file and text,
since the line numbers move), from two folders of lineprof.py files.

Usage: python protodiff.py <base dir> <proto dir> <case> [top]
"""
import collections
import json
import os
import sys

base_dir, proto_dir, case = sys.argv[1:4]
top = int(sys.argv[4]) if len(sys.argv) > 4 else 20
envs = ["p2", "p3nopa", "p3"]


def load(d, env):
    data = json.load(open(os.path.join(d, f"{case}__{env}.json")))
    acc = collections.defaultdict(float)
    for r in data["rows"]:
        acc[(r["file"].split("/")[-1], r["code"])] += r["own_us"]
    return data["plain_us"], acc


b = {e: load(base_dir, e) for e in envs}
n = {e: load(proto_dir, e) for e in envs}
print(f"{case}: sin perfilar base -> prototipo: " + ", ".join(
    f"{e} {b[e][0]:.0f} -> {n[e][0]:.0f}" for e in envs))
keys = set()
for e in envs:
    keys |= set(b[e][1]) | set(n[e][1])
rows = []
for k in keys:
    d = [b[e][1].get(k, 0.0) - n[e][1].get(k, 0.0) for e in envs]
    rows.append((d, k))
rows.sort(key=lambda r: -abs(r[0][0]))
print(f"{'ahorro p2':>10}{'p3 sin':>8}{'p3 con':>8}{'  base p2':>10}{'proto p2':>9}  linea")
for d, k in rows[:top]:
    print(f"{d[0]:10.1f}{d[1]:8.1f}{d[2]:8.1f}{b['p2'][1].get(k, 0.0):10.1f}{n['p2'][1].get(k, 0.0):9.1f}"
          f"  {k[0]}  {k[1][:68]}")
tot = [sum(b[e][1].values()) - sum(n[e][1].values()) for e in envs]
print(f"{tot[0]:10.1f}{tot[1]:8.1f}{tot[2]:8.1f}  total (suma de lineas)")
