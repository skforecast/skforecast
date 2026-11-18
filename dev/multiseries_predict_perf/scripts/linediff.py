"""
Lines of skforecast where the time difference between pandas 2 and pandas 3
goes, from the files of lineprof.py (time of each line with the pandas, numpy
and scikit-learn calls it makes, without the skforecast functions it calls).

Usage: python linediff.py <out/tag dir> <case, e.g. predict_20_wide> [top] [--by p3|p3nopa|abs]
"""
import json
import os
import sys

d, case = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 25
by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else "p3"
envs = ["p2", "p3nopa", "p3"]
data = {e: json.load(open(os.path.join(d, f"{case}__{e}.json"))) for e in envs}
rows = {e: {(r["file"], r["line"]): r for r in data[e]["rows"]} for e in envs}
keys = set().union(*[set(rows[e]) for e in envs])
tot = {e: sum(r["own_us"] for r in rows[e].values()) for e in envs}
print(f"{case}: us por unidad sin perfilar: " + ", ".join(
    f"{e} {data[e]['plain_us']:.0f}" for e in envs)
    + " | suma de lineas (perfilado): " + ", ".join(f"{e} {tot[e]:.0f}" for e in envs))
print(f"diferencia con p2 (suma de lineas): p3 sin pyarrow {tot['p3nopa'] - tot['p2']:+.0f}, "
      f"p3 con pyarrow {tot['p3'] - tot['p2']:+.0f}")
table = []
for key in keys:
    v = {e: rows[e].get(key, {}).get("own_us", 0.0) for e in envs}
    any_row = next(rows[e][key] for e in envs if key in rows[e])
    table.append((v["p3"] - v["p2"], v["p3nopa"] - v["p2"], v, any_row))
if by == "p3nopa":
    table.sort(key=lambda r: -r[1])
elif by == "abs":
    table.sort(key=lambda r: -r[2]["p3"])
else:
    table.sort(key=lambda r: -r[0])
print(f"{'p2':>8}{'p3 sin':>8}{'p3 con':>8}{'d sin':>8}{'d con':>8}{'acum':>6}{'veces':>7}  linea")
acc = 0.0
total_diff = tot["p3"] - tot["p2"] if by != "p3nopa" else tot["p3nopa"] - tot["p2"]
for d_con, d_sin, v, r in table[:top]:
    acc += d_con if by != "p3nopa" else d_sin
    share = acc / total_diff if total_diff else float("nan")
    print(f"{v['p2']:8.1f}{v['p3nopa']:8.1f}{v['p3']:8.1f}{d_sin:+8.1f}{d_con:+8.1f}{share:6.0%}"
          f"{r['hits']:7.1f}  {r['file'].split('/')[-1]}:{r['line']}  {r['code'][:70]}")
