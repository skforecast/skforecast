"""
Aggregate the JSON files of run_bench.sh: minimum over rounds of every
(case, env, tree, metric).

Usage: python agg_ms.py <out/tag dir> [--tree base|new] [--gain]
  default: tables of the tree `base` in the three environments with the
  ratios to pandas 2 and the split into fixed cost and cost per series.
  --gain: base vs new per environment.
"""
import collections
import glob
import json
import os
import re
import sys

d = sys.argv[1]
tree_sel = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else "base"
best = collections.defaultdict(lambda: float("inf"))
spread = collections.defaultdict(list)
meta = {}
for f in sorted(glob.glob(os.path.join(d, "r*_*.json"))):
    _, env, tree = os.path.basename(f)[:-5].split("_")
    data = json.load(open(f))
    meta[env] = (data["pandas"], data["storage"], data["numpy"], data["python"])
    for case, metrics in data["res"].items():
        for metric, value in metrics.items():
            if metric == "n_folds":
                continue
            best[(case, metric, env, tree)] = min(best[(case, metric, env, tree)], value)
            spread[(case, metric, env, tree)].append(value)
ENVS = ["p2", "p3nopa", "p3"]
LABEL = {"p2": "p2", "p3nopa": "p3 sin pa", "p3": "p3 con pa"}
print("Entornos:", {LABEL[e]: meta.get(e) for e in ENVS})
MAIN = {"predict": "us_per_call", "backtesting": "ms_per_fold_no_fit", "fit": "ms",
        "create_train_X_y": "ms"}


def val(case, env, tree=tree_sel, metric=None):
    metric = metric or MAIN[case.split("|")[0]]
    v = best.get((case, metric, env, tree))
    return v


cases = sorted({c for c, _, _, _ in best}, key=lambda c: (
    list(MAIN).index(c.split("|")[0]), re.sub(r"n=\d+\|", "", c),
    int(re.search(r"n=(\d+)", c).group(1)),
))

if "--gain" in sys.argv:
    print(f"\n{'case':58s}" + "".join(f"{LABEL[e] + ' base':>15}{'new':>9}{'gain':>7}" for e in ENVS))
    for c in cases:
        row = f"{c:58s}"
        for e in ENVS:
            b, nw = val(c, e, "base"), val(c, e, "new")
            row += f"{b:15.1f}{nw:9.1f}{1 - nw / b:7.1%}" if b and nw else f"{'':31}"
        print(row)
    sys.exit()

# Noise: spread of the rounds (max / min of the per-process minimum).
ratios = sorted(max(v) / min(v) for v in spread.values() if len(v) > 1)
if ratios:
    print(f"Dispersion entre rondas (max/min por caso): mediana {ratios[len(ratios) // 2]:.3f}, "
          f"p90 {ratios[int(len(ratios) * 0.9)]:.3f}, max {ratios[-1]:.3f}")

unit = {"predict": "us por llamada", "backtesting": "ms por fold (sin el fit inicial)",
        "fit": "ms", "create_train_X_y": "ms"}
last_op = None
for c in cases:
    op = c.split("|")[0]
    if op != last_op:
        print(f"\n== {op} ({unit[op]}) ==")
        print(f"{'case':58s}{'p2':>10}{'p3 sin pa':>11}{'p3 con pa':>11}{'sin/p2':>8}{'con/p2':>8}")
        last_op = op
    v = [val(c, e) for e in ENVS]
    if None in v:
        continue
    print(f"{c:58s}{v[0]:10.1f}{v[1]:11.1f}{v[2]:11.1f}{v[1] / v[0]:8.2f}{v[2] / v[0]:8.2f}")

# Fixed cost and cost per series: t(n) = fixed + per_series * n, from n = 20 and 100.
print("\n== Coste fijo y coste por serie (ajuste con n=20 y n=100; n=3 y n=1000 como comprobacion) ==")
print(f"{'familia':50s}{'env':>10}{'fijo':>10}{'por serie':>11}{'pred n=3':>10}{'real n=3':>10}"
      f"{'pred n=1000':>12}{'real n=1000':>12}")
families = collections.OrderedDict()
for c in cases:
    fam = re.sub(r"n=\d+\|", "n=*|", c)
    families.setdefault(fam, {})[int(re.search(r"n=(\d+)", c).group(1))] = c
summary = {}
for fam, by_n in families.items():
    if 20 not in by_n or 100 not in by_n:
        continue
    for e in ENVS:
        t20, t100 = val(by_n[20], e), val(by_n[100], e)
        if t20 is None or t100 is None:
            continue
        per = (t100 - t20) / 80
        fixed = t20 - 20 * per
        summary[(fam, e)] = (fixed, per, t20, t100)
        extra = ""
        for n in (3, 1000):
            real = val(by_n[n], e) if n in by_n else None
            extra += f"{fixed + per * n:{10 if n == 3 else 12}.1f}" + (
                f"{real:{10 if n == 3 else 12}.1f}" if real is not None else f"{'':{10 if n == 3 else 12}}")
        print(f"{fam:50s}{LABEL[e]:>10}{fixed:10.1f}{per:11.2f}{extra}")

print("\n== Donde esta la perdida de pandas 3 respecto a pandas 2 (misma unidad que arriba) ==")
print(f"{'familia':50s}{'env':>10}{'d fijo':>9}{'d por serie':>12}{'d n=20':>9}{'% fijo n=20':>12}"
      f"{'d n=100':>9}{'% fijo n=100':>13}")
for fam in families:
    if (fam, "p2") not in summary:
        continue
    f2, s2, a2, b2 = summary[(fam, "p2")]
    for e in ("p3nopa", "p3"):
        if (fam, e) not in summary:
            continue
        f3, s3, a3, b3 = summary[(fam, e)]
        d_fixed, d_per = f3 - f2, s3 - s2
        d20, d100 = a3 - a2, b3 - b2
        print(f"{fam:50s}{LABEL[e]:>10}{d_fixed:9.1f}{d_per:12.2f}{d20:9.1f}"
              f"{(d_fixed / d20 if d20 else float('nan')):12.0%}{d100:9.1f}"
              f"{(d_fixed / d100 if d100 else float('nan')):13.0%}")
