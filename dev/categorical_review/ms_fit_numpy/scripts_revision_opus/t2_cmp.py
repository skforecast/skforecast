import ast, sys
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.startswith(("df ", "dict ")):
            i = line.index("{")
            d[tuple(line[:i].split())] = ast.literal_eval(line[i:])
        elif line.startswith("DMV"):
            k, v = line.split(" [", 1) if " [" in line else (line, "")
            d[("DMV",) + tuple(k.split()[1:])] = v.strip()
    return d
a, b = load("t2_base.txt"), load("t2_branch.txt")
def f(x): return f"{x:.6f}" if isinstance(x, float) else str(x)[:40]
for k in a:
    va, vb = a[k], b.get(k)
    if k[0] == "DMV":
        print(" ".join(k), "same" if va == vb else f"DIFF base={va} branch={vb}")
        continue
    gap_a = (va["grid"] - va["bt"]) if isinstance(va["grid"], float) and isinstance(va["bt"], float) else None
    gap_b = (vb["grid"] - vb["bt"]) if isinstance(vb["grid"], float) and isinstance(vb["bt"], float) else None
    flag = ""
    if gap_b is not None and abs(gap_b) > 1e-9: flag += " BRANCH_OSA!=BT"
    if gap_a is not None and abs(gap_a) > 1e-9: flag += " base_osa!=bt"
    if vb["grid"] != vb["bayes"]: flag += " BRANCH grid!=bayes"
    same = va == vb
    print(f"{' '.join(k):45s} base grid={f(va['grid'])} bt={f(va['bt'])} | branch grid={f(vb['grid'])} bayes={f(vb['bayes'])} bt={f(vb['bt'])} {'' if same else 'CHANGED'}{flag}")
