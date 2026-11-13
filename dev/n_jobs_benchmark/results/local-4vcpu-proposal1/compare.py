"""Compare the before and after passes of the benchmark, cell by cell."""
import json
import statistics as st
import sys
from pathlib import Path

S = Path(sys.argv[1])  # folder with before/ and after/


def load(name):
    out = {}
    for line in open(S / name / "results.jsonl"):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        s = r["spec"]
        out[(s["case"], s["est"], s["config"])] = r
    return out


def first(r):
    return r["calls"][0]["seconds"]


def nxt(r):
    calls = r["calls"][1:]
    return st.median(c["seconds"] for c in calls) if calls else None


def f(x):
    return "    -" if x is None else f"{x:6.2f}"


before, after = load("before"), load("after")
keys = sorted(set(before) & set(after))
print(f"{'case':<15}{'est':<9}{'cfg':<5}{'nj':>3} | {'1st bef':>7}{'1st aft':>8}{'ratio':>6} | "
      f"{'nxt bef':>7}{'nxt aft':>8}{'ratio':>6} | {'fitAfter b/a':>13} | pickleMB")
for k in keys:
    b, a = before[k], after[k]
    fb, fa = first(b), first(a)
    nb, na = nxt(b), nxt(a)
    pb, pa = b.get("post", {}), a.get("post", {})
    fit_b, fit_a = pb.get("fit_after"), pa.get("fit_after")
    nj = max(b["calls"][0]["parallel_n_jobs"] or [1])
    print(f"{k[0]:<15}{k[1]:<9}{k[2]:<5}{nj:>3} | {f(fb)}{f(fa):>8}{fa / fb:6.2f} | "
          f"{f(nb)}{f(na):>8}{(na / nb if nb and na else float('nan')):6.2f} | "
          f"{f(fit_b)}/{f(fit_a)} | {pb.get('pickle_mb') or 0:6.1f}")
    if b["calls"][0].get("signature") != a["calls"][0].get("signature"):
        print("   !! different predictions signature")
print(f"\nCells compared: {len(keys)}; only before: {len(set(before) - set(after))}; "
      f"only after: {len(set(after) - set(before))}")
