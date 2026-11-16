"""Compare the before and after passes, cell by cell: python compare.py <folder>."""
import json
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1])  # folder with before/ and after/


def load(name):
    out = {}
    for line in open(D / name / "results.jsonl"):
        r = json.loads(line)
        s = r["spec"]
        if r.get("ok"):
            out[(s["cpus"], s["case"], s["est"], s["config"])] = r
    return out


def first(r):
    return r["calls"][0]["seconds"]


def nxt(r):
    return st.median(c["seconds"] for c in r["calls"][1:])


before, after = load("before"), load("after")
keys = sorted(set(before) & set(after), key=lambda k: (-k[0], k[1], k[2], k[3]))
print(f"{'cpus':>4} {'case':<15}{'est':<9}{'cfg':<5}{'nj':>3} | {'1st b':>6}{'1st a':>7}{'r':>5} | "
      f"{'nxt b':>6}{'nxt a':>7}{'r':>5} | fitAfter b/a | pickleMB")
for k in keys:
    b, a = before[k], after[k]
    nj = max(b["calls"][0]["parallel_n_jobs"] or [1])
    pb, pa = b.get("post", {}), a.get("post", {})
    print(f"{k[0]:>4} {k[1]:<15}{k[2]:<9}{k[3]:<5}{nj:>3} | {first(b):6.2f}{first(a):7.2f}"
          f"{first(a) / first(b):5.2f} | {nxt(b):6.2f}{nxt(a):7.2f}{nxt(a) / nxt(b):5.2f} | "
          f"{pb.get('fit_after') or 0:5.2f}/{pa.get('fit_after') or 0:5.2f} | {pb.get('pickle_mb') or 0:4.1f}")
    if b["calls"][0].get("signature") != a["calls"][0].get("signature"):
        print("   !! different predictions")
print(f"\nCells compared: {len(keys)}")
