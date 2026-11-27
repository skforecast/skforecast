import re, collections, statistics
rows = collections.defaultdict(list)
for line in open("logs/perf_mem.log", encoding="utf-8", errors="replace"):
    if not line.startswith("MEM"):
        continue
    p = line.split()
    tree, est, sc, op = p[1], p[2], p[3], p[4]
    peak = float(p[p.index("peak") + 1]); delta = float(p[p.index("peak-rss_before") + 1])
    rows[(est, sc, op, tree)].append((peak, delta))
keys = sorted({k[:3] for k in rows})
print("estimator scenario op | base peak-rss_before MB (2 runs) | branch (2 runs) | diff of means MB")
for k in keys:
    b = [d for _, d in rows.get(k + ("base",), [])]
    n = [d for _, d in rows.get(k + ("branch",), [])]
    if not b or not n:
        print(k, "missing", b, n); continue
    print(f"{k[0]:8} {k[1]:6} {k[2]:6} | {b} | {n} | {statistics.mean(n) - statistics.mean(b):+7.1f}")
