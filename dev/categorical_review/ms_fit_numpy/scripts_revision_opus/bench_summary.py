import json, collections, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "bench_results.jsonl")]
d = collections.defaultdict(lambda: collections.defaultdict(list))
mem = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    k = (r["est"], r["ex"], r["enc"])
    for op in ["fit", "pred", "bt", "osa"]:
        d[(k, op)][r["variant"]].extend(r["r"][op])
    for m in ["mem_fit_MB", "mem_osa_MB", "peak_ws_MB"]:
        mem[(k, m)][r["variant"]].append(r["r"][m])
print(f"{'config':32s} {'op':5s} {'base min/med/max (s)':26s} {'branch min/med/max (s)':26s} ratio(med)")
for (k, op), v in d.items():
    b, n = sorted(v["base"]), sorted(v["branch"])
    if not b or not n: continue
    print(f"{'/'.join(k):32s} {op:5s} {b[0]:.3f}/{statistics.median(b):.3f}/{b[-1]:.3f}{'':6s} "
          f"{n[0]:.3f}/{statistics.median(n):.3f}/{n[-1]:.3f}{'':6s} {statistics.median(n)/statistics.median(b):.3f}")
print()
for (k, m), v in mem.items():
    print(f"{'/'.join(k):32s} {m:11s} base={v['base']} branch={v['branch']}")
r0 = rows[0]["r"]; print("\nversions:", {x: r0[x] for x in ["py", "np", "pd", "skl"]})
