"""Profiling helper: attribute inclusive time of calls made from skforecast code."""
import cProfile
import pstats
import time
from collections import defaultdict


def _cat(filename, name):
    f = filename.replace("\\", "/")
    if "/skforecast/" in f and "/site-packages/" not in f:
        return "skforecast"
    if "/pandas/" in f:
        return "pandas"
    if "/lightgbm/" in f:
        return "estimator(lgbm)"
    if "/sklearn/" in f:
        if name == "clone":
            return "sklearn.clone"
        if "/preprocessing" in f or "_set_output" in f or name in ("transform", "fit_transform", "inverse_transform"):
            return "sklearn.transformer(OrdinalEncoder/scaler)"
        return "estimator(sklearn)"
    if "/numpy/" in f:
        return "numpy"
    if "/joblib/" in f:
        return "skforecast"
    if f == "~":
        if "numpy" in name or "ndarray" in name:
            return "numpy"
        return "builtin"
    return "other:" + f.split("/")[-1]


def profile(fn, n=1):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        fn()
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    out = defaultdict(float)
    sk_self = 0.0
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        fname, line, name = func
        c = _cat(fname, name)
        if c == "skforecast":
            sk_self += tt
            continue
        for caller, (ccc, cnc, ctt, cct) in callers.items():
            if _cat(caller[0], caller[2]) == "skforecast":
                out[c] += cct
    out["skforecast(self)"] = sk_self
    return total / n, {k: v / n for k, v in out.items()}, st


def report(label, fn, n=1, top=0, sort="cumulative", filt=None):
    tot, cats, st = profile(fn, n)
    s = sum(cats.values())
    print(f"--- {label}: profiled total {tot*1e3:.2f} ms/call (attributed {s*1e3:.2f})")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        if v / tot > 0.005:
            print(f"   {k:28s} {v*1e3:9.3f} ms  {100*v/tot:5.1f}%")
    if top:
        st.sort_stats(sort).print_stats(*( [filt] if filt else []), top)
    return st


def wall(fn, n=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def callsites(fn, n=1, cats=("pandas",), top=15):
    """Top skforecast caller functions of calls into `cats` (inclusive time)."""
    tot, _, st = profile(fn, n)
    agg = defaultdict(float)
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        c = _cat(func[0], func[2])
        if c not in cats:
            continue
        for caller, (ccc, cnc, ctt, cct) in callers.items():
            if _cat(caller[0], caller[2]) == "skforecast" and "/joblib/" not in caller[0]:
                key = f"{caller[0].split('skforecast/')[-1]}:{caller[1]} {caller[2]} -> {func[2]}"
                agg[key] += cct
    print(f"--- call sites into {cats}: profiled total {tot/n*1e3:.2f} ms/call")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:top]:
        print(f"   {v/n*1e3:8.3f} ms  {100*v/tot:5.1f}%  {k}")
