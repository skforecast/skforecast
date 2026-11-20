import timeit, numpy as np, pandas as pd
for idx in [pd.date_range("2020-01-01", periods=1000, freq="D"), pd.date_range("2020-01-01", periods=1000, freq="h", tz="Europe/Madrid"), pd.RangeIndex(0, 1000)]:
    ref = idx[[0, -1]]
    alts = {
        "idx[[0,-1]]": lambda: idx[[0, -1]],
        "idx.take([0,n-1])": lambda: idx.take([0, len(idx) - 1]),
        "idx[np.array([0,-1])]": lambda: idx[np.array([0, -1])],
        "idx[::n-1]": lambda: idx[:: len(idx) - 1],
        "idx.delete(slice(1,-1))": lambda: idx.delete(slice(1, -1)),
    }
    for k, f in alts.items():
        r = f()
        t = min(timeit.repeat(f, number=200, repeat=5)) / 200
        print(f"{type(idx).__name__:14s} {k:26s} {t*1e6:7.1f} us identical={r.identical(ref)} freq={getattr(r,'freq','-')} vs {getattr(ref,'freq','-')}")
print(pd.__version__)
