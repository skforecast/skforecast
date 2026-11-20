import numpy as np, time
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
n = 200_000
Xf = np.column_stack([rng.normal(size=(n, 24)), (np.arange(n) % 2).astype(float)])
Xo = Xf.astype(object); Xo[:, -1] = Xo[:, -1].astype(bool)
y = rng.normal(size=n)
for E in (lambda: Ridge(), lambda: LGBMRegressor(verbose=-1, n_estimators=50, n_jobs=1)):
    for X, name in ((Xf, "float"), (Xo, "object")):
        t = time.perf_counter(); m = E().fit(X, y); dt = time.perf_counter() - t
        p = m.predict(Xf[:1000])
        print(type(m).__name__, name, f"{dt*1000:.0f} ms", p[:2])
    a = E().fit(Xf, y).predict(Xf[:1000]); b = E().fit(Xo, y).predict(Xf[:1000])
    print("  identical:", np.array_equal(a, b))
t=time.perf_counter(); Xo.astype(float); print("astype object->float", (time.perf_counter()-t)*1000, "ms; bytes object", Xo.nbytes + 0, "float", Xf.nbytes)
