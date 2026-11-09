"""Mode: send_model, send_bytes, return_model, return_bytes."""
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
import cloudpickle
from xgboost import XGBRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 30)); yy = X[:, 0] + rng.normal(scale=0.1, size=5000)
def fits():
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); XGBRegressor(n_estimators=100, max_depth=6).fit(X, yy); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000)
nt = lambda: len(os.listdir("/proc/self/task"))

def use_model(m):
    if isinstance(m, bytes):
        m = pickle.loads(m)
    return float(m.predict(X[:5])[0])
def fit_model(i, as_bytes):
    m = XGBRegressor(n_estimators=20).fit(X[:1000], yy[:1000])
    return pickle.dumps(m) if as_bytes else m

mode = sys.argv[1]
before, th0 = fits(), nt()
fitted = XGBRegressor(n_estimators=20).fit(X[:1000], yy[:1000])
if mode == "send_model":
    Parallel(n_jobs=3)(delayed(use_model)(fitted) for _ in range(6))
elif mode == "send_bytes":
    b = cloudpickle.dumps(fitted)
    Parallel(n_jobs=3)(delayed(use_model)(b) for _ in range(6))
elif mode == "return_model":
    ms = Parallel(n_jobs=3)(delayed(fit_model)(i, False) for i in range(6))
elif mode == "return_bytes":
    ms = [pickle.loads(b) for b in Parallel(n_jobs=3)(delayed(fit_model)(i, True) for i in range(6))]
print(f"{mode}: XGB fit ms before {before}, after {fits()} | threads {th0} -> {nt()}")
