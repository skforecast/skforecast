import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cases
from xgboost import XGBRegressor
from skforecast.direct import ForecasterDirect

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 30)); yy = X[:, 0] + rng.normal(scale=0.1, size=5000)
def fits():
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); XGBRegressor(n_estimators=100, max_depth=6).fit(X, yy); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000)
nt = lambda: len(os.listdir("/proc/self/task"))
before, th0 = fits(), nt()
y = cases.make_y(2000)
f = ForecasterDirect(estimator=XGBRegressor(n_estimators=50), lags=24, steps=12, n_jobs=int(sys.argv[1]))
f.fit(y)
print(f"Direct n_jobs={sys.argv[1]}: XGB fit ms before {before}, after {fits()} | threads {th0} -> {nt()}")
