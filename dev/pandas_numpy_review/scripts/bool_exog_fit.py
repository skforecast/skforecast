import sys, time, warnings, tracemalloc
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
n = 100_000
idx = pd.date_range("2000", periods=n, freq="h")
rng = np.random.default_rng(0)
y = pd.Series(np.cumsum(rng.normal(size=n)), index=idx)
base = pd.DataFrame({"temp": rng.normal(size=n), "hour": idx.hour.to_numpy()}, index=idx)
cases = {"float+int": base, "float+int+bool": base.assign(holiday=(idx.dayofweek >= 5)),
         "float+int+bool (as float)": base.assign(holiday=(idx.dayofweek >= 5).astype(float))}
for est in ("ridge", "lgbm"):
    for name, ex in cases.items():
        ts = []
        for r in range(3):
            e = Ridge() if est == "ridge" else LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1)
            f = ForecasterRecursive(e, lags=24)
            tracemalloc.start() if r == 0 else None
            t = time.perf_counter(); f.fit(y, exog=ex); ts.append(time.perf_counter() - t)
            if r == 0:
                peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
        t0 = time.perf_counter(); X, _ = f.create_train_X_y(y, exog=ex); tc = time.perf_counter() - t0
        print(f"pandas {pd.__version__} {est:5s} {name:28s} fit {min(ts)*1000:7.0f} ms  peak mem {peak/1e6:6.0f} MB  create_train_X_y {tc*1000:6.0f} ms")
