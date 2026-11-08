"""Does booster_.predict (skforecast fast path) respect LGBMRegressor(n_jobs=1)?

Times 3000 single-row predictions (the recursive loop pattern) with:
- estimator.predict (sklearn API, passes num_threads from n_jobs)
- booster_.predict (what _build_predict_function does)
- booster_.predict(num_threads=1)
and the same for XGBRegressor(n_jobs=1) inplace_predict.
Prints the thread count of the process after each variant.
"""

import platform
import time

import lightgbm
import numpy as np
import psutil
import xgboost
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

proc = psutil.Process()
rng = np.random.default_rng(0)
X = rng.normal(size=(9000, 60))
y = X[:, 0] + rng.normal(size=9000)
row = X[:1]
N = 3000
print(platform.platform(), "lightgbm", lightgbm.__version__, "xgboost", xgboost.__version__)


def bench(tag, fn):
    fn(row)
    t0 = time.perf_counter()
    for _ in range(N):
        fn(row)
    dt = time.perf_counter() - t0
    print(f"  {tag:<40} {dt:6.3f} s ({1e3 * dt / N:.3f} ms/call), threads {proc.num_threads()}")


m = LGBMRegressor(n_estimators=100, n_jobs=1, verbose=-1, random_state=123).fit(X, y)
print("LGBMRegressor(n_jobs=1)")
bench("estimator.predict", m.predict)
bench("booster_.predict(num_threads=1)", lambda r: m.booster_.predict(r, num_threads=1))
bench("booster_.predict  (skforecast)", m.booster_.predict)
bench("estimator.predict, after booster", m.predict)

x = XGBRegressor(n_estimators=100, n_jobs=1, random_state=123).fit(X, y)
b = x.get_booster()
print("XGBRegressor(n_jobs=1)")
bench("estimator.predict", x.predict)
bench("inplace_predict  (skforecast)", b.inplace_predict)
