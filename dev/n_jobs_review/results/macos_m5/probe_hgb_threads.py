"""HGB fit time with all the OpenMP threads and with 4 (section 17.4).

for i in 1 2 3 4 5; do python probe_hgb_threads.py; done
for i in 1 2 3 4 5; do OMP_NUM_THREADS=4 python probe_hgb_threads.py; done
"""
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(8000, 48))
y = X[:, 0] + rng.normal(size=8000)
t = []
for _ in range(3):
    t0 = time.perf_counter()
    HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(X, y)
    t.append(time.perf_counter() - t0)
print(" ".join(f"{v:.2f}" for v in t))
