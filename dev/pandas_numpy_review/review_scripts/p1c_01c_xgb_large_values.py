"""
Option (c), item 2. XGBoost with the VALUES of the categories in the numpy
matrix and `feature_types='c'`: memory of `fit` against the largest category
value (400 rows, 4 categories, 2 trees). Each value runs in a child process
with the address space limited to 4 GB, because with 16,777,215 (the largest
value XGBoost accepts is below 2**24) the kernel killed the process for lack
of memory in a machine with 16 GB (seen once, not repeated).
"""
import _env
import subprocess
import sys

CHILD = r"""
import resource, sys
import numpy as np
from xgboost import XGBRegressor
resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
top = int(sys.argv[1])
rng = np.random.default_rng(0)
c = np.array([0, 1, 2, top])[rng.integers(0, 4, 400)].astype(float)
X = np.column_stack([rng.normal(size=400), c])
try:
    XGBRegressor(n_estimators=2, n_jobs=1, feature_types=["q", "c"], enable_categorical=True).fit(X, rng.normal(size=400))
    print(f"fit ok, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:.0f} MB")
except BaseException as e:
    print(type(e).__name__ + ": " + " ".join(str(e).split())[:200])
"""
for top in (3, 10_000, 100_000, 1_000_000, 3_000_000, -1):
    out = subprocess.run([sys.executable, "-c", CHILD, str(top)], capture_output=True, text=True)
    print(f"largest category value {top:>9}: {(out.stdout.strip() or 'no output, return code ' + str(out.returncode))}")
print(_env.versions())
