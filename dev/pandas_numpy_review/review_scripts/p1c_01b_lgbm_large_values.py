"""
Option (c), item 2. LightGBM with the VALUES of the categories in the numpy
matrix and the index as categorical: size of the model when one category has
a large integer value (the bitset of every categorical split is as long as
the largest value on its left side). 2400 rows, 8 categories, 40 trees.
A value of 50,000,000 was measured once by hand: model text 125 MB, peak RSS
1020 MB; it is not in the loop to keep the script light.
"""
import _env
import resource
import numpy as np
from lightgbm import LGBMRegressor

rng = np.random.default_rng(0)
n = 2400
for top in (7, 100_000, 5_000_000):
    cats = np.array([0, 1, 2, 3, 4, 5, 6, top])
    pos = rng.integers(0, 8, n)
    X = np.column_stack([rng.normal(size=n), cats[pos].astype(float)])
    y = rng.normal(0, 30, 8)[pos] + rng.normal(size=n)
    est = LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0).fit(X, y, categorical_feature=[1])
    print(f"largest category value {top:>9}: model text {len(est.booster_.model_to_string()) / 1e6:6.2f} MB,"
          f" peak RSS of the process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:.0f} MB")
print(_env.versions())
