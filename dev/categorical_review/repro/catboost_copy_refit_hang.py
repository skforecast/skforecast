"""
BUG-catboost-refit-categorical, CatBoost alone (no skforecast): a fitted model
with categorical features that is copied with `pickle` or `copy.deepcopy` and
fitted again with other data can block forever in `predict` (one core busy),
end with a segmentation fault or return a model different from a new one.
It is what each fold of a backtesting with refit and more than one job does:
it receives a serialized copy of the fitted forecaster and fits it again.

With `same` (the same object fitted twice) `predict` returns, but the model can
differ from a new one; with `clone` (`sklearn.base.clone` before the second
fit) it is identical to a new model.

Usage: python catboost_copy_refit_hang.py <rows> <categories> <seed> <how>
with <how> in pickle, deepcopy, same or clone. Run it with a time limit:

    timeout 40 python catboost_copy_refit_hang.py 300 6 0 pickle

Session of 2026-11-25 (CatBoost 1.2.10, Windows, Python 3.13.18): `pickle` and
`deepcopy` did not finish in 40 s with "300 6 0", "300 24 0", "1500 24 0" and
"1500 24 1" (8 of 8); `same` finished in the three tried, with a model
different from a new one (max abs diff 0.43, 0.70 and 0.13); `clone` was
identical to a new model in the three runs that finished ("300 6 0" twice and
"300 24 0", 8 to 10 s each). The result is erratic: repeating "300 6 0" later,
`pickle` ended once with a segmentation fault (exit code 139) and once blocked
for 120 s, and `deepcopy` finished with a model different from a new one (max
abs diff 2.16).
"""
import copy
import pickle
import sys
import time

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone

n, card, seed, how = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
total = n + 100
X = pd.DataFrame({
    "x": rng.normal(size=total),
    "c1": rng.integers(0, card, size=total),
    "c2": rng.integers(0, 3, size=total),
})
y = rng.normal(size=card)[X["c1"]] + 0.5 * X["c2"] + rng.normal(scale=0.3, size=total)
A, B = slice(0, n), slice(50, n + 50)
kw = dict(
    iterations=30, depth=4, random_seed=1, verbose=0, thread_count=1,
    allow_writing_files=False,
)
m = CatBoostRegressor(**kw).fit(X.iloc[A], y.iloc[A], cat_features=[1, 2])
if how == "pickle":
    m = pickle.loads(pickle.dumps(m))
elif how == "deepcopy":
    m = copy.deepcopy(m)
elif how == "clone":
    m = clone(m)
m.fit(X.iloc[B], y.iloc[B], cat_features=[1, 2])
print("second fit done, predicting...", flush=True)
t = time.perf_counter()
p = m.predict(X.iloc[B])
seconds = round(time.perf_counter() - t, 2)
new = CatBoostRegressor(**kw).fit(X.iloc[B], y.iloc[B], cat_features=[1, 2])
p_new = new.predict(X.iloc[B])
print(
    "catboost", catboost.__version__, "| predict seconds", seconds,
    "| identical to a new model:", np.array_equal(p, p_new),
    "| max abs diff", float(np.max(np.abs(p - p_new))), flush=True,
)
