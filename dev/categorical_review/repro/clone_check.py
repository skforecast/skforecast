"""
BUG-catboost-refit-categorical, aside: `sklearn.base.clone` of a
`CatBoostRegressor` with `cat_features` passed as a list to the constructor
raises `RuntimeError: Cannot clone object` (a tuple works), so that estimator
cannot be passed to a forecaster, which clones it.

Prints ok or the error of `clone` for a new estimator, a fitted one and three
ways of passing `cat_features` to the constructor, and of creating
`ForecasterRecursive`, `ForecasterRecursiveMultiSeries` and `ForecasterDirect`
with it.

Usage: python clone_check.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import sklearn
import catboost
from catboost import CatBoostRegressor
from sklearn.base import clone

warnings.simplefilter("ignore")
print("catboost", catboost.__version__, "scikit-learn", sklearn.__version__)
rng = np.random.default_rng(1)
X = pd.DataFrame({"c": rng.integers(0, 4, 60), "x": rng.normal(size=60)})
y = rng.normal(size=60)


def cb(**kw):
    return CatBoostRegressor(
        iterations=5, verbose=0, thread_count=1, allow_writing_files=False, **kw
    )


def check(label, fn):
    try:
        fn()
        print(f"{label:<58} ok")
    except Exception as e:
        print(f"{label:<58} {type(e).__name__}: {str(e)[:60]}")


check("clone, new", lambda: clone(cb()))
check("clone, fitted with cat_features in fit", lambda: clone(cb().fit(X, y, cat_features=[0])))
check("clone, new with cat_features in the constructor", lambda: clone(cb(cat_features=[0])))
check("clone, new with cat_features as tuple", lambda: clone(cb(cat_features=(0,))))
check("clone, new with cat_features by name", lambda: clone(cb(cat_features=["c"])))

from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect

check("ForecasterRecursive(cat_features in the constructor)", lambda: ForecasterRecursive(estimator=cb(cat_features=[0]), lags=3))
check("ForecasterRecursiveMultiSeries(...)", lambda: ForecasterRecursiveMultiSeries(estimator=cb(cat_features=[0]), lags=3))
check("ForecasterDirect(...)", lambda: ForecasterDirect(estimator=cb(cat_features=[0]), lags=3, steps=2))
