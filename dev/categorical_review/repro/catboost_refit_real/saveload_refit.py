"""
BUG-catboost-refit-categorical with real data: the retraining flow. Fit, save
with `save_forecaster`, load, fit again with the data shifted one week and
predict, against a new forecaster fitted with the second window.

In the session of 2026-11-25 the four combinations finished and were identical
to the new forecaster. It is not a guarantee: the loaded forecaster is a copy
of a fitted CatBoost model, the case that blocks in ../catboost_copy_refit_hang.py.

Usage: python saveload_refit.py <same|saveload> <recursive|multi>
Run it with a time limit (for example `timeout 240 python saveload_refit.py ...`),
after `python run_variant.py prepare`.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import skforecast
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.utils import load_forecaster, save_forecaster

import run_variant as rv

warnings.simplefilter("ignore")
how, kind = sys.argv[1], sys.argv[2]
print(skforecast.__version__, skforecast.__file__, flush=True)
y, exog, initial = rv.data_bikeS(rv.GUIDE)
A, B, P = slice(0, initial - 168), slice(168, initial), slice(initial, initial + 24)
if kind == "recursive":
    new = ForecasterRecursive(estimator=rv.cb(iterations=100), lags=24)
    f = ForecasterRecursive(estimator=rv.cb(iterations=100), lags=24)
    data, key = y, "y"
else:
    data = pd.DataFrame({"a": y, "b": y.rolling(3).mean().bfill()})
    new = ForecasterRecursiveMultiSeries(estimator=rv.cb(iterations=100), lags=24)
    f = ForecasterRecursiveMultiSeries(estimator=rv.cb(iterations=100), lags=24)
    key = "series"


def values(p):
    return p["pred"].to_numpy() if isinstance(p, pd.DataFrame) else p.to_numpy()


new.fit(**{key: data.iloc[B]}, exog=exog.iloc[B])
p_new = new.predict(steps=24, exog=exog.iloc[P])
f.fit(**{key: data.iloc[A]}, exog=exog.iloc[A])
if how == "saveload":
    path = rv.WORK / f"saveload_{kind}.joblib"
    save_forecaster(f, str(path), verbose=False)
    f = load_forecaster(str(path), verbose=False)
t = time.perf_counter()
f.fit(**{key: data.iloc[B]}, exog=exog.iloc[B])
print("second fit done", round(time.perf_counter() - t, 1), "s; predicting...", flush=True)
t = time.perf_counter()
p = f.predict(steps=24, exog=exog.iloc[P])
print(
    how, kind, "predict", round(time.perf_counter() - t, 2),
    "s | identical to a new forecaster:", np.array_equal(values(p), values(p_new)),
    "| max abs diff", float(np.max(np.abs(values(p) - values(p_new)))), flush=True,
)
