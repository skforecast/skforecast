"""
PERF-create-predict-X-astype: `create_predict_X` of `ForecasterRecursive`
(also Classifier, Direct and DirectMultiVariate) restores the exog dtypes with
`compat.astype(X_predict, X_predict_dtypes)`, without `changed_only`, so every
column is cast even when none changes. A backtesting with
`return_predictors=True` calls it once per fold.

Variants, applied by replacing the `compat` module seen by
`skforecast.recursive._forecaster_recursive` (the package is not changed):

- `current`: the code of 0.28.x.
- `step 1`: `changed_only=True` in `create_predict_X`. pandas `astype` with a
  dict still walks every column of the frame (a shallow copy of each one that
  does not change) and rebuilds it with `concat`, one block per column.
- `steps 1+2`: only the columns whose dtype changes are cast, assigned on a
  shallow copy of the frame.

Every variant must give the same predictors (`assert_frame_equal`: values,
dtypes, index and column order).

Usage: python predict_x_astype.py  (run it with pandas 2 and with pandas 3)

Result (2026-11-21, shared 4-CPU machine, best of 3; Python 3.12.3, numpy
2.5.5, scikit-learn 1.9.1, p2 = pandas 2.3.3, p3 = pandas 3.1.1), 200 folds
without refit unless said otherwise:

| Case | pandas | No predictors | current | step 1 | steps 1+2 |
|---|---|---|---|---|---|
| LinearRegression, 10 float exog | p2 | 0.090 s | 0.595 s | 0.165 s | 0.163 s |
| LinearRegression, 10 float exog | p3 | 0.072 s | 0.732 s | 0.195 s | 0.193 s |
| LinearRegression, 8 float + int + category | p2 | 0.251 s | 0.923 s | 0.829 s | 0.588 s |
| LinearRegression, 8 float + int + category | p3 | 0.368 s | 1.234 s | 1.229 s | 0.787 s |
| LightGBM, 10 float exog | p2 | 0.313 s | 0.788 s | 0.513 s | 0.526 s |
| LightGBM, 10 float exog | p3 | 0.339 s | 1.092 s | 0.566 s | 0.566 s |
| LightGBM, 10 float exog, refit, 20 folds | p2 | 4.241 s | 4.401 s | 4.325 s | 4.388 s |
| LightGBM, 10 float exog, refit, 20 folds | p3 | 4.076 s | 4.227 s | 4.218 s | 4.288 s |

A cProfile of 300 `create_predict_X` calls (10 float exog, p3) puts about 90 %
of their time in `astype`.
"""
import time
import types
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from _common import versions
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

import skforecast.recursive._forecaster_recursive as fr
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive
from skforecast.utils import _pandas_compat as compat

warnings.filterwarnings("ignore")
print(versions())


def astype_assign(obj, dtypes, changed_only=False):
    """Cast only the columns whose dtype changes, on a shallow copy."""
    current = obj.dtypes
    changed = {col: dt for col, dt in dtypes.items() if current[col] != dt}
    if not changed:
        return obj
    out = obj.copy(deep=False)
    for col, dt in changed.items():
        out[col] = out[col].astype(dt)
    return out


def shim(astype):
    return types.SimpleNamespace(
        concat=compat.concat, infer_freq=compat.infer_freq,
        PANDAS_GE_3=compat.PANDAS_GE_3, astype=astype,
    )


VARIANTS = {
    "current": compat,
    "step 1": shim(lambda o, d, changed_only=False: compat.astype(o, d, changed_only=True)),
    "steps 1+2": shim(astype_assign),
}

rng = np.random.default_rng(0)
n = 3000
idx = pd.date_range("2020-01-01", periods=n, freq="h")
y = pd.Series(rng.normal(size=n).cumsum(), index=idx, name="y")
exog_float = pd.DataFrame(
    rng.normal(size=(n, 10)), index=idx, columns=[f"x{i}" for i in range(10)]
)
exog_mixed = exog_float.iloc[:, :8].assign(
    i1=rng.integers(0, 5, n), cat=pd.Categorical(rng.choice(list("abc"), n))
)
CASES = [
    ("LinearRegression, 10 float exog", LinearRegression, exog_float, False, 5),
    ("LinearRegression, 8 float + int + category", LinearRegression, exog_mixed, False, 5),
    ("LightGBM, 10 float exog", lambda: LGBMRegressor(n_estimators=100, verbose=-1, n_jobs=1), exog_float, False, 5),
    ("LightGBM, 10 float exog, refit", lambda: LGBMRegressor(n_estimators=100, verbose=-1, n_jobs=1), exog_float, True, 50),
]


def run(make_estimator, exog, refit, steps, module, return_predictors):
    cv = TimeSeriesFold(steps=steps, initial_train_size=2000, refit=refit)
    best = np.inf
    with patch.object(fr, "compat", module):
        for _ in range(3):
            forecaster = ForecasterRecursive(
                make_estimator(), lags=24, categorical_features="auto"
            )
            start = time.perf_counter()
            _, pred = backtesting_forecaster(
                forecaster, y=y, exog=exog, cv=cv, metric="mean_absolute_error",
                return_predictors=return_predictors, n_jobs=1, show_progress=False,
            )
            best = min(best, time.perf_counter() - start)
    return best, pred


for label, make_estimator, exog, refit, steps in CASES:
    base, _ = run(make_estimator, exog, refit, steps, compat, False)
    times, ref = {}, None
    for name, module in VARIANTS.items():
        times[name], pred = run(make_estimator, exog, refit, steps, module, True)
        if ref is None:
            ref = pred
        pd.testing.assert_frame_equal(ref, pred)
    row = " | ".join(f"{k} {v:.3f} s" for k, v in times.items())
    print(f"{label} ({1000 // steps} folds): no predictors {base:.3f} s | {row} | same predictors")
