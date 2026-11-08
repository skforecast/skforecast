"""
PERF-contiguous-lags, script 07: do the in-sample residuals (computed with
`estimator.predict(X_train)` inside `fit`) and the prediction intervals change
when the training matrix is C-contiguous (prototype) instead of the view?

Run from the root of the repository:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_07_residuals.py
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

import skforecast.direct._forecaster_direct as m_direct
import skforecast.recursive._forecaster_recursive as m_rec
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive

warnings.filterwarnings("ignore")

rng = np.random.default_rng(123)
n = 20_000
t = np.arange(n)
y = pd.Series(
    50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(size=n).cumsum() * 0.1,
    index=pd.date_range("2000-01-01", periods=n, freq="h"), name="y",
)

print("| forecaster | estimator | lags | max abs diff in-sample residuals "
      "| max abs diff predict_interval (bootstrapping) | max abs diff conformal |")
print("|---|---|---|---|---|---|")
for Forecaster, module in [(ForecasterRecursive, m_rec), (ForecasterDirect, m_direct)]:
    for name, make in [
        ("LinearRegression", LinearRegression), ("Ridge", Ridge),
        ("Lasso", lambda: Lasso(alpha=0.1)),
        ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(
            max_iter=50, random_state=1)),
        ("LightGBM", lambda: LGBMRegressor(n_estimators=50, random_state=1,
                                           verbose=-1)),
    ]:
        for lags in (24, 168):
            out = {}
            for flag in (False, True):
                module._CONTIGUOUS_X_TRAIN = flag
                kw = dict(steps=12) if Forecaster is ForecasterDirect else {}
                f = Forecaster(estimator=make(), lags=lags, **kw)
                f.fit(y=y, store_in_sample_residuals=True)
                steps = None if Forecaster is ForecasterDirect else 12
                out[flag] = (
                    np.asarray(f.in_sample_residuals_),
                    f.predict_interval(steps=steps, method="bootstrapping",
                                       n_boot=100, random_state=1).to_numpy(),
                    f.predict_interval(steps=steps, method="conformal").to_numpy(),
                )
            module._CONTIGUOUS_X_TRAIN = True
            d = [float(np.abs(a - b).max()) for a, b in zip(out[False], out[True])]
            print(f"| {Forecaster.__name__} | {name} | {lags} | {d[0]:.2e} | "
                  f"{d[1]:.2e} | {d[2]:.2e} |", flush=True)
