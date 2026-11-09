"""
PERF-contiguous-lags, script 06: `estimator.fit` and `estimator.predict` alone
(without the forecaster) on the three layouts the forecasters produce today:
the reversed view of `sliding_window_view` (lags as an int), an F-ordered
array (lags as a non-contiguous list, MultiSeries) and a C-contiguous copy.

It separates the two calls that `ForecasterRecursive.fit` makes on the training
matrix (the `fit` and the in-sample `predict` for the residuals) to see which one
changes with the layout. Many rounds alternating the three layouts; minimum and
median per layout.

Run from the root of the repository:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_06_estimator_fit_predict.py
"""

import gc
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

warnings.filterwarnings("ignore")

ESTIMATORS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RandomForest(10)": lambda: RandomForestRegressor(
        n_estimators=10, max_depth=10, random_state=123
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=100, random_state=123
    ),
}
try:
    from lightgbm import LGBMRegressor
    ESTIMATORS["LightGBM"] = lambda: LGBMRegressor(
        n_estimators=100, random_state=123, verbose=-1
    )
except ImportError:
    pass
try:
    from xgboost import XGBRegressor
    ESTIMATORS["XGBoost"] = lambda: XGBRegressor(n_estimators=100, random_state=123)
except ImportError:
    pass


def matrices(n_rows, k):
    rng = np.random.default_rng(123)
    y = rng.normal(size=n_rows + k).cumsum()
    view = np.lib.stride_tricks.sliding_window_view(y, k + 1)[:, :k][:, ::-1]
    return {
        "view": view,
        "F": np.asfortranarray(view),
        "C": np.ascontiguousarray(view),
    }, y[k:]


def t(fn):
    gc.collect()
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


print(f"Python {platform.python_version()}, numpy {np.__version__}, "
      f"scikit-learn {sklearn.__version__}, {platform.platform()}")
print("| rows x lags | estimator | call | view (ms) | F (ms) | C (ms) | C / view |")
print("|---|---|---|---|---|---|---|")
for n_rows, k in [(20_000, 24), (20_000, 168), (200_000, 168)]:
    Xs, y = matrices(n_rows, k)
    for name, make in ESTIMATORS.items():
        if name == "RandomForest(10)" and n_rows > 20_000:
            continue
        fitted = {lay: make().fit(X, y) for lay, X in Xs.items()}
        t_one = t(lambda: make().fit(Xs["C"], y))
        rounds = 9 if t_one < 0.5 else (5 if t_one < 3 else 3)
        res = {(c, lay): [] for c in ("fit", "predict") for lay in Xs}
        for _ in range(rounds):
            for lay, X in Xs.items():
                res[("fit", lay)].append(t(lambda: make().fit(X, y)))
                res[("predict", lay)].append(t(lambda: fitted[lay].predict(X)))
        for call in ("fit", "predict"):
            mins = {lay: min(res[(call, lay)]) * 1e3 for lay in Xs}
            meds = {lay: np.median(res[(call, lay)]) * 1e3 for lay in Xs}
            print(
                f"| {n_rows:,} x {k} | {name} | {call} | "
                f"{mins['view']:.2f} ({meds['view']:.2f}) | "
                f"{mins['F']:.2f} ({meds['F']:.2f}) | "
                f"{mins['C']:.2f} ({meds['C']:.2f}) | "
                f"{mins['C'] / mins['view']:.3f} |",
                flush=True,
            )
