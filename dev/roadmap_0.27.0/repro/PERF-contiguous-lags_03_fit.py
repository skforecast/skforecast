"""
PERF-contiguous-lags, script 03: `forecaster.fit` end to end with the training
matrix as it is today (view of `sliding_window_view`, flag off) and C-contiguous
(prototype of the branch `perf/contiguous-lag-matrix`, flag on).

Both versions run in the same process, alternating the flag
`_CONTIGUOUS_X_TRAIN` of the four forecaster modules. With the flag off the code
path is the one of `0.27.x` (plus one `if`). Each measurement is the mean of
`inner` consecutive fits (at least ~0.2 s); `rounds` measurements per version,
alternating off/on; the table reports the minimum and the median.

The first fit of each version is also used to compare the predictions
(`predict(steps=24)`, `predict_proba` in the classifier): maximum absolute and
relative difference between the two versions.

Usage (from the root of the repository, one process at a time):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_03_fit.py recursive
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_03_fit.py direct
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_03_fit.py classifier
Results are appended to `PERF-contiguous-lags_03_fit.csv` next to this script.

F-ordered variant (the flag set to "F" instead of `True`), optionally with a
subset of the estimators; results in `PERF-contiguous-lags_03_fit_F.csv`:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_03_fit.py recursive F Ridge,LightGBM
"""

import gc
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor

import skforecast.direct._forecaster_direct as m_direct
import skforecast.direct._forecaster_direct_multivariate as m_dmv
import skforecast.recursive._forecaster_recursive as m_rec
import skforecast.recursive._forecaster_recursive_classifier as m_cls
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier

warnings.filterwarnings("ignore")

MODULES = (m_rec, m_cls, m_direct, m_dmv)
assert all(hasattr(m, "_CONTIGUOUS_X_TRAIN") for m in MODULES), "prototype missing"

MODE = sys.argv[2] if len(sys.argv) > 2 else "C"
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
ON_VALUE = "F" if MODE == "F" else True
OUT = Path(__file__).with_name(
    "PERF-contiguous-lags_03_fit_F.csv" if MODE == "F" else "PERF-contiguous-lags_03_fit.csv"
)


def set_contiguous(value: bool) -> None:
    # `True` means the version under test: C order, or F order with MODE "F"
    for m in MODULES:
        m._CONTIGUOUS_X_TRAIN = ON_VALUE if value else False


def optional_estimators():
    est = {}
    try:
        from lightgbm import LGBMRegressor
        est["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=100, random_state=123, verbose=-1
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        est["XGBoost"] = lambda: XGBRegressor(n_estimators=100, random_state=123)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        est["CatBoost"] = lambda: CatBoostRegressor(
            iterations=100, random_state=123, verbose=0, allow_writing_files=False
        )
    except ImportError:
        pass
    return est


REGRESSORS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": lambda: Lasso(alpha=0.1),
    "ElasticNet": lambda: ElasticNet(alpha=0.1),
    "KNeighbors": KNeighborsRegressor,
    "RandomForest(10)": lambda: RandomForestRegressor(
        n_estimators=10, max_depth=10, random_state=123
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=100, random_state=123
    ),
    **optional_estimators(),
}

SIZES = [1_000, 20_000, 200_000]
LAGS = [24, 168]


def make_y(n_rows: int, max_lag: int) -> pd.Series:
    rng = np.random.default_rng(123)
    n = n_rows + max_lag
    t = np.arange(n)
    noise = np.zeros(n)
    eps = rng.normal(scale=1.0, size=n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168) + noise
    return pd.Series(y, index=pd.RangeIndex(n), name="y")


def to_classes(y: pd.Series) -> pd.Series:
    q = y.quantile([1 / 3, 2 / 3]).to_numpy()
    return pd.Series(
        np.where(y < q[0], "low", np.where(y < q[1], "mid", "high")),
        index=y.index, name="y",
    )


def timed_fit(forecaster, y, inner: int) -> float:
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(inner):
        forecaster.fit(y=y)
    return (time.perf_counter() - t0) / inner


def predictions(forecaster, kind):
    if kind == "classifier":
        return forecaster.predict_proba(steps=24).to_numpy(dtype=float)
    if kind == "direct":
        return np.asarray(forecaster.predict(), dtype=float)
    return forecaster.predict(steps=24).to_numpy()


def run_case(kind, name, make_forecaster, y):
    results = {}
    preds = {}
    first = {}
    for flag in (False, True):
        set_contiguous(flag)
        f = make_forecaster()
        t0 = time.perf_counter()
        f.fit(y=y)
        first[flag] = time.perf_counter() - t0
        preds[flag] = predictions(f, kind)
        results[flag] = (f, [])
    t_first = max(first.values())
    inner = max(1, int(0.2 / max(t_first, 1e-6)))
    rounds = 7 if t_first < 0.5 else (5 if t_first < 3 else (3 if t_first < 15 else 2))
    for _ in range(rounds):
        for flag in (False, True):
            set_contiguous(flag)
            f, times = results[flag]
            times.append(timed_fit(f, y, inner))
    set_contiguous(True)
    off, on = results[False][1], results[True][1]
    diff = np.abs(preds[True] - preds[False])
    scale = np.maximum(np.abs(preds[False]), 1e-12)
    return dict(
        min_off_ms=min(off) * 1e3, min_on_ms=min(on) * 1e3,
        med_off_ms=np.median(off) * 1e3, med_on_ms=np.median(on) * 1e3,
        ratio_min=min(on) / min(off),
        max_abs_diff=float(diff.max()), max_rel_diff=float((diff / scale).max()),
        identical=bool(np.array_equal(preds[True], preds[False])),
        inner=inner, rounds=rounds,
    )


def main(kind: str) -> None:
    print(f"Python {platform.python_version()}, numpy {np.__version__}, pandas "
          f"{pd.__version__}, scikit-learn {sklearn.__version__}, "
          f"{platform.platform()}", flush=True)
    if kind == "recursive":
        estimators = REGRESSORS
    elif kind == "direct":
        # Direct: `n_jobs='auto'` gives 1 job with linear models and LightGBM
        # (the view reaches the estimator) and cpu_count - 1 processes with
        # the rest (joblib already sends a C copy).
        estimators = {k: v for k, v in REGRESSORS.items()
                      if k in ("LinearRegression", "Ridge", "LightGBM",
                               "HistGradientBoosting")}
    elif kind == "classifier":
        estimators = {"LogisticRegression": lambda: LogisticRegression(max_iter=1000)}
    else:
        raise ValueError(kind)
    if ONLY is not None:
        estimators = {k: v for k, v in estimators.items() if k in ONLY}

    for n_rows in SIZES:
        for max_lag in LAGS:
            y = make_y(n_rows, max_lag)
            if kind == "classifier":
                y = to_classes(y)
            for name, make_est in estimators.items():
                if name == "KNeighbors" and n_rows > 20_000:
                    continue  # in-sample predict of 200,000 rows takes minutes
                if kind == "recursive":
                    def make_forecaster():
                        return ForecasterRecursive(estimator=make_est(), lags=max_lag)
                elif kind == "direct":
                    def make_forecaster():
                        return ForecasterDirect(
                            estimator=make_est(), lags=max_lag, steps=5
                        )
                else:
                    def make_forecaster():
                        return ForecasterRecursiveClassifier(
                            estimator=make_est(), lags=max_lag
                        )
                t0 = time.perf_counter()
                res = run_case(kind, name, make_forecaster, y)
                row = dict(forecaster=kind, mode=MODE, estimator=name, rows=n_rows,
                           lags=max_lag, **res)
                pd.DataFrame([row]).to_csv(
                    OUT, mode="a", header=not OUT.exists(), index=False
                )
                print(
                    f"{kind:10s} {name:22s} {n_rows:>7,} x {max_lag:<3d} "
                    f"off {res['min_off_ms']:10.2f} ms  on {res['min_on_ms']:10.2f} ms "
                    f"ratio {res['ratio_min']:.3f}  max|diff| {res['max_abs_diff']:.2e} "
                    f"({time.perf_counter() - t0:.0f} s)",
                    flush=True,
                )


if __name__ == "__main__":
    main(sys.argv[1])
