"""
PERF-contiguous-lags, script 09: `forecaster.fit` end to end with more linear
and kernel estimators, to decide the list of option D with data.

Same method as script 03 (flag off = 0.27.x, flag on = always copy to C order),
with the helpers of `PERF-contiguous-lags_common.py`: it waits until the
1-minute load is below 4 before each case and records the load and the power
source. An estimator whose first fit takes more than `LIMIT_S` seconds is not
measured in the following (larger) sizes.

Direct uses `steps=5` and `n_jobs='auto'`: 1 job with subclasses of
`LinearModel` (and LightGBM), `cpu_count() - 1` processes with the rest (joblib
already sends a C copy to the workers).

Usage (from the root of the repository, prototype applied, one at a time):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_09_more_estimators.py recursive
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_09_more_estimators.py direct
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_09_more_estimators.py classifier
Optional second argument: comma-separated estimator names. Results are appended
to `PERF-contiguous-lags_09_more_estimators.csv`.
"""

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    Lars,
    LassoLars,
    LinearRegression,
    PoissonRegressor,
    QuantileRegressor,
    RidgeClassifier,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR

from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.utils import select_n_jobs_fit_forecaster

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).with_name("PERF-contiguous-lags_common.py")
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

OUT = Path(__file__).with_name("PERF-contiguous-lags_09_more_estimators.csv")
SIZES = [(20_000, 24), (20_000, 168), (200_000, 168)]
LIMIT_S = 30

REGRESSORS = {
    "BayesianRidge": BayesianRidge,
    "ARDRegression": ARDRegression,
    "HuberRegressor": HuberRegressor,
    "SGDRegressor": lambda: SGDRegressor(random_state=123),
    "PoissonRegressor": PoissonRegressor,
    "TweedieRegressor": TweedieRegressor,
    "LinearSVR": lambda: LinearSVR(random_state=123),
    "KernelRidge": KernelRidge,
    "Lars": Lars,
    "LassoLars": lambda: LassoLars(alpha=0.1),
    "QuantileRegressor": lambda: QuantileRegressor(alpha=0.0, solver="highs"),
    "RidgeCV": RidgeCV,
    "LinearRegression(positive=True)": lambda: LinearRegression(positive=True),
}
CLASSIFIERS = {
    "RidgeClassifier": RidgeClassifier,
    "LinearSVC": lambda: LinearSVC(random_state=123),
    "SGDClassifier": lambda: SGDClassifier(random_state=123),
}
# KernelRidge builds an n x n kernel (3.2 GB with 20,000 rows) and solves it:
# only 20,000 x 24 and only in Recursive (Direct would fit it in up to
# cpu_count() - 1 processes at the same time).
# QuantileRegressor (HiGHS) takes 13 s per fit with 20,000 x 24 and does not
# change (ratio 1.00): the larger sizes would take minutes per fit.
ONLY_SIZES = {"KernelRidge": [(20_000, 24)], "QuantileRegressor": [(20_000, 24)]}
SKIP_KIND = {"KernelRidge": {"direct"}}


def main(kind, only=None):
    print(common.environment(), flush=True)
    estimators = CLASSIFIERS if kind == "classifier" else REGRESSORS
    if only:
        estimators = {k: v for k, v in estimators.items() if k in only}
    for name, make_est in estimators.items():
        if kind in SKIP_KIND.get(name, set()):
            continue
        for n_rows, lags in ONLY_SIZES.get(name, SIZES):
            y = common.make_y(n_rows, lags)
            if kind == "classifier":
                y = common.to_classes(y)

                def make_forecaster():
                    return ForecasterRecursiveClassifier(estimator=make_est(), lags=lags)

                def predict(f):
                    # RidgeClassifier and LinearSVC have no `predict_proba`:
                    # the predicted classes are compared (as codes).
                    if hasattr(f.estimator, "predict_proba"):
                        return f.predict_proba(steps=24)
                    labels = np.asarray(f.predict(steps=24)).ravel()
                    return pd.Categorical(labels, categories=["high", "low", "mid"]).codes
            elif kind == "direct":
                def make_forecaster():
                    return ForecasterDirect(estimator=make_est(), lags=lags, steps=5)

                def predict(f):
                    return f.predict()
            else:
                def make_forecaster():
                    return ForecasterRecursive(estimator=make_est(), lags=lags)

                def predict(f):
                    return f.predict(steps=24)

            def fit(f):
                f.fit(y=y)

            load_before = common.wait_for_load()
            t0 = time.perf_counter()
            res = common.run_case(make_forecaster, fit, predict, on_value=True)
            n_jobs = (
                select_n_jobs_fit_forecaster("ForecasterDirect", make_est())
                if kind == "direct" else 1
            )
            row = dict(
                forecaster=kind, estimator=name, rows=n_rows, lags=lags,
                n_jobs=n_jobs, **res, load_before=load_before,
                power=common.power_source(),
            )
            pd.DataFrame([row]).to_csv(
                OUT, mode="a", header=not OUT.exists(), index=False
            )
            print(
                f"{kind:10s} {name:32s} {n_rows:>7,} x {lags:<3d} n_jobs {n_jobs} "
                f"off {res['off_ms']:10.2f} ms  on {res['on_ms']:10.2f} ms "
                f"ratio {res['ratio']:.3f}  max|diff| {res['max_abs_diff']:.1e} "
                f"load {load_before:.1f} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
            if res["t_first_s"] > LIMIT_S:
                print(f"{name}: first fit {res['t_first_s']:.0f} s > {LIMIT_S} s, "
                      "larger sizes skipped", flush=True)
                break


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
