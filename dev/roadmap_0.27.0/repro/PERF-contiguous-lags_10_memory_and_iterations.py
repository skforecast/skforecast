"""
PERF-contiguous-lags, script 10, two parts:

`memory`: peak `tracemalloc` memory of `forecaster.fit` (Recursive and Direct
with `steps=5`) with the flag off (0.27.x) and on (always copy), for the
estimators given as argument, with 20,000 x 168 and 200,000 x 168. The extra
peak is also given as a share of the training matrix (criterion of option D:
at most 15 %).

`iterations`: why the predictions of HuberRegressor, PoissonRegressor and
TweedieRegressor change. Their solver (lbfgs) may stop at `max_iter` (100)
without converging on this data (unscaled lags around 50). For each version it
reports the iterations, whether it converged (no `ConvergenceWarning`), the time
per iteration (which separates the effect of the layout from a different number
of iterations) and the difference of `predict(steps=24)`; also with
`Pipeline(StandardScaler, estimator)`, where the scaled data usually converge.

Usage (from the root of the repository, prototype applied):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_10_memory_and_iterations.py memory Ridge,Lars
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_10_memory_and_iterations.py iterations
"""

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    Lars,
    LassoLars,
    LinearRegression,
    PoissonRegressor,
    Ridge,
    RidgeClassifier,
    RidgeCV,
    TweedieRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).with_name("PERF-contiguous-lags_common.py")
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

ESTIMATORS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RidgeCV": RidgeCV,
    "BayesianRidge": BayesianRidge,
    "ARDRegression": ARDRegression,
    "Lars": Lars,
    "LassoLars": lambda: LassoLars(alpha=0.1),
    "LinearRegression(positive=True)": lambda: LinearRegression(positive=True),
    "Pipeline(StandardScaler, Ridge)": lambda: make_pipeline(StandardScaler(), Ridge()),
    "Pipeline(StandardScaler, LinearRegression)": lambda: make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "RidgeClassifier": RidgeClassifier,
    "HuberRegressor": HuberRegressor,
    "PoissonRegressor": PoissonRegressor,
    "TweedieRegressor": TweedieRegressor,
}
# Direct fits these in joblib processes (not subclasses of `LinearModel`), which
# `tracemalloc` does not see: only Recursive.
ONLY_RECURSIVE = {"PoissonRegressor", "TweedieRegressor"}


def memory(names):
    print(common.environment(), flush=True)
    print("| forecaster | estimator | rows x lags | matrix (MB) | peak 0.27.x (MB) "
          "| peak copy (MB) | extra / matrix |")
    print("|---|---|---|---|---|---|---|")
    for name in names:
        make = ESTIMATORS[name]
        kinds = ["classifier"] if name == "RidgeClassifier" else ["recursive", "direct"]
        if name in ONLY_RECURSIVE:
            kinds = ["recursive"]
        for kind in kinds:
            for n_rows, lags in [(20_000, 168), (200_000, 168)]:
                y = common.make_y(n_rows, lags)
                if kind == "classifier":
                    y = common.to_classes(y)

                    def make_forecaster():
                        return ForecasterRecursiveClassifier(estimator=make(), lags=lags)
                elif kind == "direct":
                    def make_forecaster():
                        return ForecasterDirect(estimator=make(), lags=lags, steps=5)
                else:
                    def make_forecaster():
                        return ForecasterRecursive(estimator=make(), lags=lags)

                common.wait_for_load()
                off, on = common.peak_memory(
                    make_forecaster, lambda f: f.fit(y=y), on_value=True
                )
                matrix = n_rows * lags * 8 / 2**20
                print(f"| {kind} | {name} | {n_rows:,} x {lags} | {matrix:.0f} | "
                      f"{off:.0f} | {on:.0f} | {(on - off) / matrix:+.0%} |", flush=True)


def iterations():
    print(common.environment(), flush=True)
    print("| estimator | rows x lags | n_iter view / C | converged view / C "
          "| ms per iteration view / C | fit view / C (ms) | max abs diff predictions |")
    print("|---|---|---|---|---|---|---|")
    cases = []
    for name, make in [
        ("HuberRegressor", HuberRegressor),
        ("PoissonRegressor", PoissonRegressor),
        ("TweedieRegressor", TweedieRegressor),
    ]:
        cases.append((name, make))
        cases.append((
            f"Pipeline(StandardScaler, {name})",
            lambda make=make: make_pipeline(StandardScaler(), make()),
        ))
    for name, make in cases:
        for n_rows, lags in [(20_000, 24), (20_000, 168)]:
            y = common.make_y(n_rows, lags)
            common.wait_for_load()
            res = {}
            for flag in (False, True):
                common.set_mode(flag)
                times, preds = [], None
                for _ in range(3):
                    f = ForecasterRecursive(estimator=make(), lags=lags)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        t0 = time.perf_counter()
                        f.fit(y=y)
                        times.append(time.perf_counter() - t0)
                    preds = f.predict(steps=24).to_numpy()
                est = f.estimator[-1] if hasattr(f.estimator, "steps") else f.estimator
                n_iter = int(np.max(est.n_iter_))
                converged = not any(
                    issubclass(x.category, ConvergenceWarning) for x in w
                )
                res[flag] = (n_iter, converged, min(times) * 1e3, preds)
            v, c = res[False], res[True]
            diff = float(np.abs(v[3] - c[3]).max())
            print(f"| {name} | {n_rows:,} x {lags} | {v[0]} / {c[0]} | {v[1]} / {c[1]} | "
                  f"{v[2] / v[0]:.2f} / {c[2] / c[0]:.2f} | {v[2]:.0f} / {c[2]:.0f} | "
                  f"{diff:.1e} |", flush=True)
    common.set_mode(True)


if __name__ == "__main__":
    if sys.argv[1] == "memory":
        memory(sys.argv[2].split(";"))
    else:
        iterations()
