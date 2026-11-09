"""
PERF-contiguous-lags, script 11: option D against 0.27.x in clean conditions
(rows of the comparison table of the pull request).

List of option D: Ridge, LinearRegression, Lars, LassoLars and RidgeClassifier
(scripts 09 and 10). The Pipelines with `StandardScaler` start with a
transformer and option D does not copy them; they are also measured with the
copy (version "always copy") to quantify what is left out.

Option D: the flag `_CONTIGUOUS_X_TRAIN = "D"` of the prototype copies the
training matrix to C order only when `_estimator_benefits_from_c_contiguous_X`
is `True` (the estimators of `_SKLEARN_C_CONTIGUOUS_ESTIMATORS`, or the last
step of a `Pipeline`). With the flag off the code path is the one of 0.27.x.

For every row: minimum time of several rounds alternating the two versions
(helpers of `PERF-contiguous-lags_common.py`), whether the predictions are
identical, and the peak `tracemalloc` memory of each version (separate pass).
Before each row it waits until the 1-minute load is below 4.

- `fit`: `forecaster.fit` of Recursive, Direct (`steps=5`) and Classifier.
- `backtesting` and `search`: the cases of script 04 with many fits.
- Controls: HistGradientBoosting and LightGBM (option D does not copy).

Run from the root of the repository (prototype applied):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_11_option_d.py [fit|backtesting]
Results are appended to `PERF-contiguous-lags_11_option_d.csv`.
"""

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    Lars,
    LassoLars,
    LinearRegression,
    Ridge,
    RidgeClassifier,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    backtesting_forecaster,
    grid_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).with_name("PERF-contiguous-lags_common.py")
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

OUT = Path(__file__).with_name("PERF-contiguous-lags_11_option_d.csv")
MODE = "D"

SIX = [(1_000, 24), (1_000, 168), (20_000, 24), (20_000, 168), (200_000, 24),
       (200_000, 168)]
THREE = [(20_000, 24), (20_000, 168), (200_000, 168)]
CONTROL = [(20_000, 168), (200_000, 168)]

FIT_CASES = [
    # forecaster kind, estimator name, factory, sizes, version under test
    # ("D": option D, True: always copy, to quantify what option D leaves out)
    ("recursive", "Ridge", Ridge, SIX, "D"),
    ("recursive", "LinearRegression", LinearRegression, SIX, "D"),
    ("direct", "Ridge", Ridge, SIX, "D"),
    ("direct", "LinearRegression", LinearRegression, SIX, "D"),
    ("recursive", "Lars", Lars, THREE, "D"),
    ("direct", "Lars", Lars, THREE, "D"),
    ("recursive", "LassoLars", lambda: LassoLars(alpha=0.1), THREE, "D"),
    ("direct", "LassoLars", lambda: LassoLars(alpha=0.1), THREE, "D"),
    ("classifier", "RidgeClassifier", RidgeClassifier, THREE, "D"),
    ("recursive", "Pipeline(StandardScaler, Ridge)",
     lambda: make_pipeline(StandardScaler(), Ridge()), SIX, "D"),
    ("recursive", "Pipeline(StandardScaler, LinearRegression)",
     lambda: make_pipeline(StandardScaler(), LinearRegression()), SIX, "D"),
    ("recursive", "Pipeline(StandardScaler, Ridge)",
     lambda: make_pipeline(StandardScaler(), Ridge()), THREE, True),
    ("recursive", "Pipeline(StandardScaler, LinearRegression)",
     lambda: make_pipeline(StandardScaler(), LinearRegression()), THREE, True),
    ("recursive", "HistGradientBoosting (control)",
     lambda: HistGradientBoostingRegressor(max_iter=100, random_state=123), CONTROL, "D"),
    ("recursive", "LightGBM (control)",
     lambda: LGBMRegressor(n_estimators=100, random_state=123, verbose=-1), CONTROL, "D"),
    ("direct", "HistGradientBoosting (control)",
     lambda: HistGradientBoostingRegressor(max_iter=100, random_state=123), CONTROL, "D"),
    ("direct", "LightGBM (control)",
     lambda: LGBMRegressor(n_estimators=100, random_state=123, verbose=-1), CONTROL, "D"),
]


def save(row):
    row = dict(row, power=common.power_source())
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def fit_rows():
    for kind, name, make, sizes, mode in FIT_CASES:
        for n_rows, lags in sizes:
            y = common.make_y(n_rows, lags)
            if kind == "classifier":
                y = common.to_classes(y)

                def make_forecaster():
                    return ForecasterRecursiveClassifier(estimator=make(), lags=lags)

                def predict(f):
                    labels = np.asarray(f.predict(steps=24)).ravel()
                    return pd.Categorical(labels, categories=["high", "low", "mid"]).codes
            elif kind == "direct":
                def make_forecaster():
                    return ForecasterDirect(estimator=make(), lags=lags, steps=5)

                def predict(f):
                    return f.predict()
            else:
                def make_forecaster():
                    return ForecasterRecursive(estimator=make(), lags=lags)

                def predict(f):
                    return f.predict(steps=24)

            def fit(f):
                f.fit(y=y)

            load = common.wait_for_load()
            t0 = time.perf_counter()
            res = common.run_case(make_forecaster, fit, predict, on_value=mode)
            mem_off, mem_on = common.peak_memory(make_forecaster, fit, on_value=mode)
            save(dict(section="fit", version="D" if mode == "D" else "always copy",
                      forecaster=kind, case=name, rows=n_rows, lags=lags,
                      n_fits=1, **res, mem_off_mb=mem_off, mem_on_mb=mem_on,
                      load_before=load))
            print(f"fit {kind:10s} {name:44s} {str(mode):4s} {n_rows:>7,} x {lags:<3d} "
                  f"off {res['off_ms']:10.2f} ms  on {res['on_ms']:10.2f} ms "
                  f"ratio {res['ratio']:.3f}  mem {mem_off:.0f} -> {mem_on:.0f} MB  "
                  f"identical {res['identical']}  load {load:.1f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)


def backtesting_rows():
    def bt_case(name, make_forecaster, y, cv, n_fits):
        def fit(holder):
            _, preds = backtesting_forecaster(
                forecaster=make_forecaster(), y=y, cv=cv,
                metric="mean_absolute_error", show_progress=False,
            )
            holder["out"] = preds["pred"].to_numpy()
        return name, fit, n_fits

    def gs_case(name, make_forecaster, y, cv, n_fits):
        def fit(holder):
            res = grid_search_forecaster(
                forecaster=make_forecaster(), y=y, cv=cv,
                param_grid={"alpha": np.logspace(-3, 3, 10)}, lags_grid=[24, 168],
                metric="mean_absolute_error", return_best=False, show_progress=False,
            )
            res = res.sort_values(["lags", "alpha"], key=lambda s: s.astype(str))
            holder["out"] = res["mean_absolute_error"].to_numpy()
        return name, fit, n_fits

    cases = []
    y = common.make_y(20_000 - 24, 24)  # 20,000 observations
    cv = TimeSeriesFold(steps=24, initial_train_size=10_000, refit=True)
    for est_name, make in [("Ridge", Ridge), ("LinearRegression", LinearRegression)]:
        for lags in (24, 168):
            cases.append(bt_case(
                f"backtesting Recursive {est_name} lags={lags}, 417 folds",
                lambda make=make, lags=lags: ForecasterRecursive(make(), lags=lags),
                y, cv, 417,
            ))
    y5 = common.make_y(5_000 - 24, 24)
    cv5 = TimeSeriesFold(steps=24, initial_train_size=2_500, refit=True)
    for lags in (24, 168):
        cases.append(bt_case(
            f"backtesting Direct(steps=24) Ridge lags={lags}, 105 folds",
            lambda lags=lags: ForecasterDirect(Ridge(), lags=lags, steps=24),
            y5, cv5, 105 * 24,
        ))
    y2 = common.make_y(2_000 - 24, 24)
    cases.append(gs_case(
        "grid_search Ridge TimeSeriesFold refit=True, 42 folds x 20 candidates",
        lambda: ForecasterRecursive(Ridge(), lags=24), y2,
        TimeSeriesFold(steps=24, initial_train_size=1_000, refit=True), 840,
    ))
    for n, initial in [(2_000, 1_500), (20_000, 15_000)]:
        yy = common.make_y(n - 24, 24)
        cases.append(gs_case(
            f"grid_search Ridge OneStepAheadFold n={n}, 20 candidates",
            lambda: ForecasterRecursive(Ridge(), lags=24), yy,
            OneStepAheadFold(initial_train_size=initial), 20,
        ))
    cases.append(bt_case(
        "backtesting Recursive LightGBM lags=24, 84 folds (control)",
        lambda: ForecasterRecursive(
            LGBMRegressor(n_estimators=50, random_state=123, verbose=-1), lags=24),
        y2, TimeSeriesFold(steps=12, initial_train_size=1_000, refit=True), 84,
    ))

    for name, fit, n_fits in cases:
        load = common.wait_for_load()
        t0 = time.perf_counter()
        res = common.run_case(dict, fit, lambda h: h["out"], on_value=MODE)
        mem_off, mem_on = common.peak_memory(dict, fit, on_value=MODE, repeats=1)
        save(dict(section="backtesting", version="D", forecaster="", case=name, rows=None,
                  lags=None, n_fits=n_fits, **res, mem_off_mb=mem_off,
                  mem_on_mb=mem_on, load_before=load))
        print(f"{name:70s} off {res['off_ms'] / 1e3:8.3f} s  on {res['on_ms'] / 1e3:8.3f} s "
              f"ratio {res['ratio']:.3f}  mem {mem_off:.0f} -> {mem_on:.0f} MB  "
              f"max|diff| {res['max_abs_diff']:.1e}  load {load:.1f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    print(common.environment(), flush=True)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("fit", "all"):
        fit_rows()
    if part in ("backtesting", "all"):
        backtesting_rows()
