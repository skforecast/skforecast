"""
PERF-contiguous-lags, script 12: the implementation of option D (branch
`perf/contiguous-lag-matrix`, without the prototype) against 0.27.x, on the
main rows of the comparison table (section 15 of PERF-contiguous-lags.md).

The two versions run in the same process. "0.27.x" is the implementation with
`_prefers_c_contiguous_X` replaced, in the three forecaster modules, by a
function that always returns `False`: the code path of 0.27.x plus one call
that returns `False`. "Option D" is the implementation as it is. Same method
as scripts 09 to 11 (helpers of `PERF-contiguous-lags_common.py`): it waits
until the 1-minute load is below 4 before each row, alternates the two
versions, takes the minimum of several rounds, checks that the predictions are
identical and measures the peak `tracemalloc` memory of each version. Each
row waits (up to 2 hours) until the machine is on AC power and the load is
below 4.

Run from the root of the repository, without the prototype applied:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_12_implementation.py
Results are appended to `PERF-contiguous-lags_12_implementation.csv`.
"""

import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

import skforecast.direct._forecaster_direct as m_direct
import skforecast.direct._forecaster_direct_multivariate as m_dmv
import skforecast.recursive._forecaster_recursive as m_rec
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    backtesting_forecaster,
    grid_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).with_name("PERF-contiguous-lags_common.py")
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

MODULES = (m_rec, m_direct, m_dmv)
assert all(hasattr(m, "_prefers_c_contiguous_X") for m in MODULES), (
    "the implementation of option D is missing"
)
assert not common.PROTOTYPE, "remove the prototype before running this script"
ORIGINAL = m_rec._prefers_c_contiguous_X


def _never(estimator):
    return False


def set_mode_implementation(value) -> None:
    for m in MODULES:
        m._prefers_c_contiguous_X = ORIGINAL if value else _never


# `run_case` and `peak_memory` call `common.set_mode`.
common.set_mode = set_mode_implementation

OUT = Path(__file__).with_name("PERF-contiguous-lags_12_implementation.csv")
THREE = [(20_000, 24), (20_000, 168), (200_000, 168)]
CONTROL = [(20_000, 168), (200_000, 168)]

FIT_CASES = [
    ("recursive", "Ridge", Ridge, THREE),
    ("recursive", "LinearRegression", LinearRegression, THREE),
    ("direct", "Ridge", Ridge, THREE),
    ("direct", "LinearRegression", LinearRegression, THREE),
    ("recursive", "HistGradientBoosting (control)",
     lambda: HistGradientBoostingRegressor(max_iter=100, random_state=123), CONTROL),
    ("recursive", "LightGBM (control)",
     lambda: LGBMRegressor(n_estimators=100, random_state=123, verbose=-1), CONTROL),
    ("direct", "HistGradientBoosting (control)",
     lambda: HistGradientBoostingRegressor(max_iter=100, random_state=123), CONTROL),
    ("direct", "LightGBM (control)",
     lambda: LGBMRegressor(n_estimators=100, random_state=123, verbose=-1), CONTROL),
]


def save(row):
    row = dict(row, power=common.power_source())
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def fit_rows():
    for kind, name, make, sizes in FIT_CASES:
        for n_rows, lags in sizes:
            y = common.make_y(n_rows, lags)
            if kind == "direct":
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

            load = common.wait_for_load(max_wait=7200, require_ac=True)
            t0 = time.perf_counter()
            res = common.run_case(make_forecaster, fit, predict, on_value=True)
            mem_off, mem_on = common.peak_memory(make_forecaster, fit, on_value=True)
            save(dict(section="fit", forecaster=kind, case=name, rows=n_rows, lags=lags,
                      n_fits=1, **res, mem_off_mb=mem_off, mem_on_mb=mem_on,
                      load_before=load))
            print(f"fit {kind:10s} {name:32s} {n_rows:>7,} x {lags:<3d} "
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
    y = common.make_y(20_000 - 24, 24)
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

    for name, fit, n_fits in cases:
        load = common.wait_for_load(max_wait=7200, require_ac=True)
        t0 = time.perf_counter()
        res = common.run_case(dict, fit, lambda h: h["out"], on_value=True)
        mem_off, mem_on = common.peak_memory(dict, fit, on_value=True, repeats=1)
        save(dict(section="backtesting", forecaster="", case=name, rows=None,
                  lags=None, n_fits=n_fits, **res, mem_off_mb=mem_off,
                  mem_on_mb=mem_on, load_before=load))
        print(f"{name:70s} off {res['off_ms'] / 1e3:8.3f} s  on {res['on_ms'] / 1e3:8.3f} s "
              f"ratio {res['ratio']:.3f}  mem {mem_off:.0f} -> {mem_on:.0f} MB  "
              f"max|diff| {res['max_abs_diff']:.1e}  load {load:.1f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    print(common.environment(), flush=True)
    fit_rows()
    backtesting_rows()
