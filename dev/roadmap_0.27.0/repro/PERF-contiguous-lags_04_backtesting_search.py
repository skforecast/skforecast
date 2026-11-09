"""
PERF-contiguous-lags, script 04: where many small fits add up. Backtesting with
`refit=True` and many folds, and grid searches with Ridge, with the training
matrix as it is today (flag off) and C-contiguous (prototype, flag on).

Same method as script 03: both versions in the same process, alternating the
flag `_CONTIGUOUS_X_TRAIN`, minimum and median of `rounds` runs per version.
The predictions of the backtesting and the metrics of the search are compared
between the two versions.

Run from the root of the repository:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_04_backtesting_search.py
Results are appended to `PERF-contiguous-lags_04_backtesting_search.csv`.
"""

import gc
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge

import skforecast.direct._forecaster_direct as m_direct
import skforecast.direct._forecaster_direct_multivariate as m_dmv
import skforecast.recursive._forecaster_recursive as m_rec
import skforecast.recursive._forecaster_recursive_classifier as m_cls
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    OneStepAheadFold,
    TimeSeriesFold,
    backtesting_forecaster,
    grid_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive

warnings.filterwarnings("ignore")

MODULES = (m_rec, m_cls, m_direct, m_dmv)
OUT = Path(__file__).with_name("PERF-contiguous-lags_04_backtesting_search.csv")


def set_contiguous(value: bool) -> None:
    for m in MODULES:
        m._CONTIGUOUS_X_TRAIN = value


def make_y(n: int) -> pd.Series:
    rng = np.random.default_rng(123)
    t = np.arange(n)
    noise = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168) + noise
    return pd.Series(
        y, index=pd.date_range("2000-01-01", periods=n, freq="h"), name="y"
    )


def backtesting_case(make_forecaster, y, cv):
    def run():
        metric, preds = backtesting_forecaster(
            forecaster=make_forecaster(), y=y, cv=cv,
            metric="mean_absolute_error", show_progress=False,
        )
        return preds["pred"].to_numpy()
    return run


def grid_case(make_forecaster, y, cv, param_grid, lags_grid):
    def run():
        res = grid_search_forecaster(
            forecaster=make_forecaster(), y=y, cv=cv, param_grid=param_grid,
            lags_grid=lags_grid, metric="mean_absolute_error",
            return_best=False, show_progress=False,
        )
        res = res.sort_values(["lags", "alpha"], key=lambda s: s.astype(str))
        return res["mean_absolute_error"].to_numpy()
    return run


def measure(name, run, n_fits, rounds=5):
    out = {}
    times = {False: [], True: []}
    for flag in (False, True):
        set_contiguous(flag)
        out[flag] = run()
    for _ in range(rounds):
        for flag in (False, True):
            set_contiguous(flag)
            gc.collect()
            t0 = time.perf_counter()
            run()
            times[flag].append(time.perf_counter() - t0)
    set_contiguous(True)
    diff = np.abs(out[True] - out[False])
    row = dict(
        case=name, n_fits=n_fits,
        min_off_s=min(times[False]), min_on_s=min(times[True]),
        med_off_s=np.median(times[False]), med_on_s=np.median(times[True]),
        ratio_min=min(times[True]) / min(times[False]),
        saved_per_fit_ms=(min(times[False]) - min(times[True])) / n_fits * 1e3,
        max_abs_diff=float(diff.max()),
        identical=bool(np.array_equal(out[True], out[False])),
        rounds=rounds,
    )
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(
        f"{name:60s} fits {n_fits:5d}  off {row['min_off_s']:7.3f} s  "
        f"on {row['min_on_s']:7.3f} s  ratio {row['ratio_min']:.3f}  "
        f"saved/fit {row['saved_per_fit_ms']:.3f} ms  max|diff| {row['max_abs_diff']:.2e}",
        flush=True,
    )


def n_folds(n, initial, steps):
    return int(np.ceil((n - initial) / steps))


def main():
    print(f"Python {platform.python_version()}, numpy {np.__version__}, pandas "
          f"{pd.__version__}, scikit-learn {sklearn.__version__}, "
          f"{platform.platform()}", flush=True)

    # Backtesting, ForecasterRecursive, refit=True --------------------------
    for n, initial, steps in [(2_000, 1_000, 12), (20_000, 10_000, 24)]:
        y = make_y(n)
        cv = TimeSeriesFold(steps=steps, initial_train_size=initial, refit=True)
        folds = n_folds(n, initial, steps)
        for lags in (24, 168):
            for est_name, make_est in [
                ("Ridge", Ridge),
                ("LinearRegression", LinearRegression),
                ("LightGBM", lambda: LGBMRegressor(
                    n_estimators=50, random_state=123, verbose=-1)),
            ]:
                if est_name == "LightGBM" and n > 2_000:
                    continue
                measure(
                    f"backtesting Recursive {est_name} lags={lags} "
                    f"n={n} train={initial} folds={folds}",
                    backtesting_case(
                        lambda: ForecasterRecursive(estimator=make_est(), lags=lags),
                        y, cv,
                    ),
                    n_fits=folds, rounds=5 if est_name != "LightGBM" else 3,
                )

    # Backtesting, ForecasterDirect (one fit per step and fold) --------------
    n, initial, steps = 5_000, 2_500, 24
    y = make_y(n)
    cv = TimeSeriesFold(steps=steps, initial_train_size=initial, refit=True)
    folds = n_folds(n, initial, steps)
    for lags in (24, 168):
        measure(
            f"backtesting Direct(steps=24) Ridge lags={lags} n={n} "
            f"train={initial} folds={folds}",
            backtesting_case(
                lambda: ForecasterDirect(estimator=Ridge(), lags=lags, steps=steps),
                y, cv,
            ),
            n_fits=folds * steps, rounds=3,
        )

    # Grid search with Ridge ------------------------------------------------
    param_grid = {"alpha": np.logspace(-3, 3, 10)}
    lags_grid = [24, 168]
    n_candidates = len(param_grid["alpha"]) * len(lags_grid)

    n, initial, steps = 2_000, 1_000, 24
    y = make_y(n)
    cv = TimeSeriesFold(steps=steps, initial_train_size=initial, refit=True)
    folds = n_folds(n, initial, steps)
    measure(
        f"grid_search Recursive Ridge TimeSeriesFold refit=True "
        f"n={n} folds={folds} candidates={n_candidates}",
        grid_case(lambda: ForecasterRecursive(estimator=Ridge(), lags=24),
                  y, cv, param_grid, lags_grid),
        n_fits=folds * n_candidates, rounds=3,
    )

    for n, initial in [(2_000, 1_500), (20_000, 15_000)]:
        y = make_y(n)
        cv = TimeSeriesFold(steps=24, initial_train_size=initial, refit=False)
        measure(
            f"grid_search Recursive Ridge TimeSeriesFold refit=False "
            f"n={n} train={initial} candidates={n_candidates}",
            grid_case(lambda: ForecasterRecursive(estimator=Ridge(), lags=24),
                      y, cv, param_grid, lags_grid),
            n_fits=n_candidates, rounds=5,
        )
        cv = OneStepAheadFold(initial_train_size=initial)
        measure(
            f"grid_search Recursive Ridge OneStepAheadFold "
            f"n={n} train={initial} candidates={n_candidates}",
            grid_case(lambda: ForecasterRecursive(estimator=Ridge(), lags=24),
                      y, cv, param_grid, lags_grid),
            n_fits=n_candidates, rounds=5,
        )


if __name__ == "__main__":
    main()
