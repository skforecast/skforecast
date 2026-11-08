"""
Data, estimators and timed operations of the n_jobs review.

Every operation is split in an untimed `setup` (build the forecaster and the
cv) and a timed `run` (the call that skforecast parallelizes). `run` returns
the object used to check that the results do not depend on `n_jobs`.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import cpu_count

warnings.filterwarnings("ignore")

SEED = 123

# Sizes
# ==============================================================================
# S: documentation example scale. L: thousands of rows, tens of series and a
# long horizon. Each size gives about 10 to 20 folds.
SINGLE = {
    "S": {"n": 1_000, "lags": 24, "steps": 24, "initial": 760},  # 10 folds
    "L": {"n": 10_000, "lags": 48, "steps": 168, "initial": 6_976},  # 18 folds
}
MULTI = {
    "S": {"n_series": 5, "n": 1_000, "lags": 24, "steps": 24, "initial": 760},
    "L": {"n_series": 50, "n": 2_000, "lags": 48, "steps": 48, "initial": 1_040},
}
DIRECT = {
    "S": {"n": 1_000, "lags": 24, "steps": 24, "initial": 760},  # 10 folds
    "L": {"n": 5_000, "lags": 48, "steps": 48, "initial": 4_520},  # 10 folds
}
# Backtesting of ForecasterDirect: each refit fits `steps` estimators, so the
# large size keeps 5 folds (the pilot took up to 314 s per call with 10).
DIRECT_BT = {
    "S": DIRECT["S"],
    "L": {"n": 5_000, "lags": 48, "steps": 48, "initial": 4_760},  # 5 folds
}
DMV = {
    "S": {
        "n_series": 5,
        "n": 1_000,
        "lags": 12,
        "steps": 12,
        "initial": 880,
    },  # 10 folds
    "L": {
        "n_series": 10,
        "n": 2_000,
        "lags": 24,
        "steps": 24,
        "initial": 1_760,
    },  # 10 folds
}
STATS = {
    "S": {"n": 360, "steps": 12, "initial": 240, "m": 12},  # 10 folds
    "L": {"n": 2_000, "steps": 36, "initial": 1_640, "m": 7},  # 10 folds
}


# Data
# ==============================================================================
def _series(
    n: int, rng: np.random.Generator, period_a: int, period_b: int
) -> np.ndarray:
    t = np.arange(n)
    level = rng.uniform(5, 50)
    scale = rng.uniform(0.5, 5)
    noise = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]
    values = (
        level
        + scale * np.sin(2 * np.pi * t / period_a + rng.uniform(0, 6.3))
        + 0.5 * scale * np.sin(2 * np.pi * t / period_b + rng.uniform(0, 6.3))
        + 0.003 * scale * t
        + noise
    )

    return values


@lru_cache(maxsize=None)
def single_data(n: int, freq: str = "h", period_a: int = 24, period_b: int = 168):
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2020-01-01", periods=n, freq=freq)
    exog = pd.DataFrame(
        {
            "exog_1": rng.normal(0, 1, n).cumsum() / 10,
            "exog_2": np.sin(2 * np.pi * np.arange(n) / period_a),
        },
        index=index,
    )
    y = pd.Series(
        _series(n, rng, period_a, period_b) + 2 * exog["exog_1"].to_numpy(),
        index=index,
        name="y",
    )

    return y, exog


@lru_cache(maxsize=None)
def class_data(n: int):
    y, exog = single_data(n)
    y_class = pd.Series(
        pd.qcut(y.diff().fillna(0), q=3, labels=False).astype(int),
        index=y.index,
        name="y",
    )

    return y_class, exog


@lru_cache(maxsize=None)
def multi_data(n_series: int, n: int):
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    series = pd.DataFrame(
        {f"s_{i:03d}": _series(n, rng, 24, 168) for i in range(n_series)},
        index=index,
    )
    exog = pd.DataFrame(
        {
            "exog_1": rng.normal(0, 1, n).cumsum() / 10,
            "exog_2": np.sin(2 * np.pi * np.arange(n) / 24),
        },
        index=index,
    )

    return series, exog


# Estimators
# ==============================================================================
def make_estimator(name: str):
    """
    Estimators of moderate size (about 100 trees), so a sequential large
    backtesting with refit lasts tens of seconds.
    """

    from catboost import CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    lgbm = {"n_estimators": 100, "random_state": SEED, "verbose": -1}
    xgb = {"n_estimators": 100, "max_depth": 6, "random_state": SEED}
    forest = {"n_estimators": 25, "max_depth": 10, "random_state": SEED}
    cat = {
        "iterations": 100,
        "depth": 6,
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }
    factories = {
        "ridge": lambda: Ridge(alpha=1.0),
        "pipe_ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=50, random_state=SEED),
        "lgbm_1": lambda: LGBMRegressor(n_jobs=1, **lgbm),
        "lgbm_none": lambda: LGBMRegressor(n_jobs=None, **lgbm),
        "lgbm_m1": lambda: LGBMRegressor(n_jobs=-1, **lgbm),
        "lgbm_4": lambda: LGBMRegressor(n_jobs=4, **lgbm),
        "xgb_1": lambda: XGBRegressor(n_jobs=1, **xgb),
        "xgb_none": lambda: XGBRegressor(n_jobs=None, **xgb),
        "xgb_m1": lambda: XGBRegressor(n_jobs=-1, **xgb),
        "rf_none": lambda: RandomForestRegressor(n_jobs=None, **forest),
        "rf_m1": lambda: RandomForestRegressor(n_jobs=-1, **forest),
        "et_none": lambda: ExtraTreesRegressor(n_jobs=None, **forest),
        "et_m1": lambda: ExtraTreesRegressor(n_jobs=-1, **forest),
        "cat_1": lambda: CatBoostRegressor(thread_count=1, **cat),
        "cat_m1": lambda: CatBoostRegressor(thread_count=-1, **cat),
        "logreg": lambda: LogisticRegression(max_iter=1000),
        "hgbc": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
        "lgbmc_1": lambda: LGBMClassifier(n_jobs=1, **lgbm),
    }

    return factories[name]()


def make_stats_estimator(name: str, m: int):
    from skforecast.stats import Arima, Sarimax

    if name == "arima":
        return Arima(order=(1, 1, 1), seasonal_order=(1, 0, 1), m=m)
    if name == "sarimax":
        return Sarimax(order=(1, 1, 1), seasonal_order=(1, 0, 1, m))
    raise ValueError(name)


# Resolution of n_jobs (what skforecast actually uses)
# ==============================================================================
def resolve_backtesting(forecaster, refit, n_jobs, stats: bool = False) -> int:
    """
    Number of jobs that the backtesting functions use, replicating their
    resolution of `n_jobs` (including the forced 1 cases).
    """

    from skforecast.model_selection._utils import select_n_jobs_backtesting

    if stats and not refit:
        return 1
    if n_jobs == "auto":
        return select_n_jobs_backtesting(forecaster=forecaster, refit=refit)
    if not isinstance(refit, bool) and refit != 1 and n_jobs != 1:
        return 1

    return n_jobs if n_jobs > 0 else cpu_count()


# Cells
# ==============================================================================
def to_array(result) -> np.ndarray:
    """
    Numeric values of a result, to compare runs with different `n_jobs`.
    """

    if isinstance(result, pd.DataFrame):
        num = result.select_dtypes(include="number")
        return num.to_numpy(dtype=float)
    if isinstance(result, pd.Series):
        return result.to_numpy(dtype=float)

    return np.asarray(result, dtype=float)


def build_cell(meta: dict):
    """
    Return `(setup, run, info)` for a cell. `setup()` builds the state
    (untimed), `run(state)` performs the timed call and `info` holds the
    resolved number of jobs and the number of folds.
    """

    op = meta["op"]
    builders = {
        "backtest": _cell_backtest,
        "backtest_ms": _cell_backtest_ms,
        "fit_direct": _cell_fit_direct,
        "fit_dmv": _cell_fit_dmv,
        "backtest_stats": _cell_backtest_stats,
        "grid": _cell_grid,
        "grid_ms": _cell_grid_ms,
    }

    return builders[op](meta)


def _cv(steps: int, initial: int, refit, lags: int | None = None):
    from skforecast.model_selection import TimeSeriesFold

    return TimeSeriesFold(
        steps=steps,
        initial_train_size=initial,
        refit=refit,
        fixed_train_size=False,
        verbose=False,
    )


def _n_folds(n: int, initial: int, steps: int) -> int:
    return int(np.ceil((n - initial) / steps))


def _forecaster_single(meta: dict, cfg: dict):
    from skforecast.direct import ForecasterDirect
    from skforecast.recursive import (
        ForecasterEquivalentDate,
        ForecasterRecursive,
        ForecasterRecursiveClassifier,
    )

    kind = meta["forecaster"]
    if kind == "recursive":
        return ForecasterRecursive(
            estimator=make_estimator(meta["estimator"]), lags=cfg["lags"]
        )
    if kind == "classifier":
        return ForecasterRecursiveClassifier(
            estimator=make_estimator(meta["estimator"]), lags=cfg["lags"]
        )
    if kind in ("direct", "direct_fix"):
        if kind == "direct_fix":
            from patched import ForecasterDirect
        return ForecasterDirect(
            estimator=make_estimator(meta["estimator"]),
            steps=cfg["steps"],
            lags=cfg["lags"],
            n_jobs=meta.get("inner_n_jobs", "auto"),
        )
    if kind == "equivalent_date":
        return ForecasterEquivalentDate(offset=24, n_offsets=2)
    raise ValueError(kind)


def _cell_backtest(meta: dict):
    from skforecast.model_selection import backtesting_forecaster

    table = DIRECT_BT if meta["forecaster"].startswith("direct") else SINGLE
    cfg = table[meta["size"]]
    if meta["forecaster"] == "classifier":
        y, exog = class_data(cfg["n"])
        metric = "accuracy_score"
    else:
        y, exog = single_data(cfg["n"])
        metric = "mean_absolute_error"
    if meta["forecaster"] == "equivalent_date":
        exog = None
    refit = meta["refit"]

    def setup():
        return _forecaster_single(meta, cfg)

    def run(forecaster):
        _, preds = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=_cv(cfg["steps"], cfg["initial"], refit),
            metric=metric,
            exog=exog,
            n_jobs=meta["n_jobs"],
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )
        return preds

    probe = setup()
    info = {
        "n_jobs_resolved": resolve_backtesting(probe, refit, meta["n_jobs"]),
        "inner_resolved": getattr(probe, "n_jobs", ""),
        "n_folds": _n_folds(cfg["n"], cfg["initial"], cfg["steps"]),
    }

    return setup, run, info


def _cell_backtest_ms(meta: dict):
    from skforecast.direct import ForecasterDirectMultiVariate
    from skforecast.model_selection import backtesting_forecaster_multiseries
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    is_dmv = meta["forecaster"].startswith("dmv")
    table = DMV if is_dmv else MULTI
    cfg = table[meta["size"]]
    series, exog = multi_data(cfg["n_series"], cfg["n"])
    refit = meta["refit"]

    def setup():
        if is_dmv:
            cls = ForecasterDirectMultiVariate
            if meta["forecaster"] == "dmv_fix":
                from patched import ForecasterDirectMultiVariate as cls
            return cls(
                estimator=make_estimator(meta["estimator"]),
                level=series.columns[0],
                steps=cfg["steps"],
                lags=cfg["lags"],
                n_jobs=meta.get("inner_n_jobs", "auto"),
            )
        return ForecasterRecursiveMultiSeries(
            estimator=make_estimator(meta["estimator"]), lags=cfg["lags"]
        )

    def run(forecaster):
        _, preds = backtesting_forecaster_multiseries(
            forecaster=forecaster,
            series=series,
            cv=_cv(cfg["steps"], cfg["initial"], refit),
            metric="mean_absolute_error",
            exog=exog,
            n_jobs=meta["n_jobs"],
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )
        return preds

    probe = setup()
    info = {
        "n_jobs_resolved": resolve_backtesting(probe, refit, meta["n_jobs"]),
        "inner_resolved": getattr(probe, "n_jobs", ""),
        "n_folds": _n_folds(cfg["n"], cfg["initial"], cfg["steps"]),
    }

    return setup, run, info


def _cell_fit_direct(meta: dict):
    from skforecast.direct import ForecasterDirect

    cfg = DIRECT[meta["size"]]
    y, exog = single_data(cfg["n"])

    def setup():
        return ForecasterDirect(
            estimator=make_estimator(meta["estimator"]),
            steps=cfg["steps"],
            lags=cfg["lags"],
            n_jobs=meta["n_jobs"],
        )

    def run(forecaster):
        forecaster.fit(y=y, exog=exog)
        return forecaster.predict(exog=_future_exog(exog, cfg["steps"]))

    info = {
        "n_jobs_resolved": setup().n_jobs,
        "inner_resolved": "",
        "n_folds": cfg["steps"],  # number of estimators fitted
    }

    return setup, run, info


def _cell_fit_dmv(meta: dict):
    from skforecast.direct import ForecasterDirectMultiVariate

    cfg = DMV[meta["size"]]
    series, exog = multi_data(cfg["n_series"], cfg["n"])

    def setup():
        return ForecasterDirectMultiVariate(
            estimator=make_estimator(meta["estimator"]),
            level=series.columns[0],
            steps=cfg["steps"],
            lags=cfg["lags"],
            n_jobs=meta["n_jobs"],
        )

    def run(forecaster):
        forecaster.fit(series=series, exog=exog)
        return forecaster.predict(exog=_future_exog(exog, cfg["steps"]))

    info = {
        "n_jobs_resolved": setup().n_jobs,
        "inner_resolved": "",
        "n_folds": cfg["steps"],
    }

    return setup, run, info


def _future_exog(exog: pd.DataFrame, steps: int) -> pd.DataFrame:
    index = pd.date_range(
        exog.index[-1] + exog.index.freq, periods=steps, freq=exog.index.freq
    )
    return pd.DataFrame(
        np.tile(exog.iloc[[-1]].to_numpy(), (steps, 1)),
        index=index,
        columns=exog.columns,
    )


def _cell_backtest_stats(meta: dict):
    from skforecast.model_selection import backtesting_stats
    from skforecast.recursive import ForecasterStats

    cfg = STATS[meta["size"]]
    y, _ = single_data(cfg["n"], freq="D", period_a=cfg["m"], period_b=4 * cfg["m"])
    refit = meta["refit"]

    def setup():
        return ForecasterStats(
            estimator=make_stats_estimator(meta["estimator"], cfg["m"])
        )

    def run(forecaster):
        _, preds = backtesting_stats(
            forecaster=forecaster,
            y=y,
            cv=_cv(cfg["steps"], cfg["initial"], refit),
            metric="mean_absolute_error",
            n_jobs=meta["n_jobs"],
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )
        return preds

    probe = setup()
    info = {
        "n_jobs_resolved": resolve_backtesting(
            probe, refit, meta["n_jobs"], stats=True
        ),
        "inner_resolved": "",
        "n_folds": _n_folds(cfg["n"], cfg["initial"], cfg["steps"]),
    }

    return setup, run, info


def _sorted_metric(results: pd.DataFrame) -> pd.Series:
    """
    Metric of each candidate in a fixed order (the search sorts by metric).
    """

    key = results["lags"].astype(str) + results["params"].astype(str)
    metric = [c for c in results.columns if c.startswith("mean_absolute_error")][0]

    return results.assign(_key=key).sort_values("_key")[metric].reset_index(drop=True)


def _cell_grid(meta: dict):
    from skforecast.model_selection import grid_search_forecaster

    cfg = SINGLE[meta["size"]]
    y, exog = single_data(cfg["n"])
    refit = meta["refit"]
    param_grid = (
        {"max_iter": [50, 100]}
        if meta["estimator"] == "hgb"
        else {"n_estimators": [50, 100]}
    )
    lags_grid = [cfg["lags"] // 2, cfg["lags"]]

    def setup():
        return _forecaster_single(meta, cfg)

    def run(forecaster):
        results = grid_search_forecaster(
            forecaster=forecaster,
            y=y,
            cv=_cv(cfg["steps"], cfg["initial"], refit),
            param_grid=param_grid,
            lags_grid=lags_grid,
            metric="mean_absolute_error",
            exog=exog,
            return_best=False,
            n_jobs=meta["n_jobs"],
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )
        return _sorted_metric(results)

    probe = setup()
    info = {
        "n_jobs_resolved": resolve_backtesting(probe, refit, meta["n_jobs"]),
        "inner_resolved": "",
        "n_folds": 4 * _n_folds(cfg["n"], cfg["initial"], cfg["steps"]),
    }

    return setup, run, info


def _cell_grid_ms(meta: dict):
    from skforecast.model_selection import grid_search_forecaster_multiseries
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    cfg = MULTI[meta["size"]]
    series, exog = multi_data(cfg["n_series"], cfg["n"])
    refit = meta["refit"]
    param_grid = {"n_estimators": [50, 100]}
    lags_grid = [cfg["lags"] // 2, cfg["lags"]]

    def setup():
        return ForecasterRecursiveMultiSeries(
            estimator=make_estimator(meta["estimator"]), lags=cfg["lags"]
        )

    def run(forecaster):
        results = grid_search_forecaster_multiseries(
            forecaster=forecaster,
            series=series,
            cv=_cv(cfg["steps"], cfg["initial"], refit),
            param_grid=param_grid,
            lags_grid=lags_grid,
            metric="mean_absolute_error",
            exog=exog,
            return_best=False,
            n_jobs=meta["n_jobs"],
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )
        return _sorted_metric(results)

    probe = setup()
    info = {
        "n_jobs_resolved": resolve_backtesting(probe, refit, meta["n_jobs"]),
        "inner_resolved": "",
        "n_folds": 4 * _n_folds(cfg["n"], cfg["initial"], cfg["steps"]),
    }

    return setup, run, info
