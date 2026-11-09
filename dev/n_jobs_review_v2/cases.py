"""
Builders of the benchmark cases of the n_jobs review (v2).

A case is a JSON-serializable dict (spec). `build(spec)` returns a callable
`run(n_jobs)` that executes the skforecast call of the case and returns a
fingerprint of its result (to check that parallel and sequential runs give the
same numbers).
"""

import hashlib
import warnings

import numpy as np
import pandas as pd


def make_y(n, seed=0, name="y"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = (
        10
        + 3 * np.sin(2 * np.pi * t / 24)
        + 2 * np.sin(2 * np.pi * t / (24 * 7))
        + 0.002 * t
        + rng.normal(0, 1, n)
    )
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(values, index=index, name=name)


def make_exog(n, n_exog, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        rng.normal(size=(n, n_exog)),
        index=index,
        columns=[f"exog_{i}" for i in range(n_exog)],
    )


def make_estimator(name, params=None):
    params = dict(params or {})
    if name == "lr":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    if name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**{"alpha": 0.01, **params})
    if name == "pipe_ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "quantile":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(**{"alpha": 0.0, "quantile": 0.5, **params})
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{"n_estimators": 100, "random_state": 0, **params})
    if name == "et":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**{"n_estimators": 100, "random_state": 0, **params})
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**{"random_state": 0, **params})
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**{"random_state": 0, "verbose": -1, **params})
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**{"random_state": 0, **params})
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            **{"random_state": 0, "verbose": 0, "allow_writing_files": False, **params}
        )
    if name == "arima":
        from skforecast.stats import Arima
        return Arima(**params)
    if name == "ets":
        from skforecast.stats import Ets
        return Ets(**params)
    if name == "sarimax":
        from skforecast.stats import Sarimax
        return Sarimax(**params)
    raise ValueError(name)


def fingerprint(obj):
    """Hash of the numbers of a DataFrame (or tuple of them), rounded to 1e-9."""
    if isinstance(obj, tuple):
        obj = [o for o in obj if isinstance(o, pd.DataFrame)]
    if not isinstance(obj, list):
        obj = [obj]
    h = hashlib.md5()
    for df in obj:
        num = df.select_dtypes("number").to_numpy(dtype=float)
        h.update(np.round(np.nan_to_num(num, nan=-999.0), 9).tobytes())
    return h.hexdigest()[:10]


def build(spec):
    from skforecast.model_selection import TimeSeriesFold

    kind = spec["kind"]
    n = spec.get("n", 2000)
    steps = spec.get("steps", 24)
    n_folds = spec.get("n_folds", 20)
    refit = spec.get("refit", True)
    fixed = spec.get("fixed_train_size", False)
    lags = spec.get("lags", 24)
    est_name = spec.get("est", "ridge")
    est_params = spec.get("est_params", {})
    n_exog = spec.get("n_exog", 0)
    initial_train_size = n - steps * n_folds

    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=initial_train_size,
        refit=refit,
        fixed_train_size=fixed,
        verbose=False,
    )

    if kind in ("backtest", "grid", "direct_fit", "backtest_direct"):
        y = make_y(n)
        exog = make_exog(n, n_exog) if n_exog else None

        if kind in ("direct_fit", "backtest_direct"):
            from skforecast.direct import ForecasterDirect

            def new_forecaster(n_jobs):
                return ForecasterDirect(
                    estimator=make_estimator(est_name, est_params),
                    steps=spec.get("direct_steps", steps),
                    lags=lags,
                    n_jobs=n_jobs,
                )
        else:
            from skforecast.recursive import ForecasterRecursive

            def new_forecaster(n_jobs=None):
                return ForecasterRecursive(
                    estimator=make_estimator(est_name, est_params), lags=lags
                )

        if kind == "direct_fit":
            def run(n_jobs):
                f = new_forecaster(n_jobs)
                f.fit(y=y, exog=exog)
                X = f.create_predict_X(steps=f.max_step, exog=None if exog is None else make_exog(n + f.max_step, n_exog).iloc[n:])
                return fingerprint(f.predict(exog=None if exog is None else make_exog(n + f.max_step, n_exog).iloc[n:]).to_frame())
            return run

        if kind == "backtest_direct":
            from skforecast.model_selection import backtesting_forecaster

            def run(n_jobs):
                f = new_forecaster(n_jobs)
                res = backtesting_forecaster(
                    forecaster=f, y=y, exog=exog, cv=cv,
                    metric="mean_absolute_error", n_jobs=1,
                    show_progress=False,
                )
                return fingerprint(res)
            return run

        if kind == "backtest":
            from skforecast.model_selection import backtesting_forecaster
            interval = spec.get("interval", None)
            n_boot = spec.get("n_boot", 250)

            def run(n_jobs):
                f = new_forecaster()
                res = backtesting_forecaster(
                    forecaster=f, y=y, exog=exog, cv=cv,
                    metric="mean_absolute_error", n_jobs=n_jobs,
                    interval=interval, n_boot=n_boot,
                    show_progress=False,
                )
                return fingerprint(res)
            return run

        if kind == "grid":
            from skforecast.model_selection import grid_search_forecaster
            param_grid = spec["param_grid"]
            lags_grid = spec.get("lags_grid", None)

            def run(n_jobs):
                f = new_forecaster()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grid_search_forecaster(
                        forecaster=f, y=y, exog=exog, cv=cv,
                        param_grid=param_grid, lags_grid=lags_grid,
                        metric="mean_absolute_error", n_jobs=n_jobs,
                        return_best=False, show_progress=False,
                    )
                return fingerprint(res[["mean_absolute_error"]])
            return run

    if kind in ("backtest_ms", "grid_ms"):
        from skforecast.recursive import ForecasterRecursiveMultiSeries
        n_series = spec.get("n_series", 10)
        series = pd.concat(
            [make_y(n, seed=i, name=f"s{i}") for i in range(n_series)], axis=1
        )
        if kind == "backtest_ms":
            from skforecast.model_selection import backtesting_forecaster_multiseries

            def run(n_jobs):
                f = ForecasterRecursiveMultiSeries(
                    estimator=make_estimator(est_name, est_params), lags=lags
                )
                res = backtesting_forecaster_multiseries(
                    forecaster=f, series=series, cv=cv,
                    metric="mean_absolute_error", n_jobs=n_jobs,
                    show_progress=False,
                )
                return fingerprint(res)
            return run
        from skforecast.model_selection import grid_search_forecaster_multiseries
        param_grid = spec["param_grid"]

        def run(n_jobs):
            f = ForecasterRecursiveMultiSeries(
                estimator=make_estimator(est_name, est_params), lags=lags
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grid_search_forecaster_multiseries(
                    forecaster=f, series=series, cv=cv, param_grid=param_grid,
                    metric="mean_absolute_error", n_jobs=n_jobs,
                    return_best=False, show_progress=False,
                )
            return fingerprint(res[["mean_absolute_error"]])
        return run

    if kind == "backtest_stats":
        from skforecast.model_selection import backtesting_stats
        from skforecast.recursive import ForecasterStats
        y = make_y(n)
        freeze_params = spec.get("freeze_params", True)

        def run(n_jobs):
            f = ForecasterStats(estimator=make_estimator(est_name, est_params))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = backtesting_stats(
                    forecaster=f, y=y, cv=cv, metric="mean_absolute_error",
                    n_jobs=n_jobs, freeze_params=freeze_params,
                    show_progress=False,
                )
            return fingerprint(res)
        return run

    raise ValueError(kind)
