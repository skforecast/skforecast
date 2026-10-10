"""
Select the model settings used in the hero animation of the home page.

The animation compares LightGBM, Chronos-2 and ARIMA on the last 56 days of
the daily electricity demand dataset. To keep that comparison honest, the
settings that are not fixed are chosen on a validation period (the 56 days
before the test period), never on the test period itself:

  1. LightGBM: lags, window and calendar features, among a few candidates.
  2. ARIMA: the subset of exogenous variables. Chronos-2 is zero-shot and uses
     every exogenous variable, so it has nothing to select.

This script backtests the candidates on the validation period and prints their
error. Copy the winners into `lightgbm_forecaster()` and `EXOG_COLUMNS` in
tools/docs/home_page/generate_home_data.py, then regenerate the data. With
`--windows`, it also repeats the comparison of the three models on other test
windows, to check that the ranking is not specific to the period shown.

Usage
-----
Run from the repository root, so that the local skforecast is imported:

    PYTHONPATH=. python tools/docs/home_page/select_model_configs.py
    PYTHONPATH=. python tools/docs/home_page/select_model_configs.py --windows
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd
from generate_home_data import (
    CHRONOS_MODEL_ID,
    HERO_FOLD,
    HERO_STEPS,
    RANDOM_STATE,
    arima_forecaster,
    check_local_skforecast,
    load_daily_electricity,
)
from lightgbm import LGBMRegressor

from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_foundation,
    backtesting_stats,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Every exogenous variable available. LightGBM and Chronos-2 use all of them.
ALL_EXOG = ["temp_max", "temp_mean", "holiday"]
# Candidate subsets of exogenous variables for ARIMA
ARIMA_EXOG_CANDIDATES = [
    [],
    ["temp_max"],
    ["temp_max", "holiday"],
    ["temp_max", "temp_mean", "holiday"],
]
# Test windows checked with --windows (first day of each 56-day window)
EXTRA_WINDOWS = ["2014-01-06", "2014-06-02"]


def lightgbm_candidates() -> dict[str, dict]:
    """Candidate configurations, as keyword arguments of ForecasterRecursive."""
    return {
        "lags 7, day of week, month": dict(
            lags=7,
            calendar_features=CalendarFeatures(features=["day_of_week", "month"]),
        ),
        "lags 14, rolling mean 7, day of week, month": dict(
            lags=14,
            window_features=RollingFeatures(stats=["mean"], window_sizes=7),
            calendar_features=CalendarFeatures(features=["day_of_week", "month"]),
        ),
        "lags 7, rolling mean 7, day of week, day of year": dict(
            lags=7,
            window_features=RollingFeatures(stats=["mean"], window_sizes=7),
            calendar_features=CalendarFeatures(features=["day_of_week", "day_of_year"]),
        ),
    }


def lightgbm_estimator() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def _cv(initial_train_size: int) -> TimeSeriesFold:
    return TimeSeriesFold(
        steps=HERO_FOLD,
        initial_train_size=initial_train_size,
        refit=False,
        verbose=False,
    )


def select(daily: pd.DataFrame, n_train: int) -> tuple[str, list[str]]:
    """
    Backtest the candidates on the HERO_STEPS days before `n_train` and return
    the best LightGBM configuration and the best ARIMA exogenous variables.
    """
    y, exog = daily["demand"].iloc[:n_train], daily[ALL_EXOG].iloc[:n_train]
    cv = _cv(n_train - HERO_STEPS)

    lightgbm = {}
    for name, kwargs in lightgbm_candidates().items():
        metric, _ = backtesting_forecaster(
            forecaster=ForecasterRecursive(estimator=lightgbm_estimator(), **kwargs),
            y=y,
            exog=exog,
            cv=cv,
            metric="mean_absolute_error",
            show_progress=False,
        )
        lightgbm[name] = metric.iloc[0, 0]
        print(f"    LightGBM {name:48s} validation MAE {lightgbm[name]:6.2f}")

    arima = {}
    for columns in ARIMA_EXOG_CANDIDATES:
        metric, _ = backtesting_stats(
            forecaster=arima_forecaster(),
            y=y,
            exog=exog[columns] if columns else None,
            cv=cv,
            metric="mean_absolute_error",
            show_progress=False,
        )
        arima[tuple(columns)] = metric.iloc[0, -1]
        label = ", ".join(columns) or "no exogenous variables"
        print(f"    ARIMA    {label:48s} validation MAE {arima[tuple(columns)]:6.2f}")

    best_lightgbm = min(lightgbm, key=lambda name: lightgbm[name])
    best_arima = list(min(arima, key=lambda columns: arima[columns]))
    print(f"  Selected LightGBM: {best_lightgbm}")
    print(f"  Selected ARIMA exogenous variables: {best_arima or 'none'}")

    return best_lightgbm, best_arima


def compare(
    daily: pd.DataFrame, n_train: int, lightgbm: str, arima_exog: list[str]
) -> None:
    """Backtest the selected LightGBM, Chronos-2 and ARIMA on the test window."""
    end = n_train + HERO_STEPS
    y, exog = daily["demand"].iloc[:end], daily[ALL_EXOG].iloc[:end]
    cv = _cv(n_train)
    lgbm, _ = backtesting_forecaster(
        forecaster=ForecasterRecursive(
            estimator=lightgbm_estimator(), **lightgbm_candidates()[lightgbm]
        ),
        y=y,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    chronos, _ = backtesting_foundation(
        forecaster=ForecasterFoundation(estimator=FoundationModel(CHRONOS_MODEL_ID)),
        series=y,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    arima, _ = backtesting_stats(
        forecaster=arima_forecaster(),
        y=y,
        exog=exog[arima_exog] if arima_exog else None,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    print(
        f"  Test MAE: LightGBM {lgbm.iloc[0, 0]:.2f} | "
        f"Chronos-2 {chronos['mean_absolute_error'].iloc[0]:.2f} | "
        f"ARIMA {arima.iloc[0, -1]:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Also compare the three models on other test windows.",
    )
    args = parser.parse_args()
    check_local_skforecast()
    warnings.filterwarnings("ignore")

    daily = load_daily_electricity()
    n_train = len(daily) - HERO_STEPS
    print(f"Test window shown on the home page, from {daily.index[n_train].date()}:")
    compare(daily, n_train, *select(daily, n_train))

    if args.windows:
        for start in EXTRA_WINDOWS:
            n_start = daily.index.get_loc(pd.Timestamp(start))
            print(f"Test window from {start}:")
            compare(daily, n_start, *select(daily, n_start))


if __name__ == "__main__":
    main()
