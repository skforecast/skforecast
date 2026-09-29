"""
Generate the data shown in the animations of the documentation home page.

The home page (docs/overrides/home.html) embeds real skforecast outputs, not
drawings. This script computes them and writes a compact JSON file that the
template includes at build time:

  1. hero   - daily electricity demand in Victoria (vic_electricity, aggregated
              to daily totals) with 56-day forecasts and a 2-fold backtest of
              LightGBM, Chronos-2 and ARIMA.
  2. global - 36 random series of australia_tourism forecast by a single
              global LightGBM model trained on all 304 series.

The JSON is committed to the repository, so building the docs does not need
LightGBM or Chronos. Rerun this script only when the data, the models or the
skforecast API shown on the home page change. See tools/docs/home_page/README.md.

Usage
-----
Run from the repository root, so that the local skforecast is imported:

    PYTHONPATH=. python tools/docs/home_page/generate_home_data.py              # both
    PYTHONPATH=. python tools/docs/home_page/generate_home_data.py --only hero  # one
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_foundation,
    backtesting_stats,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveMultiSeries,
    ForecasterStats,
)
from skforecast.stats import Arima

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs" / "overrides" / "partials" / "home-data.json"

# Hero animation: the test period is the last HERO_STEPS days of the dataset,
# evaluated with 2 backtesting folds of HERO_FOLD days.
HERO_STEPS = 56
HERO_FOLD = 28
HERO_HISTORY = 160  # days of history kept in the JSON (the chart shows 150)
INTERVAL = [0.1, 0.9]
CHRONOS_MODEL_ID = "autogluon/chronos-2-small"
# Exogenous variables given to the three models. For ARIMA, this set was chosen
# among other subsets on the validation period (select_model_configs.py).
EXOG_COLUMNS = ["temp_max", "temp_mean", "holiday"]

# Global models wall.
GLOBAL_STEPS = 8
GLOBAL_N_SERIES = 36
GLOBAL_HISTORY = 24  # quarters of history kept per series
RANDOM_STATE = 123


def check_local_skforecast() -> None:
    """
    Stop if skforecast is not imported from this repository.

    When a script is run as `python tools/docs/home_page/<script>.py`, Python puts
    the script folder, not the repository root, first in `sys.path`. Without an
    editable install, `import skforecast` then loads the installed release,
    which may lack the fixes of the working tree and give different results.
    """
    location = Path(skforecast.__file__).resolve()
    if not location.is_relative_to(ROOT):
        raise SystemExit(
            f"skforecast is imported from {location.parent}, not from this "
            "repository. Run the script from the repository root with "
            "`PYTHONPATH=. python tools/docs/home_page/...`, or install skforecast "
            "in editable mode (`pip install -e .`)."
        )


def lightgbm_forecaster() -> ForecasterRecursive:
    """
    LightGBM forecaster of the hero animation.

    The configuration was selected on the 56 days before the test period with
    tools/docs/home_page/select_model_configs.py. Rerun that script, not this
    one, to choose a new configuration.
    """
    return ForecasterRecursive(
        estimator=LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        lags=7,
        window_features=RollingFeatures(stats=["mean"], window_sizes=7),
        calendar_features=CalendarFeatures(features=["day_of_week", "day_of_year"]),
    )


def arima_forecaster() -> ForecasterStats:
    """ARIMA forecaster of the hero animation, with automatic order selection."""
    return ForecasterStats(estimator=Arima(order=None, seasonal_order=None, m=7))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_daily_electricity() -> pd.DataFrame:
    """
    Daily electricity demand (GWh), maximum and mean temperature and holidays
    in Victoria, aggregated from the half-hourly vic_electricity dataset.
    """
    data = fetch_dataset("vic_electricity", verbose=False)
    data["Date"] = pd.to_datetime(data["Date"])
    daily = data.groupby("Date").agg(
        demand=("Demand", "sum"),
        temp_max=("Temperature", "max"),
        temp_mean=("Temperature", "mean"),
        holiday=("Holiday", "max"),
    )
    # The first and last days in UTC are incomplete in local time
    daily = daily.loc["2012-01-01":"2014-12-31"].asfreq("D")
    daily.index.name = None
    daily["demand"] = daily["demand"] / 1000  # MWh to GWh
    daily["holiday"] = daily["holiday"].astype(float)

    return daily


# ---------------------------------------------------------------------------
# Hero animation
# ---------------------------------------------------------------------------


def _round(values, decimals: int = 2) -> list[float]:
    return [round(float(v), decimals) for v in values]


def _pack(predictions: pd.DataFrame, backtest: pd.DataFrame, mae: float) -> dict:
    return {
        "pred": _round(predictions["pred"]),
        "lo": _round(predictions["lower_bound"]),
        "hi": _round(predictions["upper_bound"]),
        "bt": _round(backtest["pred"]),
        "mae": round(float(mae), 2),
    }


def compute_hero() -> dict:
    """
    Forecasts, prediction intervals and backtesting of the three models shown
    in the hero animation.
    """
    daily = load_daily_electricity()
    y = daily["demand"]
    exog = daily[EXOG_COLUMNS]

    n_train = len(y) - HERO_STEPS
    y_train = y.iloc[:n_train]
    exog_train = exog.iloc[:n_train]
    exog_test = exog.iloc[n_train:]
    cv = TimeSeriesFold(
        steps=HERO_FOLD, initial_train_size=n_train, refit=False, verbose=False
    )
    models = {}

    # LightGBM with exogenous variables. Its prediction intervals use
    # out-of-sample residuals from a backtest on the days before the test set.
    forecaster = lightgbm_forecaster()
    cv_val = TimeSeriesFold(
        steps=HERO_FOLD,
        initial_train_size=n_train - HERO_STEPS,
        refit=False,
        verbose=False,
    )
    _, backtest_val = backtesting_forecaster(
        forecaster=forecaster,
        y=y_train,
        exog=exog_train,
        cv=cv_val,
        metric="mean_absolute_error",
        show_progress=False,
    )
    forecaster.fit(y=y_train, exog=exog_train)
    forecaster.set_out_sample_residuals(
        y_true=y_train.loc[backtest_val.index], y_pred=backtest_val["pred"]
    )
    predictions = forecaster.predict_interval(
        steps=HERO_STEPS,
        exog=exog_test,
        interval=INTERVAL,
        method="bootstrapping",
        n_boot=500,
        use_in_sample_residuals=False,
        use_binned_residuals=False,
    )
    metric, backtest = backtesting_forecaster(
        forecaster=lightgbm_forecaster(),
        y=y,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    models["lightgbm"] = _pack(predictions, backtest, metric.iloc[0, 0])

    # Chronos-2, zero-shot, with temperature and holidays as covariates
    forecaster = ForecasterFoundation(estimator=FoundationModel(CHRONOS_MODEL_ID))
    forecaster.fit(series=y_train, exog=exog_train)
    predictions = forecaster.predict_interval(
        steps=HERO_STEPS, exog=exog_test, interval=INTERVAL
    )
    metric, backtest = backtesting_foundation(
        forecaster=forecaster,
        series=y,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    models["chronos"] = _pack(
        predictions, backtest, metric["mean_absolute_error"].iloc[0]
    )

    # ARIMA with automatic order selection and the same exogenous variables
    forecaster = arima_forecaster()
    forecaster.fit(y=y_train, exog=exog_train)
    predictions = forecaster.predict_interval(
        steps=HERO_STEPS, exog=exog_test, interval=INTERVAL
    )
    metric, backtest = backtesting_stats(
        forecaster=arima_forecaster(),
        y=y,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    models["arima"] = _pack(predictions, backtest, metric.iloc[0, -1])

    y_test = y.iloc[n_train:].to_numpy()
    for name, values in models.items():
        lo, hi = np.array(values["lo"]), np.array(values["hi"])
        coverage = np.mean((y_test >= lo) & (y_test <= hi))
        print(
            f"  {name:9s} MAE {values['mae']:6.2f} GWh | "
            f"80% interval coverage {coverage:.0%} | "
            f"mean width {np.mean(hi - lo):.1f} GWh"
        )

    keep = HERO_HISTORY + HERO_STEPS
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in y.index[-keep:]],
        "y": _round(y.iloc[-keep:]),
        "temp": _round(daily["temp_max"].iloc[-keep:], 1),
        "train_end": y.index[n_train - 1].strftime("%Y-%m-%d"),
        "fold": HERO_FOLD,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Global models wall
# ---------------------------------------------------------------------------


def compute_global() -> dict:
    """
    Forecasts of a single global LightGBM model trained on all the series of
    australia_tourism, for a random sample of series.
    """
    tourism = fetch_dataset("australia_tourism", verbose=False)
    tourism["series"] = tourism["Region"] + " · " + tourism["Purpose"]
    series = tourism.pivot_table(
        index=tourism.index, columns="series", values="Trips"
    ).asfreq("QS")
    series.index.name = None
    train = series.iloc[:-GLOBAL_STEPS]

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        lags=8,
        window_features=RollingFeatures(stats=["mean"], window_sizes=4),
        calendar_features=CalendarFeatures(features=["quarter"]),
        encoding="ordinal",
        transformer_series=StandardScaler(),
    )
    forecaster.fit(series=train)
    predictions = forecaster.predict(steps=GLOBAL_STEPS)

    rng = np.random.default_rng(RANDOM_STATE)
    sample = sorted(
        rng.choice(series.columns.to_numpy(), size=GLOBAL_N_SERIES, replace=False)
    )
    keep = GLOBAL_HISTORY + GLOBAL_STEPS
    wall = {
        name: {
            "y": _round(series[name].iloc[-keep:], 1),
            "pred": _round(predictions.loc[predictions["level"] == name, "pred"], 1),
        }
        for name in sample
    }
    print(
        f"  {series.shape[1]} series, {GLOBAL_N_SERIES} shown, "
        f"{GLOBAL_STEPS}-quarter forecasts"
    )

    return {"steps": GLOBAL_STEPS, "n_series": int(series.shape[1]), "series": wall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--only",
        choices=["hero", "global"],
        help="Recompute only one part and keep the other from the existing file.",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help=f"Output file (default: {OUTPUT})."
    )
    args = parser.parse_args()
    check_local_skforecast()

    warnings.filterwarnings("ignore")
    data = {}
    if args.only and args.output.exists():
        data = json.loads(args.output.read_text(encoding="utf-8"))

    if args.only in (None, "hero"):
        print("Hero animation (vic_electricity, daily):")
        data["hero"] = compute_hero()
    if args.only in (None, "global"):
        print("Global models wall (australia_tourism):")
        data["global"] = compute_global()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    shown = (
        args.output.relative_to(ROOT)
        if args.output.is_relative_to(ROOT)
        else args.output
    )
    print(f"Written {shown}")


if __name__ == "__main__":
    main()
