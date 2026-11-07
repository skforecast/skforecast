"""
Backtesting of the model comparison table of the user guide
(`docs/user_guides/foundation-forecasting-models.ipynb`): `vic_electricity`
aggregated to hourly, 2012-01-01 to 2014-12-30, training until 2014-11-30,
folds of 24 steps without refit, `context_length=500` and MAE. The exogenous
variables (`Temperature`, `Holiday`) are passed only when requested.

Prints the metric, the elapsed time, the versions and the hardware, and saves
the predictions so that two runs can be compared value by value.

Usage:
    python backtest_guide.py --model-id autogluon/chronos-2-small --exog \
        --out chronos/backtest_default.csv [--kwargs '{"device": "cpu"}']
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from skforecast.datasets import fetch_dataset
from skforecast.foundation import ForecasterFoundation, FoundationModel
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

PACKAGES = (
    "skforecast", "torch", "numpy", "pandas", "scikit-learn", "transformers",
    "huggingface_hub", "chronos-forecasting", "timesfm", "uni2ts", "tabicl",
    "tabpfn", "tabpfn-time-series", "tfc-t0", "synthefy-nori", "tsicl",
)


def versions() -> dict:
    out = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            pass
    return out


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()}"


def load_data() -> pd.DataFrame:
    data = fetch_dataset(name="vic_electricity", verbose=False)
    data = data.drop(columns="Date")
    data = (
        data
        .resample(rule="h", closed="left", label="right")
        .agg({"Demand": "mean", "Temperature": "mean", "Holiday": "mean"})
    )
    return data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00", :].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-length", type=int, default=500)
    parser.add_argument("--exog", action="store_true")
    parser.add_argument("--kwargs", default="{}", help="JSON with extra FoundationModel arguments.")
    parser.add_argument("--out", default=None, help="CSV file for the predictions.")
    args = parser.parse_args()

    data = load_data()
    end_train = "2014-11-30 23:59:00"
    cv = TimeSeriesFold(
        steps=24, initial_train_size=len(data.loc[:end_train]), refit=False
    )
    kwargs = json.loads(args.kwargs)
    estimator = FoundationModel(
        model_id=args.model_id, context_length=args.context_length, **kwargs
    )
    forecaster = ForecasterFoundation(estimator=estimator)
    exog = data[["Temperature", "Holiday"]] if args.exog else None

    start = time.perf_counter()
    metrics, predictions = backtesting_foundation(
        forecaster=forecaster,
        series=data["Demand"],
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        suppress_warnings=True,
        show_progress=False,
    )
    elapsed = time.perf_counter() - start

    result = {
        "model_id": args.model_id,
        "context_length": args.context_length,
        "exog": args.exog,
        "kwargs": kwargs,
        "mae": round(float(metrics["mean_absolute_error"].iloc[0]), 4),
        "elapsed_s": round(elapsed, 2),
        "n_predictions": len(predictions),
        "versions": versions(),
        "hardware": hardware(),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        predictions.to_csv(args.out)


if __name__ == "__main__":
    main()
