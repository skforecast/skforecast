"""
T3, part 3: do the prediction intervals change when the bins of `fit` are
learned with a random sample of the rows of X_train instead of all of them?

Multiple series with heteroscedastic noise (more noise at the daily peak), so
that the bins matter. Train / calibration / test split. Nominal coverage 80 %.

Flow A: default `fit`, out-of-sample residuals of the calibration period with
`set_out_sample_residuals`, intervals on test. From `fit` only the edges of the
bins are used.
Flow B: intervals on test with in-sample residuals. From `fit`, the edges and
the stored residuals are used; with a sample, both come from the sampled rows.

python dev/preprocessing_speedups/t3_intervals.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from data import env_report  # noqa: E402

from skforecast.model_selection import (  # noqa: E402
    TimeSeriesFold,
    backtesting_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries  # noqa: E402

warnings.filterwarnings("ignore")

N_SERIES, LEN_MIN, LEN_MAX = 100, 2000, 5000
N_TEST, N_CAL, STEPS = 336, 672, 24
INTERVAL = [0.1, 0.9]


def make_series(seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(LEN_MIN, LEN_MAX + 1, size=N_SERIES)
    full_len = int(lengths.max())
    index = pd.date_range(end="2025-01-01", periods=full_len, freq="h")
    season = np.sin(2 * np.pi * np.arange(full_len) / 24)
    sigma = 0.3 + 1.5 * ((season + 1) / 2) ** 2
    series = {}
    for i, n in enumerate(lengths):
        ar = np.zeros(full_len)
        eps = rng.normal(0, 0.3, full_len)
        for t in range(1, full_len):
            ar[t] = 0.7 * ar[t - 1] + eps[t]
        y = (
            rng.uniform(10, 100)
            + rng.uniform(3, 8) * season
            + ar
            + sigma * rng.normal(0, 1, full_len)
        )
        name = f"series_{i:03d}"
        series[name] = pd.Series(y[-n:], index=index[-n:], name=name)
    return series


def evaluate(predictions):
    """Coverage, width and coverage by third of the predicted value per series."""
    p = predictions
    inside = (p["y_true"] >= p["lower_bound"]) & (p["y_true"] <= p["upper_bound"])
    third = p.groupby("level")["pred"].transform(
        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
    )
    return {
        "coverage": 100 * inside.mean(),
        "width": (p["upper_bound"] - p["lower_bound"]).mean(),
        "cov. low third": 100 * inside[third == 0].mean(),
        "cov. high third": 100 * inside[third == 2].mean(),
    }


def main():
    print(env_report())
    series = make_series()
    train = {k: v.iloc[: -(N_TEST + N_CAL)] for k, v in series.items()}

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(n_estimators=100, verbose=-1, random_state=123),
        lags=24,
        encoding="ordinal",
        transformer_series=StandardScaler(),
    )
    forecaster.fit(series=train)
    ws = forecaster.window_size
    cal = {k: v.iloc[-(N_TEST + N_CAL + ws): -N_TEST] for k, v in series.items()}
    test = {k: v.iloc[-(N_TEST + ws):] for k, v in series.items()}
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=None, refit=False)

    # Prediction over X_train, as `fit` does
    out = forecaster._create_train_X_y(series=train, store_last_window=False)
    X_train, y_train = out[0], out[1].to_numpy()
    level_slices = forecaster._get_level_row_slices(X_train=X_train)
    levels = list(level_slices.keys())
    y_pred_train = forecaster.estimator.predict(X_train)
    rows_level = np.array([level_slices[k].stop - level_slices[k].start for k in levels])
    print(
        f"\nseries: {len(levels)} | rows X_train: {len(X_train):,} | rows per series: "
        f"min {rows_level.min()}, median {int(np.median(rows_level))}, max {rows_level.max()}"
    )
    print(f"nominal coverage: {100 * (INTERVAL[1] - INTERVAL[0]):.0f} %, conformal")

    # Out-of-sample predictions of the calibration period (they do not depend
    # on the bins)
    _, pred_cal = backtesting_forecaster_multiseries(
        forecaster, series=cal, cv=cv, metric="mean_absolute_error", show_progress=False
    )
    cal_pred = {k: g["pred"] for k, g in pred_cal.groupby("level")}
    cal_true = {k: g["y_true"] for k, g in pred_cal.groupby("level")}

    def set_bins(kind, value, seed, store):
        """What `fit` would leave if it predicted only on the sampled rows."""
        rng = np.random.default_rng(seed)
        idx_all = []
        for k in levels:
            sl = level_slices[k]
            n = sl.stop - sl.start
            if kind == "full":
                m = n
            elif kind == "cap":
                m = min(n, value)
            else:
                m = max(int(round(n * value)), 1)
            idx = np.sort(rng.choice(n, size=m, replace=False)) + sl.start
            idx_all.append(idx)
            forecaster._binning_in_sample_residuals(
                level=k, y_true=y_train[idx], y_pred=y_pred_train[idx],
                store_in_sample_residuals=store, random_state=123,
            )
        idx_all = np.concatenate(idx_all)
        forecaster._binning_in_sample_residuals(
            level="_unknown_level", y_true=y_train[idx_all],
            y_pred=y_pred_train[idx_all], store_in_sample_residuals=store,
            random_state=123,
        )
        return len(idx_all)

    def backtest(use_in_sample, binned=True):
        _, pred = backtesting_forecaster_multiseries(
            forecaster, series=test, cv=cv, metric="mean_absolute_error",
            interval=INTERVAL, interval_method="conformal",
            use_in_sample_residuals=use_in_sample, use_binned_residuals=binned,
            show_progress=False,
        )
        return evaluate(pred)

    variants = [
        ("full", None), ("cap", 1000), ("cap", 500), ("cap", 250),
        ("frac", 0.25), ("frac", 0.1),
    ]
    for flow in ("A", "B"):
        title = (
            "A: out-of-sample residuals (from fit, only the edges)"
            if flow == "A"
            else "B: in-sample residuals (from fit, edges and residuals)"
        )
        print(f"\n--- Flow {title} ---")
        rows = []
        t0 = time.perf_counter()
        for kind, value in variants:
            seeds = [0] if kind == "full" else [1, 2, 3]
            results = []
            for seed in seeds:
                n_rows = set_bins(kind, value, seed, store=(flow == "B"))
                if flow == "A":
                    forecaster.set_out_sample_residuals(y_true=cal_true, y_pred=cal_pred)
                results.append(backtest(use_in_sample=(flow == "B")))
                if kind == "full":
                    ref = dict(results[0])
                    not_binned = backtest(use_in_sample=(flow == "B"), binned=False)
            res = pd.DataFrame(results)
            row = {
                "bins learned with": "all rows" if kind == "full" else f"{kind} {value}",
                "rows predicted": f"{100 * n_rows / len(X_train):.0f} %",
            }
            for col in res.columns:
                row[col] = f"{res[col].mean():.2f}"
                if kind != "full":
                    row[col] += f" [{res[col].min():.2f}, {res[col].max():.2f}]"
            rows.append(row)
        rows.append(
            {
                "bins learned with": "(reference: residuals not binned)",
                "rows predicted": "",
                **{k: f"{v:.2f}" for k, v in not_binned.items()},
            }
        )
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"reference all rows: {ref} | time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
