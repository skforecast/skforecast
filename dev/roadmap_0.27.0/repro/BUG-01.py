"""
BUG-01: backtesting with `initial_train_size=None` (TimeSeriesFold default) on an
already fitted forecaster evaluates in-sample without any warning.

Self-contained: uses the h2o demo dataset (downloaded) and falls back to a
synthetic monthly series if there is no network.

Run: python dev/roadmap_0.27.0/repro/BUG-01.py
"""

import warnings

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from lightgbm import LGBMRegressor

import skforecast
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    backtesting_forecaster_multiseries,
    grid_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries


def load_y() -> tuple[pd.Series, str]:
    try:
        from skforecast.datasets import load_demo_dataset

        return load_demo_dataset(verbose=False), "h2o (load_demo_dataset)"
    except Exception:
        rng = np.random.default_rng(0)
        idx = pd.date_range("1991-07-01", periods=204, freq="MS")
        t = np.arange(204)
        y = 0.4 + 0.004 * t + 0.15 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.05, 204)
        return pd.Series(y, index=idx, name="y"), "synthetic"


def run(forecaster, cv, y, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metric, preds = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            show_progress=False,
            **kwargs,
        )
    return metric.iloc[0, 0], preds, [f"{x.category.__name__}: {x.message}" for x in w]


def make_forecaster():
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=123, verbose=-1), lags=12
    )


print(
    f"skforecast {skforecast.__version__} | sklearn {sklearn.__version__} | "
    f"lightgbm {lightgbm.__version__} | pandas {pd.__version__}"
)
y, source = load_y()
print(f"Data: {source}, {len(y)} obs, {y.index.min().date()} .. {y.index.max().date()}")
n_test = 60
end_train = y.index[-n_test - 1]

# ---------------------------------------------------------------------------
# Case A: typical notebook flow. fit on all data, then backtesting with a
# TimeSeriesFold where `initial_train_size` was forgotten (default None).
# ---------------------------------------------------------------------------
print("\n=== Case A: fit(y) then backtesting with TimeSeriesFold(steps=12) ===")
forecaster = make_forecaster()
forecaster.fit(y=y)
mae_in, preds_in, warns_in = run(forecaster, TimeSeriesFold(steps=12), y)
print(f"training_range_: {forecaster.training_range_.tolist()}")
print(f"predictions: {len(preds_in)} ({preds_in.index.min().date()} .. {preds_in.index.max().date()})")
n_overlap = preds_in.index.isin(
    y.loc[forecaster.training_range_[0]:forecaster.training_range_[1]].index
).sum()
print(f"predictions inside training range: {n_overlap} / {len(preds_in)}")
print(f"MAE (initial_train_size=None): {mae_in:.4f}")
print(f"warnings: {warns_in}")

# Reference: proper out-of-sample backtesting over the last n_test observations,
# compared with the in-sample predictions of case A over the same period.
mae_oos, preds_oos, _ = run(
    make_forecaster(), TimeSeriesFold(steps=12, initial_train_size=len(y) - n_test), y
)
mae_in_same = (
    (y.loc[preds_oos.index] - preds_in.loc[preds_oos.index, "pred"]).abs().mean()
)
print(
    f"MAE out-of-sample last {n_test} obs (initial_train_size={len(y) - n_test}): "
    f"{mae_oos:.4f}"
)
print(f"MAE in-sample same last {n_test} obs (from case A): {mae_in_same:.4f}")
print(f"ratio out-of-sample / in-sample (same period): {mae_oos / mae_in_same:.1f}x")

# ---------------------------------------------------------------------------
# Case B: fit on a train split (data_train), then backtesting on the full series
# forgetting `initial_train_size`: train (in-sample) and test (out-of-sample)
# predictions are mixed in a single metric.
# ---------------------------------------------------------------------------
print("\n=== Case B: fit(y_train) then backtesting on full y, initial_train_size=None ===")
forecaster_b = make_forecaster()
forecaster_b.fit(y=y.loc[:end_train])
mae_b, preds_b, warns_b = run(forecaster_b, TimeSeriesFold(steps=12), y)
in_train = preds_b.index <= forecaster_b.training_range_[1]
print(f"predictions: {len(preds_b)}, inside training range: {in_train.sum()}")
print(f"MAE mixed (reported): {mae_b:.4f}  | true out-of-sample on test: {mae_oos:.4f}")
print(f"warnings: {warns_b}")

# ---------------------------------------------------------------------------
# Case C: multiseries, same pattern.
# ---------------------------------------------------------------------------
print("\n=== Case C: ForecasterRecursiveMultiSeries, fit(series) then backtesting ===")
rng = np.random.default_rng(1)
series = {
    "s1": y.rename("s1"),
    "s2": (y * 1.5 + rng.normal(0, 0.05, len(y))).rename("s2"),
}
fm = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=123, verbose=-1), lags=12
)
fm.fit(series=series)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m_in, p_in = backtesting_forecaster_multiseries(
        forecaster=fm, series=series, cv=TimeSeriesFold(steps=12),
        metric="mean_absolute_error", show_progress=False,
    )
warns_c = [f"{x.category.__name__}: {x.message}" for x in w]
fm2 = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=123, verbose=-1), lags=12
)
m_oos, _ = backtesting_forecaster_multiseries(
    forecaster=fm2, series=series,
    cv=TimeSeriesFold(steps=12, initial_train_size=len(y) - n_test),
    metric="mean_absolute_error", show_progress=False,
)
print("initial_train_size=None (in-sample):")
print(m_in.to_string(index=False))
print(f"initial_train_size={len(y) - n_test} (out-of-sample, last {n_test} obs):")
print(m_oos.to_string(index=False))
print(f"warnings: {warns_c}")

# ---------------------------------------------------------------------------
# Case D: hyperparameter search is NOT affected (set_lags/set_params reset
# is_fitted, so NotFittedError is raised for every candidate).
# ---------------------------------------------------------------------------
print("\n=== Case D: grid_search_forecaster with initial_train_size=None ===")
fd = make_forecaster()
fd.fit(y=y)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    res = grid_search_forecaster(
        fd, y=y, cv=TimeSeriesFold(steps=12), param_grid={"n_estimators": [10, 100]},
        lags_grid=[3, 12], metric="mean_absolute_error", return_best=False,
        show_progress=False,
    )
print(f"results rows: {len(res)}; last warning: {str(w[-1].message)[:90]}")

# ---------------------------------------------------------------------------
# Case E: the error raised for an unfitted forecaster suggests fitting it, which
# leads straight to case A.
# ---------------------------------------------------------------------------
print("\n=== Case E: unfitted forecaster with TimeSeriesFold(steps=12) ===")
try:
    run(make_forecaster(), TimeSeriesFold(steps=12), y)
except Exception as e:
    print(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(
    f"OBSERVED: case A MAE {mae_in:.4f} over {len(preds_in)} predictions, all inside "
    f"the training range, warnings={len(warns_in)}."
)
print(
    f"EXPECTED: a warning saying the {n_overlap} predictions are in-sample. Same "
    f"last {n_test} obs: in-sample MAE {mae_in_same:.4f} vs out-of-sample "
    f"{mae_oos:.4f} ({mae_oos / mae_in_same:.1f}x). Case B mixes both: reported "
    f"{mae_b:.4f}, real test MAE {mae_oos:.4f}."
)
