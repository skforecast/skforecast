"""
BUG-06: ForecasterFoundation uses two different point forecasts.

`predict()` returns the adapter point forecast (the mean for TabICL and Nori by
default, or TabPFN with point_estimate != 'median'), while `predict_interval()`
always builds `pred` from the 0.5 quantile, and `backtesting_foundation` measures
the metric on `pred` without `quantiles` but on `q_0.5` with `quantiles` (and
drops `pred`). In the ML forecasters, `predict_interval()['pred']` is exactly
`predict()`.

Part A: fake Nori backend (no network, deterministic).
Part B: reference behaviour of an ML forecaster.
Part C: real TabICL (only if `tabicl` is installed; downloads the checkpoint).

Run: python dev/roadmap_0.27.0/repro/BUG-06.py
"""

import importlib.util
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from skforecast.foundation import ForecasterFoundation, FoundationModel
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")


class FakeNoriRegressor:
    """
    Mimics `synthefy_nori.NoriRegressor`: in-context `fit`, and `predict` with
    `output_type` in {'mean', 'median', 'mode', 'quantiles'}. The predictive
    distribution is the empirical distribution of the context target.
    """

    def fit(self, X, y):
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X, *, output_type="mean", quantiles=None):
        n = len(X)
        if output_type == "quantiles":
            return np.tile(np.quantile(self.y_, quantiles).reshape(-1, 1), (1, n))
        value = {
            "mean": self.y_.mean(),
            "median": np.median(self.y_),
            "mode": self.y_.mean(),
        }[output_type]
        return np.full(n, value)


print("=" * 78)
print("PART A: Nori (point_estimate='mean' by default), skewed series (lognormal)")
print("=" * 78)

rng = np.random.default_rng(1)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
y = pd.Series(rng.lognormal(0, 1, 200), index=idx, name="y")

forecaster = ForecasterFoundation(FoundationModel("Synthefy/Nori"))
# ForecasterFoundation clones the estimator, so inject the fake backend after.
forecaster.estimator.adapter._model = FakeNoriRegressor()
forecaster.fit(y)
print("point_estimate:", forecaster.estimator.adapter.point_estimate)

p = forecaster.predict(steps=3)
pi = forecaster.predict_interval(steps=3, interval=[0.1, 0.9])
print("predict()['pred']          :", p["pred"].round(3).tolist())
print("predict_interval()['pred'] :", pi["pred"].round(3).tolist())
print(
    "OBSERVED: equal?", np.allclose(p["pred"], pi["pred"]),
    "| EXPECTED: True (as in ML forecasters)",
)

cv = TimeSeriesFold(steps=7, initial_train_size=100, refit=False)
m1, b1 = backtesting_foundation(
    forecaster, y, cv, metric="mean_absolute_error", show_progress=False
)
m2, b2 = backtesting_foundation(
    forecaster, y, cv, metric="mean_absolute_error",
    quantiles=[0.1, 0.5, 0.9], show_progress=False,
)
print(f"MAE backtesting without quantiles : {m1.iloc[0, 0]:.4f}")
print(f"MAE backtesting with quantiles    : {m2.iloc[0, 0]:.4f}")
print("columns without quantiles:", list(b1.columns))
print("columns with quantiles   :", list(b2.columns))
print(
    "OBSERVED: same MAE?", np.isclose(m1.iloc[0, 0], m2.iloc[0, 0]),
    "| EXPECTED: True (requesting quantiles should not change the point metric)",
)

# The point_estimate tuned by bayesian_search_foundation (evaluated without
# quantiles) only affects predict(); predict_interval() ignores it.
forecaster.set_params({"point_estimate": "median"})  # resets is_fitted
forecaster.fit(y)
p_med = forecaster.predict(steps=3)
pi_med = forecaster.predict_interval(steps=3, interval=[0.1, 0.9])
print("point_estimate='median' -> predict()['pred']         :",
      p_med["pred"].round(3).tolist())
print("point_estimate='median' -> predict_interval()['pred']:",
      pi_med["pred"].round(3).tolist(),
      "(identical to the 'mean' run:", np.allclose(pi["pred"], pi_med["pred"]), ")")

print()
print("=" * 78)
print("PART B: reference, ForecasterRecursive (ML)")
print("=" * 78)
f_ml = ForecasterRecursive(estimator=LinearRegression(), lags=7)
f_ml.fit(y=y, store_in_sample_residuals=True)
p_ml = f_ml.predict(steps=3)
pi_ml = f_ml.predict_interval(
    steps=3, interval=[0.1, 0.9], method="bootstrapping", n_boot=200
)
print("predict()                  :", p_ml.round(3).tolist())
print("predict_interval()['pred'] :", pi_ml["pred"].round(3).tolist())
print("equal?", np.allclose(p_ml.to_numpy(), pi_ml["pred"].to_numpy()))

print()
print("=" * 78)
print("PART C: real TabICL (point_estimate='mean' by default)")
print("=" * 78)
if importlib.util.find_spec("tabicl") is None:
    print("tabicl not installed: skipped (pip install 'tabicl[forecast]').")
else:
    # Intermittent daily demand (Poisson, higher at weekends): a typical
    # retail series where mean and median of the predictive distribution differ.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=600, freq="D")
    lam = 0.6 * (1 + 0.8 * (idx.dayofweek >= 5))
    demand = pd.Series(rng.poisson(lam).astype(float), index=idx, name="demand")

    t0 = time.time()
    f_tabicl = ForecasterFoundation(
        FoundationModel("soda-inria/tabicl", context_length=365)
    )
    f_tabicl.fit(demand.iloc[:-7])
    p_t = f_tabicl.predict(steps=7)
    pi_t = f_tabicl.predict_interval(steps=7, interval=[0.1, 0.9])
    print("predict()['pred']          :", p_t["pred"].round(2).tolist())
    print("predict_interval()['pred'] :", pi_t["pred"].round(2).tolist())

    cv = TimeSeriesFold(steps=7, initial_train_size=len(demand) - 56, refit=False)
    m1, b1 = backtesting_foundation(
        f_tabicl, demand, cv, metric="mean_absolute_error", show_progress=False
    )
    m2, b2 = backtesting_foundation(
        f_tabicl, demand, cv, metric="mean_absolute_error",
        quantiles=[0.1, 0.5, 0.9], show_progress=False,
    )
    diff = (b1["pred"] - b2["q_0.5"]).abs()
    print(f"MAE backtesting without quantiles : {m1.iloc[0, 0]:.4f}")
    print(f"MAE backtesting with quantiles    : {m2.iloc[0, 0]:.4f} "
          f"({100 * (m1.iloc[0, 0] / m2.iloc[0, 0] - 1):+.1f}%)")
    print(f"mean |pred - q_0.5| = {diff.mean():.3f} "
          f"({100 * diff.mean() / demand.mean():.0f}% of the mean demand)")
    print(f"time: {time.time() - t0:.0f} s")

print()
print("NOTE: TimesFM 2.5 is NOT affected: in timesfm (2p5 torch/flax) the point")
print("forecast is `full_forecast[..., 5]`, i.e. the 0.5 quantile, so predict()")
print("and predict_interval() agree. Chronos-2, TimesFM 3, Moirai-2, T0 and")
print("TS-ICL also use the median as point forecast.")
