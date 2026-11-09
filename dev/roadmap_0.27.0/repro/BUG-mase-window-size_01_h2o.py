"""
Experiment 1: reproduce the h2o observation and isolate the denominator.

- Backtesting MASE / RMSSE vs the same predictions scored with the full
  `y.iloc[:n_train]` (Hyndman and Koehler, sktime, utilsforecast).
- Same y and same predictions, denominator with the first `window_size`
  observations excluded, for window_size 0, 1, 12, 24, 48.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

from skforecast.datasets import load_demo_dataset
from skforecast.recursive import ForecasterRecursive, ForecasterEquivalentDate, ForecasterStats
from skforecast.stats import Ets
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, backtesting_stats
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error
import skforecast

warnings.filterwarnings("ignore")
print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
      f"scikit-learn {sklearn.__version__}, skforecast {skforecast.__version__} ({skforecast.__file__})")

y = load_demo_dataset(verbose=False)
n_train = len(y) - 36
print(f"h2o: {len(y)} obs, n_train={n_train}")
exog = pd.DataFrame({"month": y.index.month}, index=y.index)

def cv():
    return TimeSeriesFold(steps=12, initial_train_size=n_train, refit=False)

def scale_mae(y_tr):
    return np.nanmean(np.abs(np.diff(y_tr)))

def scale_rmse(y_tr):
    return np.sqrt(np.nanmean(np.diff(y_tr) ** 2))

metrics = ["mean_absolute_error", mean_absolute_scaled_error, root_mean_squared_scaled_error]
rows = []
preds = {}

f = ForecasterEquivalentDate(offset=12, n_offsets=1)
m, p = backtesting_forecaster(forecaster=f, y=y, cv=cv(), metric=metrics, show_progress=False)
rows.append(("seasonal naive (offset 12)", f.window_size, "False", m)); preds["snaive"] = p

f = ForecasterRecursive(estimator=Ridge(), lags=24)
m, p = backtesting_forecaster(forecaster=f, y=y, exog=exog, cv=cv(), metric=metrics, show_progress=False)
rows.append(("Ridge lags=24 + month", f.window_size, "False", m)); preds["ridge"] = p

f = ForecasterStats(estimator=Ets(m=12))
m, p = backtesting_stats(forecaster=f, y=y, cv=cv(), metric=metrics, show_progress=False, suppress_warnings=True)
rows.append(("ETS m=12 (refit forced True)", f.window_size, "True (forced)", m)); preds["ets"] = p

print("\n## Backtesting vs full y_train (y.iloc[:n_train])")
full = y.iloc[:n_train]
print("| Model | window_size | refit | MAE | MASE backtesting | MASE full y_train | RMSSE backtesting | RMSSE full y_train |")
print("|---|---|---|---|---|---|---|---|")
for (name, ws, refit, m), p in zip(rows, preds.values()):
    mae = m["mean_absolute_error"].iat[0]
    mase_full = mean_absolute_scaled_error(p["y_true"], p["pred"], full)
    rmsse_full = root_mean_squared_scaled_error(p["y_true"], p["pred"], full)
    print(f"| {name} | {ws} | {refit} | {mae:.4f} | {m['mean_absolute_scaled_error'].iat[0]:.4f} | "
          f"{mase_full:.4f} | {m['root_mean_squared_scaled_error'].iat[0]:.4f} | {rmsse_full:.4f} |")

print("\n## ETS: what y_train does backtesting_stats use? (refit=True: union of train sets)")
for end in [n_train, n_train + 12, n_train + 24]:
    p = preds["ets"]
    print(f"y.iloc[:{end}] -> scale {scale_mae(y.iloc[:end]):.5f}, "
          f"MASE {mean_absolute_scaled_error(p['y_true'], p['pred'], y.iloc[:end]):.4f}")

print("\n## Same y and predictions, denominator without the first window_size obs")
print("| window_size | scale MAE naive | scale RMSE naive | MASE snaive | MASE Ridge | MASE ETS | Ridge/ETS |")
print("|---|---|---|---|---|---|---|")
for ws in [0, 1, 12, 24, 48]:
    y_tr = y.iloc[ws:n_train]
    out = []
    for k in ["snaive", "ridge", "ets"]:
        p = preds[k]
        out.append(mean_absolute_scaled_error(p["y_true"], p["pred"], y_tr))
    print(f"| {ws} | {scale_mae(y_tr):.5f} | {scale_rmse(y_tr):.5f} | {out[0]:.4f} | {out[1]:.4f} | {out[2]:.4f} | {out[1]/out[2]:.4f} |")

print("\nMAE ratio Ridge/ETS:", round(rows[1][3]["mean_absolute_error"].iat[0] / rows[2][3]["mean_absolute_error"].iat[0], 4))
print("MASE ratio Ridge/ETS as reported by skforecast:",
      round(rows[1][3]["mean_absolute_scaled_error"].iat[0] / rows[2][3]["mean_absolute_scaled_error"].iat[0], 4))
