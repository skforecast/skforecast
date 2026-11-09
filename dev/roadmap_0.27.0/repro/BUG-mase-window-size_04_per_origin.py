import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import Ridge
from skforecast.datasets import load_demo_dataset
from skforecast.recursive import ForecasterRecursive, ForecasterEquivalentDate, ForecasterStats
from skforecast.stats import Ets
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, backtesting_stats
from skforecast.metrics import mean_absolute_scaled_error
y = load_demo_dataset(verbose=False); n = len(y) - 36
cv = lambda: TimeSeriesFold(steps=12, initial_train_size=n, refit=False)
P = {}
M = {}
M["naive"], P["naive"] = backtesting_forecaster(forecaster=ForecasterEquivalentDate(offset=12), y=y, cv=cv(), metric=["mean_absolute_error", mean_absolute_scaled_error], show_progress=False)
M["ridge"], P["ridge"] = backtesting_forecaster(forecaster=ForecasterRecursive(Ridge(), lags=24), y=y, cv=cv(), metric=["mean_absolute_error", mean_absolute_scaled_error], show_progress=False)
M["ets"], P["ets"] = backtesting_stats(forecaster=ForecasterStats(Ets(m=12)), y=y, cv=cv(), metric=["mean_absolute_error", mean_absolute_scaled_error], show_progress=False, suppress_warnings=True)
s = lambda end: np.mean(np.abs(np.diff(y.iloc[:end])))
print("| model | MAE | MASE fixed y[:168] | MASE per origin (y[:168], y[:180], y[:192]) | MASE skforecast |")
for k, p in P.items():
    e = np.abs(p["y_true"] - p["pred"]).to_numpy()
    origins = n + 12 * p["fold"].to_numpy()
    q = e / np.array([s(o) for o in origins])
    print(f"| {k} | {e.mean():.4f} | {e.mean()/s(n):.4f} | {q.mean():.4f} | {M[k]['mean_absolute_scaled_error'].iat[0]:.4f} |")
