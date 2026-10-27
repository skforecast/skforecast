"""
BUG-search-short-series, experiment 1: the ranking of a grid search flips
because the candidate with the larger window leaves the short series out.
lags_3 is better on the common series, lags_24 wins the search. No warning
is raised with TimeSeriesFold nor with OneStepAheadFold.
"""
import warnings, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, grid_search_forecaster_multiseries,
    backtesting_forecaster_multiseries)
from skforecast.recursive import ForecasterRecursiveMultiSeries

idx = pd.date_range("2016-01-01", periods=96, freq="MS")
rng = np.random.default_rng(0)
t = np.arange(96)
def season():
    # AR(1) without seasonality: lags 1..3 are enough, lags 24 only adds noise
    y = np.zeros(96)
    for i in range(1, 96):
        y[i] = 0.8 * y[i - 1] + rng.normal(0, 1)
    return 100 + y
series = {
    "S01": pd.Series(season(), index=idx),
    "S10": pd.Series(season(), index=idx),
    # short and noisy: 80 obs, 13 of them in the initial train set
    "HARD": pd.Series(rng.normal(100, 10, 80), index=idx[-80:]),
}
f = ForecasterRecursiveMultiSeries(Ridge(alpha=1e-6), lags=3, encoding="ordinal")
lags_grid = {"lags_3": 3, "lags_24": 24}
for name, cv in [
    ("TimeSeriesFold", TimeSeriesFold(steps=12, initial_train_size="2018-06-01", refit=False)),
    ("OneStepAheadFold", OneStepAheadFold(initial_train_size="2018-06-01")),
]:
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        res = grid_search_forecaster_multiseries(
            forecaster=f, series=series, cv=cv, param_grid={"alpha": [1e-6]},
            lags_grid=lags_grid, metric="mean_absolute_error",
            aggregate_metric=["average"], return_best=False, show_progress=False)
    print(f"===== {name}")
    print(res[["lags_label", "mean_absolute_error__average"]].to_string())
    print("  MissingValuesWarnings:", sum(issubclass(w.category, Warning) and "not predicted" in str(w.message) for w in c))

# per-level metrics behind each candidate (backtesting)
for lags in [3, 24]:
    f2 = ForecasterRecursiveMultiSeries(Ridge(alpha=1e-6), lags=lags, encoding="ordinal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, _ = backtesting_forecaster_multiseries(
            f2, series, TimeSeriesFold(steps=12, initial_train_size="2018-06-01", refit=False),
            "mean_absolute_error", show_progress=False)
    print(f"lags={lags}:", dict(zip(m["levels"], m["mean_absolute_error"].round(2))))
