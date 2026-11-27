"""
Reproduction of BUG-ets-freeze-params, found while reviewing the forecaster
finder study: `backtesting_stats` with an automatic `Ets` and the default
`freeze_params=True` returned forecasts of the order of -1e11 from the second
fold on in some series (first prediction of each fold of 'The Murray |
Visiting': 76.5, -1.02e+11, -1.12e+11). With `freeze_params=False` the same
folds were fine.

Fixed in commit 5f020a0 (a model with a multiplicative error or seasonality
is not valid when its level or forecasts are not positive, and the initial
states are searched from more starting points when alpha is fixed). With the
fix, both values of `freeze_params` give forecasts in the range of the series.

Usage:
    python -I repro_ets_freeze_params.py
"""
import warnings

from skforecast.datasets import fetch_dataset
from skforecast.model_selection import TimeSeriesFold, backtesting_stats
from skforecast.recursive import ForecasterStats
from skforecast.stats import Ets

warnings.filterwarnings("ignore")
d = fetch_dataset("australia_tourism", verbose=False)
d["series"] = d["Region"] + " | " + d["Purpose"]
series = d.pivot_table(index=d.index, columns="series", values="Trips").asfreq("QS")

for name in ["The Murray | Visiting", "Mackay | Visiting", "Whitsundays | Visiting"]:
    y = series[name]
    for freeze_params in (True, False):
        cv = TimeSeriesFold(steps=4, initial_train_size=68, refit=True, fixed_train_size=False)
        metric, pred = backtesting_stats(
            ForecasterStats(estimator=Ets(m=4)), y=y, cv=cv, metric="mean_absolute_error",
            freeze_params=freeze_params, n_jobs=1, show_progress=False,
        )
        print(f"{name:<24} freeze_params={freeze_params!s:<5} "
              f"MAE={metric['mean_absolute_error'].iloc[0]:.4g}  "
              f"first prediction of each fold: {[f'{v:.3g}' for v in pred['pred'].iloc[::4]]}")
