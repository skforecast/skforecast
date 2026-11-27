"""
Reproduction of BUG-ets-freeze-params, found while reviewing the forecaster
finder study: `backtesting_stats` with an automatic `Ets` and the default
`freeze_params=True` returns forecasts of the order of -1e11 from the second
fold on in some series. With `freeze_params=False` the same folds are fine.

The first fit selects the model and its smoothing parameters; `freeze_params`
then sets all of `best_params_` (model, `alpha`, `beta`, `gamma`, `phi`) on
the estimator that is refitted in the following folds.

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
