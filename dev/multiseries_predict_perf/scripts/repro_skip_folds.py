"""
Reproduces the `TimeSeriesFold` bug with `refit=int` and `skip_folds` (see
"Hallazgos sin tocar" in SEGUIMIENTO.md). Fold 3 has no refit and reports the
training range of the skipped fold 2 (0:40), but it is predicted with the model
of fold 0 (0:30): level 'b' (3 values in 0:30, 13 in 0:40) is predicted as an
unknown level and gives NaN predictions.

Usage: PYTHONPATH=<tree> python repro_skip_folds.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import skforecast
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

print(skforecast.__file__, pd.__version__)

n = 60
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=pd.RangeIndex(n)
)
df.iloc[:27, 1] = np.nan  # 'b' has 3 values in 0:30 and 13 in 0:40

cv = TimeSeriesFold(
    steps=5, initial_train_size=30, refit=2, fixed_train_size=False, skip_folds=[2]
)
print(
    cv.split(X=df, as_pandas=True)[
        ["fold", "train_start", "train_end", "test_start", "fit_forecaster"]
    ]
)

forecaster = ForecasterRecursiveMultiSeries(Ridge(), lags=5)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster, df, cv, metric="mean_absolute_error", show_progress=False
    )
print(predictions.reset_index().groupby(["fold", "level"]).size())
print("NaN predictions per level:")
print(predictions.groupby("level")["pred"].apply(lambda s: s.isna().sum()))
for w in caught:
    print(w.category.__name__, str(w.message)[:200])
