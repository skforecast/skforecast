"""
QW-13, section 8.3 of QW-13.md: number of warnings of the training rows with
NaN in a grid search with `OneStepAheadFold` and with `TimeSeriesFold`
(365 days, 2 gaps in the training set and 4 in the validation set,
`lags_grid=[3, 7, 14, 35]`, `n_jobs=1`). Run it with `PYTHONPATH` pointing to
a checkout of cb6a280d3 to get the numbers before the change.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import skforecast
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster, OneStepAheadFold, TimeSeriesFold

rng = np.random.default_rng(42)
index = pd.date_range("2023-01-01", periods=365, freq="D")
y = pd.Series(100 + rng.normal(0, 2, 365).cumsum(), index=index)
y.iloc[[100, 200, 290, 310, 330, 350]] = np.nan  # 2 gaps in train, 4 in validation
print(skforecast.__file__)
for name, cv in [("OneStepAheadFold", OneStepAheadFold(initial_train_size="2023-09-30")),
                 ("TimeSeriesFold refit=False", TimeSeriesFold(steps=14, initial_train_size="2023-09-30", refit=False))]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        grid_search_forecaster(
            forecaster=ForecasterRecursive(Ridge(), lags=7, dropna_from_series=True),
            y=y, cv=cv, param_grid={"alpha": [0.1, 1]}, lags_grid=[3, 7, 14, 35],
            metric="mean_absolute_error", return_best=False, show_progress=False, n_jobs=1,
        )
    msgs = [getattr(r.message, "message", str(r.message)) for r in rec]
    nan_rows = [m for m in msgs if m.startswith(("NaNs detected", "Some rows of the training"))]
    print(f"{name}: {len(nan_rows)} warnings of NaN training rows")
    for m in nan_rows:
        print("   ", m[:150])
