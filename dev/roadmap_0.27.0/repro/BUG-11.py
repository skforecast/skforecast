"""
BUG-11: `backtesting_forecaster(ForecasterEquivalentDate, interval=...)` with
the default `interval_method='bootstrapping'` raises an unclear
`AttributeError` instead of an actionable error.

`ForecasterEquivalentDate` is listed in `forecasters_boot_conformal`
(`skforecast/model_selection/_utils.py:222-228`) but only supports conformal
intervals.

Expected: a clear `ValueError` pointing to `interval_method="conformal"` (or
conformal by default for this forecaster).
Observed (skforecast 0.26.0): AttributeError: 'ForecasterEquivalentDate'
object has no attribute 'predict_quantiles'.

Run: python dev/roadmap_0.27.0/repro/BUG-11.py
"""

import warnings

from skforecast.datasets import fetch_dataset
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterEquivalentDate

warnings.simplefilter("ignore")
y = fetch_dataset("h2o", raw=False, verbose=False).iloc[:, 0]

forecaster = ForecasterEquivalentDate(offset=12)
cv = TimeSeriesFold(steps=12, initial_train_size=150)
for kwargs in [
    dict(interval=[0.1, 0.9]),
    dict(interval=[0.1, 0.9], interval_method="conformal"),
]:
    try:
        _, preds = backtesting_forecaster(
            forecaster, y=y, cv=cv, metric="mean_absolute_error",
            show_progress=False, **kwargs
        )
        print(kwargs, "OK", list(preds.columns))
    except Exception as e:
        print(kwargs, type(e).__name__, str(e)[:120])
