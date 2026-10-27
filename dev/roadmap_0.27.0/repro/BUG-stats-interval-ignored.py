"""
BUG-stats-interval-ignored: `ForecasterStats.predict_interval` ignores
`interval` because `alpha=0.05` is the default and takes precedence
(`skforecast/recursive/_forecaster_stats.py:1010,1078-1084`).

`skills/statistical-models/SKILL.md:72-75` uses `interval=[0.1, 0.9]` as the
example of an 80% interval.

Expected: `interval=[0.1, 0.9]` returns the 80% interval.
Observed (skforecast 0.26.0, h2o, Arima(order=(1, 1, 0)), first step):
default and `interval=[0.1, 0.9]` give the same bounds 0.4631 / 1.0723 (95%);
only `alpha=None` gives the 80% interval 0.5685 / 0.9668.
`backtesting_stats` is not affected (its `alpha` defaults to None).

Run: python dev/roadmap_0.27.0/repro/BUG-stats-interval-ignored.py
"""

import warnings

import numpy as np

from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima

warnings.simplefilter("ignore")
y = fetch_dataset("h2o", raw=False, verbose=False).iloc[:, 0]

forecaster = ForecasterStats(estimator=Arima(order=(1, 1, 0)))
forecaster.fit(y=y)
a = forecaster.predict_interval(steps=3)
b = forecaster.predict_interval(steps=3, interval=[0.1, 0.9])
c = forecaster.predict_interval(steps=3, interval=[0.1, 0.9], alpha=None)
print("default           :", a.iloc[0].round(4).to_dict())
print("interval=[0.1,0.9]:", b.iloc[0].round(4).to_dict())
print("alpha=None        :", c.iloc[0].round(4).to_dict())
print("default == interval=[0.1, 0.9] (expected False):", np.allclose(a.values, b.values))
