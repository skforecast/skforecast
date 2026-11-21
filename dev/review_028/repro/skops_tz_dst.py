"""
BUG-skops-tz-dst: with `backend='skops'`, a forecaster whose index is daily
(`freq='D'`), created in UTC and converted to a local time zone, is saved but
does not load when `last_window_` crosses a daylight saving change:
`ValueError: Inferred frequency None from passed values does not conform to
passed frequency D` (`_compose_index` in `skforecast/utils/utils.py`).

It is pre-existing (same error with 0.27.x) and does not depend on pandas 3:
it fails saving and loading with pandas 2.3. With pandas 3 the index cannot
have `freq='D'` (`tz_convert` drops it); `freq='24h'` works in both versions.

Usage (environment with pandas 2.3): python skops_tz_dst.py
To check 0.27.x: PYTHONPATH=<worktree of origin/0.27.x> python skops_tz_dst.py
"""
import warnings

import numpy as np
import pandas as pd
from _common import OUT, attempt, versions
from sklearn.linear_model import Ridge

from skforecast.recursive import ForecasterRecursive
from skforecast.utils import load_forecaster, save_forecaster

warnings.simplefilter("ignore")
print(versions())

# 60 days from 2020-02-01 cross the spring change in Europe/Madrid (2020-03-29),
# 30 days do not.
for freq in ["D", "24h"]:
    for periods in [30, 60]:
        if freq == "D" and pd.__version__ >= "3":
            continue
        idx = pd.date_range("2020-02-01", periods=periods, freq=freq, tz="UTC")
        idx = idx.tz_convert("Europe/Madrid")
        y = pd.Series(np.arange(periods, dtype=float), index=idx, name="y")
        forecaster = ForecasterRecursive(Ridge(), lags=3)
        forecaster.fit(y=y)
        file = str(OUT / f"skops_{freq}_{periods}.skops")
        save_forecaster(forecaster, file, backend="skops")
        attempt(
            f"freq={freq} periods={periods} load",
            lambda: load_forecaster(
                file, backend="skops", trusted=True, verbose=False
            ).predict(2).round(3).tolist(),
        )
