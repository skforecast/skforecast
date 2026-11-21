"""
DOC-028-review: behavior changes that the 0.28.0 release notes do not list in
"Before upgrading", or describe inaccurately.

Usage: python doc_checks.py   (run it with pandas 2.3 and with pandas 3)
To compare: PYTHONPATH=<worktree of origin/0.27.x> python doc_checks.py (pandas 2)

1. `n_jobs=0` raises `ValueError` in `ForecasterDirect` and in backtesting
   (0.27.x used all the CPUs).
2. `ForecasterEquivalentDate` with `offset=pd.offsets.Day(1)` on hourly
   Europe/Madrid data: with pandas 3 the backtesting raises `ValueError` near
   the autumn change; the note only says the predictions differ.
3. `reshape_series_wide_to_long` with business days and one missing date (a
   holiday) raises `ValueError` (0.27.x returned NaN rows on weekends); the
   note says "daily data with missing dates".
"""
import warnings

import numpy as np
import pandas as pd
from _common import attempt, versions
from sklearn.linear_model import LinearRegression

from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import reshape_series_wide_to_long
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive

warnings.simplefilter("ignore")
print(versions())

# 1. n_jobs=0
y = pd.Series(
    np.arange(60.0), index=pd.date_range("2020-01-01", periods=60, freq="D"), name="y"
)
attempt(
    "1. ForecasterDirect(n_jobs=0)",
    lambda: ForecasterDirect(LinearRegression(), steps=3, lags=3, n_jobs=0) and None,
)
attempt(
    "1. backtesting_forecaster(n_jobs=0)",
    lambda: backtesting_forecaster(
        ForecasterRecursive(LinearRegression(), lags=3), y=y,
        cv=TimeSeriesFold(steps=5, initial_train_size=40, refit=True),
        metric="mean_absolute_error", n_jobs=0, show_progress=False,
    )[0].iloc[0, 0],
)

# 2. ForecasterEquivalentDate with Day(1) on hourly tz-aware data (autumn change)
idx = pd.date_range("2021-10-24 12:00", periods=24 * 8, freq="h", tz="Europe/Madrid")
y_h = pd.Series(np.arange(len(idx), dtype=float) % 24, index=idx, name="y")
attempt(
    "2. ForecasterEquivalentDate(Day(1)) backtesting",
    lambda: backtesting_forecaster(
        ForecasterEquivalentDate(offset=pd.offsets.Day(1), n_offsets=2), y=y_h,
        cv=TimeSeriesFold(steps=12, initial_train_size=100, refit=False),
        metric="mean_absolute_error", show_progress=False,
    )[0].iloc[0, 0],
)

# 3. Business days with a holiday
dates = pd.DatetimeIndex(list(pd.date_range("2020-01-01", periods=20, freq="B")))
dates = dates.delete(5)
wide = pd.DataFrame({"a": np.arange(19.0), "b": np.arange(19.0) * 10}, index=dates)
attempt(
    "3. reshape_series_wide_to_long, business days with a holiday",
    lambda: (lambda out: f"{len(out)} rows, {int(out['value'].isna().sum())} NaN")(
        reshape_series_wide_to_long(wide)
    ),
)
