"""
BUG-utc-midnight-anchored-freq: an index with a time zone, an anchored
frequency ('W', 'MS') and dates at a local time that is UTC midnight (19:00 of
America/New_York in winter, 20:00 in summer) loses all its values in
`reshape_series_wide_to_long` and `reshape_series_long_to_dict`, and the
predictions of a forecaster trained with it have other dates (Saturdays
instead of Sundays, the last day of the month instead of the first).

Without a daylight saving change in the index, `_is_utc_anchored_index` takes
it as an index in fixed UTC steps (UTC midnight, not local midnight), and
`_date_range_from_index` generates the dates in UTC with a frequency whose
anchor is the local calendar: local Sundays are Mondays in UTC, so 'W-SUN'
gives other dates.

Found on 2026-11-22 while fixing BUG-splitter-long-phase, whose fix protects
the 'long' outputs of `TimeSeriesSplitter` (the series are stacked when their
dates are not in the grid) but not the functions of `preprocessing`.

Usage: python utc_midnight_anchored_freq.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python utc_midnight_anchored_freq.py

Result (2026-11-23; values kept of 6, expected 6), the same in 0.27.0 and in
0.28.x, with pandas 2.3.3 and 3.1.1 (0.27.0 only with pandas 2.3.3):
    case                         wide_to_long   long_to_dict   predict index
    W,  Sundays 19:00 (winter)   0              0              Saturdays 19:00
    MS, day 1 at 20:00 (summer)  0              0              last day of month
    MS, day 1 at 19:00 (summer)  6              6              right (control)
    D,  19:00 (winter)           6              6              right (control)
`reshape_series_long_to_dict` is called with `freq`. Not a regression.
"""
import warnings

import numpy as np
import pandas as pd
from _common import versions
from sklearn.linear_model import LinearRegression

from skforecast.preprocessing import (
    reshape_series_long_to_dict,
    reshape_series_wide_to_long,
)
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")
print(versions())
cases = [
    ("W", "2022-11-13 19:00"),
    ("MS", "2023-04-01 20:00"),
    ("MS", "2023-04-01 19:00"),
    ("D", "2022-11-13 19:00"),
]
for freq, start in cases:
    index = pd.date_range(start, periods=6, freq=freq, tz="America/New_York")
    wide = pd.DataFrame({"a": np.arange(6.0)}, index=index)
    long = reshape_series_wide_to_long(wide)
    n_long = long["value"].notna().sum()

    data = pd.DataFrame(
        {"series_id": "a", "datetime": index, "value": np.arange(6.0)}
    )
    series_dict = reshape_series_long_to_dict(
        data, series_id="series_id", index="datetime", values="value", freq=freq
    )
    n_dict = series_dict["a"].notna().sum()

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    forecaster.fit(y=wide["a"])
    expected = pd.date_range(index[-1], periods=3, freq=freq)[1:]
    predicted = forecaster.predict(steps=2).index
    print(
        f"freq={freq} start={start}: wide_to_long kept {n_long} of 6, "
        f"long_to_dict kept {n_dict} of 6, predict index "
        f"{'right' if predicted.equals(expected) else list(map(str, predicted))}"
        f"{'' if predicted.equals(expected) else f', expected {list(map(str, expected))}'}"
    )
