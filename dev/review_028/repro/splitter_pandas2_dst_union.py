"""
BUG-splitter-pandas2-dst-union: with pandas 2, the 'wide', 'long' and
'long_multi_index' outputs of `TimeSeriesSplitter` raise a bare
`AssertionError` (no message) for a dictionary of daily series with a time
zone and different spans, when a split joins two overlapping series and the
last date of the earlier one is the day of a daylight saving change.

The error comes from pandas: `DatetimeIndex.union` (called by
`pd.DataFrame.from_dict`) joins overlapping indexes with the same frequency by
appending the dates of the second, and checks that the result keeps the
frequency. In pandas 2, 'D' is a fixed step of 24 hours, but `date_range` with
a time zone generates calendar days, so after the change the step is 23 or 25
hours and the check fails. pandas 3 ('D' is a calendar day) does not raise.

Found on 2026-11-23 by the equivalence battery of BUG-splitter-long-phase.

Usage: python splitter_pandas2_dst_union.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python splitter_pandas2_dst_union.py

Result (2026-11-23), the same in 0.27.0 and in 0.28.x:
    pandas 2.3.3: 'dict' right; 'wide' and 'long' AssertionError;
                  DatetimeIndex.union AssertionError
    pandas 3.1.1: all right
`ForecasterRecursiveMultiSeries` (fit, predict, backtesting) with the same
dictionary works with both. With identical indexes, as the class docstring
asks, the union does not take this path (see BUG-splitter-positional-cuts).
Not a regression.
"""
import warnings

import numpy as np
import pandas as pd
from _common import versions

from skforecast.experimental._splitter import TimeSeriesSplitter

warnings.simplefilter("ignore")
print(versions())

# The change in Europe/Madrid is on 2023-10-29: 'a' ends that day and 'b'
# goes on after it.
dates = pd.date_range("2023-10-20", periods=20, freq="D", tz="Europe/Madrid")
series = {
    "a": pd.Series(np.arange(10.0), index=dates[0:10]),
    "b": pd.Series(np.arange(10.0), index=dates[5:15]),
}
for output_format in ["dict", "wide", "long"]:
    try:
        TimeSeriesSplitter(series).split_by_size(
            train_size=10, output_format=output_format
        )
        print(f"split_by_size(train_size=10), '{output_format}': right")
    except AssertionError:
        print(f"split_by_size(train_size=10), '{output_format}': AssertionError")

try:
    dates[0:10].union(dates[5:15])
    print("DatetimeIndex.union: right")
except AssertionError:
    print("DatetimeIndex.union: AssertionError")
