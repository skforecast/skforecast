"""
BUG-wide-to-long-inferred-freq: `reshape_series_wide_to_long` loses values
when the index has no frequency and pandas infers one that is not the one of
the data. With hourly dates in two spans of 9 hours separated by a gap,
`pandas.infer_freq` returns a business hour frequency ('bh') instead of None,
so the `ValueError` for dates that are not regularly spaced is not raised and
`asfreq('bh')` keeps only the dates at business hours.

Found on 2026-11-22 while fixing BUG-splitter-long-phase, whose fix protects
`TimeSeriesSplitter` (it sets the frequency of the series) but not this
function.

Usage: python wide_to_long_inferred_freq.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python wide_to_long_inferred_freq.py

Result (2026-11-22; values kept of 36, expected a `ValueError` or 36), the same
without time zone, with UTC and with Europe/Madrid:
    0.27.x (pandas 2.3.3)             2, with a daily frequency (4 rows)
    0.28.x (pandas 2.3.3 and 3.1.1)   2, with the inferred 'bh' (18 rows)
Not a regression in the number of values kept: in 0.27.x the frequency was not
inferred and `asfreq` used a daily one.
"""
import warnings

import numpy as np
import pandas as pd
from _common import versions

from skforecast.preprocessing import reshape_series_wide_to_long

warnings.simplefilter("ignore")
print(versions())
for tz in [None, "UTC", "Europe/Madrid"]:
    index = pd.date_range("2023-10-02", periods=34, freq="h", tz=tz)
    index = pd.DatetimeIndex(index[:9].append(index[25:]))
    wide = pd.DataFrame(
        {"a": np.arange(18.0), "b": np.arange(18.0) + 100}, index=index
    )
    try:
        long = reshape_series_wide_to_long(wide)
        print(
            f"tz={tz}: values kept {long['value'].notna().sum()} of "
            f"{wide.notna().sum().sum()}, rows {len(long)}, "
            f"frequency of 'a': {long.loc['a'].index.freq}"
        )
    except ValueError as exc:
        print(f"tz={tz}: ValueError: {str(exc)[:80]}")
