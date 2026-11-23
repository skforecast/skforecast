"""
BUG-splitter-positional-cuts: `TimeSeriesSplitter` cuts all the series of a
group by the positions of the first one, so the series of a dictionary with
different spans are cut at other dates, with no warning. The class docstring
asks for "identical indexes" in a dictionary, but nothing checks it. The same
series in a long DataFrame with a MultiIndex are rejected (`ValueError`, no
frequency), so only dictionaries are affected.

Found on 2026-11-22 by the code review of BUG-splitter-long-phase.

Usage: python splitter_positional_cuts.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python splitter_positional_cuts.py

Result (2026-11-23), the same in 0.27.0 and in 0.28.x, with pandas 2.3.3 and
3.1.1 (0.27.0 only with pandas 2.3.3). Measured with the 'dict' output; the
other formats are built from it:
    split_by_date(end_train='2023-01-07'), 'a' from 01-01 and 'c' from 01-05
        train of 'c': 2023-01-05 to 2023-01-11 (4 dates after `end_train`)
    split_by_size(train_size=7), 'a' with 10 values and 'long' with 14
        'long': 7 in train and 3 in test (4 values in no split)
Not a regression.
"""
import warnings

import numpy as np
import pandas as pd
from _common import versions

from skforecast.experimental._splitter import TimeSeriesSplitter

warnings.simplefilter("ignore")
print(versions())

a = pd.Series(
    np.arange(10.0), index=pd.date_range("2023-01-01", periods=10, freq="D")
)
c = pd.Series(
    np.arange(10.0), index=pd.date_range("2023-01-05", periods=10, freq="D")
)
train, _ = TimeSeriesSplitter({"a": a, "c": c}).split_by_date(
    end_train="2023-01-07", output_format="dict"
)
for name, s in train.items():
    print(
        f"split_by_date(end_train='2023-01-07'), train of '{name}': "
        f"{s.index.min().date()} to {s.index.max().date()}"
    )

long_frame = pd.concat({"a": a, "c": c}, names=["series_id", "datetime"])
try:
    TimeSeriesSplitter(long_frame.to_frame("value"))
    print("long DataFrame with a MultiIndex and the same series: accepted")
except ValueError as exc:
    print(
        "long DataFrame with a MultiIndex and the same series: ValueError: "
        f"...{str(exc)[-42:]}"
    )

long = pd.Series(
    np.arange(14.0), index=pd.date_range("2023-01-01", periods=14, freq="D")
)
train, test = TimeSeriesSplitter({"a": a, "long": long}).split_by_size(
    train_size=7, output_format="dict"
)
for name, s in [("a", a), ("long", long)]:
    print(
        f"split_by_size(train_size=7), '{name}' with {len(s)} values: "
        f"{len(train[name])} in train, {len(test[name])} in test"
    )
