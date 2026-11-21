"""
BUG-splitter-long-phase: the 'long' outputs of `TimeSeriesSplitter` lose the
values of a series with the same frequency as the others but a different phase
(`_wide_with_freq` in `skforecast/experimental/_splitter.py` calls `asfreq`
from the minimum date of the union of the indexes). The 'dict' and 'wide'
outputs are right.

Usage: python splitter_long_phase.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python splitter_long_phase.py

Result (2026-11-21, pandas 2.3):
    case         0.27.x   0.28.x   (non-NaN values of 'b' in train, expected 7)
    7D + 3D      7        0        <- regression
    D + 6h       0        0
    3h + 1h      0        0        (0.27.x also lost 'a': 1 of 7)
    MS + 14D     7        7
"""
import warnings

import numpy as np
import pandas as pd
from _common import versions

from skforecast.experimental._splitter import TimeSeriesSplitter

warnings.simplefilter("ignore")
print(versions())
for freq, phase in [("7D", "3D"), ("D", "6h"), ("3h", "1h"), ("MS", "14D")]:
    a = pd.Series(
        np.arange(10.0), index=pd.date_range("2023-01-01", periods=10, freq=freq)
    )
    start_b = pd.Timestamp("2023-01-01") + pd.Timedelta(phase)
    b = pd.Series(
        np.arange(10.0) + 100, index=pd.date_range(start_b, periods=10, freq=freq)
    )
    splitter = TimeSeriesSplitter({"a": a, "b": b})
    train, _ = splitter.split_by_size(train_size=7, output_format="long")
    n_a = train.loc[train["series_id"] == "a", "value"].notna().sum()
    n_b = train.loc[train["series_id"] == "b", "value"].notna().sum()
    print(f"freq={freq} phase={phase}: non-NaN in train a={n_a} b={n_b} (expected 7, 7)")
