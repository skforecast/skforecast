"""
BUG-njobs-temp-folder: since #1503, a forecaster larger than 1 MB is always
sent to the workers through joblib's temporary folder. If that folder cannot
be written, the backtesting fails with an opaque
`PicklingError: Could not pickle the task to send it to the workers`.
0.27.x works with the same forecaster (none of its arrays is over 1 MB).

Fixed in #1535 (branch `fix/njobs-temp-folder`, 2026-11-23): the error now
explains the cause and how to solve it.

Usage: python njobs_temp_folder.py
To compare: PYTHONPATH=<worktree of origin/0.27.x> python njobs_temp_folder.py
"""
import os

import numpy as np
import pandas as pd
from _common import OUT, attempt, versions
from sklearn.ensemble import RandomForestRegressor

from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

# NOTE: A path below a regular file cannot be created in any operating system.
# `/dev/null/not-writable` is a valid path on Windows, where joblib creates it
# (`C:\dev\null\not-writable`) and the error is not reproduced. joblib reads
# the variable in each call, so it can be set after the imports.
not_a_folder = OUT / "not_a_folder"
not_a_folder.write_text("")
os.environ["JOBLIB_TEMP_FOLDER"] = str(not_a_folder / "not-writable")

print(versions())
rng = np.random.default_rng(0)
n = 1500
y = pd.Series(
    np.cumsum(rng.normal(size=n)),
    index=pd.date_range("2000-01-01", periods=n, freq="h"),
    name="y",
)
forecaster = ForecasterRecursive(
    RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=1), lags=10
)
cv = TimeSeriesFold(steps=100, initial_train_size=1000, refit=True)
attempt(
    "backtesting n_jobs=2",
    lambda: backtesting_forecaster(
        forecaster, y=y, cv=cv, metric="mean_absolute_error", n_jobs=2,
        show_progress=False,
    )[0].iloc[0, 0],
)
