"""
BUG-load-pytz: a forecaster saved with pandas 2 (joblib or pickle) whose index
has a named time zone does not load in a pandas 3 environment without pytz.

Usage:
    python load_pytz.py save   # environment with pandas 2.3
    python load_pytz.py load   # environment with pandas 3 and no pytz

Result with 0.28.x (2026-11-21): None and 'UTC' load with every backend;
'Europe/Madrid' raises `ModuleNotFoundError: No module named 'pytz'` with
joblib and pickle, without the note of `load_forecaster`. skops loads.
"""
import importlib.util
import sys
import warnings

import numpy as np
import pandas as pd
from _common import OUT, attempt, versions

warnings.simplefilter("ignore")
TZS = [None, "UTC", "Europe/Madrid"]
BACKENDS = {"joblib": "joblib", "pickle": "pkl", "skops": "skops"}


def path(tz, ext):
    return OUT / f"pytz_{str(tz).replace('/', '_')}.{ext}"


print(versions())
if sys.argv[1] == "save":
    from sklearn.linear_model import Ridge

    from skforecast.recursive import ForecasterRecursive
    from skforecast.utils import save_forecaster

    for tz in TZS:
        idx = pd.date_range("2021-01-01", periods=30, freq="D", tz=tz)
        y = pd.Series(np.arange(30.0), index=idx, name="y")
        forecaster = ForecasterRecursive(Ridge(), lags=3)
        forecaster.fit(y=y)
        for backend, ext in BACKENDS.items():
            save_forecaster(forecaster, str(path(tz, ext)), backend=backend)
    print(f"saved in {OUT}")
else:
    from skforecast.utils import load_forecaster

    print("pytz installed:", importlib.util.find_spec("pytz") is not None)
    for tz in TZS:
        for backend, ext in BACKENDS.items():
            attempt(
                f"{tz} {backend}",
                lambda: load_forecaster(
                    str(path(tz, ext)), backend=backend, trusted=True, verbose=False
                ).predict(2).round(3).tolist(),
            )
