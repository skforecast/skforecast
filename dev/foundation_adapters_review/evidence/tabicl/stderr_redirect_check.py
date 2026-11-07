"""
Check whether the stderr redirection used by TabICLAdapter to hide the tqdm bar
(`show_progress=False`) also hides the UserWarning that tabicl emits when the
future covariates contain NaN. Run as a plain script (no pytest, no
catch_warnings) so warnings are printed to sys.stderr as a user would see them:
    ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/stderr_redirect_check.py <show_progress>
"""

import importlib.metadata as md
import sys

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

show = sys.argv[1] == "True"
print("tabicl", md.version("tabicl"), "| skforecast", md.version("skforecast"), "| show_progress", show)
idx = pd.date_range("2024-01-01", periods=126, freq="h")
y = pd.Series(10 + np.sin(np.arange(126) / 4), index=idx, name="y").iloc[:120]
ex = pd.DataFrame({"x": np.cos(np.arange(126) / 4)}, index=idx)
ex_future = ex.iloc[120:].copy()
ex_future.iloc[0, 0] = np.nan
m = FoundationModel("soda-inria/tabicl", context_length=64, show_progress=show)
m.fit(series=y, exog=ex.iloc[:120])
m.predict(steps=6, exog=ex_future)
print("done")
