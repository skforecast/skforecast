"""
What `MoiraiAdapter` returns when the context window of a series is all NaN
(the series has values only before the last `context_length` observations).

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/all_nan_context.py
"""

from __future__ import annotations
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
print("python", platform.python_version(), "| skforecast", version("skforecast"),
      "| uni2ts", version("uni2ts"), "| torch", version("torch"))
idx = pd.date_range("2024-01-01", periods=400, freq="h")
values = np.r_[np.random.default_rng(0).normal(100, 1, 200), np.full(200, np.nan)]
series = {"ok": pd.Series(np.random.default_rng(1).normal(100, 1, 400), index=idx),
          "nan_tail": pd.Series(values, index=idx)}
fm = FoundationModel(model_id="Salesforce/moirai-2.0-R-small", context_length=128)
try:
    fm.fit(series=series)
    pred = fm.predict(steps=4, quantiles=[0.1, 0.5, 0.9])
    print(pred)
except Exception as exc:  # noqa: BLE001
    print(type(exc).__name__, ":", str(exc)[:400])
