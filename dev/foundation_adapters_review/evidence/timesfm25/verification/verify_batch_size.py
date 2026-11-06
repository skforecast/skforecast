"""
per_core_batch_size=4 vs default 1 through the adapter (forecast_config_kwargs):
same values up to float32 noise? Correctness only, no timing.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_batch_size.py
"""

import logging
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
print(f"timesfm {version('timesfm')} | torch {version('torch')}")
rng = np.random.default_rng(7)
idx = pd.date_range("2021-01-01", periods=400, freq="h")
series = {f"s{i}": pd.Series(rng.normal(0, 1, 400).cumsum() + 50, index=idx) for i in range(5)}
out = {}
for b in [1, 4]:
    fm = FoundationModel(model_id="google/timesfm-2.5-200m-pytorch", context_length=256,
                         forecast_config_kwargs={"per_core_batch_size": b})
    fm.fit(series=series)
    out[b] = fm.predict(steps=24, quantiles=[0.1, 0.5, 0.9])
    print(f"per_core_batch_size={b}: global_batch_size {fm.adapter._model.global_batch_size}")
cols = ["q_0.1", "q_0.5", "q_0.9"]
print(f"max |b4 - b1|: {np.max(np.abs(out[4][cols].to_numpy() - out[1][cols].to_numpy())):.2e}")
