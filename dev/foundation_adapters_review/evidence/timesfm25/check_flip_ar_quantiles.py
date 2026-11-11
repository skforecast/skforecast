"""
Upstream fix af8f595 (timesfm 3.0.0, "apply quantile flip to autoregressive
outputs in force_flip_invariance"): before it, with the default
`force_flip_invariance=True`, the quantile channels of the autoregressive part
of the horizon (steps beyond output_patch_len = 128) were averaged with the
unflipped channels of the negated series. This script checks, through the
adapter, whether the quantiles are ordered (q0.1 <= q0.5 <= q0.9) in the first
128 steps and beyond, with the installed timesfm. Not an accuracy comparison:
only the ordering of the quantiles returned by the adapter.

Run from the repository root with each environment, e.g.:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_flip_ar_quantiles.py
"""

import logging
from importlib.metadata import version

import numpy as np
import pandas as pd
from skforecast.foundation import FoundationModel

logging.disable(logging.CRITICAL)
print(f"timesfm {version('timesfm')} | torch {version('torch')} "
      f"| huggingface_hub {version('huggingface_hub')}")

rng = np.random.default_rng(1)
n = 1000
t = np.arange(n)
idx = pd.date_range("2024-01-01", periods=n, freq="h")
series = {
    "a": pd.Series(50 + 5 * np.sin(2 * np.pi * t / 24) + 0.02 * t + rng.normal(0, 1, n), index=idx),
    "b": pd.Series(10 + 3 * np.cos(2 * np.pi * t / 168) + rng.normal(0, 0.5, n), index=idx),
}
steps = 384
fm = FoundationModel(model_id="google/timesfm-2.5-200m-pytorch", context_length=512)
fm.fit(series=series)
pred = fm.predict(steps=steps, quantiles=[0.1, 0.5, 0.9])
for name in series:
    sub = pred[pred["level"] == name]
    q = sub[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    ordered = (q[:, 0] <= q[:, 1]) & (q[:, 1] <= q[:, 2])
    width = q[:, 2] - q[:, 0]
    print(f"series {name}: ordered steps 1-128: {ordered[:128].mean():.2f}, "
          f"steps 129-{steps}: {ordered[128:].mean():.2f} | "
          f"mean width q0.9-q0.1 steps 1-128: {width[:128].mean():.3f}, "
          f"steps 129-{steps}: {width[128:].mean():.3f}")
