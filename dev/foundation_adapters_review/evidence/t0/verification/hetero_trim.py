"""
Does trim_leading_missing=True (tfc-t0 0.6.0) remove the effect of an all-NaN
covariate row? Direct backend calls: series `a` alone (no covariates) vs `a`
in a batch with `b`, which has one future covariate (row of `a` all NaN).
"""

from importlib.metadata import version

import numpy as np
from t0 import T0Forecaster

print("tfc-t0", version("tfc-t0"))
model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()
rng = np.random.default_rng(3)
T, H = 300, 12
t = np.arange(T + H)
a = (100 + 10 * np.sin(t[:T] / 6) + rng.standard_normal(T)).astype(np.float32)
b = (50 + 5 * np.cos(t[:T] / 4) + rng.standard_normal(T)).astype(np.float32)
cov = np.full((2, 1, T + H), np.nan, dtype=np.float32)
cov[1, 0] = np.sin(t / 4)
for trim in (False, True):
    alone = model.predict(a[None], H, quantile_levels=[0.5], trim_leading_missing=trim).quantiles.numpy()[0, :, 0]
    batch = model.predict(np.stack([a, b]), H, quantile_levels=[0.5], future_covariates=cov,
                          trim_leading_missing=trim).quantiles.numpy()[0, :, 0]
    print(f"trim_leading_missing={trim}: a alone vs in batch, max rel diff =",
          float(np.max(np.abs(batch - alone) / np.abs(alone))))
