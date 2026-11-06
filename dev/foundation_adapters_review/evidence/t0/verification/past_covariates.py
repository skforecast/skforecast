"""
Shape check of the past-covariate route of tfc-t0 0.6.0: a [B, V, T] context
(extra variates) combined with future_covariates. Prints the output shape and
the variate types the backend assigns to the rows. No accuracy measured.
"""

from importlib.metadata import version

import numpy as np
from t0 import T0Forecaster, TimeSeries

print("tfc-t0", version("tfc-t0"))
model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()
rng = np.random.default_rng(0)
ctx = rng.standard_normal((2, 3, 128)).astype(np.float32)
fut = rng.standard_normal((2, 1, 128 + 16)).astype(np.float32)
out = model.predict(ctx, 16, quantile_levels=[0.5], future_covariates=fut)
print("[B=2, V=3, T] + future_covariates [2, 1, T+16] -> quantiles", tuple(out.quantiles.shape))
import torch

ts = TimeSeries.from_array(torch.from_numpy(ctx), future_covariates=torch.from_numpy(fut))
print("variate types of the rows (last step):", ts.variate_type.numpy()[:, -1].tolist())
