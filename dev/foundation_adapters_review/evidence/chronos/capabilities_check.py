"""
Check the capability class attributes of ChronosAdapter against the installed
chronos-forecasting backend (no timing, no accuracy).
"""

import importlib.metadata as md
from pathlib import Path

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import ChronosAdapter

for package in ["skforecast", "chronos-forecasting", "torch", "transformers"]:
    print(f"{package:20s} {md.version(package)}")
for attr in ChronosAdapter._REQUIRED_CLASS_ATTRIBUTES:
    print(f"  {attr:36s} {getattr(ChronosAdapter, attr)!r}")

model = FoundationModel(model_id="autogluon/chronos-2-small", context_length=128)
model.adapter._load_pipeline()
pipe = model.adapter._pipeline
rng = np.random.default_rng(0)

# supports_heterogeneous_covariates=False: the backend rejects mixed schemas.
a = {"target": rng.normal(size=50).astype(np.float32),
     "past_covariates": {"x": rng.normal(size=50).astype(np.float32)}}
b = {"target": rng.normal(size=60).astype(np.float32)}
try:
    pipe.predict_quantiles([a, b], prediction_length=4, quantile_levels=[0.5])
    print("heterogeneous covariate keys in one call: accepted")
except ValueError as exc:
    print("heterogeneous covariate keys in one call: ValueError:", str(exc)[:110])

# supports_past_only_covariates=True: past-only covariate through the adapter
# equals the direct call with only `past_covariates`.
idx = pd.date_range("2024-01-01", periods=200, freq="D")
y = pd.Series(rng.normal(size=200).cumsum() + 20, index=idx, name="y")
x = pd.DataFrame({"x": rng.normal(size=200)}, index=idx)
pred = model.predict(steps=7, context=y, context_exog=x, quantiles=[0.1, 0.5, 0.9])
q, _ = pipe.predict_quantiles(
    [{"target": y.iloc[-128:].to_numpy(np.float32),
      "past_covariates": {"x": x["x"].iloc[-128:].to_numpy(np.float32)}}],
    prediction_length=7, quantile_levels=[0.1, 0.5, 0.9],
)
diff = np.abs(pred[["q_0.1", "q_0.5", "q_0.9"]].to_numpy() - q[0].squeeze(0).numpy()).max()
print("past-only covariate, adapter vs direct max_abs_diff:", diff)

# requires_equal_length_context=False and supports_nan_in_series=True.
ys = {"short": y.iloc[-30:].copy(), "long": y.copy()}
ys["long"].iloc[100:110] = np.nan
pred = model.predict(steps=7, context=ys)
print("different lengths + NaN, finite:", bool(np.isfinite(pred["pred"]).all()))

# weights_in_hf_cache=True, weights_repo_id=None: weights cached under model_id.
from huggingface_hub.constants import HF_HUB_CACHE
cache = Path(HF_HUB_CACHE) / "models--autogluon--chronos-2-small"
print("weights in HF cache under model_id:", cache.exists())

# SUPPORTED_QUANTILES=None and trained quantiles of the checkpoint.
print("trained quantiles:", pipe.quantiles)
print("model_context_length:", pipe.model_context_length,
      "model_prediction_length:", pipe.model_prediction_length)
