"""
Check whether forcing cross_learning=False for a single series (as
ChronosAdapter.predict does) changes anything in the backend: direct call to
Chronos2Pipeline.predict_quantiles with one input (target + covariates) and
cross_learning True vs False. Also check what happens when predict_kwargs
contains cross_learning.
"""

import importlib.metadata as md

import numpy as np
import pandas as pd
from chronos import BaseChronosPipeline

from skforecast.foundation import FoundationModel

for package in ["skforecast", "chronos-forecasting", "torch", "transformers"]:
    print(f"{package:20s} {md.version(package)}")

rng = np.random.default_rng(0)
pipe = BaseChronosPipeline.from_pretrained("autogluon/chronos-2-small", device_map="cpu")
inp = [{
    "target": rng.normal(size=200).astype(np.float32),
    "past_covariates": {
        "x": rng.normal(size=200).astype(np.float32),
        "c": rng.choice(["a", "b"], size=200).astype(object),
    },
    "future_covariates": {
        "x": rng.normal(size=12).astype(np.float32),
        "c": rng.choice(["a", "b"], size=12).astype(object),
    },
}]
out = {}
for cl in [False, True]:
    q, _ = pipe.predict_quantiles(inp, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], cross_learning=cl)
    out[cl] = q[0].numpy()
print("single input, cross_learning True vs False max_abs_diff:", float(np.abs(out[True] - out[False]).max()))

y = pd.Series(rng.normal(size=100), index=pd.date_range("2024-01-01", periods=100, freq="h"), name="y")
fm = FoundationModel(model_id="autogluon/chronos-2-small", predict_kwargs={"cross_learning": True})
try:
    fm.predict(steps=6, context=y)
    print("predict_kwargs with cross_learning: no error")
except Exception as exc:  # noqa: BLE001
    print("predict_kwargs with cross_learning:", type(exc).__name__, str(exc)[:120])
