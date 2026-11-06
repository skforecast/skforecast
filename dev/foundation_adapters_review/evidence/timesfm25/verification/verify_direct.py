"""
Independent re-check of TimesFM25Adapter against direct timesfm calls (CPU).

Sections:
1. FoundationModel.predict (point and quantiles 0.1..0.9) vs
   TimesFM_2p5_200M_torch.forecast on the same trimmed arrays. Uses different
   data, seeds and context than the investigator script, plus series shorter
   than context_length and a leading-NaN series.
2. Point channel: is the point forecast channel 5 (q0.5) or channel 0?
3. torch_compile of the loaded instance and config.json content.
4. forecast_config_kwargs with return_backcast / max_context, context 16384.
5. Nullable Float64 with pd.NA.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_direct.py
"""

import json
import logging
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

import timesfm
from huggingface_hub import hf_hub_download
from skforecast.foundation import FoundationModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
print(f"python {platform.python_version()} | {platform.machine()} | "
      f"torch threads {torch.get_num_threads()} | cuda {torch.cuda.is_available()}")
for pkg in ["timesfm", "torch", "huggingface_hub", "numpy", "pandas", "skforecast"]:
    print(f"{pkg} {version(pkg)}")

rng = np.random.default_rng(123)


def mk(n, start, name):
    idx = pd.date_range(start, periods=n, freq="D")
    t = np.arange(n)
    v = 10 + 3 * np.sin(2 * np.pi * t / 7) + 0.05 * t + rng.normal(0, 1, n)
    return pd.Series(v, index=idx, name=name)


a = mk(700, "2020-01-01", "a")
b = mk(90, "2020-03-01", "b")    # shorter than context_length
c = mk(500, "2020-02-01", "c")
c.iloc[:20] = np.nan             # leading NaN
c.iloc[300:310] = np.nan         # interior NaN
series = {"a": a, "b": b, "c": c}
CL, STEPS = 320, 30
Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("\n== 1. adapter vs direct")
fm = FoundationModel(model_id=MODEL_ID, context_length=CL)
fm.fit(series=series)
pp = fm.predict(steps=STEPS)
pq = fm.predict(steps=STEPS, quantiles=Q)
m = fm.adapter._model
print("adapter model class:", type(m).__name__, "| compiled:", m.forecast_config)

direct = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
direct.compile(timesfm.ForecastConfig(max_context=CL, max_horizon=STEPS))
inputs = [series[k].to_numpy()[-CL:] for k in series]
dp, dq = direct.forecast(horizon=STEPS, inputs=inputs)
print("direct shapes", dp.shape, dq.shape)
mp = mq = 0.0
for i, k in enumerate(series):
    ap = pp.loc[pp["level"] == k, "pred"].to_numpy()
    mp = max(mp, float(np.max(np.abs(ap - dp[i]))))
    sub = pq.loc[pq["level"] == k]
    for q in Q:
        mq = max(mq, float(np.max(np.abs(sub[f"q_{q}"].to_numpy() - dq[i, :, round(q * 10)]))))
print(f"max diff point {mp:.3e} | max diff quantiles {mq:.3e}")
print("any NaN in adapter output:", bool(pp["pred"].isna().any()))

print("\n== 2. point channel")
print("point == ch5:", bool(np.array_equal(dp, dq[..., 5])),
      "| point == ch0:", bool(np.array_equal(dp, dq[..., 0])),
      f"| max|ch0-ch5| {np.max(np.abs(dq[..., 0] - dq[..., 5])):.4f}")

print("\n== 3. torch_compile")
cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
print("config.json:", cfg)
print("loaded torch_compile:", m.torch_compile)

print("\n== 4. edge cases")


def case(label, **kw):
    try:
        f = FoundationModel(model_id=MODEL_ID, **kw)
        f.fit(series={"a": a})
        out = f.predict(steps=STEPS)
        print(f"{label}: OK rows {len(out)}")
    except Exception as exc:
        print(f"{label}: {type(exc).__name__}: {str(exc)[:220]}")


case("init only with max_context (no error expected at init)", context_length=CL,
     forecast_config_kwargs={"max_context": 64}) 
case("return_backcast=True", context_length=CL, forecast_config_kwargs={"return_backcast": True})
case("context_length=16384", context_length=16384)
case("context_length=16000", context_length=16000)

print("\n== 5. nullable Float64")
fn = FoundationModel(model_id=MODEL_ID, context_length=CL)
fn.adapter._model = m
fn.fit(series={"c": c.astype("Float64")})
o = fn.predict(steps=STEPS)["pred"].to_numpy()
print(f"max |Float64 - float64 direct| {np.max(np.abs(o - dp[2])):.3e}")
