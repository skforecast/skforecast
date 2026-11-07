"""
Correctness checks of TimesFM25Adapter against a direct call to the timesfm
backend on the same input (CPU).

1. FoundationModel.predict (point and quantiles) for three series of different
   lengths, one with interior NaN, against TimesFM_2p5_200M_torch.forecast on
   the same trimmed arrays. Expected: identical values.
2. Which channel the point forecast is (mean channel 0 or quantile 0.5,
   channel 5).
3. Effective `torch_compile` of the instance loaded by the adapter, and
   whether the `_TimesFMCompat` subclass is still needed (load without it).
4. Edge cases of `forecast_config_kwargs` and `context_length`
   (`return_backcast`, `max_context`, a context above the 16384 limit).
5. `per_core_batch_size` > 1 returns the same values as the default (1).
6. A series with pandas nullable dtype `Float64` and `pd.NA`.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/compare_direct.py
"""

import platform
import traceback
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

import timesfm
from skforecast.foundation import FoundationModel

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LENGTH = 200
STEPS = 24
QUANTILES = [0.1, 0.5, 0.9]

print(f"python {platform.python_version()} | {platform.platform()} | "
      f"cpu threads {torch.get_num_threads()}")
for pkg in ["timesfm", "torch", "huggingface_hub", "numpy", "pandas", "skforecast"]:
    print(f"{pkg} {version(pkg)}")
print()

rng = np.random.default_rng(0)


def make_series(n, start, level, name):
    idx = pd.date_range(start, periods=n, freq="h")
    t = np.arange(n)
    values = level + 5 * np.sin(2 * np.pi * t / 24) + 0.01 * t + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx, name=name)


s1 = make_series(400, "2024-01-01", 50, "s1")
s2 = make_series(150, "2024-01-05", 20, "s2")
s3 = make_series(300, "2024-01-03", 100, "s3")
s3.iloc[250:255] = np.nan
series = {"s1": s1, "s2": s2, "s3": s3}

# 1. FoundationModel vs direct backend call
# ==============================================================================
print("== 1. FoundationModel vs direct call")
fm = FoundationModel(model_id=MODEL_ID, context_length=CONTEXT_LENGTH)
fm.fit(series=series)
pred_point = fm.predict(steps=STEPS)
pred_q = fm.predict(steps=STEPS, quantiles=QUANTILES)
adapter_model = fm.adapter._model
print("adapter model class:", type(adapter_model).__name__,
      "| mro:", [c.__name__ for c in type(adapter_model).__mro__[:3]])
print("compiled ForecastConfig:", adapter_model.forecast_config)
print("global_batch_size:", adapter_model.global_batch_size,
      "| model device:", adapter_model.model.device)

direct = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
direct.compile(timesfm.ForecastConfig(max_context=CONTEXT_LENGTH, max_horizon=STEPS))
inputs = [series[k].to_numpy()[-CONTEXT_LENGTH:] for k in series]
d_point, d_quant = direct.forecast(horizon=STEPS, inputs=inputs)
print("direct output shapes:", d_point.shape, d_quant.shape)

max_diff_point = 0.0
max_diff_q = 0.0
for i, name in enumerate(series):
    a_point = pred_point.loc[pred_point["level"] == name, "pred"].to_numpy()
    max_diff_point = max(max_diff_point, float(np.max(np.abs(a_point - d_point[i]))))
    sub = pred_q.loc[pred_q["level"] == name]
    for q in QUANTILES:
        a_q = sub[f"q_{q}"].to_numpy()
        d_q = d_quant[i, :, round(q * 10)]
        max_diff_q = max(max_diff_q, float(np.max(np.abs(a_q - d_q))))
print(f"max |adapter - direct| point: {max_diff_point:.3e}")
print(f"max |adapter - direct| quantiles {QUANTILES}: {max_diff_q:.3e}")
print("predict_quantiles columns:", pred_q.columns.tolist())

# 2. Point forecast channel
# ==============================================================================
print("\n== 2. Point forecast channel")
print("point == quantile channel 5 (q0.5):",
      bool(np.array_equal(d_point, d_quant[..., 5])))
print("point == channel 0 (mean):", bool(np.array_equal(d_point, d_quant[..., 0])))
print(f"max |channel 0 - channel 5|: {np.max(np.abs(d_quant[..., 0] - d_quant[..., 5])):.4f}")

# 3. torch_compile and the _TimesFMCompat workaround
# ==============================================================================
print("\n== 3. torch_compile and _from_pretrained workaround")
print("adapter-loaded instance torch_compile:", adapter_model.torch_compile)
print("adapter-loaded model.forward type:", type(adapter_model.model.forward).__name__)
print("direct (no workaround) instance torch_compile:", direct.torch_compile)
print("direct load without _TimesFMCompat: OK (no TypeError)")
try:
    forced = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=True)
    print("from_pretrained(torch_compile=True) -> torch_compile:", forced.torch_compile,
          "| model.forward type:", type(forced.model.forward).__name__)
    del forced
except Exception as exc:
    print("from_pretrained(torch_compile=True) error:", type(exc).__name__, exc)

# 4. Edge cases of forecast_config_kwargs and context_length
# ==============================================================================
print("\n== 4. Edge cases")


def run_case(label, **adapter_kwargs):
    try:
        fm_case = FoundationModel(model_id=MODEL_ID, **adapter_kwargs)
        fm_case.adapter._model = None
        fm_case.fit(series={"s1": s1})
        raw = fm_case.adapter.predict(
            steps=STEPS, context={"s1": s1.iloc[-fm_case.adapter.context_length:]},
            context_exog=None, exog=None, quantiles=None,
        )
        print(f"{label}: adapter.predict shape {raw['s1'].shape}")
        out = fm_case.predict(steps=STEPS)
        print(f"{label}: FoundationModel.predict rows {len(out)}")
    except Exception as exc:
        print(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")


run_case("return_backcast=True", context_length=CONTEXT_LENGTH,
         forecast_config_kwargs={"return_backcast": True})
run_case("max_context in kwargs", context_length=CONTEXT_LENGTH,
         forecast_config_kwargs={"max_context": 64})
run_case("context_length=16384", context_length=16384)
run_case("context_length=16000", context_length=16000)
run_case("use_continuous_quantile_head, steps<=1024", context_length=CONTEXT_LENGTH,
         forecast_config_kwargs={"use_continuous_quantile_head": True})

# 5. per_core_batch_size
# ==============================================================================
print("\n== 5. per_core_batch_size")
batched = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
batched.compile(timesfm.ForecastConfig(
    max_context=CONTEXT_LENGTH, max_horizon=STEPS, per_core_batch_size=4))
b_point, b_quant = batched.forecast(horizon=STEPS, inputs=list(inputs))
print("global_batch_size:", batched.global_batch_size)
print(f"max |batch 4 - batch 1| point: {np.max(np.abs(b_point - d_point)):.3e}, "
      f"quantiles: {np.max(np.abs(b_quant - d_quant)):.3e}")
del batched

# 6. Nullable Float64 with pd.NA
# ==============================================================================
print("\n== 6. Nullable Float64 series with pd.NA")
s_nullable = s3.astype("Float64")
print("dtype:", s_nullable.dtype, "| n NA:", int(s_nullable.isna().sum()))
try:
    fm_nullable = FoundationModel(model_id=MODEL_ID, context_length=CONTEXT_LENGTH)
    fm_nullable.adapter._model = adapter_model
    fm_nullable.fit(series={"s3": s_nullable})
    out = fm_nullable.predict(steps=STEPS)
    a = out["pred"].to_numpy()
    print(f"OK, max |nullable - float64 direct|: {np.max(np.abs(a - d_point[2])):.3e}")
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:300]}")
    traceback.print_exc(limit=3)
