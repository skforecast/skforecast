"""
Registry coverage of the checkpoints of theforecastingcompany on the Hub and
what happens when they are loaded through the public API; plus the
`torch_dtype` parameter of the adapter through the public API.

Usage (from the repository root, in the review env):
    python dev/foundation_adapters_review/evidence/t0/registry_and_loading.py
"""

from __future__ import annotations

import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel, get_model_info
from skforecast.foundation._adapters import _resolve_adapter

print(
    "python", platform.python_version(), "| torch", torch.__version__,
    "| tfc-t0", version("tfc-t0"), "| skforecast", version("skforecast"),
)

HUB_IDS = [
    "theforecastingcompany/t0-alpha",
    "theforecastingcompany/t0-beta",
    "theforecastingcompany/t0-gamma",
    "theforecastingcompany/t0-alpha-onnx-fp16",
    "theforecastingcompany/t0-alpha-onnx-int8",
    "theforecastingcompany/t0-beta-onnx-fp16",
    "theforecastingcompany/t0-beta-onnx-int8",
]
for model_id in HUB_IDS:
    try:
        cls = _resolve_adapter(model_id)
        info = get_model_info(model_id)
        print(f"{model_id}: {cls.__name__}, license={info.license!r}")
    except Exception as exc:
        print(f"{model_id}: {type(exc).__name__}: {exc}")

y = pd.Series(
    50 + 10 * np.sin(np.arange(300) * 2 * np.pi / 24),
    index=pd.date_range("2024-01-01", periods=300, freq="h"),
)

print("\nLoading an ONNX repository through FoundationModel:")
try:
    fm = FoundationModel("theforecastingcompany/t0-alpha-onnx-int8", context_length=64)
    fm.fit(series=y)
    fm.predict(steps=5)
    print("  predicted without error")
except Exception as exc:
    print(f"  {type(exc).__name__}: {str(exc)[:400]}")

print("\nFoundationModel(..., torch_dtype=torch.bfloat16) through the public API:")
try:
    fm = FoundationModel("theforecastingcompany/t0-alpha", context_length=64, torch_dtype=torch.bfloat16)
    fm.fit(series=y)
    print(fm.predict(steps=3))
except Exception as exc:
    print(f"  {type(exc).__name__}: {str(exc)[:400]}")
