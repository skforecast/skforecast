"""
Workaround check of `T0Adapter._load_model`: which exception does
`T0Forecaster.from_pretrained` raise today for (a) a repository that does not
exist and (b) an ONNX repository of theforecastingcompany, which the registry
prefix `theforecastingcompany/t0` accepts. The adapter turns a `TypeError`
into an `OSError` that says the configuration could not be downloaded.

Usage (from the repository root, in the review env):
    python dev/foundation_adapters_review/evidence/t0/check_load_errors.py
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version

from huggingface_hub import hf_hub_download

print(
    "python", platform.python_version(), "| tfc-t0", version("tfc-t0"),
    "| huggingface_hub", version("huggingface_hub"),
)

from t0 import T0Forecaster  # noqa: E402

for model_id in (
    "theforecastingcompany/t0-does-not-exist",
    "theforecastingcompany/t0-alpha-onnx-int8",
):
    print(f"\n== {model_id}")
    try:
        cfg = hf_hub_download(model_id, "config.json")
        with open(cfg) as fh:
            print("config.json keys:", sorted(json.load(fh))[:15])
    except Exception as exc:
        print("config.json:", type(exc).__name__)
    try:
        T0Forecaster.from_pretrained(model_id)
        print("from_pretrained: loaded")
    except Exception as exc:
        print("from_pretrained:", type(exc).__name__, ":", str(exc)[:300])
