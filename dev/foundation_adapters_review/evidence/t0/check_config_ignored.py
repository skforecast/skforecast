"""
Which fields of the published `config.json` the installed tfc-t0 keeps when it
loads t0-beta and t0-gamma. tfc-t0<0.5 (t0-beta) and <0.6 (t0-gamma) accept the
unknown keys through `**_` in `T0Forecaster.__init__` and load without error,
so the registry of skforecast serves them on the version the docs recommend
(`tfc-t0<0.4`). Uses the weights already in the Hugging Face cache.

Usage (from the repository root, in each review env):
    python dev/foundation_adapters_review/evidence/t0/check_config_ignored.py
"""

from __future__ import annotations

import dataclasses
import json
import platform
from importlib.metadata import version

from huggingface_hub import hf_hub_download

print("python", platform.python_version(), "| tfc-t0", version("tfc-t0"))

from t0 import T0Forecaster  # noqa: E402

for model_id in ("theforecastingcompany/t0-beta", "theforecastingcompany/t0-gamma"):
    with open(hf_hub_download(model_id, "config.json")) as fh:
        published = json.load(fh)
    model = T0Forecaster.from_pretrained(model_id)
    loaded = dataclasses.asdict(model.config)
    print(f"\n== {model_id}: loaded without error")
    for key in ("scaler_eps", "scaler_eps_mode", "scaler_zero_no_value", "xpos_scale_layout"):
        if key in published:
            print(f"  {key}: published={published[key]!r} loaded={loaded.get(key, '<field absent>')!r}")
