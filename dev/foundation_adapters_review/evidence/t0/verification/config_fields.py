"""
Load t0-beta and t0-gamma with the installed tfc-t0 and print which fields of
their config.json survive in the model configuration, plus whether a forecast
runs through the current adapter call (0.3.2) without error.
"""

import json
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
from huggingface_hub import hf_hub_download
from t0 import T0Forecaster

print("tfc-t0", version("tfc-t0"))
for name in ["t0-beta", "t0-gamma"]:
    repo = f"theforecastingcompany/{name}"
    hub_cfg = json.load(open(hf_hub_download(repo, "config.json")))
    model = T0Forecaster.from_pretrained(repo).eval()
    cfg = asdict(model.config)
    missing = {k: v for k, v in hub_cfg.items() if k not in cfg and k != "dtype"}
    differ = {k: (v, cfg[k]) for k, v in hub_cfg.items() if k in cfg and cfg[k] != v and k != "dtype"}
    print(name, "| fields of config.json dropped:", missing, "| differing:", differ)
    ctx = np.sin(np.arange(256, dtype=np.float32) / 10)[None]
    if tuple(int(p) for p in version("tfc-t0").split(".")[:2]) < (0, 4):
        out = model.predict(context=ctx, horizon=8, quantiles=[0.5])
    else:
        out = model.predict(ctx, 8, quantile_levels=[0.5])
    print(name, "| predict OK, shape", tuple(out.quantiles.shape))
    del model
