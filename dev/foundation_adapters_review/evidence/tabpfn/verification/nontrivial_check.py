"""
Check that the adapter vs direct comparison of compare_direct.py is not
trivially zero: with the same tiny random TabPFN-3.5 (ModelSpecs), the
predictions must change when the input changes (other series, exog dropped,
exog perturbed), and the adapter must still match the direct call.

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/verification/nontrivial_check.py
"""

import importlib.metadata as md
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tabpfn.architectures import tabpfn_v3_5
from tabpfn.base import ModelSpecs
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
print("python :", sys.version.split()[0], platform.platform())
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "torch", "numpy", "pandas"]:
    print(f"{package:<19}:", md.version(package))

torch.manual_seed(0)
config = tabpfn_v3_5.TabPFNV3p5Config(
    max_num_classes=3, num_buckets=10, embed_dim=16, nlayers=1, icl_num_heads=2,
    icl_num_kv_heads_test=1, dist_embed_num_heads=2, dist_embed_num_blocks=1,
    dist_embed_num_inducing_points=4, feat_agg_num_heads=2, feat_agg_num_blocks=1,
    feat_agg_num_cls_tokens=1,
)
SPECS = ModelSpecs(
    model=tabpfn_v3_5.get_architecture(config).eval(),
    architecture_config=config,
    inference_config=InferenceConfig(PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")]),
)
CONFIG = {"model_path": SPECS, "device": "cpu", "n_estimators": 1, "random_state": 0}
STEPS = 12

rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=200 + STEPS, freq="h")
x1 = rng.normal(0, 1, len(idx))
y = 100 + 10 * np.sin(2 * np.pi * idx.hour / 24) + 5 * x1 + rng.normal(0, 0.5, len(idx))
series = pd.Series(y[:200], index=idx[:200], name="s")
exog_h = pd.DataFrame({"x1": x1[:200]}, index=idx[:200])
exog_f = pd.DataFrame({"x1": x1[200:]}, index=idx[200:])


def adapter(series, exog_h, exog_f):
    m = FoundationModel("priorlabs/tabpfn-ts", context_length=200, tabpfn_model_config=CONFIG)
    m.fit(series=series, exog=exog_h)
    return m.predict(steps=STEPS, exog=exog_f, quantiles=[0.1, 0.5, 0.9])[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()


def direct(series, exog_h, exog_f):
    ctx = pd.DataFrame({"item_id": "s", "timestamp": series.index, "target": series.to_numpy()})
    fut = pd.DataFrame({"item_id": "s", "timestamp": idx[200:]})
    if exog_h is not None:
        ctx["x1"] = exog_h["x1"].to_numpy()
        fut["x1"] = exog_f["x1"].to_numpy()
    pipe = TabPFNTSPipeline(max_context_length=200, tabpfn_mode=TabPFNMode.LOCAL, tabpfn_model_config=CONFIG)
    return pipe.predict_df(ctx, fut, quantiles=[0.1, 0.5, 0.9])[[0.1, 0.5, 0.9]].to_numpy()


a = adapter(series, exog_h, exog_f)
d = direct(series, exog_h, exog_f)
print("adapter vs direct, exog          : max abs diff", float(np.abs(a - d).max()))
print("spread of predictions (std q0.5) :", float(a[:, 1].std()))
print("q0.1 < q0.5 < q0.9 everywhere    :", bool((a[:, 0] <= a[:, 1]).all() and (a[:, 1] <= a[:, 2]).all()))
b = adapter(series, None, None)
print("with vs without exog             : max abs diff", float(np.abs(a - b).max()))
c = adapter(series, exog_h, exog_f.assign(x1=exog_f["x1"] + 3))
print("future exog +3 vs original       : max abs diff", float(np.abs(a - c).max()))
e = adapter(series * 2, exog_h, exog_f)
print("series x2 vs original            : max abs diff", float(np.abs(a - e).max()))
