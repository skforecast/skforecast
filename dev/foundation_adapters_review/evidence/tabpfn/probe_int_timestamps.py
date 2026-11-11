"""
Check whether tabpfn-time-series accepts integer timestamps, which decides if
the synthetic daily DatetimeIndex of TabPFNAdapter (RangeIndex series) is
still needed. Uses a tiny random TabPFN-3.5 (ModelSpecs), no real weights.

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/probe_int_timestamps.py
"""

import importlib.metadata as md
import warnings

import numpy as np
import pandas as pd
import torch
from tabpfn.architectures import tabpfn_v3_5
from tabpfn.base import ModelSpecs
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn_time_series import TabPFNTSPipeline
from tabpfn_time_series.features import RunningIndexFeature

warnings.simplefilter("ignore")
for package in ["tabpfn-time-series", "tabpfn", "pandas", "numpy"]:
    print(f"{package:<19}:", md.version(package))

torch.manual_seed(0)
config = tabpfn_v3_5.TabPFNV3p5Config(
    max_num_classes=3, num_buckets=10, embed_dim=16, nlayers=1, icl_num_heads=2,
    icl_num_kv_heads_test=1, dist_embed_num_heads=2, dist_embed_num_blocks=1,
    dist_embed_num_inducing_points=4, feat_agg_num_heads=2, feat_agg_num_blocks=1,
    feat_agg_num_cls_tokens=1,
)
specs = ModelSpecs(
    model=tabpfn_v3_5.get_architecture(config).eval(),
    architecture_config=config,
    inference_config=InferenceConfig(PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")]),
)
cfg = {"model_path": specs, "device": "cpu", "n_estimators": 1, "random_state": 0}
ctx = pd.DataFrame({"item_id": "a", "timestamp": np.arange(50), "target": np.random.default_rng(0).normal(size=50)})
fut = pd.DataFrame({"item_id": "a", "timestamp": np.arange(50, 53)})
for label, features in [("default features", None), ("[RunningIndexFeature()]", [RunningIndexFeature()])]:
    kwargs = {"tabpfn_model_config": cfg}
    if features is not None:
        kwargs["temporal_features"] = features
    try:
        out = TabPFNTSPipeline(**kwargs).predict_df(context_df=ctx, future_df=fut, quantiles=[0.5])
        print(f"integer timestamps, {label}: OK, index dtype {out.index.get_level_values('timestamp').dtype}")
    except Exception as exc:  # noqa: BLE001
        print(f"integer timestamps, {label}: {type(exc).__name__}: {str(exc)[:200]}")

with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    for expr in [
        'pd.Timestamp("2000-01-01") + pd.Timedelta(days=5)',
        'pd.date_range("2000-01-01", periods=5 + 3, freq="D")[5:]',
    ]:
        try:
            eval(expr)
            print("no DeprecationWarning:", expr)
        except DeprecationWarning as exc:
            print("DeprecationWarning   :", expr, "->", str(exc)[:90])
