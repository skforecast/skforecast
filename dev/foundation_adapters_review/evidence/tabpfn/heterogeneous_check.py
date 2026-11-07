"""
Check that `supports_heterogeneous_covariates=True` holds with the installed
backend: the forecast of a series must not change when it is predicted in the
same call as another series that has an extra covariate (the long frame then
holds an all-NaN column for the first series). Uses a tiny random TabPFN-3.5
(ModelSpecs, no real weights), so only the equality matters, not the values.

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/heterogeneous_check.py
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
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "torch", "pandas", "numpy"]:
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

rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=212, freq="h")
x1, x2 = rng.normal(size=212), rng.normal(size=212)
s1 = pd.Series(10 + x1[:200] + rng.normal(size=200), index=idx[:200], name="s1")
s2 = pd.Series(5 + x2[:200] + rng.normal(size=200), index=idx[:200], name="s2")
e_hist = {
    "s1": pd.DataFrame({"x1": x1[:200]}, index=idx[:200]),
    "s2": pd.DataFrame({"x1": x1[:200], "x2": x2[:200]}, index=idx[:200]),
}
e_fut = {
    "s1": pd.DataFrame({"x1": x1[200:]}, index=idx[200:]),
    "s2": pd.DataFrame({"x1": x1[200:], "x2": x2[200:]}, index=idx[200:]),
}

for label, features in [
    ("default features (1.4.0, with KnownCovariateLagFeature)", None),
    ("features of 1.3.0 (no KnownCovariateLagFeature)", [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]),
]:
    kwargs = {"tabpfn_model_config": cfg}
    if features is not None:
        kwargs["temporal_features"] = features
    alone = FoundationModel(model_id="priorlabs/tabpfn-ts", **kwargs)
    alone.fit(series={"s1": s1}, exog={"s1": e_hist["s1"]})
    p_alone = alone.predict(steps=12, exog={"s1": e_fut["s1"]}, quantiles=[0.1, 0.5, 0.9])
    both = FoundationModel(model_id="priorlabs/tabpfn-ts", **kwargs)
    both.fit(series={"s1": s1, "s2": s2}, exog=e_hist)
    p_both = both.predict(steps=12, exog=e_fut, quantiles=[0.1, 0.5, 0.9])
    a = p_alone[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    b = p_both.loc[p_both["level"] == "s1", ["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    print(f"{label}: max|s1 alone - s1 with s2(extra x2)| = {np.abs(a - b).max():.3e}")
