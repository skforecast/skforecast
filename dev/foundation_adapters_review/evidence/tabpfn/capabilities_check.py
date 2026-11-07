"""
Probes of behaviours of TabPFNAdapter with the installed backend that do not
need the real weights (tiny random TabPFN-3.5 passed as `ModelSpecs`, see
compare_direct.py):

1. Backend warnings raised inside `predict_df` while the adapter redirects
   stderr (`show_progress=False`, the default) do not reach the user.
2. All-NaN (or single valid value) context: what the backend does.
3. TabPFNTSPipeline signature of the installed version vs the kwargs the
   adapter passes, and whether `enable_torch_compile` in `tabpfn_model_config`
   can be turned off from the adapter (code path only, no CUDA here).

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/capabilities_check.py
"""

import contextlib
import importlib.metadata as md
import inspect
import io
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tabpfn.architectures import tabpfn_v3_5
from tabpfn.base import ModelSpecs
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn_time_series import TabPFNTSPipeline

from skforecast.foundation import FoundationModel

print("python", sys.version.split()[0])
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "torch", "numpy", "pandas"]:
    print(f"{package:<19}:", md.version(package))
print()

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

idx = pd.date_range("2024-01-01", periods=212, freq="h")
rng = np.random.default_rng(0)
y = pd.Series(10 + rng.normal(size=200), index=idx[:200], name="y")
exog_hist = pd.DataFrame({"x": rng.normal(size=200)}, index=idx[:200])
exog_fut = pd.DataFrame({"x": rng.normal(size=12)}, index=idx[200:])
exog_fut.iloc[3, 0] = np.nan  # triggers the backend UserWarning about NaN covariates

print("1. Backend warnings inside predict_df")
for show_progress in [False, True]:
    model = FoundationModel(
        model_id="priorlabs/tabpfn-ts", tabpfn_model_config=CONFIG,
        show_progress=show_progress,
    )
    model.fit(series=y, exog=exog_hist)
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("default")
        with contextlib.redirect_stderr(buffer):
            model.predict(steps=12, exog=exog_fut)
    text = buffer.getvalue()
    print(
        f"   show_progress={show_progress!s:<5}: user stderr contains backend "
        f"warning 'covariate values in future_tsdf are missing': "
        f"{'covariate values in future_tsdf are missing' in text}"
    )
print()

print("2. Context with too few valid values")
for label, values in [
    ("all NaN", np.full(200, np.nan)),
    ("one valid value", np.r_[np.full(199, np.nan), 5.0]),
]:
    s = pd.Series(values, index=idx[:200], name="y")
    model = FoundationModel(model_id="priorlabs/tabpfn-ts", tabpfn_model_config=CONFIG)
    try:
        model.fit(series=s)
        p = model.predict(steps=3)
        print(f"   {label:<16}: no error, pred = {np.round(p['pred'].to_numpy(), 3).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:<16}: {type(exc).__name__}: {str(exc)[:200]}")
print()

print("3. TabPFNTSPipeline.__init__ signature")
print("  ", inspect.signature(TabPFNTSPipeline.__init__))
src = inspect.getsource(TabPFNTSPipeline.__init__)
line = [s.strip() for s in src.splitlines() if "enable_torch_compile" in s]
print("   lines with enable_torch_compile:", line)
