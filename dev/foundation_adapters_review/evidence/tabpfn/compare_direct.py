"""
Compare the output of TabPFNAdapter (through FoundationModel) with a direct
call to `TabPFNTSPipeline.predict_df` on the same input.

The real TabPFN-3.5 weights need a TABPFN_TOKEN (license acceptance), which is
not available here. The comparison therefore uses a tiny randomly initialized
TabPFN-3.5 architecture passed in memory as `tabpfn.base.ModelSpecs` (the path
added in tabpfn-time-series 1.4.0, the same fixture as the upstream test
`tests/test_model_specs.py`). The numbers say nothing about the quality of the
real checkpoint: the script only checks that the adapter translates the inputs
and the outputs exactly like a direct call to the backend.

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/compare_direct.py
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

print("python             :", sys.version.split()[0], platform.platform())
for package in [
    "skforecast", "tabpfn-time-series", "tabpfn", "tabpfn-client",
    "torch", "numpy", "pandas",
]:
    print(f"{package:<19}:", md.version(package))
print("cuda available     :", torch.cuda.is_available())
print()


def make_specs() -> ModelSpecs:
    torch.manual_seed(0)
    config = tabpfn_v3_5.TabPFNV3p5Config(
        max_num_classes=3,
        num_buckets=10,
        embed_dim=16,
        nlayers=1,
        icl_num_heads=2,
        icl_num_kv_heads_test=1,
        dist_embed_num_heads=2,
        dist_embed_num_blocks=1,
        dist_embed_num_inducing_points=4,
        feat_agg_num_heads=2,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=1,
    )
    return ModelSpecs(
        model=tabpfn_v3_5.get_architecture(config).eval(),
        architecture_config=config,
        inference_config=InferenceConfig(
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        ),
    )


SPECS = make_specs()
CONFIG = {"model_path": SPECS, "device": "cpu", "n_estimators": 1, "random_state": 0}
CONTEXT_LENGTH = 200
STEPS = 12

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
hour = idx.hour.to_numpy()
x1 = np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(idx))
x2 = rng.normal(0, 1, len(idx))
s1 = 50 + 5 * np.sin(2 * np.pi * hour / 24) + 2 * x1 + rng.normal(0, 0.5, len(idx))
s2 = 20 + 3 * np.cos(2 * np.pi * hour / 24) + x2 + rng.normal(0, 0.3, len(idx))

# series_1: 300 obs (trimmed to CONTEXT_LENGTH), series_2: 150 obs with NaN
series_1 = pd.Series(s1[:300], index=idx[:300], name="series_1")
series_2 = pd.Series(s2[150:300], index=idx[150:300], name="series_2")
series_2.iloc[[10, 11, 70]] = np.nan
future_idx = idx[300:]
exog_hist = {
    "series_1": pd.DataFrame({"x1": x1[:300]}, index=idx[:300]),
    "series_2": pd.DataFrame({"x1": x1[150:300], "x2": x2[150:300]}, index=idx[150:300]),
}
exog_future = {
    "series_1": pd.DataFrame({"x1": x1[300:]}, index=future_idx),
    "series_2": pd.DataFrame({"x1": x1[300:], "x2": x2[300:]}, index=future_idx),
}


def direct(context, exog_hist, exog_future, quantiles, temporal_features=None):
    """Build the long frames by hand, as in the README of tabpfn-time-series."""
    ctx_parts, fut_parts = [], []
    for name, s in context.items():
        s = s.iloc[-CONTEXT_LENGTH:]
        part = pd.DataFrame({"item_id": name, "timestamp": s.index, "target": s.to_numpy()})
        if exog_hist is not None:
            part = part.join(exog_hist[name].loc[s.index].reset_index(drop=True))
        ctx_parts.append(part)
        if isinstance(s.index, pd.DatetimeIndex):
            fidx = pd.date_range(s.index[-1], periods=STEPS + 1, freq=s.index.freq or pd.infer_freq(s.index))[1:]
        else:
            fidx = None
        fpart = pd.DataFrame({"item_id": name, "timestamp": fidx})
        if exog_future is not None:
            fpart = fpart.join(exog_future[name].reset_index(drop=True))
        fut_parts.append(fpart)
    kwargs = {
        "max_context_length": CONTEXT_LENGTH,
        "tabpfn_mode": TabPFNMode.LOCAL,
        "tabpfn_output_selection": "median",
        "tabpfn_model_config": CONFIG,
    }
    if temporal_features is not None:
        kwargs["temporal_features"] = temporal_features
    pipe = TabPFNTSPipeline(**kwargs)
    out = pipe.predict_df(
        context_df=pd.concat(ctx_parts, ignore_index=True),
        future_df=pd.concat(fut_parts, ignore_index=True),
        quantiles=quantiles if quantiles is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    return out


def via_adapter(context, exog_hist, exog_future, quantiles):
    model = FoundationModel(
        model_id="priorlabs/tabpfn-ts",
        context_length=CONTEXT_LENGTH,
        tabpfn_model_config=CONFIG,
    )
    model.fit(series=context, exog=exog_hist)
    return model.predict(steps=STEPS, exog=exog_future, quantiles=quantiles)


def compare(label, context, exog_hist, exog_future, quantiles):
    out_direct = direct(context, exog_hist, exog_future, quantiles)
    out_adapter = via_adapter(context, exog_hist, exog_future, quantiles)
    max_diff = 0.0
    for name in context:
        group = out_direct.loc[name]
        if quantiles is None:
            ref = group["target"].to_numpy()
            got = out_adapter.loc[out_adapter["level"] == name, "pred"].to_numpy()
        else:
            ref = group[list(quantiles)].to_numpy()
            got = out_adapter.loc[
                out_adapter["level"] == name, [f"q_{q}" for q in quantiles]
            ].to_numpy()
        assert ref.shape == got.shape, (ref.shape, got.shape)
        max_diff = max(max_diff, float(np.nanmax(np.abs(ref - got))))
        assert np.isfinite(got).all()
    print(f"{label:<55} max|adapter - direct| = {max_diff:.3e}")
    return out_direct, out_adapter


print("Output columns of predict_df (real backend):")
out = direct({"series_1": series_1}, None, None, [0.1, 0.5, 0.9])
print("  index names :", list(out.index.names))
print("  columns     :", [(c, type(c).__name__) for c in out.columns])
print()

compare("single series, point", {"series_1": series_1}, None, None, None)
compare("single series, quantiles [0.1, 0.5, 0.9]", {"series_1": series_1}, None, None, [0.1, 0.5, 0.9])
compare("single series, quantiles [0.025, 0.975] (outside default)", {"series_1": series_1}, None, None, [0.025, 0.975])
ctx2 = {"series_1": series_1, "series_2": series_2}
compare("2 series, different length, NaN, point", ctx2, None, None, None)
compare("2 series, heterogeneous known exog, quantiles", ctx2, exog_hist, exog_future, [0.1, 0.5, 0.9])
compare("2 series, heterogeneous known exog, point", ctx2, exog_hist, exog_future, None)

# Point forecast vs 0.5 quantile (tabpfn_output_selection="median")
o = direct({"series_1": series_1}, None, None, [0.5])
print()
print("target vs 0.5 quantile, max abs diff:", float(np.abs(o["target"] - o[0.5]).max()))

# Default temporal features of the installed backend
from tabpfn_time_series.pipeline import TABPFN_TS_DEFAULT_FEATURES  # noqa: E402

print("TABPFN_TS_DEFAULT_FEATURES:", [type(f).__name__ for f in TABPFN_TS_DEFAULT_FEATURES])

# Does the known covariate lag feature see the exog passed by the adapter?
pipe = TabPFNTSPipeline(max_context_length=CONTEXT_LENGTH, tabpfn_model_config=CONFIG)
from tabpfn_time_series import TimeSeriesDataFrame  # noqa: E402

adapter = FoundationModel(model_id="priorlabs/tabpfn-ts", tabpfn_model_config=CONFIG).adapter
ctx_df = adapter._build_context_df(["series_1"], {"series_1": series_1.iloc[-CONTEXT_LENGTH:]}, {"series_1": exog_hist["series_1"].iloc[-CONTEXT_LENGTH:]}, True)
fut_df = adapter._build_future_df(["series_1"], {"series_1": series_1.iloc[-CONTEXT_LENGTH:]}, {"series_1": exog_future["series_1"]}, STEPS, True)
train, test = pipe.featurize(TimeSeriesDataFrame.from_data_frame(ctx_df), TimeSeriesDataFrame.from_data_frame(fut_df))
print("featurized columns (adapter input, default features):", list(train.columns))

# RangeIndex series
s_range = pd.Series(series_1.to_numpy(), name="series_1")
model = FoundationModel(model_id="priorlabs/tabpfn-ts", context_length=CONTEXT_LENGTH, tabpfn_model_config=CONFIG)
model.fit(series=s_range)
p = model.predict(steps=STEPS, quantiles=[0.1, 0.9])
print("RangeIndex: index", list(p.index[:3]), "... finite:", bool(np.isfinite(p[["q_0.1", "q_0.9"]].to_numpy()).all()))

# Categorical (pandas category) known covariate passed through unchanged
cat_hist = pd.DataFrame({"x1": x1[:300], "day": pd.Categorical(np.where(idx[:300].dayofweek < 5, "wd", "we"))}, index=idx[:300])
cat_fut = pd.DataFrame({"x1": x1[300:], "day": pd.Categorical(np.where(future_idx.dayofweek < 5, "wd", "we"), categories=["wd", "we"])}, index=future_idx)
model = FoundationModel(model_id="priorlabs/tabpfn-ts", context_length=CONTEXT_LENGTH, tabpfn_model_config=CONFIG)
try:
    model.fit(series=series_1, exog=cat_hist)
    p = model.predict(steps=STEPS, exog=cat_fut)
    print("category covariate: predict ran, finite:", bool(np.isfinite(p["pred"]).all()))
except Exception as exc:  # noqa: BLE001
    print("category covariate: error", type(exc).__name__, str(exc)[:300])
str_hist = cat_hist.assign(day=cat_hist["day"].astype(str))
str_fut = cat_fut.assign(day=cat_fut["day"].astype(str))
model = FoundationModel(model_id="priorlabs/tabpfn-ts", context_length=CONTEXT_LENGTH, tabpfn_model_config=CONFIG)
try:
    model.fit(series=series_1, exog=str_hist)
    p = model.predict(steps=STEPS, exog=str_fut)
    print("string covariate: predict ran, finite:", bool(np.isfinite(p["pred"]).all()))
except Exception as exc:  # noqa: BLE001
    print("string covariate: error", type(exc).__name__, str(exc)[:300])
