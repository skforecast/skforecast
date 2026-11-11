"""
Discriminative version of the adapter vs direct comparison. The tiny random
TabPFN-3.5 used by compare_direct.py returns forecasts that are constant over
the horizon and insensitive to covariates (see nontrivial_check.log), so equal
forecasts do not prove that timestamps and exog are aligned. Here the frames
that the adapter passes to `predict_df` are captured and featurized with the
backend's own preprocessing (`_featurize_context_future`), and compared with
the featurized frames of a hand-built direct input (README format).

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/verification/feature_level_check.py
"""

import importlib.metadata as md
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from tabpfn_time_series import TabPFNTSPipeline, TimeSeriesDataFrame
from tabpfn_time_series import pipeline as tspipe

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
print("python :", sys.version.split()[0], platform.platform())
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "numpy", "pandas"]:
    print(f"{package:<19}:", md.version(package))

CONTEXT_LENGTH = 200
STEPS = 12
rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
x1 = rng.normal(0, 1, len(idx))
x2 = rng.normal(0, 1, len(idx))
s1 = 50 + 5 * np.sin(2 * np.pi * idx.hour / 24) + 2 * x1
s2 = 20 + 3 * np.cos(2 * np.pi * idx.hour / 24) + x2
series_1 = pd.Series(s1[:300], index=idx[:300], name="series_1")
series_2 = pd.Series(s2[150:300], index=idx[150:300], name="series_2")
series_2.iloc[[10, 11, 70]] = np.nan
fidx = idx[300:]
exog_hist = {
    "series_1": pd.DataFrame({"x1": x1[:300]}, index=idx[:300]),
    "series_2": pd.DataFrame({"x1": x1[150:300], "x2": x2[150:300]}, index=idx[150:300]),
}
exog_future = {
    "series_1": pd.DataFrame({"x1": x1[300:]}, index=fidx),
    "series_2": pd.DataFrame({"x1": x1[300:], "x2": x2[300:]}, index=fidx),
}

captured = {}


def fake_predict_df(self, context_df, future_df, quantiles=None, **kwargs):
    captured["context_df"] = context_df.copy()
    captured["future_df"] = future_df.copy()
    rows = []
    for name, g in future_df.groupby("item_id", sort=False):
        for ts in g["timestamp"]:
            rows.append((name, ts))
    index = pd.MultiIndex.from_tuples(rows, names=["item_id", "timestamp"])
    out = pd.DataFrame({"target": 0.0}, index=index)
    for q in quantiles:
        out[q] = 0.0
    return out


TabPFNTSPipeline.predict_df = fake_predict_df

model = FoundationModel("priorlabs/tabpfn-ts", context_length=CONTEXT_LENGTH)
model.adapter._model = TabPFNTSPipeline.__new__(TabPFNTSPipeline)
ctx = {"series_1": series_1, "series_2": series_2}
model.fit(series=ctx, exog=exog_hist)
model.predict(steps=STEPS, exog=exog_future, quantiles=[0.1, 0.9])

ctx_parts, fut_parts = [], []
for name, s in ctx.items():
    s = s.iloc[-CONTEXT_LENGTH:]
    part = pd.DataFrame({"item_id": name, "timestamp": s.index, "target": s.to_numpy()})
    part = part.join(exog_hist[name].loc[s.index].reset_index(drop=True))
    ctx_parts.append(part)
    fpart = pd.DataFrame({"item_id": name, "timestamp": fidx})
    fpart = fpart.join(exog_future[name].reset_index(drop=True))
    fut_parts.append(fpart)
direct_ctx = pd.concat(ctx_parts, ignore_index=True)
direct_fut = pd.concat(fut_parts, ignore_index=True)

ft = tspipe.FeatureTransformer(tspipe.TABPFN_TS_DEFAULT_FEATURES)
results = []
for label, c, f in [
    ("adapter", captured["context_df"], captured["future_df"]),
    ("direct", direct_ctx, direct_fut),
]:
    tr, te = tspipe._featurize_context_future(
        TimeSeriesDataFrame.from_data_frame(c),
        TimeSeriesDataFrame.from_data_frame(f),
        ft,
        CONTEXT_LENGTH,
    )
    results.append((tr, te))
    print(f"{label:<8} train shape {tr.shape} test shape {te.shape}")

(tr_a, te_a), (tr_d, te_d) = results
cols = sorted(tr_d.columns)
print("same columns       :", sorted(tr_a.columns) == cols, cols[:8], "...")
pd.testing.assert_frame_equal(tr_a[cols].sort_index(), tr_d[cols].sort_index(), check_dtype=False)
pd.testing.assert_frame_equal(te_a[cols].sort_index(), te_d[cols].sort_index(), check_dtype=False)
print("featurized train and test frames identical (adapter vs direct): True")
print("future timestamps (adapter, series_2):", list(te_a.loc["series_2"].index[:2]))
