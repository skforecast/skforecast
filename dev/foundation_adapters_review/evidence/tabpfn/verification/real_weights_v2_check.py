"""
Discriminative checks with real (non random) weights that do not need a
TABPFN_TOKEN: the TabPFN-v2 regressor checkpoint of the public, non gated repo
Prior-Labs/TabPFN-v2-reg (tabpfn 9.2.0 does not ask for license acceptance for
v2), passed as `tabpfn_model_config={"model_path": <ckpt>}`. This is NOT the
checkpoint the adapter uses by default (TabPFN-3.5); it only provides a model
whose forecasts depend on the covariates and on the timestamps, so that
equalities are meaningful. No accuracy is measured.

Download first (about 44 MB, outside the repository):
    ~/.venvs/fm-review-tabpfn/bin/python -I -c "from huggingface_hub import \
hf_hub_download; import os; hf_hub_download('Prior-Labs/TabPFN-v2-reg', \
'tabpfn-v2-regressor.ckpt', local_dir=os.path.expanduser('~/fm-review-tmp-tabpfn-verif/v2'))"

Run from the repository root:
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/verification/real_weights_v2_check.py
"""

import importlib.metadata as md
import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
torch.set_num_threads(2)
print("python :", sys.version.split()[0], platform.platform())
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "torch", "numpy", "pandas"]:
    print(f"{package:<19}:", md.version(package))

CKPT = os.path.expanduser("~/fm-review-tmp-tabpfn-verif/v2/tabpfn-v2-regressor.ckpt")
CONFIG = {"model_path": CKPT, "device": "cpu", "n_estimators": 2, "random_state": 0}
STEPS = 12
CL = 150
Q = [0.1, 0.5, 0.9]
QC = [f"q_{q}" for q in Q]

rng = np.random.default_rng(3)
idx = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
x1 = rng.normal(0, 1, len(idx))
x2 = rng.normal(0, 1, len(idx))
y1 = 50 + 5 * np.sin(2 * np.pi * idx.hour / 24) + 4 * x1 + rng.normal(0, 0.3, len(idx))
y2 = 20 + 3 * np.cos(2 * np.pi * idx.hour / 24) + 3 * x2 + rng.normal(0, 0.3, len(idx))
s1 = pd.Series(y1[:300], index=idx[:300], name="s1")
s2 = pd.Series(y2[100:300], index=idx[100:300], name="s2")
s2.iloc[[5, 6, 50]] = np.nan
fidx = idx[300:]
eh = {
    "s1": pd.DataFrame({"x1": x1[:300]}, index=idx[:300]),
    "s2": pd.DataFrame({"x1": x1[100:300], "x2": x2[100:300]}, index=idx[100:300]),
}
ef = {
    "s1": pd.DataFrame({"x1": x1[300:]}, index=fidx),
    "s2": pd.DataFrame({"x1": x1[300:], "x2": x2[300:]}, index=fidx),
}


def adapter(series, eh, ef, features=None):
    kw = {"tabpfn_model_config": CONFIG}
    if features is not None:
        kw["temporal_features"] = features
    m = FoundationModel("priorlabs/tabpfn-ts", context_length=CL, **kw)
    m.fit(series=series, exog=eh)
    return m.predict(steps=STEPS, exog=ef, quantiles=Q)


def direct(series, eh, ef):
    cp, fp = [], []
    for name, s in series.items():
        s = s.iloc[-CL:]
        c = pd.DataFrame({"item_id": name, "timestamp": s.index, "target": s.to_numpy()})
        f = pd.DataFrame({"item_id": name, "timestamp": fidx})
        if eh is not None:
            c = c.join(eh[name].loc[s.index].reset_index(drop=True))
            f = f.join(ef[name].reset_index(drop=True))
        cp.append(c)
        fp.append(f)
    pipe = TabPFNTSPipeline(max_context_length=CL, tabpfn_mode=TabPFNMode.LOCAL, tabpfn_model_config=CONFIG)
    return pipe.predict_df(pd.concat(cp, ignore_index=True), pd.concat(fp, ignore_index=True), quantiles=Q)


def get(out, name):
    return out.loc[out["level"] == name, QC].to_numpy()


both = {"s1": s1, "s2": s2}
a = adapter(both, eh, ef)
d = direct(both, eh, ef)
diff = max(np.abs(get(a, n) - d.loc[n][Q].to_numpy()).max() for n in both)
print("\n[F2] adapter vs direct, 2 series, NaN, heterogeneous exog: max abs diff", diff)
print("     std of q0.5 over the horizon (s1):", round(float(get(a, "s1")[:, 1].std()), 4))
a_shift = adapter(both, eh, {k: v.assign(x1=v["x1"] + 3) for k, v in ef.items()})
print("     sensitivity: future x1 + 3 changes the forecast by", round(float(np.abs(get(a, "s1") - get(a_shift, "s1")).max()), 4))
a_noexog = adapter(both, None, None)
d_noexog = direct(both, None, None)
diff = max(np.abs(get(a_noexog, n) - d_noexog.loc[n][Q].to_numpy()).max() for n in both)
print("     adapter vs direct, without exog: max abs diff", diff)

# Wrong alignment would be detected: shift the future exog by one step
d_bad = direct(both, eh, {k: v.shift(1).bfill() for k, v in ef.items()})
print("     control: direct with future exog shifted 1 step vs adapter:",
      round(float(np.abs(get(a, "s1") - d_bad.loc["s1"][Q].to_numpy()).max()), 4))

print("\n[F15] s1 alone vs s1 batched with s2 (s2 has extra covariate x2)")
for label, feats in [
    ("default features of the installed backend", None),
    ("features of 1.3.0", [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]),
]:
    alone = adapter({"s1": s1}, {"s1": eh["s1"]}, {"s1": ef["s1"]}, feats)
    batched = adapter(both, eh, ef, feats)
    print(f"     {label}: max abs diff", float(np.abs(get(alone, "s1") - get(batched, "s1")).max()))
