"""
Capability check for `supports_categorical_covariates` with real (non random)
weights that need no token: the public TabPFN-v2 regressor checkpoint
(Prior-Labs/TabPFN-v2-reg), not the TabPFN-3.5 default of the adapter. The
target has a jump driven only by a known future covariate "regime" (two
levels). The forecast should follow the regime in the horizon when the
covariate is passed as pandas category, as str and as 0/1 numbers. Only the
direction of the effect is checked, not accuracy.

Run from the repository root (see real_weights_v2_check.py for the download):
    TABPFN_DISABLE_TELEMETRY=1 ~/.venvs/fm-review-tabpfn/bin/python \
        dev/foundation_adapters_review/evidence/tabpfn/verification/categorical_v2_check.py
"""

import importlib.metadata as md
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
torch.set_num_threads(2)
print("python :", sys.version.split()[0])
for package in ["skforecast", "tabpfn-time-series", "tabpfn", "torch", "numpy", "pandas"]:
    print(f"{package:<19}:", md.version(package))

CKPT = os.path.expanduser("~/fm-review-tmp-tabpfn-verif/v2/tabpfn-v2-regressor.ckpt")
CONFIG = {"model_path": CKPT, "device": "cpu", "n_estimators": 2, "random_state": 0}
STEPS = 24
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=200 + STEPS, freq="h")
regime = (rng.random(len(idx)) < 0.3)
regime[200:] = np.r_[np.zeros(12, bool), np.ones(12, bool)]
y = 10 + 30 * regime + rng.normal(0, 0.5, len(idx))
series = pd.Series(y[:200], index=idx[:200], name="y")
labels = np.where(regime, "high", "low")

variants = {
    "category": pd.Categorical(labels, categories=["low", "high"]),
    "str": labels.astype(object),
    "numeric 0/1": regime.astype(float),
}
for label, values in variants.items():
    col = pd.Series(values, index=idx, name="regime")
    m = FoundationModel("priorlabs/tabpfn-ts", context_length=200, tabpfn_model_config=CONFIG)
    try:
        m.fit(series=series, exog=col.iloc[:200].to_frame())
        p = m.predict(steps=STEPS, exog=col.iloc[200:].to_frame())["pred"].to_numpy()
        print(f"{label:<12}: mean pred low-regime steps {p[:12].mean():6.2f} | high-regime steps {p[12:].mean():6.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label:<12}: {type(exc).__name__}: {str(exc)[:200]}")
m = FoundationModel("priorlabs/tabpfn-ts", context_length=200, tabpfn_model_config=CONFIG)
m.fit(series=series)
p = m.predict(steps=STEPS)["pred"].to_numpy()
print(f"{'no exog':<12}: mean pred low-regime steps {p[:12].mean():6.2f} | high-regime steps {p[12:].mean():6.2f}")
