"""
Options measured on tfc-t0>=0.6 (through the diagnostic shim) for the two
batch effects found in compare_direct.py:

- left NaN padding: `trim_leading_missing=True` forwarded to `predict`.
- heterogeneous covariates: `supports_heterogeneous_covariates=False` (set on
  the class in this process only), so FoundationModel groups the series by
  covariate signature, vs `trim_leading_missing=True` alone.

Each line compares the forecast of a series in a batch with its forecast alone.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_guide import load_data  # noqa: E402
from t0_shim import NewApiProxy, load_backend_model  # noqa: E402

from skforecast.foundation import FoundationModel  # noqa: E402
from skforecast.foundation._adapters import T0Adapter  # noqa: E402

print(
    "python", platform.python_version(), "| torch", torch.__version__,
    "| numpy", np.__version__, "| pandas", pd.__version__,
    "| tfc-t0", version("tfc-t0"), "| skforecast", version("skforecast"),
)
MODEL_ID = "theforecastingcompany/t0-alpha"
STEPS = 24
CTX = 500
BACKEND = load_backend_model(MODEL_ID)
data = load_data()
demand = data["Demand"].loc[:"2014-11-30 23:00:00"]
exog_hist = data[["Temperature", "Holiday"]].loc[:"2014-11-30 23:00:00"]
exog_future = data[["Temperature", "Holiday"]].loc["2014-12-01 00:00:00":].iloc[:STEPS]


def fm(**extra):
    model = FoundationModel(model_id=MODEL_ID, context_length=8192)
    model.adapter._model = NewApiProxy(BACKEND, **extra)
    return model


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def level(df, name):
    return df.loc[df["level"] == name, "pred"].to_numpy()


short = demand.iloc[-200:]
long = data["Temperature"].loc[:"2014-11-30 23:00:00"].iloc[-CTX:] * 100
a = demand.iloc[-CTX:]
b = demand.iloc[-CTX:] * 0.8
xb = exog_hist.iloc[-CTX:][["Temperature"]]

for heterogeneous in (True, False):
    T0Adapter.supports_heterogeneous_covariates = heterogeneous
    for extra in ({}, {"trim_leading_missing": True}):
        tag = f"supports_heterogeneous_covariates={heterogeneous}, predict_extra={extra}"
        alone = fm(**extra).fit(series={"s": short}).predict(steps=STEPS)["pred"].to_numpy()
        batch = fm(**extra).fit(series={"s": short, "l": long}).predict(steps=STEPS)
        print(f"[padding] {tag}: max rel diff = {rel(level(batch, 's'), alone):.3g}")

        alone = fm(**extra).fit(series={"a": a}).predict(steps=STEPS)["pred"].to_numpy()
        model = fm(**extra).fit(series={"a": a, "b": b}, exog={"a": None, "b": xb})
        batch = model.predict(steps=STEPS, exog={"a": None, "b": exog_future[["Temperature"]]})
        print(f"[het. covariates] {tag}: max rel diff = {rel(level(batch, 'a'), alone):.3g}")

        # Same check when the series without the column is the shorter one, so
        # padding and pooled covariates combine.
        alone = fm(**extra).fit(series={"s": short}).predict(steps=STEPS)["pred"].to_numpy()
        model = fm(**extra).fit(series={"s": short, "b": b}, exog={"s": None, "b": xb})
        batch = model.predict(steps=STEPS, exog={"s": None, "b": exog_future[["Temperature"]]})
        print(f"[padding + het. covariates] {tag}: max rel diff = {rel(level(batch, 's'), alone):.3g}")
