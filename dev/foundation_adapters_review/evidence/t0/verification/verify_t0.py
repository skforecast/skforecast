"""
Independent re-check of the T0Adapter findings (verification pass).

Runs on tfc-t0 0.3.2 and 0.6.0. On >=0.4 the current adapter call is first
executed unchanged (expected TypeError) and then a monkeypatch with the call of
proposal T0-P1 (positional `model_input`, `quantile_levels`) is used, so the
numbers come from the code change being proposed, not from a proxy.

Data: synthetic series (seeded) plus the vic_electricity demand, different
from the investigator choices, to avoid reusing the same inputs.

Usage (from the repository root):
    ~/.venvs/fm-review-t0/bin/python dev/foundation_adapters_review/evidence/t0/verification/verify_t0.py
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from backtest_guide import load_data  # noqa: E402

from skforecast.foundation import FoundationModel  # noqa: E402
from skforecast.foundation._adapters import T0Adapter  # noqa: E402

from t0 import T0Forecaster  # noqa: E402

TFC = tuple(int(p) for p in version("tfc-t0").split(".")[:3])
NEW_API = TFC >= (0, 4, 0)
MODEL_ID = "theforecastingcompany/t0-alpha"
STEPS = 12

print(
    "python", platform.python_version(), "| torch", torch.__version__,
    "| numpy", np.__version__, "| pandas", pd.__version__,
    "| tfc-t0", version("tfc-t0"), "| huggingface_hub", version("huggingface_hub"),
    "| skforecast", version("skforecast"), "|", platform.machine(),
)
torch.manual_seed(0)

BACKEND = T0Forecaster.from_pretrained(MODEL_ID).to("cpu").eval()


ORIGINAL_PREDICT = T0Adapter.predict
PREDICT_EXTRA: dict = {}


def p1_predict(self, steps, context, context_exog, exog, quantiles):
    """T0Adapter.predict with the backend call of proposal T0-P1 (new API)."""
    self._load_model()
    model = self._model

    class _Call:
        def predict(_self, context, horizon, quantiles, future_covariates=None):
            return model.predict(
                context, horizon, quantile_levels=quantiles,
                future_covariates=future_covariates, **PREDICT_EXTRA,
            )

    self._model = _Call()
    try:
        return ORIGINAL_PREDICT(self, steps, context, context_exog, exog, quantiles)
    finally:
        self._model = model


def use_p1(flag):
    T0Adapter.predict = p1_predict if (flag and NEW_API) else ORIGINAL_PREDICT


def fm(p1=True, context_length=8192):
    use_p1(p1)
    f = FoundationModel(model_id=MODEL_ID, context_length=context_length)
    f.adapter._model = BACKEND
    return f


def direct(ctx, levels, cov=None, **extra):
    if NEW_API:
        out = BACKEND.predict(ctx, STEPS, quantile_levels=levels, future_covariates=cov, **extra)
    else:
        out = BACKEND.predict(context=ctx, horizon=STEPS, quantiles=levels, future_covariates=cov, **extra)
    return out.quantiles.detach().cpu().numpy()


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))


def check(name, fn):
    print(f"\n=== {name}")
    try:
        fn()
    except Exception as exc:
        print("ERROR", type(exc).__name__, ":", str(exc)[:300])
        traceback.print_exc(limit=2, file=sys.stdout)


data = load_data()
demand = data["Demand"].loc[:"2014-06-30 23:00:00"]
temp = data[["Temperature"]].loc[:"2014-06-30 23:00:00"]
temp_future = data[["Temperature"]].loc["2014-07-01 00:00:00":].iloc[:STEPS]
rng = np.random.default_rng(7)
idx = demand.index


def c_current_call():
    y = demand.iloc[-300:]
    out = fm(False, 300).fit(series=y).predict(steps=STEPS)
    print("current adapter call OK, shape", out.shape)


def c_p1_vs_direct():
    q = [0.7, 0.2, 0.5]
    y = demand.iloc[-300:]
    out = fm(True, 300).fit(series=y).predict(steps=STEPS, quantiles=q)
    a = out[[f"q_{x}" for x in q]].to_numpy()
    s = sorted(q)
    ref = direct(y.to_numpy(np.float32)[None], s)[0][:, [s.index(x) for x in q]]
    print("no exog: max abs diff =", float(np.max(np.abs(a - ref))))
    x = temp.iloc[-300:]
    out = fm(True, 300).fit(series=y, exog=x).predict(steps=STEPS, exog=temp_future, quantiles=q)
    a = out[[f"q_{x}" for x in q]].to_numpy()
    cov = np.concatenate([x.to_numpy(np.float32), temp_future.to_numpy(np.float32)]).T[None]
    ref = direct(y.to_numpy(np.float32)[None], s, cov)[0][:, [s.index(x) for x in q]]
    print("exog:    max abs diff =", float(np.max(np.abs(a - ref))))


def c_padding():
    short = demand.iloc[-150:].rename("short")
    long = pd.Series(1000 + 50 * rng.standard_normal(400), index=idx[-400:], name="long")
    alone = fm(True).fit(series={"short": short}).predict(steps=STEPS)["pred"].to_numpy()
    b = fm(True).fit(series={"short": short, "long": long}).predict(steps=STEPS)
    inb = b.loc[b["level"] == "short", "pred"].to_numpy()
    print("short alone vs in batch (NaN padding as MISSING): max rel diff =", rel(inb, alone))
    if TFC >= (0, 6, 0):
        PREDICT_EXTRA["trim_leading_missing"] = True
        try:
            alone_t = fm(True).fit(series={"short": short}).predict(steps=STEPS)["pred"].to_numpy()
            b = fm(True).fit(series={"short": short, "long": long}).predict(steps=STEPS)
            inb_t = b.loc[b["level"] == "short", "pred"].to_numpy()
        finally:
            PREDICT_EXTRA.clear()
        print("with trim_leading_missing=True: alone vs in batch max rel diff =", rel(inb_t, alone_t))
        print("trim on a full-length single series changes it? max abs diff =", float(np.max(np.abs(alone_t - alone))))
    if TFC >= (0, 3, 0):
        from t0 import MaskType
        ctx = np.full((2, 400), np.nan, dtype=np.float32)
        ctx[0, -150:] = short.to_numpy(np.float32)
        ctx[1] = long.to_numpy(np.float32)
        mask = np.where(np.isnan(ctx), int(MaskType.MISSING), int(MaskType.VALID)).astype(np.int8)
        mask[0, :-150] = int(MaskType.PAD)
        m = direct(ctx, [0.5], mask=mask)[0, :, 0]
        print("direct with mask=PAD vs alone: max rel diff =", rel(m, alone[:]))


def c_hetero():
    a = demand.iloc[-300:].rename("a")
    bser = (demand.iloc[-300:] * 1.3 + 200).rename("b")
    xb = temp.iloc[-300:]
    alone = fm(True, 300).fit(series={"a": a}).predict(steps=STEPS)["pred"].to_numpy()
    f = fm(True, 300).fit(series={"a": a, "b": bser}, exog={"a": None, "b": xb})
    out = f.predict(steps=STEPS, exog={"a": None, "b": temp_future})
    inb = out.loc[out["level"] == "a", "pred"].to_numpy()
    print("hetero=True (current): a alone vs in batch with b+exog: max rel diff =", rel(inb, alone))
    old = T0Adapter.supports_heterogeneous_covariates
    T0Adapter.supports_heterogeneous_covariates = False
    try:
        f = fm(True, 300).fit(series={"a": a, "b": bser}, exog={"a": None, "b": xb})
        out = f.predict(steps=STEPS, exog={"a": None, "b": temp_future})
        inb2 = out.loc[out["level"] == "a", "pred"].to_numpy()
        print("hetero=False: a alone vs in batch: max rel diff =", rel(inb2, alone))
    finally:
        T0Adapter.supports_heterogeneous_covariates = old
    # direct: an all-NaN covariate row vs no covariates
    ctx = a.to_numpy(np.float32)[None]
    cov = np.full((1, 1, 300 + STEPS), np.nan, dtype=np.float32)
    d0 = direct(ctx, [0.5])[0, :, 0]
    d1 = direct(ctx, [0.5], cov)[0, :, 0]
    print("direct: all-NaN covariate row vs no covariates: max rel diff =", rel(d1, d0))


def c_quantiles():
    y = demand.iloc[-300:]
    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    out = fm(True, 300).fit(series=y).predict(steps=STEPS, quantiles=q)
    v = out[[f"q_{x}" for x in q]].to_numpy()
    print("q0.01 == q0.1:", bool(np.allclose(v[:, 0], v[:, 1])), "| q0.99 == q0.9:", bool(np.allclose(v[:, 4], v[:, 3])))


def c_dtype():
    y = demand.iloc[-300:]
    use_p1(True)
    # the adapter loads its own model with its own _load_model (model.to(dtype))
    f = FoundationModel(model_id=MODEL_ID, context_length=300, torch_dtype=torch.bfloat16)
    try:
        f.fit(series=y).predict(steps=STEPS)
        print("adapter torch_dtype=bfloat16 (model.to(dtype)): OK")
    except Exception as exc:
        print("adapter torch_dtype=bfloat16 (model.to(dtype)): ERROR", type(exc).__name__, str(exc)[:150])
    m = T0Forecaster.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cpu").eval()
    print("from_pretrained(dtype=bf16) parameter dtype:", next(m.parameters()).dtype)
    ctx = y.to_numpy(np.float32)[None]
    if NEW_API:
        r = m.predict(ctx, STEPS, quantile_levels=[0.5])
    else:
        r = m.predict(context=ctx, horizon=STEPS, quantiles=[0.5])
    print("from_pretrained(dtype=bf16) predict OK, output dtype", r.quantiles.dtype)


check("A. current adapter call unchanged", c_current_call)
if NEW_API:
    check("B. P1 call vs direct backend", c_p1_vs_direct)
check("C. left NaN padding", c_padding)
check("D. heterogeneous covariates", c_hetero)
check("E. extreme quantile levels", c_quantiles)
check("F. torch_dtype", c_dtype)
