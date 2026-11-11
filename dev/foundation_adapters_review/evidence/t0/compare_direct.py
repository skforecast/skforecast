"""
Numeric checks of T0Adapter against direct calls to the tfc-t0 backend, on
the same inputs (vic_electricity hourly, as in the user guide).

Runs on tfc-t0 0.3.2 (adapter unchanged) and on tfc-t0>=0.5 (adapter call
translated by the diagnostic shim of `t0_shim.py`). Prints one block per
check, with the versions it ran with.

1. adapter vs direct call, single series without and with exog, quantiles
   requested out of order.
2. batching with a longer series (left NaN padding by FoundationModel):
   forecast of the short series alone vs in the batch; on >=0.6 also the
   direct call with `mask=MaskType.PAD` and with `trim_leading_missing=True`.
3. heterogeneous covariates: forecast of a series without exog alone vs in
   a batch with a series that has exog (pooled NaN covariate row).
4. quantile levels outside the trained range of t0-alpha ([0.1, 0.9]).
5. `torch_dtype=torch.bfloat16` of the adapter (`model.to(dtype)`) vs the
   upstream route `from_pretrained(dtype=...)` (autocast, fp32 weights).

Usage (from the repository root):
    python dev/foundation_adapters_review/evidence/t0/compare_direct.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_guide import load_data  # noqa: E402
from t0_shim import backend_version, load_backend_model, make_adapter_model  # noqa: E402

from skforecast.foundation import FoundationModel  # noqa: E402

MODEL_ID = "theforecastingcompany/t0-alpha"
STEPS = 24
NEW_API = backend_version() >= (0, 4, 0)

print(
    "python", platform.python_version(), "| torch", torch.__version__,
    "| numpy", np.__version__, "| pandas", pd.__version__,
    "| tfc-t0", version("tfc-t0"), "| skforecast", version("skforecast"),
)
torch.manual_seed(0)

BACKEND = load_backend_model(MODEL_ID)
ADAPTER_MODEL = make_adapter_model(MODEL_ID)


def direct(context, quantiles, future_covariates=None, **extra):
    """Direct backend call, returns numpy [B, steps, Q]."""
    if NEW_API:
        out = BACKEND.predict(
            context, STEPS, quantile_levels=quantiles,
            future_covariates=future_covariates, **extra,
        )
    else:
        out = BACKEND.predict(
            context=context, horizon=STEPS, quantiles=quantiles,
            future_covariates=future_covariates, **extra,
        )
    return out.quantiles.detach().cpu().numpy()


def foundation(context_length=8192, model=None):
    fm = FoundationModel(model_id=MODEL_ID, context_length=context_length)
    fm.adapter._model = ADAPTER_MODEL if model is None else model
    return fm


def check(name, fn):
    print(f"\n=== {name}")
    try:
        fn()
    except Exception as exc:  # report and continue with the next check
        print("ERROR", type(exc).__name__, ":", str(exc)[:400])
        traceback.print_exc(limit=3, file=sys.stdout)


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))


data = load_data()
demand = data["Demand"].loc[:"2014-11-30 23:00:00"]
exog_hist = data[["Temperature", "Holiday"]].loc[:"2014-11-30 23:00:00"]
exog_future = data[["Temperature", "Holiday"]].loc["2014-12-01 00:00:00":].iloc[:STEPS]
CTX = 500


def check_single():
    quantiles = [0.9, 0.1, 0.5]
    y = demand.iloc[-CTX:]
    fm = foundation(CTX).fit(series=y)
    adapter = fm.predict(steps=STEPS, quantiles=quantiles)[[f"q_{q}" for q in quantiles]].to_numpy()
    ref = direct(y.to_numpy(np.float32)[None, :], sorted(quantiles))[0]
    ref = ref[:, [sorted(quantiles).index(q) for q in quantiles]]
    print("no exog: max abs diff adapter vs direct =", float(np.max(np.abs(adapter - ref))))

    x = exog_hist.iloc[-CTX:]
    fm = foundation(CTX).fit(series=y, exog=x)
    adapter = fm.predict(steps=STEPS, exog=exog_future, quantiles=quantiles)[[f"q_{q}" for q in quantiles]].to_numpy()
    cov = np.concatenate([x.to_numpy(np.float32), exog_future.to_numpy(np.float32)]).T[None, :, :]
    ref = direct(y.to_numpy(np.float32)[None, :], sorted(quantiles), future_covariates=cov)[0]
    ref = ref[:, [sorted(quantiles).index(q) for q in quantiles]]
    print("exog:    max abs diff adapter vs direct =", float(np.max(np.abs(adapter - ref))))


def check_padding():
    short = demand.iloc[-200:].rename("short")
    long = (data["Temperature"].loc[:"2014-11-30 23:00:00"].iloc[-CTX:] * 100).rename("long")
    alone = foundation().fit(series={"short": short}).predict(steps=STEPS)["pred"].to_numpy()
    batch = foundation().fit(series={"short": short, "long": long}).predict(steps=STEPS)
    in_batch = batch.loc[batch["level"] == "short", "pred"].to_numpy()
    print("short alone vs in batch with a 500-long series (NaN left padding, no mask):")
    print("  max rel diff of the median =", rel(in_batch, alone))
    if NEW_API:
        from t0 import MaskType

        ctx = np.full((2, CTX), np.nan, dtype=np.float32)
        ctx[0, -200:] = short.to_numpy(np.float32)
        ctx[1] = long.to_numpy(np.float32)
        mask = np.where(np.isnan(ctx), int(MaskType.MISSING), int(MaskType.VALID)).astype(np.int8)
        mask[0, :-200] = int(MaskType.PAD)
        ref_alone = direct(short.to_numpy(np.float32)[None, :], [0.5])[0, :, 0]
        print("  direct alone vs adapter alone, max abs diff =", float(np.max(np.abs(ref_alone - alone))))
        padded = direct(ctx, [0.5], mask=mask)[0, :, 0]
        print("  direct batch with mask=PAD vs alone, max rel diff =", rel(padded, ref_alone))
        trimmed = direct(ctx, [0.5], trim_leading_missing=True)[0, :, 0]
        print("  direct batch with trim_leading_missing=True vs alone, max rel diff =", rel(trimmed, ref_alone))
        nomask = direct(ctx, [0.5])[0, :, 0]
        print("  direct batch without mask vs alone, max rel diff =", rel(nomask, ref_alone))


def check_heterogeneous_covariates():
    a = demand.iloc[-CTX:].rename("a")
    b = (demand.iloc[-CTX:] * 0.8).rename("b")
    xb = exog_hist.iloc[-CTX:][["Temperature"]]
    alone = foundation(CTX).fit(series={"a": a}).predict(steps=STEPS)["pred"].to_numpy()
    fm = foundation(CTX).fit(series={"a": a, "b": b}, exog={"a": None, "b": xb})
    out = fm.predict(steps=STEPS, exog={"a": None, "b": exog_future[["Temperature"]]})
    in_batch = out.loc[out["level"] == "a", "pred"].to_numpy()
    print("series without exog alone vs in a batch with a series that has exog:")
    print("  max rel diff of the median =", rel(in_batch, alone))


def check_extreme_quantiles():
    y = demand.iloc[-CTX:]
    q = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    out = foundation(CTX).fit(series=y).predict(steps=STEPS, quantiles=q)
    vals = out[[f"q_{x}" for x in q]].to_numpy()
    print("t0-alpha trained levels: 0.1 .. 0.9; requested", q)
    print("  q0.01 == q0.1 at every step:", bool(np.allclose(vals[:, 0], vals[:, 2])))
    print("  q0.99 == q0.9 at every step:", bool(np.allclose(vals[:, 6], vals[:, 4])))
    print("  mean width 1%-99% / mean width 10%-90% =", float(np.mean(vals[:, 6] - vals[:, 0]) / np.mean(vals[:, 4] - vals[:, 2])))


def check_dtype():
    y = demand.iloc[-CTX:]
    fp32 = foundation(CTX).fit(series=y).predict(steps=STEPS)["pred"].to_numpy()
    try:
        cast = make_adapter_model(MODEL_ID)
        cast = cast.to(torch.bfloat16) if not NEW_API else cast
        if NEW_API:
            cast.model = cast.model.to(torch.bfloat16)
        bf = foundation(CTX, model=cast).fit(series=y).predict(steps=STEPS)["pred"].to_numpy()
        print("adapter route model.to(bfloat16): max rel diff vs fp32 =", rel(bf, fp32))
    except Exception as exc:
        print("adapter route model.to(bfloat16): ERROR", type(exc).__name__, ":", str(exc)[:300])
    auto = make_adapter_model(MODEL_ID, dtype=torch.bfloat16)
    bf = foundation(CTX, model=auto).fit(series=y).predict(steps=STEPS)["pred"].to_numpy()
    print("upstream route from_pretrained(dtype=bfloat16), autocast: max rel diff vs fp32 =", rel(bf, fp32))


check("1. adapter vs direct call", check_single)
check("2. left NaN padding in a batch", check_padding)
check("3. heterogeneous covariates in a batch", check_heterogeneous_covariates)
check("4. quantile levels outside the trained range", check_extreme_quantiles)
check("5. bfloat16", check_dtype)
