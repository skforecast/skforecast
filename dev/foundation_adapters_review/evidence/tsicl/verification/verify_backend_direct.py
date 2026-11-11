# Independent re-check of the TS-ICL findings, calling `tsicl.TSICL.forecast`
# directly (backend level) and through FoundationModel (adapter level).
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/verification/verify_backend_direct.py
import platform
import traceback
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import TSICLAdapter

warnings.simplefilter("ignore")
torch.manual_seed(0)

print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} pandas={pd.__version__} "
    f"skforecast={version('skforecast')} machine={platform.machine()} "
    f"cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}"
)

rng = np.random.default_rng(123)
N = 450
H = 1400
idx = pd.date_range("2025-03-01", periods=N + H, freq="h")
t = np.arange(len(idx))
x1 = (np.sin(2 * np.pi * t / 29.0) + rng.normal(0, 0.05, len(idx))).astype(np.float32)
x2 = np.cos(2 * np.pi * t / 24.0).astype(np.float32)
ya = (100 + 8 * x1 + rng.normal(0, 0.4, len(idx))).astype(np.float32)
yb = (5 + 2 * x1 + rng.normal(0, 0.2, len(idx))).astype(np.float32)

CTX = 400
STEPS = 24

fm = FoundationModel("taharnbl/TS-ICL", context_length=CTX, device="cpu")
fm.adapter._load_model()
tsicl = fm.adapter._model


def fc(inputs, steps=STEPS, q=(0.5,), ctx=CTX):
    _, out = tsicl.forecast(
        inputs=inputs, prediction_length=steps, quantile_levels=list(q),
        context_length=ctx, device=torch.device("cpu"), denormalize=True,
        squeeze_output=False,
    )
    return out


def arr(out, i):
    return np.asarray(out[i][0], dtype=float)


def rep(name, a, b):
    d = float(np.max(np.abs(a - b)))
    print(f"  {name}: max_abs_diff={d:.3e}")


def inp(y, n_ctx, past=None, fut=None, start=0):
    d = {"target": y[N - n_ctx:N]}
    if past is not None:
        d["past_covariates"] = {k: v[N - n_ctx:N] for k, v in past.items()}
    if fut is not None:
        d["future_covariates"] = {k: v[N:N + STEPS] for k, v in fut.items()}
    return d


print("\n== V0 output type of the backend with squeeze_output=False")
o_eq = fc([inp(ya, CTX), inp(yb, CTX)])
o_ne = fc([inp(ya, CTX), inp(yb, 200)])
print(f"  equal lengths: {type(o_eq).__name__} shape={tuple(o_eq.shape) if hasattr(o_eq, 'shape') else [tuple(x.shape) for x in o_eq]}")
print(f"  different lengths: {type(o_ne).__name__} shapes={[tuple(x.shape) for x in o_ne]}")

print("\n== V1 adapter (FoundationModel) vs direct backend, different lengths, exog, quantiles")
QS = [0.05, 0.5, 0.95]
sa = pd.Series(ya[:N], index=idx[:N], name="a")
sb = pd.Series(yb[N - 250:N], index=idx[N - 250:N], name="b")
ex = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
p = fm.predict(steps=STEPS, context={"a": sa, "b": sb}, quantiles=QS)
ref = fc([inp(ya, CTX), inp(yb, 250)], q=QS)
for i, name in enumerate(["a", "b"]):
    v = p.loc[p["level"] == name, [f"q_{q}" for q in QS]].to_numpy(dtype=float)
    rep(f"{name} no exog q={QS}", v, arr(ref, i))
p = fm.predict(
    steps=STEPS, context={"a": sa},
    context_exog={"a": ex.iloc[:N]}, exog={"a": ex.iloc[N:N + STEPS]}, quantiles=QS,
)
ref = fc([inp(ya, CTX, past={"x1": x1, "x2": x2}, fut={"x1": x1, "x2": x2})], q=QS)
v = p.loc[p["level"] == "a", [f"q_{q}" for q in QS]].to_numpy(dtype=float)
rep("a with past+future exog", v, arr(ref, 0))

print("\n== V2 backend: past-only covariates vs none (single series)")
none = arr(fc([inp(ya, CTX)]), 0)
past = arr(fc([inp(ya, CTX, past={"x1": x1})]), 0)
full = arr(fc([inp(ya, CTX, past={"x1": x1}, fut={"x1": x1})]), 0)
rep("past-only vs none", past, none)
rep("past+future vs none", full, none)

print("\n== V3 backend: mixed (x1 past-only, x2 past+future) vs x2 only")
mixed = arr(fc([inp(ya, CTX, past={"x1": x1, "x2": x2}, fut={"x2": x2})]), 0)
only2 = arr(fc([inp(ya, CTX, past={"x2": x2}, fut={"x2": x2})]), 0)
rep("mixed vs x2 only", mixed, only2)

print("\n== V4 backend: two series, different lengths, future exog")
joint = arr(fc([
    inp(ya, CTX, past={"x1": x1}, fut={"x1": x1}),
    inp(yb, 200, past={"x1": x1}, fut={"x1": x1}),
]), 0)
rep("a joint (b shorter) vs a alone with exog", joint, full)
rep("a joint (b shorter) vs a without exog", joint, none)
joint_eq = arr(fc([
    inp(ya, CTX, past={"x1": x1}, fut={"x1": x1}),
    inp(yb, CTX, past={"x1": x1}, fut={"x1": x1}),
]), 0)
rep("a joint (same length) vs a alone with exog", joint_eq, full)

print("\n== V5 backend: NaN in the past exog of series b, same length")
x1_nan = x1.copy()
x1_nan[N - 30:N - 20] = np.nan
joint_nan = arr(fc([
    inp(ya, CTX, past={"x1": x1}, fut={"x1": x1}),
    {"target": yb[N - CTX:N], "past_covariates": {"x1": x1_nan[N - CTX:N]},
     "future_covariates": {"x1": x1[N:N + STEPS]}},
]), 0)
rep("a joint (b exog NaN) vs a joint (no NaN)", joint_nan, joint_eq)
rep("a joint (no NaN) vs a alone", joint_eq, full)

print("\n== V6 quantile grid: SUPPORTED_QUANTILES rejected by the backend")
rejected = []
for q in TSICLAdapter.SUPPORTED_QUANTILES:
    try:
        tsicl._get_quantile_indices([q])
    except ValueError:
        rejected.append(q)
print(f"  rejected={rejected}")
print(f"  100*0.07={100 * 0.07!r} in linspace grid: {100 * 0.07 in np.linspace(1, 99, 99).tolist()}")
try:
    fm.predict(steps=STEPS, context={"a": sa}, quantiles=[0.07])
    print("  FoundationModel q=0.07: ok")
except Exception as exc:
    print(f"  FoundationModel q=0.07: {type(exc).__name__}: {str(exc)[:120]}")

print("\n== V7 rollout beyond max_target_length")
print(f"  max_target_length={tsicl.max_target_length} max_context_length={tsicl.max_context_length}")
for steps, with_exog in [(700, False), (1344, False), (700, True)]:
    try:
        if with_exog:
            d = {"target": ya[N - CTX:N], "past_covariates": {"x1": x1[N - CTX:N]},
                 "future_covariates": {"x1": x1[N:N + steps]}}
        else:
            d = {"target": ya[N - CTX:N]}
        o = fc([d], steps=steps)
        print(f"  steps={steps} exog={with_exog}: shape={tuple(o[0].shape)} finite={bool(np.isfinite(np.asarray(o[0])).all())}")
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"  steps={steps} exog={with_exog}: {type(exc).__name__}: {str(exc).splitlines()[0][:150]} at {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")

print("\n== V8 device and context_length clipping")
print(f"  _get_device(mps) -> {tsicl._get_device(torch.device('mps'))}")
fc([inp(ya, CTX)], ctx=10000)
print(f"  context_length=10000 -> this_context_length={tsicl.this_context_length}")
