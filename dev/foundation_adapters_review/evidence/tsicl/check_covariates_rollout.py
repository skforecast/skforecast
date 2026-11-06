# Check, through FoundationModel and with the real TS-ICL weights, how the
# backend handles the inputs skforecast sends it:
#
#   E1  past-only covariates (only `context_exog`) vs no exog
#   E2  mixed covariates (one known in the future, one past-only) vs only the
#       future-known one
#   E3  two series with future exog and different context lengths in the same
#       call: is the forecast of a series the same as when it is predicted alone?
#   E4  two series of the same length, one with NaN in its exog: does it change
#       the forecast of the other one?
#   E5  horizons longer than `max_target_len` (672): rollout, with and without
#       future exog
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/check_covariates_rollout.py
import platform
import traceback
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")

print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} pandas={pd.__version__} "
    f"skforecast={version('skforecast')} machine={platform.machine()} "
    f"cuda={torch.cuda.is_available()}"
)

CONTEXT_LENGTH = 512
STEPS = 24

rng = np.random.default_rng(0)
n = 600
h_max = 1400
index = pd.date_range("2024-01-01", periods=n + h_max, freq="h")
t = np.arange(len(index))
exog_all = pd.DataFrame(
    {
        # covariate with a strong effect on the target, not periodic with the
        # daily cycle so that the target history alone does not reveal it
        "drv": np.sin(2 * np.pi * t / 37.0) + rng.normal(0, 0.05, len(index)),
        "aux": np.cos(2 * np.pi * t / 24.0),
    },
    index=index,
)
y = 30 + 6 * exog_all["drv"].to_numpy() + rng.normal(0, 0.3, len(index))
y2 = 10 + 4 * exog_all["drv"].to_numpy() + rng.normal(0, 0.3, len(index))
s1 = pd.Series(y[:n], index=index[:n], name="s1")
s2 = pd.Series(y2[:n], index=index[:n], name="s2")
exog = exog_all.iloc[:n]
future = exog_all.iloc[n:n + STEPS]
truth = y[n:n + STEPS]

model = FoundationModel("taharnbl/TS-ICL", context_length=CONTEXT_LENGTH, device="cpu")


def pred(context, steps=STEPS, context_exog=None, exog=None, level="s1"):
    out = model.predict(steps=steps, context=context, context_exog=context_exog, exog=exog)
    return out.loc[out["level"] == level, "pred"].to_numpy(dtype=float)


def report(name, a, b):
    d = np.max(np.abs(a - b))
    print(f"{name}: max_abs_diff={d:.3e} identical={d == 0.0}")


print("\n== E1 past-only covariates")
p_none = pred({"s1": s1})
p_past = pred({"s1": s1}, context_exog={"s1": exog[["drv"]]})
p_full = pred({"s1": s1}, context_exog={"s1": exog[["drv"]]}, exog={"s1": future[["drv"]]})
report("past-only vs no exog", p_past, p_none)
report("past+future vs no exog", p_full, p_none)
print(f"MAE no exog={np.mean(np.abs(p_none - truth)):.4f} past-only={np.mean(np.abs(p_past - truth)):.4f} past+future={np.mean(np.abs(p_full - truth)):.4f}")

print("\n== E2 mixed covariates: 'drv' past-only, 'aux' known in the future")
p_mixed = pred({"s1": s1}, context_exog={"s1": exog[["drv", "aux"]]}, exog={"s1": future[["aux"]]})
p_aux = pred({"s1": s1}, context_exog={"s1": exog[["aux"]]}, exog={"s1": future[["aux"]]})
report("mixed (drv past-only + aux future) vs aux only", p_mixed, p_aux)

print("\n== E3 different context lengths in the same call, future exog")
s2_short = s2.iloc[-300:]
alone = pred({"s1": s1}, context_exog={"s1": exog[["drv"]]}, exog={"s1": future[["drv"]]})
joint_same_len = pred(
    {"s1": s1, "s2": s2},
    context_exog={"s1": exog[["drv"]], "s2": exog[["drv"]]},
    exog={"s1": future[["drv"]], "s2": future[["drv"]]},
)
joint_diff_len = pred(
    {"s1": s1, "s2": s2_short},
    context_exog={"s1": exog[["drv"]], "s2": exog[["drv"]].iloc[-300:]},
    exog={"s1": future[["drv"]], "s2": future[["drv"]]},
)
report("s1 joint (same length) vs alone", joint_same_len, alone)
report("s1 joint (s2 shorter) vs alone", joint_diff_len, alone)
report("s1 joint (s2 shorter) vs no exog", joint_diff_len, p_none)

print("\n== E4 NaN in the exog of another series of the same call")
exog_nan = exog[["drv"]].copy()
exog_nan.iloc[-50:-40] = np.nan
joint_nan = pred(
    {"s1": s1, "s2": s2},
    context_exog={"s1": exog[["drv"]], "s2": exog_nan},
    exog={"s1": future[["drv"]], "s2": future[["drv"]]},
)
report("s1 joint (s2 exog with NaN) vs alone", joint_nan, alone)

print("\n== E5 rollout (max_target_len=672)")
for steps, with_exog in [(700, False), (1344, False), (700, True), (1344, True)]:
    try:
        if with_exog:
            out = model.predict(
                steps=steps, context={"s1": s1}, context_exog={"s1": exog[["drv"]]},
                exog={"s1": exog_all[["drv"]].iloc[n:n + steps]},
            )
        else:
            out = model.predict(steps=steps, context={"s1": s1})
        values = out["pred"].to_numpy(dtype=float)
        print(f"steps={steps} exog={with_exog}: rows={len(out)} finite={np.isfinite(values).all()}")
    except Exception as exc:
        print(f"steps={steps} exog={with_exog}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        tb = traceback.extract_tb(exc.__traceback__)
        print("    raised at", tb[-1].filename.split("site-packages/")[-1], tb[-1].lineno)
