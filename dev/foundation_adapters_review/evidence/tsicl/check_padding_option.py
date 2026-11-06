# Evaluate, with the real TS-ICL weights, whether declaring
# `requires_equal_length_context=True` (FoundationModel left-pads the context
# and the historical exog of a call with NaN) avoids the loss of the future
# covariates when the series of a call have different lengths (see E3 of
# check_covariates_rollout.py), and whether the padding changes the forecast of
# a series compared with predicting it alone. The class attribute is
# monkeypatched in this process only.
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/check_padding_option.py
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import TSICLAdapter

warnings.simplefilter("ignore")

print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} pandas={pd.__version__} "
    f"skforecast={version('skforecast')} machine={platform.machine()}"
)

CONTEXT_LENGTH = 512
STEPS = 24
rng = np.random.default_rng(0)
n = 600
index = pd.date_range("2024-01-01", periods=n + STEPS, freq="h")
t = np.arange(len(index))
exog_all = pd.DataFrame(
    {"drv": np.sin(2 * np.pi * t / 37.0) + rng.normal(0, 0.05, len(index))},
    index=index,
)
y = 30 + 6 * exog_all["drv"].to_numpy() + rng.normal(0, 0.3, len(index))
y2 = 10 + 4 * exog_all["drv"].to_numpy() + rng.normal(0, 0.3, len(index))
s1 = pd.Series(y[:n], index=index[:n])
s2 = pd.Series(y2[:n], index=index[:n]).iloc[-300:]
exog = exog_all.iloc[:n]
future = exog_all.iloc[n:]


def run(model, context, context_exog, exog_future):
    out = model.predict(steps=STEPS, context=context, context_exog=context_exog, exog=exog_future)
    return {lv: out.loc[out["level"] == lv, "pred"].to_numpy(dtype=float) for lv in context}


def diff(a, b):
    return f"{np.max(np.abs(a - b)):.3e}"


for padded in [False, True]:
    TSICLAdapter.requires_equal_length_context = padded
    model = FoundationModel("taharnbl/TS-ICL", context_length=CONTEXT_LENGTH, device="cpu")
    ex = {"s1": exog, "s2": exog.iloc[-300:]}
    fut = {"s1": future, "s2": future}
    joint = run(model, {"s1": s1, "s2": s2}, ex, fut)
    a1 = run(model, {"s1": s1}, {"s1": exog}, {"s1": future})["s1"]
    a2 = run(model, {"s2": s2}, {"s2": ex["s2"]}, {"s2": future})["s2"]
    joint_noexog = run(model, {"s1": s1, "s2": s2}, None, None)
    n1 = run(model, {"s1": s1}, None, None)["s1"]
    n2 = run(model, {"s2": s2}, None, None)["s2"]
    print(f"\nrequires_equal_length_context={padded}")
    print(f"  with exog: s1 joint vs alone={diff(joint['s1'], a1)}  s2 joint vs alone={diff(joint['s2'], a2)}")
    print(f"  no exog  : s1 joint vs alone={diff(joint_noexog['s1'], n1)}  s2 joint vs alone={diff(joint_noexog['s2'], n2)}")
