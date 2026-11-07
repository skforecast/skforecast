# Compare the output of TSICLAdapter (through FoundationModel) with a direct
# call to `tsicl.TSICL.forecast` on the same input.
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/compare_direct.py
import platform
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

CONTEXT_LENGTH = 256
STEPS = 24
QUANTILES = [0.1, 0.5, 0.9]

rng = np.random.default_rng(42)
n = 400
index = pd.date_range("2024-01-01", periods=n + STEPS, freq="h")
hour = index.hour.to_numpy()
exog_all = pd.DataFrame(
    {
        "temp": np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(index)),
        "flag": (hour >= 8) & (hour <= 18),
    },
    index=index,
)
y1 = (
    50 + 5 * np.sin(2 * np.pi * hour / 24) + 3 * exog_all["temp"].to_numpy()
    + rng.normal(0, 0.5, len(index))
)
y2 = 20 + 3 * np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.3, len(index))
series = pd.DataFrame({"s1": y1[:n], "s2": y2[:n]}, index=index[:n])
series.iloc[100:110, 1] = np.nan  # NaN inside the context of s2
exog = exog_all.iloc[:n]
exog_future = exog_all.iloc[n:]

model = FoundationModel("taharnbl/TS-ICL", context_length=CONTEXT_LENGTH, device="cpu")
model.fit(series={"s1": series["s1"], "s2": series["s2"]}, exog={"s1": exog, "s2": exog})
backend = None


def direct(inputs, quantiles):
    global backend
    if backend is None:
        backend = model.adapter._model
    _, q = backend.forecast(
        inputs=inputs,
        prediction_length=STEPS,
        quantile_levels=quantiles,
        context_length=CONTEXT_LENGTH,
        device=torch.device("cpu"),
        denormalize=True,
        squeeze_output=False,
    )
    return [qi[0].numpy() for qi in q]


def ctx(name):
    return series[name].iloc[-CONTEXT_LENGTH:].to_numpy(dtype=np.float32)


def cov(df):
    return {
        "temp": df["temp"].to_numpy(dtype=np.float32),
        "flag": df["flag"].to_numpy(dtype=np.float32),
    }


cases = {}

# Case 1: two series, no exog, quantiles.
pred = model.predict(steps=STEPS, context={"s1": series["s1"], "s2": series["s2"]}, quantiles=QUANTILES)
ref = direct([{"target": ctx("s1")}, {"target": ctx("s2")}], QUANTILES)
cases["no_exog_quantiles"] = (pred, ref)

# Case 2: two series with past and future exog, point forecast (q=0.5).
pred = model.predict(steps=STEPS, exog={"s1": exog_future, "s2": exog_future})
ref = direct(
    [
        {"target": ctx(name), "past_covariates": cov(exog.iloc[-CONTEXT_LENGTH:]),
         "future_covariates": cov(exog_future)}
        for name in ["s1", "s2"]
    ],
    [0.5],
)
cases["exog_point"] = (pred, ref)

for case, (pred, ref) in cases.items():
    print(f"\n== {case}")
    print(pred.head(3).to_string())
    for i, name in enumerate(["s1", "s2"]):
        values = pred.loc[pred["level"] == name].drop(columns="level").to_numpy(dtype=float)
        if values.shape[1] > ref[i].shape[1]:
            # predict returns `pred` (the median) plus the quantile columns
            values = values[:, 1:]
        diff = np.max(np.abs(values - ref[i]))
        print(f"{name}: shape adapter={values.shape} direct={ref[i].shape} max_abs_diff={diff:.3e}")

# Determinism of the backend: same call twice.
a = direct([{"target": ctx("s1")}], QUANTILES)[0]
b = direct([{"target": ctx("s1")}], QUANTILES)[0]
print(f"\nrepeat call max_abs_diff={np.max(np.abs(a - b)):.3e}")
