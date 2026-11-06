"""
Compare the output of TabICLAdapter (through FoundationModel) with a direct
call to `tabicl.forecast.TabICLForecaster.predict_df` on the same input.

Cases:
  A. Single hourly series with a numeric exogenous variable, quantiles.
  B. Three series of different lengths, one of them with NaN in the middle,
     point forecast (`target` column of the backend).
  C. Two series with different covariate columns (FoundationModel groups
     them and calls the adapter once per group).

Run from the repository root with the review environment:
    ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/compare_direct.py
"""

import importlib.metadata as md
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tabicl.forecast import TabICLForecaster

from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore", category=UserWarning)

print(
    "python", sys.version.split()[0],
    "| tabicl", md.version("tabicl"),
    "| torch", torch.__version__,
    "| numpy", np.__version__,
    "| pandas", pd.__version__,
    "| skforecast", md.version("skforecast"),
    "|", platform.machine(), platform.system(),
    "| cuda", torch.cuda.is_available(),
)

CONTEXT_LENGTH = 128
STEPS = 12
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
hour = idx.hour.to_numpy()
exog_all = pd.DataFrame({"exog_1": np.cos(2 * np.pi * hour / 24)}, index=idx)
y_all = pd.Series(
    50 + 5 * np.sin(2 * np.pi * hour / 24) + 2 * exog_all["exog_1"].to_numpy()
    + rng.normal(0, 0.5, len(idx)),
    index=idx,
    name="y",
)
y = y_all.iloc[:300]
exog = exog_all.iloc[:300]
exog_future = exog_all.iloc[300:]


def long_df(name, s, ex=None):
    df = pd.DataFrame({"item_id": name, "timestamp": s.index, "target": s.to_numpy(float)})
    if ex is not None:
        df = pd.concat([df, ex.reset_index(drop=True)], axis=1)
    return df


def future_long_df(name, f_index, ex=None):
    df = pd.DataFrame({"item_id": name, "timestamp": f_index})
    if ex is not None:
        df = pd.concat([df, ex.reset_index(drop=True)], axis=1)
    return df


def report(case, a, b):
    diff = float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    print(f"[{case}] shape adapter={np.shape(a)} direct={np.shape(b)} max_abs_diff={diff:.3e} "
          f"allclose={np.allclose(a, b, rtol=1e-6, atol=1e-6)}")


# Case A ---------------------------------------------------------------------
quantiles = [0.1, 0.5, 0.9]
model = FoundationModel("soda-inria/tabicl", context_length=CONTEXT_LENGTH)
model.fit(series=y, exog=exog)
pred_a = model.predict(steps=STEPS, exog=exog_future, quantiles=quantiles)
direct = TabICLForecaster(max_context_length=CONTEXT_LENGTH)
ctx = long_df("y", y.iloc[-CONTEXT_LENGTH:], exog.iloc[-CONTEXT_LENGTH:])
fut = future_long_df("y", exog_future.index, exog_future)
res = direct.predict_df(context_df=ctx, future_df=fut, quantiles=quantiles)
print("[A] backend output columns:", list(res.columns), "| index names:", res.index.names,
      "| type:", type(res).__name__)
report("A quantiles", pred_a[[f"q_{q}" for q in quantiles]].to_numpy(), res.loc["y"][quantiles].to_numpy())
pred_a_point = model.predict(steps=STEPS, exog=exog_future)
res_point = direct.predict_df(context_df=ctx, future_df=fut)
report("A point (mean)", pred_a_point["pred"].to_numpy(), res_point.loc["y"]["target"].to_numpy())

# Case B ---------------------------------------------------------------------
s1 = y.copy().rename("s1")
s2 = (y.iloc[-90:] * 0.5 + 3).rename("s2")
s3 = (y.iloc[-200:] - 10).rename("s3")
s3.iloc[50:60] = np.nan
series = {"s1": s1, "s2": s2, "s3": s3}
model_b = FoundationModel("soda-inria/tabicl", context_length=CONTEXT_LENGTH)
model_b.fit(series=series)
pred_b = model_b.predict(steps=STEPS)
ctx_b = pd.concat([long_df(k, v.iloc[-CONTEXT_LENGTH:]) for k, v in series.items()], ignore_index=True)
f_index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=STEPS, freq="h")
fut_b = pd.concat([future_long_df(k, f_index) for k in series], ignore_index=True)
res_b = TabICLForecaster(max_context_length=CONTEXT_LENGTH).predict_df(context_df=ctx_b, future_df=fut_b)
for k in series:
    report(f"B point {k}", pred_b.loc[pred_b["level"] == k, "pred"].to_numpy(), res_b.loc[k]["target"].to_numpy())

# Case C ---------------------------------------------------------------------
exog_c = {"s1": exog, "s2": exog.rename(columns={"exog_1": "exog_2"}).iloc[-90:]}
exog_c_future = {"s1": exog_future, "s2": exog_future.rename(columns={"exog_1": "exog_2"})}
model_c = FoundationModel("soda-inria/tabicl", context_length=CONTEXT_LENGTH)
model_c.fit(series={"s1": s1, "s2": s2}, exog=exog_c)
pred_c = model_c.predict(steps=STEPS, exog=exog_c_future, quantiles=[0.2, 0.8])
for k in ["s1", "s2"]:
    ctx_k = long_df(k, series[k].iloc[-CONTEXT_LENGTH:], exog_c[k].iloc[-CONTEXT_LENGTH:])
    fut_k = future_long_df(k, f_index, exog_c_future[k])
    res_k = TabICLForecaster(max_context_length=CONTEXT_LENGTH).predict_df(
        context_df=ctx_k, future_df=fut_k, quantiles=[0.2, 0.8]
    )
    report(f"C quantiles {k}", pred_c.loc[pred_c["level"] == k, ["q_0.2", "q_0.8"]].to_numpy(),
           res_k.loc[k][[0.2, 0.8]].to_numpy())
