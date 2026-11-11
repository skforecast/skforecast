"""
Recompute the MAE stored in the investigator's backtest_exog.csv and check that
the first fold (24 steps after 2014-11-30 23:00) is reproduced by a single
FoundationModel.predict call and by a direct predict_batch call with the same
inputs. This is a correctness check of the saved predictions, not a benchmark.

Run from the repository root:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/verification/check_guide_fold0.py
"""

import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.datasets import fetch_dataset
from skforecast.foundation import FoundationModel

warnings.simplefilter("ignore")
for pkg in ("skforecast", "timesfm", "torch", "numpy", "pandas"):
    print(pkg, version(pkg))
print("python", platform.python_version())

csv = pd.read_csv(
    "dev/foundation_adapters_review/evidence/timesfm3/backtest_exog.csv",
    index_col=0, parse_dates=True,
)
print("MAE from csv", float(np.mean(np.abs(csv["y_true"] - csv["pred"]))))

# Same data preparation as dev/foundation_adapters_review/evidence/common/backtest_guide.py
data = fetch_dataset(name="vic_electricity", verbose=False)
data = data.drop(columns="Date")
data = data.resample(rule="h", closed="left", label="right").agg(
    {"Demand": "mean", "Temperature": "mean", "Holiday": "mean"}
)
data = data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00"].copy()
end_train = "2014-11-30 23:59:00"
train = data.loc[:end_train]
fut = data.loc[end_train:].iloc[:24]
exog_cols = ["Temperature", "Holiday"]

fm = FoundationModel(model_id="google/timesfm-3.0-pytorch", context_length=500, device="cpu")
fm.fit(series=train["Demand"], exog=train[exog_cols])
pred = fm.predict(steps=24, exog=fut[exog_cols])
fold0 = csv[csv["fold"] == 0]["pred"].to_numpy()
print("fold 0 index matches", bool((csv[csv["fold"] == 0].index == fut.index).all()))
print("fold 0 FoundationModel vs csv maxdiff", float(np.max(np.abs(pred["pred"].to_numpy() - fold0))))

model = fm.adapter._model
ctx = train["Demand"].iloc[-500:].to_numpy()
pf = np.stack([
    np.concatenate([train[c].iloc[-500:].to_numpy(), fut[c].to_numpy()]).astype(np.float32)
    for c in exog_cols
])
direct = list(model.predict_batch(
    contexts=[ctx], horizon=24, past_future_covariates=[pf], padding_mode="edge"
))[0].forecast
print("fold 0 direct predict_batch(edge) vs FoundationModel maxdiff",
      float(np.max(np.abs(direct - pred["pred"].to_numpy()))))

# Run-to-run variation of the same direct call (float32 CPU kernels).
direct2 = list(model.predict_batch(
    contexts=[ctx], horizon=24, past_future_covariates=[pf], padding_mode="edge"
))[0].forecast
print("direct vs direct (second call) maxdiff", float(np.max(np.abs(direct - direct2))))
fm_ctx_dtype = fm.adapter.context_["Demand"].dtype if isinstance(fm.adapter.context_, dict) else None
print("stored context dtype", fm_ctx_dtype)

# Capture the arguments the adapter passes to predict_batch and compare inputs.
captured = {}
orig = model.predict_batch


def spy(**kwargs):
    captured.update(kwargs)
    return orig(**kwargs)


model.predict_batch = spy
pred2 = fm.predict(steps=24, exog=fut[exog_cols])
model.predict_batch = orig
print("adapter kwargs", sorted(k for k in captured))
print("context equal", bool(np.array_equal(captured["contexts"][0], ctx)), captured["contexts"][0].dtype)
print("past_only", None if captured["past_only_covariates"] is None else [a is None for a in captured["past_only_covariates"]])
cap_pf = captured["past_future_covariates"][0]
print("pf shape", cap_pf.shape, pf.shape, "maxdiff", float(np.max(np.abs(cap_pf - pf))))
replay = list(orig(**captured))[0].forecast
print("replay of captured kwargs vs FoundationModel maxdiff", float(np.max(np.abs(replay - pred2["pred"].to_numpy()))))
print("pf equals reversed column order", bool(np.array_equal(cap_pf, pf[::-1])))
