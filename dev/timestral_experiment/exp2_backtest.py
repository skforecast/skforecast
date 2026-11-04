"""
Backtesting of the Timestral prototype against Chronos-2, with the same setup
as docs/user_guides/foundation-forecasting-models.ipynb: vic_electricity
hourly Demand, exog Temperature and Holiday, test December 2014, 24 steps per
fold, no refit.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import importlib.metadata as md

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from timestral_proto import register
from skforecast.datasets import fetch_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

register()

data = fetch_dataset(name="vic_electricity", verbose=False)
data = data.drop(columns="Date")
data = (
    data.resample(rule="h", closed="left", label="right")
    .agg({"Demand": "mean", "Temperature": "mean", "Holiday": "mean"})
)
data = data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00", :].copy()
end_train = "2014-11-30 23:59:00"

cv = TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_train]), refit=False)

configs = [
    # (model_id, context_length, use_exog)
    ("autogluon/chronos-2-small", 500, True),
    ("amazon/chronos-2", 500, True),
    ("mistralai/Timestral-5M", 500, True),
    ("mistralai/Timestral-28M", 500, True),
    ("mistralai/Timestral-123M", 500, True),
    ("mistralai/Timestral-209M", 500, True),
    ("mistralai/Timestral-28M", 500, False),
    ("mistralai/Timestral-209M", 500, False),
    ("autogluon/chronos-2-small", 4096, True),
    ("amazon/chronos-2", 4096, True),
    ("mistralai/Timestral-28M", 4096, True),
    ("mistralai/Timestral-209M", 4096, True),
    ("amazon/chronos-2", 8192, True),
    ("mistralai/Timestral-28M", 8192, True),
    ("mistralai/Timestral-209M", 8192, True),
]
import os
RESULTS = "results_exp2.csv"
done = set()
if os.path.exists(RESULTS):
    prev = pd.read_csv(RESULTS)
    done = set(zip(prev.model_id, prev.context_length, prev.exog))
configs = [c for c in configs if c not in done]

# end_train is 23:59, so loc[end_train:] starts at the first test hour (720 points).
y_true = data.loc[end_train:, "Demand"]
rows = []
for model_id, context_length, use_exog in configs:
    kwargs = {"device_map": "cpu"} if "chronos" in model_id else {"device": "cpu"}
    estimator = FoundationModel(model_id=model_id, context_length=context_length, **kwargs)
    forecaster = ForecasterFoundation(estimator=estimator)
    # Warm-up (weights download and load) outside the timing
    forecaster.fit(series=data.loc[:end_train, "Demand"])
    forecaster.predict(steps=24)
    t0 = time.perf_counter()
    metric, preds = backtesting_foundation(
        forecaster        = forecaster,
        series            = data["Demand"],
        exog              = data[["Temperature", "Holiday"]] if use_exog else None,
        cv                = cv,
        metric            = "mean_absolute_error",
        quantiles         = [0.1, 0.5, 0.9],
        show_progress     = False,
        suppress_warnings = True,
    )
    elapsed = time.perf_counter() - t0
    p = preds.loc[y_true.index]
    mae_median = np.mean(np.abs(p["q_0.5"] - y_true))
    coverage = np.mean((y_true >= p["q_0.1"]) & (y_true <= p["q_0.9"]))
    rows.append({
        "model_id": model_id, "context_length": context_length, "exog": use_exog,
        "mae": round(mae_median, 2), "coverage_80": round(coverage, 3),
        "seconds": round(elapsed, 1),
    })
    print(rows[-1], flush=True)
    pd.DataFrame([rows[-1]]).to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)

print(pd.read_csv(RESULTS).to_string(index=False))
print(
    "Python", sys.version.split()[0], "| torch", torch.__version__,
    "| numpy", np.__version__, "| pandas", pd.__version__,
    "| timestral", md.version("timestral"),
    "| chronos-forecasting", md.version("chronos-forecasting"),
    "| skforecast", md.version("skforecast"),
    "| threads", torch.get_num_threads(),
)
