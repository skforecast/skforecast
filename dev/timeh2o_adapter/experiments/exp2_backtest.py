"""
Experiment 2: backtesting with the setup of the comparison table of
docs/user_guides/foundation-forecasting-models.ipynb: vic_electricity hourly
Demand, exog Temperature and Holiday, test 2014-12-01 00:00 to 2014-12-30
23:00 (30 folds of 24 steps, 720 predictions), refit=False.

TimeH2O goes through the prototype adapter registered at runtime. Chronos-2
and Timestral-209M are measured again in this environment as a reference
(Timestral through the prototype of dev/timestral_adapter/experiments).

Columns of the CSV:
- exog: "num" (Temperature and Holiday as floats, as in the user guide),
  "bool" (Holiday cast to bool, which h2ofm encodes as a categorical) or
  "none".
- patch_ensemble: value of the h2ofm option (TimeH2O only).
- mae: MAE of the median over the 720 test hours.
- cov80, cov90, cov98: empirical coverage of the [0.1, 0.9], [0.05, 0.95]
  and [0.01, 0.99] intervals.
- width80: mean width of the [0.1, 0.9] interval.
- seconds: time of the backtesting, without loading the weights.

Rows already in results_exp2.csv are skipped: delete the CSV to start over.

Usage: HF_HUB_OFFLINE=1 python -W ignore exp2_backtest.py > exp2.log 2>&1

The Timestral rows import the prototype of dev/timestral_adapter/experiments,
written before PR #1450: it does not declare `requires_equal_length_context`,
so on `0.27.x` from fc7efd0 this script fails at import time. See the exp2
row of section 12 of the report.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "../../timestral_adapter/experiments")
from common import versions  # noqa: E402
from timeh2o_proto import register as register_timeh2o  # noqa: E402
from timestral_proto import register as register_timestral  # noqa: E402
from skforecast.datasets import fetch_dataset  # noqa: E402
from skforecast.foundation import FoundationModel, ForecasterFoundation  # noqa: E402
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation  # noqa: E402

register_timeh2o()
register_timestral()
print(versions(), flush=True)

data = fetch_dataset(name="vic_electricity", verbose=False)
data = data.drop(columns="Date")
data = (
    data.resample(rule="h", closed="left", label="right")
    .agg({"Demand": "mean", "Temperature": "mean", "Holiday": "mean"})
)
data = data.loc["2012-01-01 00:00:00":"2014-12-30 23:00:00", :].copy()
end_train = "2014-11-30 23:59:00"
cv = TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_train]), refit=False)

QUANTILES = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
V1, TINY = "h2oai/timeh2o-v1", "h2oai/timeh2o-v1-tiny"
configs = [
    # (model_id, context_length, exog, patch_ensemble)
    ("autogluon/chronos-2-small", 500, "num", None),
    ("amazon/chronos-2", 4096, "num", None),
    ("mistralai/Timestral-209M", 4096, "num", None),
    (V1, 500, "num", True),
    (V1, 2048, "num", True),
    (V1, 4096, "num", True),
    (V1, 8192, "num", True),
    (V1, 16384, "num", True),
    (V1, 500, "num", False),
    (V1, 4096, "num", False),
    (V1, 8192, "num", False),
    (V1, 500, "none", True),
    (V1, 4096, "none", True),
    (V1, 4096, "bool", True),
    (TINY, 500, "none", None),
    (TINY, 4096, "none", None),
    (TINY, 4096, "num", None),
    # Added after the first results: the longest context of the model
    (V1, 16384, "none", True),
    (V1, 16384, "bool", True),
    (TINY, 16384, "none", None),
    ("mistralai/Timestral-209M", 16384, "num", None),
]

RESULTS = "results_exp2.csv"
done = set()
if os.path.exists(RESULTS):
    prev = pd.read_csv(RESULTS, keep_default_na=False)
    done = set(zip(
        prev.model_id, prev.context_length, prev.exog, prev.patch_ensemble.astype(str)
    ))

# end_train is 23:59, so loc[end_train:] starts at the first test hour (720 points).
y_true = data.loc[end_train:, "Demand"]
for model_id, context_length, exog_kind, patch_ensemble in configs:
    if (model_id, context_length, exog_kind, "" if patch_ensemble is None else str(patch_ensemble)) in done:
        continue
    if "chronos" in model_id:
        kwargs = {"device_map": "cpu"}
    elif model_id.startswith("h2oai"):
        kwargs = {"device": "cpu"}
        if patch_ensemble is not None:
            kwargs["predict_kwargs"] = {"patch_ensemble": patch_ensemble}
    else:
        kwargs = {"device": "cpu"}
    exog = None
    if exog_kind == "num":
        exog = data[["Temperature", "Holiday"]]
    elif exog_kind == "bool":
        exog = data[["Temperature", "Holiday"]].astype({"Holiday": bool})

    estimator = FoundationModel(model_id=model_id, context_length=context_length, **kwargs)
    forecaster = ForecasterFoundation(estimator=estimator)
    # Warm-up (load of the weights) outside the timing
    forecaster.fit(series=data.loc[:end_train, "Demand"])
    forecaster.predict(steps=24)
    start = time.perf_counter()
    _, preds = backtesting_foundation(
        forecaster        = forecaster,
        series            = data["Demand"],
        exog              = exog,
        cv                = cv,
        metric            = "mean_absolute_error",
        quantiles         = QUANTILES,
        show_progress     = False,
        suppress_warnings = True,
    )
    elapsed = time.perf_counter() - start
    p = preds.loc[y_true.index]
    assert len(p) == 720

    def coverage(lower, upper):
        return float(np.mean((y_true >= p[f"q_{lower}"]) & (y_true <= p[f"q_{upper}"])))

    row = {
        "model_id": model_id,
        "context_length": context_length,
        "exog": exog_kind,
        "patch_ensemble": "" if patch_ensemble is None else str(patch_ensemble),
        "mae": round(float(np.mean(np.abs(p["q_0.5"] - y_true))), 2),
        "cov80": round(coverage(0.1, 0.9), 3),
        "cov90": round(coverage(0.05, 0.95), 3),
        "cov98": round(coverage(0.01, 0.99), 3),
        "width80": round(float(np.mean(p["q_0.9"] - p["q_0.1"])), 1),
        "seconds": round(elapsed, 1),
    }
    print(row, flush=True)
    pd.DataFrame([row]).to_csv(
        RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False
    )

print(pd.read_csv(RESULTS, keep_default_na=False).to_string(index=False))
