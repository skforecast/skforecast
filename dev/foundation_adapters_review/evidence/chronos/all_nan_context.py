"""
What reaches ChronosAdapter when the context window of a series is entirely
NaN (upstream issue amazon-science/chronos-forecasting#544, open on
2026-11-06: Chronos-2 returns quantiles around +-1e5 for an all-NaN context).
"""

import importlib.metadata as md
import warnings

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

for package in ["skforecast", "chronos-forecasting", "torch"]:
    print(f"{package:20s} {md.version(package)}")

idx = pd.date_range("2024-01-01", periods=300, freq="h")
values = 10 + np.sin(np.arange(300) / 5)
values[200:] = np.nan  # 100 trailing NaN, longer than context_length=64
y_nan_tail = pd.Series(values, index=idx, name="tail_nan")
y_ok = pd.Series(10 + np.cos(np.arange(300) / 5), index=idx, name="ok")

model = FoundationModel(model_id="autogluon/chronos-2-small", context_length=64)
cases = {
    "single series, last 100 values NaN": dict(context=y_nan_tail),
    "two series (dict), one with last 100 values NaN": dict(
        context={"tail_nan": y_nan_tail, "ok": y_ok}
    ),
    "single series fitted, last 100 values NaN": None,
}
for name, kwargs in cases.items():
    print("==", name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if kwargs is None:
                model.fit(series=y_nan_tail)
                pred = model.predict(steps=6, quantiles=[0.1, 0.5, 0.9])
            else:
                pred = model.predict(steps=6, quantiles=[0.1, 0.5, 0.9], **kwargs)
            print(pred.groupby("level").agg(["min", "max"]).round(2).to_string())
            print("  first prediction timestamp:", pred.index.min())
        except Exception as exc:  # noqa: BLE001
            print("  ", type(exc).__name__, str(exc)[:150])
    print("  warnings:", sorted({f"{w.category.__name__}: {str(w.message)[:100]}" for w in caught}))
