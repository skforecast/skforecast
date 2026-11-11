"""
Cross-cutting probes of shared code in skforecast/foundation (no backend
weights needed: the Chronos-2 fake pipeline of the unit tests is used).

Run from the repository root with a review environment, for example:

    ~/.venvs/fm-review-chronos/bin/python \
        dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.py
"""

import platform
import sys
import warnings

import numpy as np
import pandas as pd

import skforecast
from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import _ADAPTER_REGISTRY, _resolve_adapter
from skforecast.foundation.tests.tests_foundation_models.fixtures_adapters import (
    FakePipeline,
)

print("python", sys.version.split()[0], platform.platform())
print(
    "numpy", np.__version__, "| pandas", pd.__version__,
    "| skforecast", skforecast.__version__,
)

idx = pd.date_range("2024-01-01", periods=200, freq="h")

# 1. All-NaN context window reaches the adapter on the user path
print("\n1. All-NaN trailing context window (context_length=50)")
y = pd.Series(np.arange(200, dtype=float), index=idx, name="y")
y.iloc[-80:] = np.nan
fake = FakePipeline()
m = FoundationModel("autogluon/chronos-2-small", pipeline=fake, context_length=50)
m.fit(series=y)
try:
    pred = m.predict(steps=3)
    ctx = fake.last_inputs[0]["target"]
    print("   predict ran, context passed to backend all NaN:",
          bool(np.isnan(np.asarray(ctx, dtype=float)).all()))
except Exception as exc:
    print("   raised:", type(exc).__name__, exc)

# 2. Quantile types accepted by FoundationModel and forwarded unchanged
print("\n2. Quantile level types forwarded to the adapter")
for levels in ([0, 0.5, 1], [np.float32(0.1), np.float32(0.9)]):
    fake = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=fake)
    m.fit(series=y.iloc[:100])
    out = m.predict(steps=2, quantiles=levels)
    print("   requested", levels, "-> backend got types",
          [type(q).__name__ for q in fake.last_quantile_levels],
          "| columns", list(out.columns))

# 3. Registry prefixes accept ids that are not loadable forecasters
print("\n3. _resolve_adapter on ids reported by the per-adapter reviews")
for model_id in [
    "Synthefy/Nori-Causal",
    "theforecastingcompany/t0-alpha-onnx-int8",
    "google/timesfm-2.5-200m-flax",
    "google/timesfm-2.5-200m-transformers",
    "google/timesfm-3.0-flax",
    "soda-inria/tabicl-v1-does-not-exist",
    "priorlabs/tabpfn-v2",
    "Salesforce/moirai-2.0-R-large",
    "taharnbl/TS-ICL-v2",
]:
    try:
        cls = _resolve_adapter(model_id)
        print(f"   {model_id:45s} -> {cls.__name__}")
    except ValueError:
        print(f"   {model_id:45s} -> ValueError")

# 4. ImportError message quoting of backend_package
print("\n4. backend_package per adapter")
for prefix, cls in _ADAPTER_REGISTRY.items():
    print(f"   {cls.__name__:18s} {cls.backend_package!r}")

# 5. pd.Timedelta(days=n) used by TabICL/TabPFN _get_future_timestamps
print("\n5. pd.Timedelta(days=n) with the installed numpy/pandas")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _ = pd.Timestamp("2000-01-01") + pd.Timedelta(days=5)
print("   warnings:", [f"{w.category.__name__}: {str(w.message)[:70]}" for w in caught])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    a = pd.date_range("2000-01-01", periods=5 + 3, freq="D")[5:]
    b = pd.date_range(pd.Timestamp("2000-01-01") + pd.Timedelta(5, unit="D"),
                      periods=3, freq="D")
print("   date_range slice warnings:", len(caught), "| equal:", a.equals(b))
