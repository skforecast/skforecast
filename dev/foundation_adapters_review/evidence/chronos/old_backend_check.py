"""
Check ChronosAdapter with an older chronos-forecasting release (the adapter
does not pin a minimum version). Run with an environment that has the old
release installed, for example ~/.venvs/fm-review-chronos-old.
"""

import importlib.metadata as md
import platform
import warnings

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

print("python", platform.python_version())
for package in ["skforecast", "chronos-forecasting", "torch", "transformers"]:
    print(f"{package:20s} {md.version(package)}")

y = pd.Series(
    50 + 5 * np.sin(np.arange(200) * 2 * np.pi / 24),
    index=pd.date_range("2024-01-01", periods=200, freq="h"),
    name="y",
)
for kwargs in [{}, {"torch_dtype": "bfloat16"}]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = FoundationModel(model_id="autogluon/chronos-2-small", **kwargs)
            pred = model.predict(steps=12, context=y)
            print(kwargs, "OK", pred.shape)
        except Exception as exc:  # noqa: BLE001
            print(kwargs, type(exc).__name__, str(exc)[:160])
    print("  warnings:", sorted({f"{w.category.__name__}: {str(w.message)[:100]}" for w in caught}))
