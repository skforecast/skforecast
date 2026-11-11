"""
Check what `MoiraiAdapter.set_params(context_length=...)` reloads, and whether
the weights of `Moirai2Module` depend on `context_length`.

1. Counts the calls to `Moirai2Module.from_pretrained` when `context_length`
   changes, with the module loaded by the adapter and with an injected
   `module`.
2. Compares the predictions of a reloaded module with those of the same module
   object reused under the new `context_length` (only the `Moirai2Forecast`
   wrapper rebuilt). Equal predictions mean that resetting only the wrapper is
   enough (no change of the forecasts).

No timing is measured.

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/set_params_reload.py
"""

from __future__ import annotations
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

MODEL_ID = "Salesforce/moirai-2.0-R-small"


def main() -> None:
    warnings.simplefilter("ignore")
    print("python", platform.python_version(), "| skforecast", version("skforecast"),
          "| uni2ts", version("uni2ts"), "| torch", version("torch"))
    from uni2ts.model import moirai2
    from uni2ts.model.moirai2 import Moirai2Module

    calls = {"n": 0}
    original = Moirai2Module.from_pretrained.__func__

    def counting(cls, *args, **kwargs):
        calls["n"] += 1
        return original(cls, *args, **kwargs)

    Moirai2Module.from_pretrained = classmethod(counting)
    moirai2.Moirai2Module = Moirai2Module

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=1500, freq="h")
    y = pd.Series(100 + 10 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1, 1500),
                  index=idx, name="y")

    # Module loaded by the adapter.
    fm = FoundationModel(model_id=MODEL_ID, context_length=512)
    fm.fit(series=y)
    fm.predict(steps=24)
    module_first = fm.adapter._module
    print("from_pretrained calls after first predict:", calls["n"])
    fm.set_params(context_length=1024)
    print("adapter._module is None after set_params(context_length=1024):",
          fm.adapter._module is None)
    fm.fit(series=y)
    reloaded = fm.predict(steps=24, quantiles=[0.1, 0.5, 0.9])
    print("from_pretrained calls after predict with the new context_length:", calls["n"])

    # Same module object, only the wrapper rebuilt (what a reset of
    # `_forecast_obj` alone would do).
    fm.adapter._module = module_first
    fm.adapter._forecast_obj = None
    reused = fm.predict(steps=24, quantiles=[0.1, 0.5, 0.9])
    diff = np.max(np.abs(reloaded[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
                         - reused[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()))
    print("max|reloaded module - reused module| =", f"{diff:.3e}")

    # Injected module.
    calls["n"] = 0
    fm2 = FoundationModel(model_id=MODEL_ID, context_length=512, module=module_first)
    fm2.fit(series=y)
    fm2.predict(steps=24)
    print("injected module, from_pretrained calls after predict:", calls["n"])
    fm2.set_params(context_length=1024)
    print("injected module dropped after set_params(context_length=1024):",
          fm2.adapter._module is None)
    fm2.fit(series=y)
    fm2.predict(steps=24)
    print("injected module, from_pretrained calls after set_params + predict:", calls["n"])


if __name__ == "__main__":
    main()
