"""
Check proposal TFM3-P1 (probe for a missing torch in TimesFM3Adapter._load_model).

1. With the real installed timesfm and torch blocked, run the current adapter
   and a patched copy of `_load_model` that uses
   `from timesfm3 import TimesFM3Forecaster` as the probe and
   `getattr` in try/except (AttributeError, ImportError) instead of `hasattr`.
2. Simulate upstream PR #552 (not merged): a fake `timesfm` whose module
   `__getattr__` raises ImportError, and show what the current code raises.

Run from the repository root with each review environment:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/verification/check_p1_probe.py
    ~/.venvs/fm-review-timesfm3-v300/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/verification/check_p1_probe.py
"""

import sys
import types
import warnings
from importlib.metadata import version

from skforecast.foundation._adapters import TimesFM3Adapter, _import_timesfm

print("python", sys.version.split()[0])
print("timesfm", version("timesfm"))
warnings.simplefilter("ignore")


def patched_load_model(self):
    """Probe proposed in TFM3-P1 (only the torch-missing branch)."""
    timesfm = _import_timesfm(type(self))
    try:
        getattr(timesfm, "TimesFM3Forecaster")
        return "TimesFM3Forecaster available"
    except (AttributeError, ImportError):
        pass
    try:
        from timesfm3 import TimesFM3Forecaster  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"backend could not be imported ({exc}). torch is missing") from exc
    raise ImportError("generic: does not provide TimesFM3Forecaster")


sys.modules["torch"] = None
for label, func in (
    ("current _load_model", TimesFM3Adapter._load_model),
    ("patched _load_model", patched_load_model),
):
    for name in [m for m in sys.modules if m.startswith("timesfm")]:
        del sys.modules[name]
    adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
    try:
        print(f"{label}: returned {func(adapter)}")
    except ImportError as exc:
        print(f"{label}: ImportError: {exc}")

# Simulation of upstream PR #552: module-level __getattr__ raising ImportError.
fake = types.ModuleType("timesfm")


def fake_getattr(name):
    if name == "TimesFM3Forecaster":
        raise ImportError(
            "timesfm.TimesFM3Forecaster could not be imported (No module named "
            "'torch'). Install the optional backend with: pip install \"timesfm[torch]\""
        )
    raise AttributeError(name)


fake.__getattr__ = fake_getattr
sys.modules["timesfm"] = fake
adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
try:
    TimesFM3Adapter._load_model(adapter)
except ImportError as exc:
    print("PR #552 simulation, current code: ImportError propagated from hasattr:", exc)
try:
    patched_load_model(adapter)
except ImportError as exc:
    print("PR #552 simulation, patched code:", exc)
