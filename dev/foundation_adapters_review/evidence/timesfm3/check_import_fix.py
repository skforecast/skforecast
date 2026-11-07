"""
Check which import surfaces the real cause (torch missing) for TimesFM 3.0 in
the installed `timesfm`, to choose the probe used by
TimesFM3Adapter._load_model. Torch is blocked with `sys.modules["torch"] = None`.

Run from the repository root with each review environment:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/check_import_fix.py
    ~/.venvs/fm-review-timesfm3-v300/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/check_import_fix.py
"""

import importlib
import sys
from importlib.metadata import version

print("python", sys.version.split()[0])
print("timesfm", version("timesfm"))
sys.modules["torch"] = None


def probe(label, func):
    for name in [m for m in sys.modules if m.startswith("timesfm3")]:
        del sys.modules[name]
    try:
        func()
        print(f"{label}: OK (no error)")
    except ImportError as exc:
        print(f"{label}: ImportError: {exc}")


probe("import timesfm3 (current adapter probe)", lambda: importlib.import_module("timesfm3"))
probe(
    "from timesfm3 import TimesFM3Forecaster (proposed probe)",
    lambda: getattr(importlib.import_module("timesfm3"), "TimesFM3Forecaster"),
)
