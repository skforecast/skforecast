"""
Check the ImportError raised by TimesFM3Adapter._load_model when torch is not
importable, with the real installed `timesfm` package (torch is blocked with
`sys.modules["torch"] = None`, which makes `import torch` raise ImportError).

Run from the repository root:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/check_import_errors.py
"""

import sys
import warnings
from importlib.metadata import version

# skforecast (and scipy, through scikit-learn) are imported before blocking
# torch: scipy inspects `sys.modules["torch"]` at import time.
from skforecast.foundation._adapters import TimesFM3Adapter

assert "torch" not in sys.modules
sys.modules["torch"] = None

print("timesfm", version("timesfm"))

import timesfm  # noqa: E402

print("hasattr(timesfm, 'TimesFM3Forecaster'):", hasattr(timesfm, "TimesFM3Forecaster"))
try:
    import timesfm3  # noqa: F401

    print("import timesfm3: OK (lazy package, torch not needed to import it)")
except ImportError as exc:
    print("import timesfm3: ImportError", exc)

adapter = TimesFM3Adapter(model_id="google/timesfm-3.0-pytorch")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        adapter._load_model()
    except ImportError as exc:
        print("ImportError raised by the adapter:")
        print("  ", exc)
print("warnings:", [w.category.__name__ for w in caught])
