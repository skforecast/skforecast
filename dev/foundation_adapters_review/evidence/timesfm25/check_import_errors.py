"""
What a user sees when `timesfm` is installed without the `torch` extra
(`pip install timesfm`): `import timesfm` succeeds, because timesfm wraps the
torch import of TimesFM 2.5 in `try/except ImportError: pass`, so the adapter
fails later when it accesses `timesfm.TimesFM_2p5_200M_torch`.

torch is hidden by setting `sys.modules["torch"] = None` after importing
skforecast (scipy probes `sys.modules["torch"]` while it is imported) and
before importing timesfm, which makes `import torch` raise ImportError.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_import_errors.py
"""

import sys
from importlib.metadata import version

from skforecast.foundation._adapters import TimesFM25Adapter  # noqa: E402

print(f"timesfm {version('timesfm')} | huggingface_hub {version('huggingface_hub')}")
assert "torch" not in sys.modules or sys.modules["torch"] is None
sys.modules["torch"] = None

import timesfm  # noqa: E402

print("import timesfm: OK")
print("hasattr(timesfm, 'TimesFM_2p5_200M_torch'):", hasattr(timesfm, "TimesFM_2p5_200M_torch"))
print("hasattr(timesfm, 'ForecastConfig'):", hasattr(timesfm, "ForecastConfig"))

adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
try:
    adapter._load_model()
except Exception as exc:
    print(f"adapter._load_model(): {type(exc).__name__}: {exc}")
