"""
Is the `_TimesFMCompat` subclass of TimesFM25Adapter._load_model still needed?

It absorbs `proxies` and `resume_download`, which huggingface_hub < 1.0 passes
to `_from_pretrained`. The script loads the checkpoint (a) directly with
`timesfm.TimesFM_2p5_200M_torch.from_pretrained`, without the workaround, and
(b) through the adapter, with the workaround, and prints the result of each.
Run it in environments with different timesfm and huggingface_hub versions.

Run from the repository root, e.g.:
    ~/.venvs/fm-review-timesfm25-oldhf/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_workaround.py
"""

import inspect
import logging
from importlib.metadata import version

import timesfm
from huggingface_hub import hub_mixin
from skforecast.foundation._adapters import TimesFM25Adapter

logging.disable(logging.CRITICAL)
MODEL_ID = "google/timesfm-2.5-200m-pytorch"

print(f"timesfm {version('timesfm')} | huggingface_hub {version('huggingface_hub')} "
      f"| torch {version('torch')}")
src = inspect.getsource(hub_mixin.ModelHubMixin.from_pretrained)
print("huggingface_hub from_pretrained passes proxies/resume_download:",
      "proxies=proxies" in src, "resume_download=resume_download" in src)
init_sig = inspect.signature(timesfm.TimesFM_2p5_200M_torch.__init__)
print("TimesFM_2p5_200M_torch.__init__ signature:", init_sig)

try:
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    print("(a) direct from_pretrained, no workaround: OK, torch_compile =",
          model.torch_compile)
except Exception as exc:
    print(f"(a) direct from_pretrained, no workaround: {type(exc).__name__}: {exc}")

try:
    adapter = TimesFM25Adapter(model_id=MODEL_ID)
    adapter._load_model()
    print("(b) adapter._load_model (with workaround): OK, torch_compile =",
          adapter._model.torch_compile)
except Exception as exc:
    print(f"(b) adapter._load_model (with workaround): {type(exc).__name__}: {exc}")
