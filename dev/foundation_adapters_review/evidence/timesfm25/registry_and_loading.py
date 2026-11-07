"""
Registry prefixes and loading of every TimesFM 2.5 checkpoint on the Hub.

1. `_resolve_adapter` for every google/timesfm-* repository.
2. `get_model_info` of the default checkpoint.
3. Loading `google/timesfm-2.5-200m-flax` through the adapter (no
   model.safetensors in the repository).
4. Whether `google/timesfm-2.5-200m-transformers` could be loaded by
   `TimesFM_2p5_200M_torch`: its tensor names (read from the safetensors header,
   without downloading the weights) against the state_dict keys of the module,
   which `load_checkpoint` loads with `strict=True`.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/registry_and_loading.py
"""

import logging
from importlib.metadata import version

from huggingface_hub import get_safetensors_metadata
from skforecast.foundation import get_model_info
from skforecast.foundation._adapters import TimesFM25Adapter, _resolve_adapter

logging.disable(logging.CRITICAL)
print(f"timesfm {version('timesfm')} | huggingface_hub {version('huggingface_hub')} "
      f"| skforecast {version('skforecast')}")

REPOS = [
    "google/timesfm-1.0-200m-pytorch",
    "google/timesfm-2.0-500m-pytorch",
    "google/timesfm-2.5-200m-pytorch",
    "google/timesfm-2.5-200m-flax",
    "google/timesfm-2.5-200m-transformers",
    "google/timesfm-3.0-pytorch",
]
print("\n== 1. _resolve_adapter")
for repo in REPOS:
    try:
        print(f"{repo:40s} -> {_resolve_adapter(repo).__name__}")
    except ValueError as exc:
        print(f"{repo:40s} -> ValueError: {str(exc)[:80]}")

print("\n== 2. get_model_info")
info = get_model_info("google/timesfm-2.5-200m-pytorch")
for key, value in (info.items() if isinstance(info, dict) else vars(info).items()):
    print(f"  {key}: {value}")

print("\n== 3. load google/timesfm-2.5-200m-flax through the adapter")
adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-flax")
try:
    adapter._load_model()
    print("  loaded:", type(adapter._model).__name__)
except Exception as exc:
    print(f"  {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")

print("\n== 4. google/timesfm-2.5-200m-transformers vs the torch module keys")
import timesfm  # noqa: E402

module_keys = set(timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch_module().state_dict())
meta = get_safetensors_metadata("google/timesfm-2.5-200m-transformers")
hub_keys = set()
for file_meta in meta.files_metadata.values():
    hub_keys |= set(file_meta.tensors)
ref = get_safetensors_metadata("google/timesfm-2.5-200m-pytorch")
ref_keys = set()
for file_meta in ref.files_metadata.values():
    ref_keys |= set(file_meta.tensors)
print(f"  module keys: {len(module_keys)}")
print(f"  -pytorch keys equal to module keys: {ref_keys == module_keys}")
print(f"  -transformers keys: {len(hub_keys)}, missing for strict load: "
      f"{len(module_keys - hub_keys)}, unexpected: {len(hub_keys - module_keys)}")
print("  => load_state_dict(strict=True) would raise:", module_keys != hub_keys)
