"""
Re-check of the google/timesfm-2.5 prefix: which Hub repos it routes to
TimesFM25Adapter and whether they load. The -transformers weights are not
downloaded: their tensor names are read from the safetensors header and
compared with the state_dict keys of the timesfm torch module.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_registry_checkpoints.py
"""

import logging
import warnings
from importlib.metadata import version

from huggingface_hub import HfApi, get_safetensors_metadata
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

from skforecast.foundation import FoundationModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
print(f"timesfm {version('timesfm')} | huggingface_hub {version('huggingface_hub')}")

api = HfApi()
for c in api.list_repo_commits("google/timesfm-2.5-200m-pytorch"):
    print("pytorch repo commit:", c.commit_id[:7], c.created_at.date(), c.title)

expected = set(TimesFM_2p5_200M_torch_module().state_dict().keys())
meta = get_safetensors_metadata("google/timesfm-2.5-200m-transformers")
names = set()
for fm in meta.files_metadata.values():
    names |= set(fm.tensors.keys())
print(f"-transformers tensors {len(names)} | expected {len(expected)} | "
      f"expected missing in file {len(expected - names)}")
meta_pt = get_safetensors_metadata("google/timesfm-2.5-200m-pytorch")
names_pt = set()
for fm in meta_pt.files_metadata.values():
    names_pt |= set(fm.tensors.keys())
print(f"-pytorch tensors {len(names_pt)} | expected missing in file {len(expected - names_pt)}")

for mid in ["google/timesfm-2.5-200m-flax", "google/timesfm-2.0-500m-pytorch",
            "google/timesfm-3.0-pytorch"]:
    try:
        fm = FoundationModel(model_id=mid)
        print(f"{mid}: adapter {type(fm.adapter).__name__}")
        if "flax" in mid:
            fm.adapter._load_model()
    except Exception as exc:
        print(f"{mid}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
