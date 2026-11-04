"""
Extract the skeleton from `dev/research_timestral_adapter.md` and write two
importable modules next to this script:

- `shared_changes.py`: the shared-code changes (blocks `shared_utils` and
  `resolve_device`), which would go to `skforecast/foundation/_utils.py` and
  `skforecast/foundation/_adapters.py`.
- `timestral_skeleton.py`: the adapter (block `adapter`), with absolute
  imports instead of the package-relative ones.
- `foundation_loop.txt`: the new loop over the groups of
  `FoundationModel.predict` (block `foundation_loop`), as plain text that
  `run_checks.py` inserts in the current source of the method.

Usage: python extract_skeleton.py ../../research_timestral_adapter.md
"""
import re
import sys
from pathlib import Path

HEADER_SHARED = '''"""
Shared-code changes of the Timestral adapter, extracted from
dev/research_timestral_adapter.md. Generated file, do not edit.
"""

from __future__ import annotations
import warnings
import pandas as pd


'''

HEADER_ADAPTER = '''"""
TimestralAdapter skeleton, extracted from dev/research_timestral_adapter.md.
Generated file, do not edit.
"""

from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd

from skforecast.exceptions import FeatureOutOfRangeWarning
from skforecast.foundation._adapter_base import _AdapterBase
from skforecast.foundation._utils import (
    _validate_positive_int,
    _validate_model_id_prefix,
    _validate_supported_quantiles,
    _tensor_to_numpy,
    _apply_set_params,
    _warn_if_non_commercial,
    get_exog_signature,
)
from shared_changes import _match_quantile_indices, _resolve_torch_device


'''

doc = Path(sys.argv[1]).read_text()
blocks = dict(re.findall(
    r"<!-- skeleton:(\w+) -->\n```python\n(.*?)\n```", doc, flags=re.DOTALL
))
missing = {"shared_utils", "resolve_device", "adapter", "foundation_loop"} - set(blocks)
if missing:
    sys.exit(f"Blocks not found in the document: {sorted(missing)}")

here = Path(__file__).parent
(here / "shared_changes.py").write_text(
    HEADER_SHARED + blocks["shared_utils"] + "\n\n\n" + blocks["resolve_device"] + "\n"
)
(here / "timestral_skeleton.py").write_text(HEADER_ADAPTER + blocks["adapter"] + "\n")
(here / "foundation_loop.txt").write_text(blocks["foundation_loop"] + "\n")
print("written shared_changes.py, timestral_skeleton.py and foundation_loop.txt")
