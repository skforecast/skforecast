"""
Extract the skeleton from `dev/timeh2o_adapter/research_timeh2o_adapter.md`
and write three importable modules next to this script:

- `shared_changes.py`: the shared helper (block `shared_utils`), which would
  go to `skforecast/foundation/_utils.py`.
- `timeh2o_skeleton.py`: the adapter (block `adapter`), with absolute imports
  instead of the package-relative ones.
- `license_entry.py`: the entry of `_MODEL_LICENSES` (block `license`), as a
  dict named `LICENSE_ENTRY`.

The adapter relies on PR #1450 (`requires_equal_length_context` in the
contract and `_resolve_torch_device` with `mps_fallback_message`), so
`run_checks.py` must import a skforecast that includes it (see its
docstring).

Usage: python extract_skeleton.py ../../research_timeh2o_adapter.md
"""
import re
import sys
from pathlib import Path

HEADER_ADAPTER = '''"""
TimeH2OAdapter skeleton, extracted from dev/timeh2o_adapter/research_timeh2o_adapter.md.
Generated file, do not edit.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from skforecast.foundation._adapter_base import _AdapterBase
from skforecast.foundation._adapters import _resolve_torch_device
from skforecast.foundation._utils import (
    _validate_positive_int,
    _validate_model_id_prefix,
    _apply_set_params,
    _warn_if_non_commercial,
    get_exog_signature,
)
from shared_changes import _warn_if_flat_known_covariates


'''

HEADER_SHARED = '''"""
Shared helper of the TimeH2O adapter, extracted from
dev/timeh2o_adapter/research_timeh2o_adapter.md. Generated file, do not edit.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from skforecast.exceptions import FeatureOutOfRangeWarning


'''

HEADER_LICENSE = '''"""
Entry of `_MODEL_LICENSES` for TimeH2O, extracted from
dev/timeh2o_adapter/research_timeh2o_adapter.md. Generated file, do not edit.
"""

LICENSE_ENTRY = {
'''

doc = Path(sys.argv[1]).read_text()
blocks = dict(re.findall(
    r"<!-- skeleton:(\w+) -->\n```python\n(.*?)\n```", doc, flags=re.DOTALL
))
missing = {"adapter", "license", "shared_utils"} - set(blocks)
if missing:
    sys.exit(f"Blocks not found in the document: {sorted(missing)}")

here = Path(__file__).parent
(here / "shared_changes.py").write_text(HEADER_SHARED + blocks["shared_utils"] + "\n")
(here / "timeh2o_skeleton.py").write_text(HEADER_ADAPTER + blocks["adapter"] + "\n")
(here / "license_entry.py").write_text(HEADER_LICENSE + blocks["license"] + "\n}\n")
print("written shared_changes.py, timeh2o_skeleton.py and license_entry.py")
