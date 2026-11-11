"""
Simulate an environment with tfc-t0 0.6.0 but without torch (as installed by
`pip install tfc-t0`) by blocking the torch import, and print the error the
adapter raises.
"""

import sys
from importlib.metadata import version



class _BlockTorch:
    """Meta path finder that makes `import torch` raise ImportError."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named {name!r} (blocked for this check)")
        return None


sys.meta_path.insert(0, _BlockTorch())

print("tfc-t0", version("tfc-t0"))
import t0  # noqa: E402,F401

print("import t0: OK")
from skforecast.foundation._adapters import T0Adapter  # noqa: E402

adapter = T0Adapter("theforecastingcompany/t0-alpha")
try:
    adapter._load_model()
except ImportError as exc:
    print("adapter ImportError:", exc)
    print("cause:", repr(exc.__cause__))
