"""Check which model ids the registry prefixes of TabICLAdapter accept."""

import importlib.metadata as md
import sys

from skforecast.foundation._adapters import _resolve_adapter

print("python", sys.version.split()[0], "| tabicl", md.version("tabicl"), "| skforecast", md.version("skforecast"))

CANDIDATES = [
    "soda-inria/tabicl",
    "soda-inria/tabicl-v2",
    "soda-inria/tabicl-regressor-v2-20260212",
    "soda-inria/TabICL",
    "jingang/TabICL",
    "jingang/tabicl",
    "tabicl",
]
for model_id in CANDIDATES:
    try:
        cls = _resolve_adapter(model_id).__name__
    except ValueError as exc:
        cls = f"ValueError: {str(exc)[:60]}"
    print(f"{model_id!r:45} -> {cls}")
