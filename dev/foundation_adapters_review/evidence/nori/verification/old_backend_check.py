"""
Load Synthefy/Nori and Synthefy/Nori-1-5 with an older synthefy-nori to check
the minimum backend version that Nori-1-5 needs. Direct backend calls only.

Run with a temporary environment holding synthefy-nori==0.21.1:
    ~/.venvs/fm-review-nori-old/bin/python dev/foundation_adapters_review/evidence/nori/verification/old_backend_check.py
"""

import platform
from importlib.metadata import version

import numpy as np

print("python       :", platform.python_version())
for pkg in ["synthefy-nori", "torch", "numpy"]:
    print(f"{pkg:13s}:", version(pkg))

from synthefy_nori import NoriRegressor  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 3)).astype(np.float32)
y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 80)
for model in ["Synthefy/Nori", "Synthefy/Nori-1-5", "nori-1-5"]:
    try:
        reg = NoriRegressor(model=model).fit(X[:60], y[:60])
        p = reg.predict(X[60:])
        print(f"{model:18s}: ok, {p.shape}")
    except Exception as exc:
        print(f"{model:18s}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
