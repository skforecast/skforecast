"""
Check whether an older synthefy-nori accepts NaN in the feature matrix X
(context and query rows) and NaN in y. Direct backend calls only.

Run with a temporary environment holding the older backend, e.g.
synthefy-nori==0.18.2:
    ~/.venvs/fm-review-nori-old/bin/python dev/foundation_adapters_review/evidence/nori/verification/old_backend_nan_x.py
"""

import platform
from importlib.metadata import version

import numpy as np

print("python       :", platform.python_version())
for pkg in ["synthefy-nori", "torch", "numpy", "scikit-learn"]:
    print(f"{pkg:13s}:", version(pkg))

from synthefy_nori import NoriRegressor  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 4)).astype(np.float32)
y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(0, 0.1, 120)
Xc, yc, Xq = X[:100].copy(), y[:100].copy(), X[100:].copy()
Xc[::4, 1] = np.nan
Xq[::3, 1] = np.nan
try:
    reg = NoriRegressor(model="Synthefy/Nori").fit(Xc, yc)
    p = reg.predict(Xq)
    q = reg.predict(Xq, output_type="quantiles", quantiles=[0.9, 0.1])
    print(f"NaN in X: ok, finite point {np.isfinite(p).all()}, finite quantiles {np.isfinite(q).all()}")
except Exception as exc:
    print(f"NaN in X: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
try:
    yy = yc.copy()
    yy[2] = np.nan
    reg = NoriRegressor(model="Synthefy/Nori").fit(X[:100], yy)
    p = reg.predict(X[100:])
    print(f"NaN in y: accepted, finite point {np.isfinite(p).all()}")
except Exception as exc:
    print(f"NaN in y: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
