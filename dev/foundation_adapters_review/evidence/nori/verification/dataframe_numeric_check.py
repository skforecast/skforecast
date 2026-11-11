"""
Check whether NoriRegressor gives the same output when the same numeric
feature matrix is passed as a numpy array or as a pandas DataFrame (the input
type NORI-P6 would need for native categorical covariates), and that a
DataFrame with a string column is accepted. Direct backend calls only.

    ~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/verification/dataframe_numeric_check.py
"""

import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

print("python       :", platform.python_version())
for pkg in ["synthefy-nori", "synthefy", "torch", "numpy", "pandas"]:
    print(f"{pkg:13s}:", version(pkg))

from synthefy_nori import NoriRegressor  # noqa: E402

rng = np.random.default_rng(1)
X = rng.normal(size=(150, 5)).astype(np.float32)
X[::7, 2] = np.nan
y = np.nan_to_num(X) @ np.array([1.0, -2.0, 0.5, 0.0, 0.3]) + rng.normal(0, 0.1, 150)
cols = [f"f{i}" for i in range(5)]
Xc, Xq, yc = X[:120], X[120:], y[:120]

for model in ["Synthefy/Nori", "Synthefy/Nori-1-5"]:
    a = NoriRegressor(model=model).fit(Xc, yc)
    d = NoriRegressor(model=model).fit(pd.DataFrame(Xc, columns=cols), yc)
    pa, pd_ = a.predict(Xq), d.predict(pd.DataFrame(Xq, columns=cols))
    qa = a.predict(Xq, output_type="quantiles", quantiles=[0.1, 0.9])
    qd = d.predict(pd.DataFrame(Xq, columns=cols), output_type="quantiles", quantiles=[0.1, 0.9])
    print(f"{model}: numpy vs numeric DataFrame max diff point {np.max(np.abs(pa - pd_)):.3e}, "
          f"quantiles {np.max(np.abs(qa - qd)):.3e}")

frame = pd.DataFrame(Xc, columns=cols)
frame["cat"] = rng.choice(["a", "b", "c"], size=len(frame))
query = pd.DataFrame(Xq, columns=cols)
query["cat"] = rng.choice(["a", "b", "z"], size=len(query))
r = NoriRegressor(model="Synthefy/Nori").fit(frame, yc)
print("string column with unseen level 'z' in the query: finite", np.isfinite(r.predict(query)).all())
