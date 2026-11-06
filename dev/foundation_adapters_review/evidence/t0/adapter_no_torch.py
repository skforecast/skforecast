"""
Error the user sees with `pip install tfc-t0` (0.6.0, no torch) when the
adapter loads the model. Run before installing the torch extra.
"""
from importlib.metadata import version

import pandas as pd

from skforecast.foundation import ForecasterFoundation, FoundationModel

print("tfc-t0", version("tfc-t0"), "skforecast", version("skforecast"))
try:
    import torch  # noqa: F401
    print("torch installed:", torch.__version__)
except ImportError:
    print("torch not installed")

y = pd.Series(range(100), index=pd.date_range("2024-01-01", periods=100, freq="h"), dtype=float)
f = ForecasterFoundation(estimator=FoundationModel("theforecastingcompany/t0-alpha"))
f.fit(series=y)
try:
    f.predict(steps=5)
except Exception as exc:
    print(type(exc).__name__, ":", exc)
    print("cause:", type(exc.__cause__).__name__, ":", exc.__cause__)
