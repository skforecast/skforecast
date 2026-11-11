# Check whether every level of `TSICLAdapter.SUPPORTED_QUANTILES` is accepted by
# the backend, both directly (`TSICL._get_quantile_indices`) and through
# FoundationModel.predict.
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/check_quantile_grid.py
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import TSICLAdapter

warnings.simplefilter("ignore")
print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} skforecast={version('skforecast')}"
)

levels = TSICLAdapter.SUPPORTED_QUANTILES
model = FoundationModel("taharnbl/TS-ICL", context_length=128, device="cpu")
model.adapter._load_model()
backend = model.adapter._model

rejected_direct = []
for level in levels:
    try:
        backend._get_quantile_indices([level])
    except ValueError:
        rejected_direct.append(level)
print(f"levels in SUPPORTED_QUANTILES: {len(levels)}")
print(f"rejected by TSICL._get_quantile_indices: {len(rejected_direct)} -> {rejected_direct}")

rng = np.random.default_rng(0)
series = pd.Series(
    rng.normal(size=200), index=pd.date_range("2024-01-01", periods=200, freq="h")
)
rejected_fm = []
for level in levels:
    try:
        model.predict(steps=4, context=series, quantiles=[level])
    except ValueError as exc:
        rejected_fm.append((level, str(exc).splitlines()[0][:90]))
print(f"rejected through FoundationModel.predict: {len(rejected_fm)}")
for level, msg in rejected_fm[:5]:
    print(f"    {level}: {msg}")
