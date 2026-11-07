# Check whether the 8 levels of SUPPORTED_QUANTILES rejected by tsicl 0.2.1
# (exact float membership test in TSICL._get_quantile_indices) can be passed
# as exact numbers (decimal.Decimal / fractions.Fraction), and that the
# returned columns are the expected ones of the backend quantile grid.
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/verification/check_quantile_exact_types.py
import platform
import warnings
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import version

import numpy as np
import torch

from skforecast.foundation._adapters import TSICLAdapter

warnings.simplefilter("ignore")
print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} skforecast={version('skforecast')}"
)

adapter = TSICLAdapter("taharnbl/TS-ICL", device="cpu")
adapter._load_model()
m = adapter._model
bad = [0.07, 0.14, 0.28, 0.29, 0.55, 0.56, 0.57, 0.58]
for q in bad:
    idx_dec = m._get_quantile_indices([Decimal(str(q))])
    idx_fr = m._get_quantile_indices([Fraction(str(q))])
    print(f"q={q}: Decimal idx={idx_dec} Fraction idx={idx_fr} expected={round(q * 100) - 1}")

rng = np.random.default_rng(0)
y = (50 + 5 * np.sin(np.arange(400) * 2 * np.pi / 24) + rng.normal(0, 0.5, 400)).astype(np.float32)
good = [0.06, 0.08, 0.5]
levels = [0.06, Decimal("0.07"), 0.08, 0.5]
_, q_dec = m.forecast(
    inputs=[{"target": y}], prediction_length=24, quantile_levels=levels,
    context_length=400, device=torch.device("cpu"), squeeze_output=False,
)
_, q_ref = m.forecast(
    inputs=[{"target": y}], prediction_length=24, quantile_levels=good,
    context_length=400, device=torch.device("cpu"), squeeze_output=False,
)
a = np.asarray(q_dec[0][0])
r = np.asarray(q_ref[0][0])
print(f"output shape={a.shape}")
print(f"q0.06 equal to reference: {np.max(np.abs(a[:, 0] - r[:, 0])):.3e}")
print(f"q0.08 equal to reference: {np.max(np.abs(a[:, 2] - r[:, 1])):.3e}")
print(f"q0.06 <= q0.07 <= q0.08 for all steps: {bool(np.all(a[:, 0] <= a[:, 1]) and np.all(a[:, 1] <= a[:, 2]))}")
