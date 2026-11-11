"""
Edge case of supports_nan_in_series=True: a series whose trimmed context is
entirely NaN (e.g. a long gap at the end of the window). Checks the backend
helpers directly and the adapter through FoundationModel.

Run from the repository root:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_all_nan_context.py
"""

import logging
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
from timesfm.timesfm_2p5.timesfm_2p5_base import linear_interpolation, strip_leading_nans

from skforecast.foundation import FoundationModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
print(f"timesfm {version('timesfm')} | numpy {version('numpy')} | pandas {version('pandas')}")

try:
    out = linear_interpolation(strip_leading_nans(np.full(10, np.nan)))
    print("backend helpers, all-NaN input:", out)
except Exception as exc:
    print(f"backend helpers, all-NaN input: {type(exc).__name__}: {exc}")

idx = pd.date_range("2020-01-01", periods=300, freq="D")
ok = pd.Series(np.sin(np.arange(300) / 5) + 10, index=idx)
gap = ok.copy()
gap.iloc[-80:] = np.nan
for label, ctx_len in [("context fully NaN (context_length=64)", 64),
                       ("context partly NaN (context_length=128)", 128)]:
    try:
        fm = FoundationModel(model_id="google/timesfm-2.5-200m-pytorch", context_length=ctx_len)
        fm.fit(series={"ok": ok, "gap": gap})
        pred = fm.predict(steps=5)
        print(f"{label}: OK, NaN in output: {bool(pred['pred'].isna().any())}")
    except Exception as exc:
        print(f"{label}: {type(exc).__name__}: {str(exc)[:250]}")
