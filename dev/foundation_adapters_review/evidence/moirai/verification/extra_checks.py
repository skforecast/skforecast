"""
Extra checks of the verification of MoiraiAdapter (correctness only, no timing).

1. Context longer than max_seq_len * patch_size of the checkpoint (8192): the
   call must run and the adapter output must match a direct uni2ts call.
2. Location reported by the LicenseWarning issued on the first load.

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/verification/extra_checks.py
"""

from __future__ import annotations
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.exceptions import LicenseWarning

MODEL_ID = "Salesforce/moirai-2.0-R-small"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    print("python", platform.python_version(), "| skforecast", version("skforecast"),
          "| uni2ts", version("uni2ts"), "| torch", version("torch"),
          "| numpy", version("numpy"))

    rng = np.random.default_rng(0)
    n = 13000
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    y = pd.Series(100 + 10 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1, n),
                  index=idx, name="y")

    print("\n2. LicenseWarning location on first load")
    fm = FoundationModel(model_id=MODEL_ID, context_length=12000, device="cpu")
    fm.fit(series=y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = fm.predict(steps=24, quantiles=QUANTILES)  # user line
    lic = [w for w in caught if issubclass(w.category, LicenseWarning)]
    print("   LicenseWarning at:",
          [f"{w.filename.split('skforecast/')[-1]}:{w.lineno}" for w in lic])

    print("\n1. context_length=12000 (> 512 patches * 16 = 8192)")
    module = fm.adapter._module
    print("   module.max_seq_len, patch_size:", module.max_seq_len, module.patch_size)
    from uni2ts.model.moirai2 import Moirai2Forecast
    direct = Moirai2Forecast(
        module=module, prediction_length=24, context_length=12000, target_dim=1,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).eval().predict([y.iloc[-12000:].to_numpy().reshape(-1, 1)])[0].T
    adapter_q = pred[[f"q_{q}" for q in QUANTILES]].to_numpy()
    print("   finite:", np.isfinite(adapter_q).all(),
          "| max|adapter - direct| =", f"{np.max(np.abs(adapter_q - direct)):.3e}")


if __name__ == "__main__":
    main()
