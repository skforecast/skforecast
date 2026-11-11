"""
Small checks of `MoiraiAdapter` behaviour that do not need a GPU:

1. Message raised when `uni2ts` is installed but has no `uni2ts.model.moirai2`
   (uni2ts < 2.0, which is what a resolver picks on Python >= 3.13). The
   missing submodule is simulated with `sys.modules[...] = None`.
2. Where the MPS fallback warning points to (its `stacklevel`), simulating an
   Apple Silicon machine with `torch.backends.mps.is_available = True`. No MPS
   device is used: the adapter falls back to CPU before moving the model.
3. dtype of the predictions returned by `FoundationModel.predict`.

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/misc_checks.py
"""

from __future__ import annotations
import platform
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import ForecasterFoundation, FoundationModel
from skforecast.foundation._adapters import MoiraiAdapter

MODEL_ID = "Salesforce/moirai-2.0-R-small"


def main() -> None:
    print("python", platform.python_version(), "| skforecast", version("skforecast"),
          "| uni2ts", version("uni2ts"), "| torch", version("torch"))
    print()

    print("1. uni2ts without uni2ts.model.moirai2")
    saved = sys.modules.get("uni2ts.model.moirai2")
    sys.modules["uni2ts.model.moirai2"] = None
    try:
        MoiraiAdapter(model_id=MODEL_ID)._load_module()
    except ImportError as exc:
        print(f"   {type(exc).__name__}: {exc}")
        print(f"   cause: {type(exc.__cause__).__name__}: {exc.__cause__}")
    finally:
        if saved is None:
            sys.modules.pop("uni2ts.model.moirai2", None)
        else:
            sys.modules["uni2ts.model.moirai2"] = saved
    print()

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    y = pd.Series(rng.normal(10, 1, 300), index=idx, name="y")

    print("2. MPS fallback warning location (simulated MPS availability)")
    import torch

    torch.backends.mps.is_available = lambda: True
    for device in ["mps", "mps:0", "auto"]:
        for entry in ["FoundationModel.predict", "ForecasterFoundation.predict"]:
            fm = FoundationModel(model_id=MODEL_ID, context_length=128, device=device)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if entry == "FoundationModel.predict":
                    fm.fit(series=y)
                    fm.predict(steps=3)
                else:
                    forecaster = ForecasterFoundation(estimator=fm)
                    forecaster.fit(series=y)
                    forecaster.predict(steps=3)
                    fm = forecaster.estimator
            mps = [w for w in caught if "MPS" in str(w.message)]
            where = [f"{w.filename.split('skforecast/')[-1]}:{w.lineno}" for w in mps]
            # Lines of this script: 67 is fm.predict, 71 is forecaster.predict, 90 is main().
            print(f"   device={device!r:8s} via {entry:29s} -> warnings {len(mps)}, at {where},"
                  f" model device {fm.adapter._forecast_obj.device}")
    print()

    print("3. dtype of the predictions")
    fm = FoundationModel(model_id=MODEL_ID, context_length=128, device="cpu")
    fm.fit(series=y)
    print("   pred dtype:", fm.predict(steps=3)["pred"].dtype)
    raw = fm.adapter.predict(steps=3, context={"y": y}, context_exog=None, exog=None,
                             quantiles=[0.1, 0.9])
    print("   adapter.predict array dtype:", raw["y"].dtype, raw["y"].shape)


if __name__ == "__main__":
    main()
