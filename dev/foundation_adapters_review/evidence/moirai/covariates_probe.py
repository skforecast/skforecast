"""
Probe whether `Moirai2Forecast.predict` of the installed uni2ts accepts
covariates (`feat_dynamic_real`, known in the future, and
`past_feat_dynamic_real`, past only) for Moirai-2 checkpoints.

It only checks whether the call runs and whether the output changes when an
informative covariate is added; it does not evaluate accuracy.

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/covariates_probe.py
"""

from __future__ import annotations
import platform
import traceback
import warnings
from importlib.metadata import version

import numpy as np

MODEL_ID = "Salesforce/moirai-2.0-R-small"
CTX = 256
STEPS = 24
T = 400


def build(module, feat_dim: int, past_feat_dim: int):
    from uni2ts.model.moirai2 import Moirai2Forecast

    return Moirai2Forecast(
        module=module,
        prediction_length=STEPS,
        context_length=CTX,
        target_dim=1,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=past_feat_dim,
    ).eval()


def run(name, fn):
    try:
        out = fn()
        print(f"{name}: OK, shape {out.shape}")
        return out
    except Exception as exc:  # noqa: BLE001
        last = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
        print(f"    raised at {last.filename.split('site-packages/')[-1]}:{last.lineno}")
        return None


def main() -> None:
    warnings.simplefilter("ignore")
    print("python", platform.python_version(), "| uni2ts", version("uni2ts"),
          "| torch", version("torch"), "| numpy", version("numpy"))
    from uni2ts.model.moirai2 import Moirai2Module

    module = Moirai2Module.from_pretrained(MODEL_ID).eval()

    rng = np.random.default_rng(0)
    n = T + STEPS
    cov = rng.normal(0, 1, n).cumsum()
    y = 50 + 5 * cov + rng.normal(0, 0.1, n)
    past = y[:T].reshape(-1, 1)
    cov_full = cov.reshape(-1, 1)          # past + future
    cov_past = cov[:T].reshape(-1, 1)      # past only

    base = run("no covariates", lambda: build(module, 0, 0).predict([past]))
    fut = run(
        "feat_dynamic_real (len T + steps)",
        lambda: build(module, 1, 0).predict([past], feat_dynamic_real=[cov_full]),
    )
    fut_ctx = run(
        "feat_dynamic_real (len T)",
        lambda: build(module, 1, 0).predict([past], feat_dynamic_real=[cov_past]),
    )
    pst = run(
        "past_feat_dynamic_real (len T)",
        lambda: build(module, 0, 1).predict([past], past_feat_dynamic_real=[cov_past]),
    )
    for name, out in [("feat_dynamic_real (len T + steps)", fut),
                      ("feat_dynamic_real (len T)", fut_ctx),
                      ("past_feat_dynamic_real (len T)", pst)]:
        if out is not None and base is not None and out.shape == base.shape:
            print(f"max|{name} - no covariates| = {np.max(np.abs(out - base)):.3e}")


if __name__ == "__main__":
    main()
