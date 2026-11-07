"""
Check whether the TimesFM 3.0 forecast of a series depends on the other series
of the same `predict_batch` call (backend left-padding to the batch context).

Run from the repository root:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/batch_invariance.py
"""

from __future__ import annotations
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

MODEL_ID = "google/timesfm-3.0-pytorch"
STEPS = 48


def make_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (50 + 0.02 * t + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)).astype(np.float32)


def forecast(model, contexts):
    return [o.forecast for o in model.predict_batch(contexts=contexts, horizon=STEPS)]


def rel(a, b) -> str:
    d = float(np.max(np.abs(a - b)))
    return f"maxabs={d:.5f} rel_to_mean={d / float(np.mean(np.abs(a))):.5f}"


def main() -> None:
    print("python", platform.python_version())
    for package in ("skforecast", "timesfm", "torch", "numpy", "pandas"):
        print(package, version(package))
    warnings.simplefilter("ignore")

    fm = FoundationModel(model_id=MODEL_ID, context_length=2048, device="cpu")
    fm.adapter._load_model()
    model = fm.adapter._model
    print("per_core_batch_size", model.config.per_core_batch_size, "input_patch_length", model.config.input_patch_length)
    print()

    print("== 1. Same length (512) in one batch vs alone")
    ctxs = [make_series(512, s) for s in range(4)]
    together = forecast(model, ctxs)
    for i, c in enumerate(ctxs):
        print(f"  series {i}:", rel(forecast(model, [c])[0], together[i]))
    print()

    print("== 2. Short series alone vs batched with one longer series")
    for n_short, n_long in ((100, 120), (100, 128), (100, 129), (100, 200), (100, 512), (100, 2048), (37, 512), (300, 512), (500, 512)):
        short = make_series(n_short, 1)
        long_ = make_series(n_long, 2)
        alone = forecast(model, [short])[0]
        mixed = forecast(model, [short, long_])[0]
        ctx_alone = int(np.ceil(n_short / 32) * 32)
        ctx_mixed = int(np.ceil(max(n_short, n_long) / 32) * 32)
        print(f"  short={n_short:<4} long={n_long:<5} batch_ctx {ctx_alone}->{ctx_mixed}:", rel(alone, mixed))
    print()

    print("== 3. Through FoundationModel: predict(levels=[x]) vs predict() filtered")
    idx = pd.date_range("2024-01-01", periods=2048, freq="h")
    series = {}
    for i, n in enumerate((2048, 1000, 300, 100)):
        series[f"s{n}"] = pd.Series(make_series(n, 10 + i), index=idx[-n:])
    fm.fit(series=series)
    p_all = fm.predict(steps=STEPS)
    for name in series:
        p_one = fm.predict(steps=STEPS, levels=[name])
        a = p_all.loc[p_all["level"] == name, "pred"].to_numpy()
        print(f"  {name:<6}", rel(p_one["pred"].to_numpy(), a))


if __name__ == "__main__":
    main()
