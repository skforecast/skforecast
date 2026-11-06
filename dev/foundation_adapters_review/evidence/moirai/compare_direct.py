"""
Correctness checks of `MoiraiAdapter` against a direct call to uni2ts.

Compares the predictions returned through `FoundationModel` with a direct call
to `Moirai2Forecast.predict` on the same input, and probes the capabilities the
adapter declares (NaN in the series, series of different length in one call,
quantile order, long horizons, long contexts, routing of model ids).

No timing is measured: this is a correctness check only.

Usage (from the repository root):
    ~/.venvs/fm-review-moirai/bin/python \
        dev/foundation_adapters_review/evidence/moirai/compare_direct.py
"""

from __future__ import annotations
import os
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import MoiraiAdapter, _resolve_adapter

MODEL_ID = "Salesforce/moirai-2.0-R-small"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def print_versions() -> None:
    print("python", platform.python_version())
    for package in ("skforecast", "uni2ts", "torch", "numpy", "pandas", "scipy",
                    "huggingface_hub", "gluonts"):
        print(f"{package} {version(package)}")
    print("cpus", os.cpu_count(), platform.system(), platform.machine())
    print()


def make_series(n: int, seed: int, freq: str = "h") -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq=freq)
    hour = index.hour.to_numpy()
    values = (
        100 + 0.05 * np.arange(n) + 10 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 1, n)
    )
    return pd.Series(values, index=index, name=f"s{seed}")


def direct_forecast(module, context_length: int, arrays: list[np.ndarray],
                    steps: int) -> np.ndarray:
    """Direct call to uni2ts, as in the README of uni2ts."""
    from uni2ts.model.moirai2 import Moirai2Forecast

    model = Moirai2Forecast(
        module=module,
        prediction_length=steps,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).eval()
    return model.predict(arrays)


def model_quantiles(fm: FoundationModel, series: dict, steps: int) -> dict:
    fm.fit(series=series)
    pred = fm.predict(steps=steps, quantiles=QUANTILES)
    out = {}
    for level in series:
        out[level] = pred.loc[pred["level"] == level, [f"q_{q}" for q in QUANTILES]].to_numpy()
    return out


def section(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    warnings.simplefilter("ignore")
    print_versions()

    section("1. Routing of model ids (_resolve_adapter)")
    for model_id in [
        "Salesforce/moirai-2.0-R-small",
        "Salesforce/moirai-2.0-R-base",
        "Salesforce/moirai-2.0-R-large",
        "Salesforce/moirai-1.1-R-small",
        "Salesforce/moirai-moe-1.0-R-small",
        "Salesforce/moirai-agent",
        "salesforce/moirai-2.0-R-small",
        "ChengShengchuan/moirai-2.0-R-small",
    ]:
        try:
            print(f"{model_id:40s} -> {_resolve_adapter(model_id).__name__}")
        except ValueError as exc:
            print(f"{model_id:40s} -> ValueError: {str(exc)[:60]}")
    print()

    section("2. Quantile levels of the checkpoint vs SUPPORTED_QUANTILES")
    fm = FoundationModel(model_id=MODEL_ID, context_length=512)
    s1 = make_series(700, 1)
    fm.fit(series={"s1": s1})
    fm.predict(steps=3)
    module = fm.adapter._module
    print("module.quantile_levels  :", list(module.quantile_levels))
    print("SUPPORTED_QUANTILES     :", MoiraiAdapter.SUPPORTED_QUANTILES)
    print("equal                   :",
          list(module.quantile_levels) == MoiraiAdapter.SUPPORTED_QUANTILES)
    print("patch_size, num_predict_token, max_seq_len:",
          module.patch_size, module.num_predict_token, module.max_seq_len)
    print("forecast obj device     :", fm.adapter._forecast_obj.device)
    print("param dtype             :", next(module.parameters()).dtype)
    print()

    section("3. Adapter vs direct uni2ts call (same input, same context_length)")
    cases = {
        "equal length, 2 series, steps=24": (
            {"a": make_series(700, 1), "b": make_series(700, 2)}, 24, 512),
        "different lengths (700, 300, 40), steps=24": (
            {"a": make_series(700, 1), "b": make_series(300, 2),
             "c": make_series(40, 3)}, 24, 512),
        "long horizon steps=150 (> num_predict_token * patch_size = 64)": (
            {"a": make_series(700, 1)}, 150, 512),
        "default context_length=2048, series of 3000": (
            {"a": make_series(3000, 4)}, 48, 2048),
    }
    for name, (series, steps, ctx) in cases.items():
        fm = FoundationModel(model_id=MODEL_ID, context_length=ctx)
        fm.adapter._module = module
        adapter_q = model_quantiles(fm, series, steps)
        arrays = [
            s.iloc[-ctx:].to_numpy(dtype=np.float64).reshape(-1, 1)
            for s in series.values()
        ]
        raw = direct_forecast(module, ctx, arrays, steps)
        print(f"{name}: raw shape {raw.shape}")
        for i, level in enumerate(series):
            direct = raw[i].T
            diff = np.max(np.abs(adapter_q[level] - direct))
            print(f"  {level}: adapter {adapter_q[level].shape} max|adapter - direct| = {diff:.3e}")
        # Batch independence: each series alone vs in the batch.
        for i, (level, s) in enumerate(series.items()):
            alone = direct_forecast(module, ctx, [arrays[i]], steps)[0].T
            diff = np.max(np.abs(adapter_q[level] - alone))
            print(f"  {level}: max|batched - alone| = {diff:.3e}")
    print()

    section("4. NaN in the series (supports_nan_in_series=True)")
    s = make_series(600, 5)
    s_nan = s.copy()
    s_nan.iloc[100:130] = np.nan
    s_nan.iloc[-10:-5] = np.nan
    fm = FoundationModel(model_id=MODEL_ID, context_length=512)
    fm.adapter._module = module
    q_nan = model_quantiles(fm, {"x": s_nan}, 24)["x"]
    raw = direct_forecast(module, 512, [s_nan.iloc[-512:].to_numpy().reshape(-1, 1)], 24)
    print("interior NaN: finite:", np.isfinite(q_nan).all(),
          "| max|adapter - direct| =", f"{np.max(np.abs(q_nan - raw[0].T)):.3e}")
    s_lead = s.copy()
    s_lead.iloc[:50] = np.nan
    try:
        q_lead = model_quantiles(fm, {"x": s_lead}, 24)["x"]
        print("leading NaN: finite:", np.isfinite(q_lead).all())
    except Exception as exc:  # noqa: BLE001
        print("leading NaN:", type(exc).__name__, str(exc)[:200])
    print()

    section("5. Very short series and long contexts")
    for n, ctx in [(5, 512), (17, 512), (9000, 8192)]:
        fm = FoundationModel(model_id=MODEL_ID, context_length=ctx)
        fm.adapter._module = module
        try:
            q = model_quantiles(fm, {"x": make_series(n, 6)}, 24)["x"]
            print(f"n={n}, context_length={ctx}: shape {q.shape}, finite {np.isfinite(q).all()}")
        except Exception as exc:  # noqa: BLE001
            print(f"n={n}, context_length={ctx}: {type(exc).__name__}: {str(exc)[:200]}")
    print()

    section("6. Point forecast column equals the 0.5 quantile")
    fm = FoundationModel(model_id=MODEL_ID, context_length=512)
    fm.adapter._module = module
    fm.fit(series={"a": make_series(700, 1)})
    point = fm.predict(steps=24)["pred"].to_numpy()
    q = fm.predict(steps=24, quantiles=[0.5])["q_0.5"].to_numpy()
    print("max|pred - q_0.5| =", f"{np.max(np.abs(point - q)):.3e}")
    print()


if __name__ == "__main__":
    main()
