"""
Compare the output of TimesFM3Adapter (through FoundationModel) with direct
calls to `timesfm.TimesFM3Forecaster.predict_batch` on the same inputs, and
check the capabilities declared by the adapter against the real backend.

Run from the repository root with the review environment:
    ~/.venvs/fm-review-timesfm3/bin/python \
        dev/foundation_adapters_review/evidence/timesfm3/compare_direct.py
"""

from __future__ import annotations
import platform
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import _resolve_adapter

MODEL_ID = "google/timesfm-3.0-pytorch"


def header() -> None:
    import torch

    print("python", platform.python_version())
    for package in ("skforecast", "timesfm", "torch", "numpy", "pandas", "huggingface_hub"):
        print(package, version(package))
    print("torch threads", torch.get_num_threads())
    print()


def make_series(n: int, seed: int, start: str = "2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h")
    t = np.arange(n)
    values = 50 + 0.02 * t + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx, name=f"s{seed}")


def make_exog(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = index.hour.to_numpy()
    return pd.DataFrame(
        {
            "fut": np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(index)),
            "past": rng.normal(0, 1, len(index)),
        },
        index=index,
    )


def maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    header()
    warnings.simplefilter("ignore")

    print("== Registry routing")
    for model_id in (
        "google/timesfm-3.0-pytorch",
        "google/timesfm-3.0-flax",
        "google/timesfm-3.5-pytorch",
        "google/timesfm-2.5-200m-transformers",
    ):
        try:
            print(f"  {model_id} -> {_resolve_adapter(model_id).__name__}")
        except ValueError as exc:
            print(f"  {model_id} -> ValueError: {str(exc)[:80]}")
    print()

    fm = FoundationModel(model_id=MODEL_ID, context_length=512, device="cpu")
    adapter = fm.adapter
    t0 = time.perf_counter()
    adapter._load_model()
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s")
    model = adapter._model
    print("config.quantiles", list(model.config.quantiles))
    print("config.median_quantile_index", model.config.median_quantile_index)
    print("config.per_core_batch_size", model.config.per_core_batch_size)
    print("global_context", model.global_context)
    print("device", model.device, "dtype", next(model.model.parameters()).dtype)
    print()

    # A. Single series, no exog, point and all quantiles
    print("== A. Single series, no exog (steps=24)")
    y = make_series(400, 1)
    fm.fit(series=y)
    pred = fm.predict(steps=24)
    pred_q = fm.predict(steps=24, quantiles=adapter.SUPPORTED_QUANTILES)
    ctx = y.iloc[-512:].to_numpy()
    out = list(model.predict_batch(contexts=[ctx], horizon=24, return_quantiles=True, padding_mode="none"))[0]
    print("  raw quantiles shape", np.asarray(out.quantiles).shape)
    print("  point vs direct forecast   maxdiff", maxdiff(pred["pred"], out.forecast))
    q_cols = [f"q_{q}" for q in adapter.SUPPORTED_QUANTILES]
    print("  quantiles vs direct        maxdiff", maxdiff(pred_q[q_cols].to_numpy(), out.quantiles))
    print("  q_0.5 vs point             maxdiff", maxdiff(pred_q["q_0.5"], pred["pred"]))
    single = model.predict(context=ctx, horizon=24)
    print("  point vs TimesFM3Forecaster.predict maxdiff", maxdiff(pred["pred"], single.forecast))
    print("  quantiles monotone", bool(np.all(np.diff(pred_q[q_cols].to_numpy(), axis=1) >= 0)))
    print()

    # B. Horizons not multiple of the output patch (64)
    print("== B. Horizons")
    for steps in (1, 63, 64, 65, 200):
        p = fm.predict(steps=steps)
        print(f"  steps={steps:<4} rows={len(p)} finite={bool(np.isfinite(p['pred']).all())}")
    print()

    # C. Batch invariance with series of different lengths
    print("== C. Multi-series of different lengths: batch vs one by one")
    series = {f"s{i}": make_series(n, i) for i, n in enumerate((100, 300, 450, 37, 512, 260))}
    fm.fit(series=series)
    p_all = fm.predict(steps=48)
    worst = 0.0
    for name in series:
        p_one = fm.predict(steps=48, levels=[name])
        worst = max(worst, maxdiff(p_all.loc[p_all["level"] == name, "pred"], p_one["pred"]))
    print("  max |batch - single| over 6 series", worst)
    ctxs = [s.iloc[-512:].to_numpy() for s in series.values()]
    direct = [o.forecast for o in model.predict_batch(contexts=ctxs, horizon=48)]
    d = max(
        maxdiff(p_all.loc[p_all["level"] == name, "pred"], direct[i])
        for i, name in enumerate(series)
    )
    print("  adapter vs direct predict_batch maxdiff", d)
    for bs in (1, 4, 16):
        model.config.per_core_batch_size = bs
        alt = [o.forecast for o in model.predict_batch(contexts=ctxs, horizon=48)]
        print(f"  per_core_batch_size={bs:<3} vs default maxdiff", max(maxdiff(a, b) for a, b in zip(alt, direct)))
    model.config.per_core_batch_size = 4
    print()

    # D. Exogenous variables: past-only + known-future
    print("== D. Exog: past-only 'past' + known-future 'fut' (steps=24)")
    y = make_series(400, 7)
    full_idx = pd.date_range(y.index[0], periods=len(y) + 24, freq="h")
    ex = make_exog(full_idx, 7)
    ex_hist, ex_fut = ex.iloc[:400], ex.iloc[400:][["fut"]]
    fm.fit(series=y, exog=ex_hist)
    p = fm.predict(steps=24, exog=ex_fut)
    ctx = y.iloc[-512:].to_numpy()
    po = [ex_hist[["past"]].to_numpy(dtype=np.float32).T]
    pf = [np.concatenate([ex_hist["fut"].to_numpy(), ex_fut["fut"].to_numpy()]).astype(np.float32)[None, :]]
    edge = list(model.predict_batch(contexts=[ctx], horizon=24, past_only_covariates=po, past_future_covariates=pf, padding_mode="edge"))[0]
    none = list(model.predict_batch(contexts=[ctx], horizon=24, past_only_covariates=po, past_future_covariates=pf, padding_mode="none"))[0]
    print("  adapter vs direct edge     maxdiff", maxdiff(p["pred"], edge.forecast))
    print("  direct edge vs none        maxdiff", maxdiff(edge.forecast, none.forecast),
          "rel", maxdiff(edge.forecast, none.forecast) / float(np.mean(np.abs(edge.forecast))))
    for steps in (64, 100):
        ex_f = ex.reindex(pd.date_range(y.index[0], periods=len(y) + steps, freq="h")).ffill().iloc[400:][["fut"]]
        pp = fm.predict(steps=steps, exog=ex_f)
        print(f"  exog steps={steps:<4} rows={len(pp)} finite={bool(np.isfinite(pp['pred']).all())}")
    print()

    # E. Exog dtypes: int and bool columns
    print("== E. Exog dtypes int64 / bool / category")
    ex2 = ex.copy()
    ex2["fut_int"] = (ex2["fut"] > 0).astype("int64")
    ex2["fut_bool"] = ex2["fut"] > 0
    fm.fit(series=y, exog=ex2.iloc[:400][["fut_int", "fut_bool"]])
    pp = fm.predict(steps=24, exog=ex2.iloc[400:][["fut_int", "fut_bool"]])
    print("  int/bool finite", bool(np.isfinite(pp["pred"]).all()))
    ex2["cat"] = pd.Categorical(np.where(ex2["fut"] > 0, "a", "b"))
    fm.fit(series=y, exog=ex2.iloc[:400][["cat"]])
    try:
        fm.predict(steps=24, exog=ex2.iloc[400:][["cat"]])
        print("  category accepted")
    except ValueError as exc:
        print("  category -> ValueError:", str(exc)[:110])
    print()

    # F. NaN in the target
    print("== F. NaN in the series")
    y_nan = make_series(400, 3).copy()
    y_nan.iloc[100:110] = np.nan
    y_nan.iloc[:20] = np.nan
    fm.fit(series=y_nan)
    pp = fm.predict(steps=24)
    print("  inner + leading NaN finite", bool(np.isfinite(pp["pred"]).all()))
    y_nan.iloc[-5:] = np.nan
    fm.fit(series=y_nan)
    pp = fm.predict(steps=24)
    print("  trailing NaN finite", bool(np.isfinite(pp["pred"]).all()))
    print()

    # G. Many covariates (config.max_variates = 32)
    print("== G. 40 known-future covariates (config max_variates=32)")
    y = make_series(300, 11)
    idx = pd.date_range(y.index[0], periods=324, freq="h")
    rng = np.random.default_rng(0)
    many = pd.DataFrame(rng.normal(size=(324, 40)), index=idx, columns=[f"c{i}" for i in range(40)])
    fm.fit(series=y, exog=many.iloc[:300])
    try:
        pp = fm.predict(steps=24, exog=many.iloc[300:])
        print("  40 covariates finite", bool(np.isfinite(pp["pred"]).all()))
    except Exception as exc:  # noqa: BLE001
        print("  40 covariates ->", type(exc).__name__, str(exc)[:150])
    print()

    # H. Low precision (bfloat16) on CPU
    print("== H. bfloat16 weights on CPU")
    import torch

    try:
        model.model.to(torch.bfloat16)
        list(model.predict_batch(contexts=[make_series(300, 1).to_numpy()], horizon=24))
        print("  bfloat16 ran")
    except Exception as exc:  # noqa: BLE001
        print("  bfloat16 ->", type(exc).__name__, str(exc)[:200])
    finally:
        model.model.to(torch.float32)


if __name__ == "__main__":
    main()
