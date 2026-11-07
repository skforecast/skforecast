"""
Correctness checks of ChronosAdapter against a direct call to the
chronos-forecasting backend on the same input.

Usage (from the repository root):
    ~/.venvs/fm-review-chronos/bin/python \
        dev/foundation_adapters_review/evidence/chronos/compare_direct.py [model_id ...]

Every check prints the maximum absolute difference between the adapter output
(through FoundationModel, as a user calls it) and the direct backend call, or
the behaviour observed. No timing and no accuracy comparisons.
"""

import importlib.metadata as md
import logging
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from skforecast.foundation import FoundationModel

MODEL_IDS = sys.argv[1:] or ["autogluon/chronos-2-small"]
QUANTILES = [0.1, 0.5, 0.9]
STEPS = 24
CONTEXT_LENGTH = 256

print("python", platform.python_version(), platform.machine(), platform.system())
for package in [
    "skforecast", "chronos-forecasting", "torch", "transformers", "accelerate",
    "numpy", "pandas", "huggingface-hub",
]:
    print(f"{package:20s} {md.version(package)}")
print("torch threads", torch.get_num_threads(), "cuda", torch.cuda.is_available())
print()

# Data: three hourly series of different lengths, one with NaN, a numeric
# covariate, a string covariate and a bool covariate.
rng = np.random.default_rng(42)
n_total = 400
index = pd.date_range("2024-01-01", periods=n_total + STEPS, freq="h")
hour = index.hour.to_numpy()
exog_all = pd.DataFrame(
    {
        "temp": (np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(index))),
        "kind": np.where(hour < 8, "night", np.where(hour < 18, "day", "evening")),
        "flag": (hour % 7 == 0),
    },
    index=index,
)
base = 50 + 5 * np.sin(2 * np.pi * hour / 24) + 3 * exog_all["temp"].to_numpy()
series_all = {
    "s1": pd.Series(base + rng.normal(0, 0.5, len(index)), index=index),
    "s2": pd.Series(20 + 0.5 * base + rng.normal(0, 0.3, len(index)), index=index),
    "s3": pd.Series(100 - 0.3 * base + rng.normal(0, 1.0, len(index)), index=index),
}
lengths = {"s1": n_total, "s2": 300, "s3": 180}
context = {
    k: v.iloc[n_total - lengths[k]:n_total].copy() for k, v in series_all.items()
}
context["s2"].iloc[50:60] = np.nan
for name in context:
    context[name].name = name
context_exog = {
    k: exog_all.loc[context[k].index] for k in context
}
future_exog = {k: exog_all.iloc[n_total:] for k in context}


def as_backend_dict(y, x_past=None, x_future=None):
    """
    Build the backend input independently of the adapter code. The context is
    trimmed to the last CONTEXT_LENGTH observations, as FoundationModel does.
    """
    y = y.iloc[-CONTEXT_LENGTH:]
    out = {"target": y.to_numpy(dtype=np.float32)}
    if x_past is not None:
        x_past = x_past.iloc[-CONTEXT_LENGTH:]
        out["past_covariates"] = {
            "temp": x_past["temp"].to_numpy(dtype=np.float32),
            "kind": x_past["kind"].to_numpy(dtype=object),
            "flag": x_past["flag"].to_numpy(dtype=np.float32),
        }
    if x_future is not None:
        out["future_covariates"] = {
            "temp": x_future["temp"].to_numpy(dtype=np.float32),
            "kind": x_future["kind"].to_numpy(dtype=object),
            "flag": x_future["flag"].to_numpy(dtype=np.float32),
        }
    return out


def adapter_matrix(pred, level):
    """Quantile matrix (steps, n_quantiles) of one level from predict output."""
    cols = [f"q_{q}" for q in QUANTILES]
    return pred.loc[pred["level"] == level, cols].to_numpy(dtype=float)


def direct(pipeline, inputs, quantiles=QUANTILES, **kwargs):
    q, _ = pipeline.predict_quantiles(
        inputs=inputs, prediction_length=STEPS, quantile_levels=quantiles, **kwargs
    )
    return [t.squeeze(0).numpy().astype(float) for t in q]


def report(name, a, b):
    diff = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
    print(f"  {name:62s} max_abs_diff={diff:.3e}")
    return diff


for model_id in MODEL_IDS:
    print(f"==== {model_id}")
    with warnings.catch_warnings(record=True) as load_warnings:
        warnings.simplefilter("always")
        fm = FoundationModel(model_id=model_id, context_length=CONTEXT_LENGTH)
        fm.adapter._load_pipeline()
    pipeline = fm.adapter._pipeline
    print("  pipeline class:", type(pipeline).__name__,
          "| model device:", pipeline.model.device,
          "| model dtype:", pipeline.model.dtype,
          "| hf_device_map:", getattr(pipeline.model, "hf_device_map", None))
    print("  model_context_length:", pipeline.model_context_length,
          "| model_prediction_length:", pipeline.model_prediction_length,
          "| n training quantiles:", len(pipeline.quantiles))
    print("  warnings while loading:", [str(w.message)[:120] for w in load_warnings])

    # A. Single series, no exog.
    pred = fm.predict(steps=STEPS, context=context["s1"], quantiles=QUANTILES)
    ref = direct(pipeline, [as_backend_dict(context["s1"])])[0]
    report("A single series, no exog", adapter_matrix(pred, "s1"), ref)

    # B. Three series, different lengths, NaN, numeric + string + bool exog.
    for cross_learning in [False, True]:
        fm.set_params(cross_learning=cross_learning)
        pred = fm.predict(
            steps=STEPS, context=context, context_exog=context_exog,
            exog=future_exog, quantiles=QUANTILES,
        )
        inputs = [
            as_backend_dict(context[k], context_exog[k], future_exog[k])
            for k in context
        ]
        refs = direct(pipeline, inputs, cross_learning=cross_learning)
        for i, k in enumerate(context):
            report(
                f"B multi-series exog cross_learning={cross_learning} level={k}",
                adapter_matrix(pred, k), refs[i],
            )
    fm.set_params(cross_learning=False)

    if model_id != MODEL_IDS[0]:
        print()
        continue

    # C. pandas Categorical covariate with integer categories.
    print("  C pandas Categorical covariate with integer categories")
    cat_past = pd.DataFrame(
        {"dow": pd.Categorical(context["s1"].index.dayofweek)},
        index=context["s1"].index,
    )
    cat_future = pd.DataFrame(
        {"dow": pd.Categorical(exog_all.index[n_total:].dayofweek)},
        index=exog_all.index[n_total:],
    )
    pred = fm.predict(
        steps=STEPS, context=context["s1"], context_exog=cat_past,
        exog=cat_future, quantiles=QUANTILES,
    )
    y_np = context["s1"].iloc[-CONTEXT_LENGTH:].to_numpy(dtype=np.float32)
    cat_past_trim = cat_past.iloc[-CONTEXT_LENGTH:]
    ref_numeric = direct(pipeline, [{
        "target": y_np,
        "past_covariates": {"dow": cat_past_trim["dow"].to_numpy().astype(np.float32)},
        "future_covariates": {"dow": cat_future["dow"].to_numpy().astype(np.float32)},
    }])[0]
    ref_categorical = direct(pipeline, [{
        "target": y_np,
        "past_covariates": {"dow": cat_past_trim["dow"].astype(str).to_numpy(dtype=object)},
        "future_covariates": {"dow": cat_future["dow"].astype(str).to_numpy(dtype=object)},
    }])[0]
    print("    dtype of the array the adapter builds:",
          fm.adapter._to_covariate_array(cat_past["dow"]).dtype)
    report("C adapter vs backend with dow as numeric", adapter_matrix(pred, "s1"), ref_numeric)
    report("C adapter vs backend with dow as category (str)", adapter_matrix(pred, "s1"), ref_categorical)

    # D. Same column name, numeric in one series and string in another,
    # in the same backend call (same covariate signature, so same group).
    print("  D same covariate name, numeric in s1 and string in s3, one group")
    past_d = {
        "s1": context_exog["s1"][["temp"]],
        "s3": context_exog["s3"][["kind"]].rename(columns={"kind": "temp"}),
    }
    fut_d = {
        "s1": future_exog["s1"][["temp"]],
        "s3": future_exog["s3"][["kind"]].rename(columns={"kind": "temp"}),
    }
    ctx_d = {"s1": context["s1"], "s3": context["s3"]}
    try:
        fm.predict(
            steps=STEPS, context=ctx_d, context_exog=past_d, exog=fut_d,
            quantiles=QUANTILES,
        )
        print("    no error raised")
    except TypeError as exc:
        print("    FoundationModel raises TypeError:", str(exc)[:90])
    pred_alone = fm.predict(
        steps=STEPS, context=context["s1"], context_exog=past_d["s1"],
        exog=fut_d["s1"], quantiles=QUANTILES,
    )
    past_same = {"s1": context_exog["s1"][["temp"]], "s3": context_exog["s3"][["temp"]]}
    fut_same = {"s1": future_exog["s1"][["temp"]], "s3": future_exog["s3"][["temp"]]}
    pred_pair_numeric = fm.predict(
        steps=STEPS, context=ctx_d, context_exog=past_same, exog=fut_same,
        quantiles=QUANTILES,
    )
    report("D batch independence: s1 with s3 (both numeric) vs s1 alone",
           adapter_matrix(pred_pair_numeric, "s1"), adapter_matrix(pred_alone, "s1"))

    # E. Quantiles outside the trained range [0.01, 0.99].
    print("  E quantiles outside the trained range")
    chronos_logger = logging.getLogger("chronos.chronos2.pipeline")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    chronos_logger.addHandler(handler)
    with warnings.catch_warnings(record=True) as w_e:
        warnings.simplefilter("always")
        pred_e = fm.predict(steps=STEPS, context=context["s1"], quantiles=[0.001, 0.01, 0.99, 0.999])
    chronos_logger.removeHandler(handler)
    v = pred_e[["q_0.001", "q_0.01", "q_0.99", "q_0.999"]].to_numpy(dtype=float)
    report("E q_0.001 vs q_0.01 (clamped by the backend)", v[:, 0], v[:, 1])
    report("E q_0.999 vs q_0.99 (clamped by the backend)", v[:, 3], v[:, 2])
    print("    python warnings:", [str(x.message)[:80] for x in w_e])
    print("    backend log records:", [r.getMessage().strip()[:80] for r in records])

    # F. Horizon above the model prediction length.
    print("  F steps above model_prediction_length")
    long_steps = pipeline.model_prediction_length + 16
    with warnings.catch_warnings(record=True) as w_f:
        warnings.simplefilter("always")
        pred_f = fm.predict(steps=long_steps, context=context["s1"])
    print("    rows:", len(pred_f), "| finite:", bool(np.isfinite(pred_f["pred"]).all()))
    print("    warnings:", sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w_f}))

    # G. cross_learning and the backend batch_size.
    print("  G cross_learning=True, effect of the backend batch_size")
    rng_g = np.random.default_rng(7)
    ctx_g = {
        "g0": series_all["s1"].iloc[n_total - 60:n_total],
        "g1": series_all["s2"].iloc[n_total - 60:n_total],
        "g2": series_all["s3"].iloc[n_total - 60:n_total],
        "g3": pd.Series(
            np.cumsum(rng_g.normal(0, 1, 60)) + 30, index=index[n_total - 60:n_total]
        ),
    }
    fm.set_params(cross_learning=True)
    pred_g256 = fm.predict(steps=STEPS, context=ctx_g, quantiles=QUANTILES)
    fm.set_params(predict_kwargs={"batch_size": 2})
    pred_g2 = fm.predict(steps=STEPS, context=ctx_g, quantiles=QUANTILES)
    fm.set_params(cross_learning=False, predict_kwargs=None)
    pred_gnone = fm.predict(steps=STEPS, context=ctx_g, quantiles=QUANTILES)
    report("G g0 cross_learning batch_size=256 vs batch_size=2",
           adapter_matrix(pred_g256, "g0"), adapter_matrix(pred_g2, "g0"))
    report("G g0 cross_learning batch_size=2 vs cross_learning=False",
           adapter_matrix(pred_g2, "g0"), adapter_matrix(pred_gnone, "g0"))
    report("G g3 cross_learning batch_size=2 vs cross_learning=False",
           adapter_matrix(pred_g2, "g3"), adapter_matrix(pred_gnone, "g3"))

    # H. dtype and device arguments.
    print("  H torch_dtype=torch.bfloat16 and device_map='cpu'")
    with warnings.catch_warnings(record=True) as w_h:
        warnings.simplefilter("always")
        fm_bf16 = FoundationModel(
            model_id=model_id, context_length=CONTEXT_LENGTH,
            torch_dtype=torch.bfloat16, device_map="cpu",
        )
        pred_bf16 = fm_bf16.predict(steps=STEPS, context=context["s1"], quantiles=QUANTILES)
    p16 = fm_bf16.adapter._pipeline
    print("    model dtype:", p16.model.dtype, "| device:", p16.model.device)
    print("    warnings:", sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w_h}))
    raw = fm_bf16.adapter.predict(
        steps=STEPS, context={"s1": context["s1"]}, context_exog=None,
        exog=None, quantiles=QUANTILES,
    )
    print("    adapter output dtype:", raw["s1"].dtype, "shape:", raw["s1"].shape)
    ref16 = direct(p16, [as_backend_dict(context["s1"])])[0]
    report("H bf16 adapter vs direct bf16", adapter_matrix(pred_bf16, "s1"), ref16)
    print()
