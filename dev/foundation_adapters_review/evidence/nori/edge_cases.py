"""
Edge cases of NoriAdapter with the real backend (Synthefy/Nori, 6M):
steps=1, duplicated and unsorted quantiles, RangeIndex series, two series
with different covariates and lengths, categorical covariate, `point_estimate`
values, device requests that the machine cannot serve, and the console output
of the backend.

Run from the repository root:
    ~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/edge_cases.py [model_id]
"""

import contextlib
import io
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Synthefy/Nori"
for pkg in ["synthefy-nori", "torch", "numpy", "pandas", "skforecast"]:
    print(f"{pkg:14s}:", version(pkg))
print("model_id      :", MODEL_ID)
print()

rng = np.random.default_rng(2)
idx = pd.date_range("2024-01-01", periods=212, freq="h")
temp = pd.Series(np.cos(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, len(idx)), index=idx, name="temp")
y = pd.Series(30 + 4 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.3, len(idx)), index=idx, name="y")
y_train, exog_train, exog_fut = y.iloc[:200], temp.iloc[:200].to_frame(), temp.iloc[200:].to_frame()


def run(name, func):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                out = func()
            print(f"[ok]  {name}: {out}")
        except Exception as exc:
            print(f"[err] {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        stdout = buf.getvalue().strip()
        if stdout:
            print(f"      stdout of the backend: {stdout.splitlines()[0][:120]!r} ({len(stdout.splitlines())} lines)")
        for w in caught:
            print(f"      warning {w.category.__name__}: {str(w.message)[:120]}")


fm = FoundationModel(model_id=MODEL_ID, context_length=200)
fm.fit(series=y_train, exog=exog_train)

run("steps=1 point shape", lambda: fm.predict(steps=1, exog=exog_fut.iloc[:1]).shape)
run("steps=1 quantiles shape", lambda: fm.predict(steps=1, exog=exog_fut.iloc[:1], quantiles=[0.1, 0.9]).shape)
run(
    "duplicated + unsorted quantiles [0.9, 0.5, 0.5, 0.1]",
    lambda: fm.predict(steps=3, exog=exog_fut.iloc[:3], quantiles=[0.9, 0.5, 0.5, 0.1]).iloc[0, 1:].astype(float).round(3).to_dict(),
)
run("quantile 0.0 (accepted by FoundationModel, rejected by the adapter)",
    lambda: fm.predict(steps=3, exog=exog_fut.iloc[:3], quantiles=[0.0, 0.5]).shape)


def point_estimates():
    out = {}
    for pe in ["mean", "median", "mode"]:
        f = FoundationModel(model_id=MODEL_ID, context_length=200, point_estimate=pe)
        f.fit(series=y_train)
        out[pe] = f.predict(steps=3)["pred"].round(4).tolist()
    return out


run("point_estimate mean / median / mode", point_estimates)


def range_index():
    f = FoundationModel(model_id=MODEL_ID, context_length=200)
    f.fit(series=y_train.reset_index(drop=True))
    return f.predict(steps=3)["pred"].round(3).tolist()


run("RangeIndex series", range_index)


def heterogeneous():
    s2 = (y_train.iloc[50:] * 2).rename("y2")
    f = FoundationModel(model_id=MODEL_ID, context_length=200)
    f.fit(series={"y": y_train, "y2": s2}, exog={"y": exog_train, "y2": None})
    p = f.predict(steps=3, exog={"y": exog_fut.iloc[:3], "y2": None})
    return p.shape, sorted(p["level"].unique())


run("two series, different length and covariates", heterogeneous)


def categorical():
    cat_train = pd.DataFrame({"temp": exog_train["temp"], "kind": pd.Categorical(np.where(idx[:200].hour < 12, "am", "pm"))}, index=idx[:200])
    cat_fut = pd.DataFrame({"temp": exog_fut["temp"], "kind": pd.Categorical(np.where(idx[200:].hour < 12, "am", "pm"))}, index=idx[200:])
    f = FoundationModel(model_id=MODEL_ID, context_length=200)
    f.fit(series=y_train, exog=cat_train)
    return f.predict(steps=3, exog=cat_fut.iloc[:3]).shape


run("categorical covariate", categorical)


def device(dev):
    def _f():
        f = FoundationModel(model_id=MODEL_ID, context_length=200, nori_config={"device": dev})
        f.fit(series=y_train)
        return f.predict(steps=2).shape
    return _f


run("nori_config device='cuda' on a CPU machine", device("cuda"))
run("nori_config device='mps' on a CPU machine", device("mps"))
