"""
Probe the capability attributes of TabICLAdapter against tabicl 2.x behaviour.

Each probe prints OK/ERROR with the exception type and message so that the log
can be re-checked. Run from the repository root:
    ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/capabilities_check.py
"""

import contextlib
import importlib.metadata as md
import io
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tabicl.forecast import TabICLForecaster

from skforecast.foundation import FoundationModel

print(
    "python", sys.version.split()[0], "| tabicl", md.version("tabicl"),
    "| torch", torch.__version__, "| pandas", pd.__version__,
    "| skforecast", md.version("skforecast"),
)

STEPS = 6
rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=120 + STEPS, freq="h")
y_all = pd.Series(10 + np.sin(np.arange(len(idx)) / 4) + rng.normal(0, 0.1, len(idx)), index=idx, name="y")
y = y_all.iloc[:120]
exog_all = pd.DataFrame({"x": np.cos(np.arange(len(idx)) / 4)}, index=idx)
exog, exog_future = exog_all.iloc[:120], exog_all.iloc[120:]


def probe(name, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
            msg = f"OK {out}" if out is not None else "OK"
        except Exception as exc:  # noqa: BLE001
            msg = f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
    cats = sorted({f"{x.category.__name__}: {str(x.message)[:110]}" for x in w
                   if x.category.__name__ not in ("DeprecationWarning", "FutureWarning")})
    print(f"[{name}] {msg}")
    for c in cats:
        print(f"    warning -> {c}")


def fm(**kw):
    m = FoundationModel("soda-inria/tabicl", context_length=64, **kw)
    return m


# 1. Quantile levels given as int (accepted by FoundationModel validation).
def q_int():
    m = fm(); m.fit(series=y)
    p = m.predict(steps=STEPS, quantiles=[0, 0.5, 1])
    return p.columns.tolist()
probe("quantiles=[0, 0.5, 1] (int levels)", q_int)

# 2. Quantile levels as numpy float32.
def q_f32():
    m = fm(); m.fit(series=y)
    p = m.predict(steps=STEPS, quantiles=list(np.array([0.1, 0.9], dtype=np.float32)))
    return p.columns.tolist()
probe("quantiles as numpy float32", q_f32)

# 3. Quantile levels as tuple of python floats.
def q_tuple():
    m = fm(); m.fit(series=y)
    p = m.predict(steps=STEPS, quantiles=(0.05, 0.95))
    return p.shape
probe("quantiles as tuple (0.05, 0.95)", q_tuple)

# 4. Categorical (string) covariate, direct backend call.
def cat_direct():
    ctx = pd.DataFrame({"item_id": "y", "timestamp": y.index, "target": y.to_numpy(),
                        "c": np.where(np.arange(120) % 2 == 0, "a", "b")})
    fut = pd.DataFrame({"item_id": "y", "timestamp": exog_future.index,
                        "c": np.where(np.arange(STEPS) % 2 == 0, "a", "b")})
    with contextlib.redirect_stderr(io.StringIO()):
        r = TabICLForecaster(max_context_length=64).predict_df(context_df=ctx, future_df=fut, quantiles=[0.5])
    return r["target"].to_numpy()[:3].round(3).tolist()
probe("string covariate, direct predict_df", cat_direct)

# 5. Categorical (pandas category) covariate through FoundationModel.
def cat_fm():
    ex = pd.DataFrame({"c": pd.Categorical(np.where(np.arange(len(idx)) % 2 == 0, "a", "b"))}, index=idx)
    m = fm(); m.fit(series=y, exog=ex.iloc[:120])
    p = m.predict(steps=STEPS, exog=ex.iloc[120:])
    return p["pred"].to_numpy()[:3].round(3).tolist()
probe("category covariate through FoundationModel", cat_fm)

# 6. Integer timestamps passed directly to the backend (what the adapter avoids
#    with its synthetic daily DatetimeIndex).
def int_ts():
    ctx = pd.DataFrame({"item_id": "y", "timestamp": np.arange(120), "target": y.to_numpy()})
    fut = pd.DataFrame({"item_id": "y", "timestamp": np.arange(120, 120 + STEPS)})
    with contextlib.redirect_stderr(io.StringIO()):
        r = TabICLForecaster(max_context_length=64).predict_df(context_df=ctx, future_df=fut)
    return f"timestamps parsed as {r.index.get_level_values('timestamp')[:2].tolist()}"
probe("integer timestamps, direct predict_df", int_ts)

# 7. RangeIndex series through FoundationModel (synthetic daily index).
def range_idx():
    m = fm(); m.fit(series=y.reset_index(drop=True))
    p = m.predict(steps=STEPS)
    return p.index[:2].tolist()
probe("RangeIndex series through FoundationModel", range_idx)

# 8. Series with NaN inside and with only one valid value.
def nan_series():
    s = y.copy(); s.iloc[10:30] = np.nan
    s1 = y.copy(); s1.iloc[:-1] = np.nan
    m = fm(); m.fit(series={"a": s, "b": s1})
    p = m.predict(steps=STEPS)
    return p.groupby("level")["pred"].first().round(3).to_dict()
probe("NaN in series / one valid value", nan_series)

# 9. Past-only covariate (column only in context exog).
def past_only():
    ex_future = exog_future.copy()
    m = fm(); m.fit(series=y, exog=exog.assign(past=1.0))
    p = m.predict(steps=STEPS, exog=ex_future)
    return p.shape
probe("past-only covariate", past_only)

# 10. NaN in future covariate.
def nan_future():
    ex_future = exog_future.copy(); ex_future.iloc[0, 0] = np.nan
    m = fm(); m.fit(series=y, exog=exog)
    p = m.predict(steps=STEPS, exog=ex_future)
    return bool(np.isfinite(p["pred"]).all())
probe("NaN in future covariate", nan_future)

# 11. Progress bar on CPU: is it written to stderr, and suppressed by default?
def progress(show):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        m = fm(show_progress=show); m.fit(series={"a": y, "b": y + 1})
        m.predict(steps=STEPS)
    text = buf.getvalue()
    return f"stderr chars={len(text)}; contains 'Predicting time series'={'Predicting time series' in text}"
probe("show_progress=False (default)", lambda: progress(False))
probe("show_progress=True", lambda: progress(True))

# 12. Device forwarded through tabicl_config.
def device_cfg():
    m = fm(tabicl_config={"device": "cpu", "n_estimators": 2}); m.fit(series=y)
    return m.predict(steps=STEPS).shape
probe("tabicl_config={'device': 'cpu', 'n_estimators': 2}", device_cfg)

# 13. Checkpoint selection: model_id is not forwarded; checkpoint_version is.
def ckpt():
    m = fm(tabicl_config={"checkpoint_version": "tabicl-regressor-v2-20260212.ckpt"}); m.fit(series=y)
    return m.predict(steps=STEPS).shape
probe("tabicl_config checkpoint_version (explicit default)", ckpt)

def model_id_label():
    m = FoundationModel("soda-inria/tabicl-v1-does-not-exist", context_length=64)
    m.fit(series=y)
    return f"adapter={type(m.adapter).__name__}, shape={m.predict(steps=STEPS).shape}"
probe("model_id with arbitrary suffix", model_id_label)
