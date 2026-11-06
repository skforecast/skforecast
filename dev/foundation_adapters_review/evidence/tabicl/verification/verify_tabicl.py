"""
Independent re-check of the TabICLAdapter findings (tabicl 2.x).

Each check runs in a fresh subprocess so that warnings are printed to the real
stderr exactly as a user would see them (no pytest, no outer catch_warnings).
Nothing in skforecast/ is modified: the prototypes of the proposals are applied
with monkeypatches inside the subprocess.

Run from the repository root with the review environment:
    ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/verification/verify_tabicl.py
"""

import importlib.metadata as md
import platform
import subprocess
import sys
import textwrap

import torch

print(
    "python", sys.version.split()[0],
    "| tabicl", md.version("tabicl"),
    "| torch", torch.__version__,
    "| numpy", md.version("numpy"),
    "| pandas", md.version("pandas"),
    "| huggingface-hub", md.version("huggingface-hub"),
    "| skforecast", md.version("skforecast"),
    "|", platform.machine(), platform.system(),
    "| cuda", torch.cuda.is_available(),
)

COMMON = textwrap.dedent(
    """
    import contextlib, io, sys, warnings
    import numpy as np
    import pandas as pd
    from tabicl.forecast import TabICLForecaster
    from skforecast.foundation import FoundationModel
    import skforecast.foundation._adapters as adapters

    idx = pd.date_range("2024-01-01", periods=126, freq="h")
    y = pd.Series(10 + np.sin(np.arange(126) / 4), index=idx, name="y").iloc[:120]
    ex = pd.DataFrame({"x": np.cos(np.arange(126) / 4)}, index=idx)
    ex_future = ex.iloc[120:].copy()
    ex_future_nan = ex_future.copy()
    ex_future_nan.iloc[0, 0] = np.nan
    """
)


def run(name, body):
    code = COMMON + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    err = proc.stderr
    lines = [ln for ln in err.splitlines() if "Predicting time series" not in ln]
    bar = "Predicting time series" in err
    print(f"### {name}")
    print(f"    returncode={proc.returncode} | tqdm bar on stderr={bar}")
    for ln in proc.stdout.strip().splitlines():
        print(f"    stdout: {ln}")
    for ln in lines:
        if ln.strip():
            print(f"    stderr: {ln.strip()[:200]}")


# 1. Backend warning with the default show_progress=False and with True.
for show in (False, True):
    run(
        f"F15 NaN in future exog, show_progress={show}",
        f"""
        m = FoundationModel("soda-inria/tabicl", context_length=64, show_progress={show})
        m.fit(series=y, exog=ex.iloc[:120])
        m.predict(steps=6, exog=ex_future_nan)
        print("done")
        """,
    )

# 2. Once hidden, the default filter registry also hides it in a later call
#    with show_progress=True in the same process.
run(
    "F15 first call hidden, second call show_progress=True (same process)",
    """
    m = FoundationModel("soda-inria/tabicl", context_length=64)
    m.fit(series=y, exog=ex.iloc[:120])
    m.predict(steps=6, exog=ex_future_nan)
    m.set_params(show_progress=True)  # FoundationModel.set_params resets the fit
    m.fit(series=y, exog=ex.iloc[:120])
    m.predict(steps=6, exog=ex_future_nan)
    print("done")
    """,
)

# 3. Prototype of P1 checked without any outer catch_warnings.
run(
    "P1 prototype (re-emit recorded warnings), show_progress=False",
    """
    _orig = contextlib.redirect_stderr

    @contextlib.contextmanager
    def redirect_keep_warnings(target):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _orig(target):
                yield target
        for w in caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    adapters.contextlib = type("patched", (), {
        "redirect_stderr": staticmethod(redirect_keep_warnings),
        "nullcontext": staticmethod(contextlib.nullcontext),
    })
    m = FoundationModel("soda-inria/tabicl", context_length=64)
    m.fit(series=y, exog=ex.iloc[:120])
    m.predict(steps=6, exog=ex_future_nan)
    print("done")
    """,
)

# 4. Quantile levels: int, numpy float32, python float; and P2 prototype.
run(
    "F10/P2 quantile level types",
    """
    m = FoundationModel("soda-inria/tabicl", context_length=64)
    m.fit(series=y)
    ref = m.predict(steps=6, quantiles=[0.0, 0.5, 1.0])
    for label, q in [("int [0, 0.5, 1]", [0, 0.5, 1]),
                     ("np.float32", list(np.array([0.1, 0.9], dtype=np.float32))),
                     ("np.float64", list(np.array([0.1, 0.9], dtype=np.float64)))]:
        try:
            m.predict(steps=6, quantiles=q)
            print(label, "OK")
        except Exception as exc:
            print(label, "ERROR", type(exc).__name__, exc)
    orig = adapters.TabICLAdapter.predict
    def patched(self, steps, context, context_exog, exog, quantiles):
        if quantiles is not None:
            quantiles = [float(q) for q in quantiles]
        return orig(self, steps, context, context_exog, exog, quantiles)
    adapters.TabICLAdapter.predict = patched
    p = m.predict(steps=6, quantiles=[0, 0.5, 1])
    print("P2 columns:", p.columns.tolist())
    print("P2 equal to float levels:", np.array_equal(ref.iloc[:, 1:].to_numpy(), p.iloc[:, 1:].to_numpy()))
    p32 = m.predict(steps=6, quantiles=list(np.array([0.1, 0.9], dtype=np.float32)))
    print("P2 float32 columns:", p32.columns.tolist())
    """,
)

# 5. Adapter vs direct backend call with NaN inside the context window.
run(
    "F8 NaN inside the context window, adapter vs direct predict_df",
    """
    warnings.simplefilter("ignore")
    s = y.copy(); s.iloc[-40:-30] = np.nan
    m = FoundationModel("soda-inria/tabicl", context_length=64)
    m.fit(series={"s": s})
    p = m.predict(steps=6, quantiles=[0.1, 0.5, 0.9])
    ctx = pd.DataFrame({"item_id": "s", "timestamp": s.index[-64:], "target": s.iloc[-64:].to_numpy(float)})
    fut = pd.DataFrame({"item_id": "s", "timestamp": pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=6, freq="h")})
    with contextlib.redirect_stderr(io.StringIO()):
        r = TabICLForecaster(max_context_length=64).predict_df(context_df=ctx, future_df=fut, quantiles=[0.1, 0.5, 0.9])
    d = np.abs(p[["q_0.1", "q_0.5", "q_0.9"]].to_numpy() - r.loc["s"][[0.1, 0.5, 0.9]].to_numpy()).max()
    print("max_abs_diff (same 64-row window incl. 10 NaN):", d)
    with contextlib.redirect_stderr(io.StringIO()):
        r2 = TabICLForecaster(max_context_length=64).predict_df(
            context_df=pd.DataFrame({"item_id": "s", "timestamp": s.index, "target": s.to_numpy(float)}),
            future_df=fut, quantiles=[0.1, 0.5, 0.9])
    d2 = np.abs(p[["q_0.1", "q_0.5", "q_0.9"]].to_numpy() - r2.loc["s"][[0.1, 0.5, 0.9]].to_numpy()).max()
    print("max_abs_diff vs direct call with full series (backend drops NaN then keeps last 64 valid):", d2)
    """,
)

# 6. temporal_features given as strings through the adapter.
run(
    "F18 temporal_features as strings",
    """
    warnings.simplefilter("ignore")
    for tf in (["index", "datetime"], ["fourier"]):
        try:
            m = FoundationModel("soda-inria/tabicl", context_length=64, temporal_features=tf)
            m.fit(series=y)
            print(tf, "OK", m.predict(steps=6).shape)
        except Exception as exc:
            print(tf, "ERROR", type(exc).__name__, str(exc)[:120])
    """,
)

# 7. Categorical covariate and integer timestamps, direct backend.
run(
    "F9/F14 categorical covariate and integer timestamps (direct backend)",
    """
    warnings.simplefilter("ignore")
    ctx = pd.DataFrame({"item_id": "y", "timestamp": y.index, "target": y.to_numpy(),
                        "c": np.where(np.arange(120) % 2 == 0, "a", "b")})
    fut = pd.DataFrame({"item_id": "y", "timestamp": ex_future.index,
                        "c": np.where(np.arange(6) % 2 == 0, "a", "b")})
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            TabICLForecaster(max_context_length=64).predict_df(context_df=ctx, future_df=fut)
        print("string covariate OK")
    except Exception as exc:
        print("string covariate ERROR", type(exc).__name__, str(exc)[:110])
    ctx = pd.DataFrame({"item_id": "y", "timestamp": np.arange(120), "target": y.to_numpy()})
    fut = pd.DataFrame({"item_id": "y", "timestamp": np.arange(120, 126)})
    with contextlib.redirect_stderr(io.StringIO()):
        r = TabICLForecaster(max_context_length=64).predict_df(context_df=ctx, future_df=fut)
    print("integer timestamps parsed as", r.index.get_level_values("timestamp")[:2].tolist())
    """,
)

# 8. model_id handling.
run(
    "F7 model_id resolution",
    """
    from skforecast.foundation._adapters import _resolve_adapter
    for mid in ["soda-inria/tabicl", "soda-inria/tabicl-v1-does-not-exist", "jingang/TabICL"]:
        try:
            print(mid, "->", _resolve_adapter(mid).__name__)
        except ValueError as exc:
            print(mid, "-> ValueError", str(exc)[:60])
    """,
)
