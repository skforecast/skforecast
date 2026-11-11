"""
PERF-01: backtesting_foundation calls the model once per fold and spends
pandas time per series and fold preparing the context and the exog.

This script measures:

1. Real model (autogluon/chronos-2-small on CPU), current
   `backtesting_foundation` versus a prototype that builds the context of
   every (series, fold) pair with positional numpy slicing and sends all of
   them to `ForecasterFoundation.predict` in a single call (Chronos-2 batches
   internally with batch_size=256). Predictions are compared.

   A. 1 hourly series, 6000 obs, 200 folds of 24 steps, context_length=512.
   B. 50 daily series, 600 obs, 30 folds of 7 steps, context_length=256.
   C. 5 hourly series, 3000 obs, 100 folds of 24 steps, context_length=512.

2. Pure pandas overhead (adapter.predict mocked to return zeros), 2000 daily
   series, 730 obs, 10 folds of 7 steps, without and with exog (2 columns):
   current versus the same prototype. A cProfile of the current path is
   written next to the script outputs.

Each timing is repeated REPS times and the median is reported. The first
model load and first predict are timed separately (warm-up). numba is not on
this code path (no njit in skforecast/foundation or model_selection/_utils),
so there is no numba compile time to separate.

Caveat: other agents share the 4 CPUs of this machine, timings are noisy.

Run: python dev/roadmap_0.27.0/repro/PERF-01.py [--skip-real] [--skip-mock]
"""

from __future__ import annotations

import cProfile
import io
import os
import platform
import pstats
import statistics
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

import skforecast
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

warnings.filterwarnings("ignore")

REPS = int(os.environ.get("PERF01_REPS", 3))
OUT_DIR = os.environ.get(
    "PERF01_OUT",
    "/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/"
    "scratchpad/perf01",
)
os.makedirs(OUT_DIR, exist_ok=True)
SEP = "||"


def versions() -> None:
    import numba
    import torch

    try:
        import statsmodels

        sm = statsmodels.__version__
    except ImportError:
        sm = "not installed"
    from importlib.metadata import version

    print(f"python {platform.python_version()} | numpy {np.__version__} | "
          f"pandas {pd.__version__} | numba {numba.__version__} | "
          f"statsmodels {sm} | skforecast {skforecast.__version__}")
    print(f"torch {torch.__version__} (threads={torch.get_num_threads()}) | "
          f"chronos-forecasting {version('chronos-forecasting')} | "
          f"transformers {version('transformers')}")
    print(f"cpus={os.cpu_count()} loadavg={os.getloadavg()}")


def timeit(fn, reps: int = REPS) -> tuple[float, list[float], object]:
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, out


# ---------------------------------------------------------------------------
# Prototype: all folds in one predict call, positional slicing with numpy
# ---------------------------------------------------------------------------
def backtesting_foundation_batched(
    forecaster: ForecasterFoundation,
    series: dict[str, pd.Series],
    cv: TimeSeriesFold,
    exog: dict[str, pd.DataFrame] | None = None,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """
    Prototype (gap=0, no levels, point forecast). Returns predictions with the
    same columns as backtesting_foundation (level, fold, pred).
    """
    cv = deepcopy(cv)
    cv.refit = True
    cv.fixed_train_size = False
    cv.set_params({"window_size": forecaster.window_size,
                   "return_all_indexes": False, "verbose": False})
    folds = cv.split(X=series, as_pandas=False)
    span_index = cv._extract_index(X=series)
    L = forecaster.context_length

    fold_ids = np.array([f[0] for f in folds])
    train_end = np.array([f[1][1] for f in folds])       # exclusive, span iloc
    test_start = np.array([f[4][0] for f in folds])
    test_end = np.array([f[4][1] for f in folds])        # exclusive
    steps = folds[0][3][1] - folds[0][3][0]
    # Labels, looked up once for all folds
    lab_train_end = span_index[train_end - 1]
    lab_test_start = span_index[test_start]
    lab_test_end = span_index[test_end - 1]

    ctx, cexog, fexog = {}, ({} if exog is not None else None), (
        {} if exog is not None else None)
    for name, s in series.items():
        idx = s.index
        notna = ~np.isnan(s.to_numpy())
        cum = np.concatenate(([0], np.cumsum(notna)))
        first_valid = int(np.argmax(notna))
        te = idx.searchsorted(lab_train_end, side="right")
        ts = idx.searchsorted(lab_test_start, side="left")
        tt = idx.searchsorted(lab_test_end, side="right")
        start = np.maximum(first_valid, te - L)
        keep = ((cum[tt] - cum[ts]) > 0) & (te > start) & ((cum[te] - cum[start]) > 0)
        e = exog.get(name) if exog is not None else None
        if e is not None:
            eidx = e.index
            e_start = eidx.searchsorted(idx[np.minimum(start, len(idx) - 1)], "left")
            e_te = eidx.searchsorted(lab_train_end, side="right")
            e_fs = eidx.searchsorted(lab_test_start, side="left")
        for k in np.flatnonzero(keep):
            key = f"{name}{SEP}{fold_ids[k]}"
            ctx[key] = s.iloc[start[k]:te[k]]
            if e is not None:
                cexog[key] = e.iloc[e_start[k]:e_te[k]]
                fexog[key] = e.iloc[e_fs[k]:e_fs[k] + steps]

    keys = list(ctx)
    chunk_size = chunk_size or len(keys)
    preds = []
    for i in range(0, len(keys), chunk_size):
        kk = keys[i:i + chunk_size]
        preds.append(forecaster.predict(
            steps=steps,
            context={k: ctx[k] for k in kk},
            context_exog=None if cexog is None else {k: cexog[k] for k in kk},
            exog=None if fexog is None else {k: fexog[k] for k in kk},
            check_inputs=False,
        ))
    pred = pd.concat(preds)
    lvl = pred["level"].str.rsplit(SEP, n=1, expand=True)
    pred["level"] = lvl[0].to_numpy()
    pred.insert(1, "fold", lvl[1].astype(int).to_numpy())
    return pred


def compare(p_cur: pd.DataFrame, p_new: pd.DataFrame) -> str:
    a = p_cur.rename_axis("idx").reset_index()
    if "level" not in a:
        a["level"] = p_new["level"].iloc[0]
    b = p_new.rename_axis("idx").reset_index()
    m = a.merge(b, on=["idx", "level", "fold"], suffixes=("_cur", "_new"))
    diff = np.abs(m["pred_cur"] - m["pred_new"]).max()
    return f"rows {len(m)}/{len(a)}/{len(b)}, max abs diff {diff:.1e}"


# ---------------------------------------------------------------------------
# 1. Real model
# ---------------------------------------------------------------------------
def real_model() -> None:
    print("\n=== 1. Real model: chronos-2-small, CPU ===")
    rng = np.random.default_rng(0)

    # A. single hourly series, many folds
    n, steps, nfolds, L = 6000, 24, 200, 512
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = pd.Series(10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, n),
                  index=idx, name="y")
    t0 = time.perf_counter()
    f = ForecasterFoundation(FoundationModel(
        "autogluon/chronos-2-small", context_length=L, device_map="cpu"))
    f.fit(series=y)
    f.predict(steps=steps)
    print(f"warm-up (model load + first predict): {time.perf_counter() - t0:.2f} s")

    cv = TimeSeriesFold(steps=steps, initial_train_size=n - steps * nfolds)
    med_cur, ts_cur, (m_cur, p_cur) = timeit(lambda: backtesting_foundation(
        f, series=y, cv=cv, metric="mean_absolute_error", show_progress=False))
    med_new, ts_new, p_new = timeit(
        lambda: backtesting_foundation_batched(f, {"y": y}, cv))
    mae_new = np.mean(np.abs(y.loc[p_new.index].to_numpy() - p_new["pred"].to_numpy()))
    print(f"A. 1 series x {nfolds} folds x {steps} steps (L={L}): "
          f"current median {med_cur:.2f} s {np.round(ts_cur, 2).tolist()} | "
          f"batched median {med_new:.2f} s {np.round(ts_new, 2).tolist()} | "
          f"speed-up {med_cur / med_new:.2f}x | {compare(p_cur, p_new)} | "
          f"MAE {m_cur.iloc[0, 0]:.6f} vs {mae_new:.6f}")

    # Model-only time per fold (no skforecast prep): one direct predict call
    ctx1 = {"y": y.iloc[-L - steps:-steps]}
    med_one, _, _ = timeit(lambda: f.predict(steps=steps, context=ctx1,
                                              check_inputs=False), reps=20)
    print(f"   one predict call, 1 context (model + FoundationModel): "
          f"{med_one * 1000:.1f} ms -> x{nfolds} = {med_one * nfolds:.2f} s")

    # B. 50 daily series, 30 folds
    n_series, n, steps, nfolds, L = 50, 600, 7, 30, 256
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    series = {
        f"s{i}": pd.Series(10 + 3 * np.sin(2 * np.pi * t / 7 + i)
                           + rng.normal(0, 1, n).cumsum() * 0.1, index=idx)
        for i in range(n_series)
    }
    f = ForecasterFoundation(FoundationModel(
        "autogluon/chronos-2-small", context_length=L, device_map="cpu"))
    f.fit(series=series)
    f.predict(steps=steps)
    cv = TimeSeriesFold(steps=steps, initial_train_size=n - steps * nfolds)
    med_cur, ts_cur, (m_cur, p_cur) = timeit(lambda: backtesting_foundation(
        f, series=series, cv=cv, metric="mean_absolute_error",
        show_progress=False))
    med_new, ts_new, p_new = timeit(
        lambda: backtesting_foundation_batched(f, series, cv))
    print(f"B. {n_series} series x {nfolds} folds x {steps} steps (L={L}): "
          f"current median {med_cur:.2f} s {np.round(ts_cur, 2).tolist()} | "
          f"batched median {med_new:.2f} s {np.round(ts_new, 2).tolist()} | "
          f"speed-up {med_cur / med_new:.2f}x | {compare(p_cur, p_new)}")

    # C. 5 hourly series, 100 folds (intermediate case)
    n_series, n, steps, nfolds, L = 5, 3000, 24, 100, 512
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    t = np.arange(n)
    series = {
        f"s{i}": pd.Series(10 + 3 * np.sin(2 * np.pi * t / 24 + i)
                           + rng.normal(0, 1, n), index=idx)
        for i in range(n_series)
    }
    f = ForecasterFoundation(FoundationModel(
        "autogluon/chronos-2-small", context_length=L, device_map="cpu"))
    f.fit(series=series)
    f.predict(steps=steps)
    cv = TimeSeriesFold(steps=steps, initial_train_size=n - steps * nfolds)
    med_cur, ts_cur, (m_cur, p_cur) = timeit(lambda: backtesting_foundation(
        f, series=series, cv=cv, metric="mean_absolute_error",
        show_progress=False))
    med_new, ts_new, p_new = timeit(
        lambda: backtesting_foundation_batched(f, series, cv))
    print(f"C. {n_series} series x {nfolds} folds x {steps} steps (L={L}): "
          f"current median {med_cur:.2f} s {np.round(ts_cur, 2).tolist()} | "
          f"batched median {med_new:.2f} s {np.round(ts_new, 2).tolist()} | "
          f"speed-up {med_cur / med_new:.2f}x | {compare(p_cur, p_new)}")


# ---------------------------------------------------------------------------
# 2. Pure overhead with a mocked adapter
# ---------------------------------------------------------------------------
def mocked() -> None:
    print("\n=== 2. Overhead only (adapter.predict mocked), 2000 daily series ===")
    rng = np.random.default_rng(0)
    n, N, steps, nfolds, L = 730, 2000, 7, 10, 256
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    series = {f"s{i}": pd.Series(rng.normal(size=n), index=idx) for i in range(N)}
    exog = {
        k: pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)},
                        index=idx)
        for k in series
    }
    f = ForecasterFoundation(FoundationModel("autogluon/chronos-2-small",
                                             context_length=L))
    f.estimator.adapter.predict = (
        lambda steps, context, context_exog, exog, quantiles:
        {k: np.zeros((steps, 1)) for k in context}
    )
    f.fit(series=series, exog=exog)
    cv = TimeSeriesFold(steps=steps, initial_train_size=n - steps * nfolds)

    for label, ex in [("no exog", None), ("exog 2 cols", exog)]:
        med_cur, ts_cur, (_, p_cur) = timeit(lambda: backtesting_foundation(
            f, series=series, exog=ex, cv=cv, metric="mean_absolute_error",
            show_progress=False))
        med_new, ts_new, p_new = timeit(
            lambda: backtesting_foundation_batched(f, series, cv, exog=ex))
        print(f"{label}: current median {med_cur:.2f} s "
              f"({med_cur / nfolds:.2f} s/fold, "
              f"{med_cur / (nfolds * N) * 1e3:.2f} ms per series-fold) "
              f"{np.round(ts_cur, 2).tolist()} | prototype median {med_new:.2f} s "
              f"({med_new / (nfolds * N) * 1e3:.2f} ms per series-fold) "
              f"{np.round(ts_new, 2).tolist()} | {med_cur / med_new:.1f}x | "
              f"{compare(p_cur, p_new)}")

        for tag, call in [
            ("current", lambda: backtesting_foundation(
                f, series=series, exog=ex, cv=cv,
                metric="mean_absolute_error", show_progress=False)),
            ("prototype", lambda: backtesting_foundation_batched(
                f, series, cv, exog=ex)),
        ]:
            pr = cProfile.Profile()
            pr.enable()
            call()
            pr.disable()
            s = io.StringIO()
            st = pstats.Stats(pr, stream=s)
            st.sort_stats("cumulative").print_stats(
                r"skforecast|pandas/core/(indexing|generic|frame)", 25)
            path = os.path.join(
                OUT_DIR, f"prof_{tag}_{label.replace(' ', '_')}.txt")
            with open(path, "w") as fh:
                fh.write(s.getvalue())
            # Short summary of skforecast functions (cumulative time)
            rows = []
            for (file, line, func), (cc, nc, tt, ct, _) in st.stats.items():
                if "skforecast" in file and "site-packages" not in file:
                    rows.append((ct, f"{os.path.basename(file)}:{line} {func}"))
            rows.sort(reverse=True)
            print(f"   cProfile {tag} ({label}), top skforecast cumulative:")
            for ct, name in rows[:7]:
                print(f"     {ct:6.2f} s  {name}")


if __name__ == "__main__":
    versions()
    if "--skip-real" not in sys.argv:
        real_model()
    if "--skip-mock" not in sys.argv:
        mocked()
    print(f"\nloadavg at end={os.getloadavg()}")
