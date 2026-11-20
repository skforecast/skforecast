import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cProfile
import pstats
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

N = 3000
STEPS = 24
LAGS = 24


def make_data(n=N, exog_kind="num"):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = pd.Series(
        10 + np.sin(2 * np.pi * t / 7) + 0.01 * t + rng.normal(0, 0.3, n),
        index=idx, name="y",
    )
    if exog_kind is None:
        exog = None
    elif exog_kind == "num":
        exog = pd.DataFrame(
            rng.normal(size=(n, 3)), index=idx, columns=["e1", "e2", "e3"]
        )
    elif exog_kind == "cat":
        days = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
        exog = pd.DataFrame(
            {"dow": pd.Categorical(days[t % 7], categories=list(days))}, index=idx
        )
    elif exog_kind == "mixed":
        days = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
        exog = pd.DataFrame(
            rng.normal(size=(n, 3)), index=idx, columns=["e1", "e2", "e3"]
        )
        exog["dow"] = pd.Categorical(days[t % 7], categories=list(days))
    return y, exog


def estimators():
    return {
        "Ridge": Ridge(),
        "LGBM": LGBMRegressor(n_estimators=100, verbose=-1, n_jobs=1),
    }


def classify(filename):
    f = filename
    if "/pandas/" in f:
        return "pandas"
    if "/sklearn/" in f or "/scipy/" in f:
        return "sklearn"
    if "/lightgbm/" in f:
        return "lightgbm"
    if "/numpy/" in f:
        return "numpy"
    if "/skforecast/" in f:
        return "skforecast"
    if "/joblib/" in f:
        return "joblib"
    if f == "~":
        return "builtin"
    return "other"


def profile_breakdown(fn, n_rep=1, top=0):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_rep):
        fn()
    pr.disable()
    st = pstats.Stats(pr)
    totals = defaultdict(float)
    # Builtins (C functions) are attributed to the module of their callers,
    # proportionally to the time spent from each caller.
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        cat = classify(func[0])
        if cat == "builtin" and callers:
            tot = sum(v[2] for v in callers.values()) or 1.0
            for cfunc, v in callers.items():
                ccat = classify(cfunc[0])
                if ccat == "builtin":
                    ccat = "builtin"
                totals[ccat] += tt * v[2] / tot
        else:
            totals[cat] += tt
    total = sum(totals.values())
    out = {k: (v, 100 * v / total) for k, v in sorted(totals.items(), key=lambda x: -x[1])}
    if top:
        st.sort_stats("cumulative").print_stats(top)
    return total, out, st


def fmt_breakdown(total, out):
    return f"total={total:.3f}s | " + ", ".join(
        f"{k}:{p:.0f}%" for k, (v, p) in out.items() if p >= 1
    )


def timeit(fn, n=20, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)
