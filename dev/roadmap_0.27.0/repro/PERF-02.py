"""
PERF-02: parallelism in stats backtesting and BLAS thread oversubscription in
the fit of Ets and Sarimax.

Parts (run all with no argument, or one part by name):

- env: versions, BLAS thread pools, CPU count and load average.
- compile: numba compile time with an empty cache vs warm cache (subprocess).
- workers: one-off cost of starting loky workers (import + numba cache load).
- bt: backtesting_stats with refit=True, n_jobs=1, n_jobs=1 inside
  threadpool_limits(1, user_api="blas") ("1+lim"), n_jobs=3 (the value that
  `cpu_count() - 1` gives on this 4 core machine) and 4. Same predictions?
  Cases: Arima fixed order, Ets auto (today, because of BUG-08 every fold
  re-runs the automatic search), Ets fixed (what BUG-08 fixed would cost),
  Sarimax fixed order. Also refit=False as reference (only Sarimax honours
  it: backtesting_stats forces refit=True for Arima, Ets and Arar).
- blas: single fit of Ets auto and Sarimax, alone and with N simultaneous
  processes or busy-loop processes, with and without
  threadpool_limits(1, user_api="blas").
- pool: 8 series Ets auto, sequential vs joblib vs multiprocessing.Pool(4).
- profile: cProfile of Ets auto fit with 2 simultaneous processes.

Data: synthetic daily series with weekly seasonality, trend and noise.
Timings are medians of `--repeats` runs. The machine is shared with other
agents: the load average is printed before each part.

Run: python dev/roadmap_0.27.0/repro/PERF-02.py [part] [--repeats 3]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def make_series(n: int = 1000, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (
        50
        + 0.02 * t
        + 6 * np.sin(2 * np.pi * t / 7)
        + 2 * np.cos(2 * np.pi * t / 7)
        + np.cumsum(rng.normal(0, 0.3, n))
        + rng.normal(0, 1.5, n)
    )
    return pd.Series(
        y, index=pd.date_range("2020-01-01", periods=n, freq="D"), name="y"
    )


def load() -> str:
    return " ".join(open("/proc/loadavg").read().split()[:3])


def header(name: str) -> None:
    print(f"\n=== {name} (load avg {load()}) ===", flush=True)


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
def part_env(args) -> None:
    import joblib
    import numba
    import scipy
    import sklearn
    import statsmodels
    import threadpoolctl
    import scipy.optimize  # noqa: F401  (loads scipy's OpenBLAS)
    import skforecast

    header("env")
    print("python", sys.version.split()[0])
    for m in (np, pd, scipy, numba, sklearn, statsmodels, joblib, threadpoolctl,
              skforecast):
        print(f"{m.__name__} {m.__version__}")
    print("os.cpu_count()", os.cpu_count(),
          "| sched_getaffinity", len(os.sched_getaffinity(0)))
    for info in threadpoolctl.threadpool_info():
        print(
            f"  {info['user_api']:6s} {info['internal_api']:8s} "
            f"threads={info['num_threads']} version={info['version']} "
            f"{os.path.basename(info['filepath'])}"
        )
    from skforecast.model_selection._utils import select_n_jobs_backtesting
    from skforecast.recursive import ForecasterStats
    from skforecast.stats import Ets
    print(
        "select_n_jobs_backtesting(ForecasterStats, refit=True) =",
        select_n_jobs_backtesting(ForecasterStats(estimator=Ets(m=7)), True),
    )


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
COMPILE_CODE = r"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
t0 = time.perf_counter()
from skforecast.stats import Ets, Arima
t_import = time.perf_counter() - t0
rng = np.random.default_rng(0)
y = 50 + 6*np.sin(2*np.pi*np.arange(300)/7) + rng.normal(0, 1, 300)
t0 = time.perf_counter(); Ets(m=7, model="AAA").fit(y); t_ets1 = time.perf_counter() - t0
t0 = time.perf_counter(); Ets(m=7, model="AAA").fit(y); t_ets2 = time.perf_counter() - t0
t0 = time.perf_counter(); Arima(order=(2,1,1), seasonal_order=(1,0,1), m=7).fit(y); t_ar1 = time.perf_counter() - t0
t0 = time.perf_counter(); Arima(order=(2,1,1), seasonal_order=(1,0,1), m=7).fit(y); t_ar2 = time.perf_counter() - t0
print(f"{t_import:.2f} {t_ets1:.2f} {t_ets2:.3f} {t_ar1:.2f} {t_ar2:.3f}")
"""


def part_compile(args) -> None:
    header("compile (first fit in a new process)")
    for label, cache_dir in (("cold cache", tempfile.mkdtemp()), ("warm cache", None)):
        env = dict(os.environ)
        if cache_dir:
            env["NUMBA_CACHE_DIR"] = cache_dir
        out = subprocess.run(
            [sys.executable, "-c", COMPILE_CODE], env=env, capture_output=True,
            text=True, check=True,
        ).stdout.split()
        print(
            f"  {label}: import {out[0]} s | Ets first fit {out[1]} s, second "
            f"{out[2]} s | Arima first fit {out[3]} s, second {out[4]} s"
        )


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _tiny_fit(i):
    from skforecast.stats import Ets
    rng = np.random.default_rng(i)
    y = 50 + 6 * np.sin(2 * np.pi * np.arange(200) / 7) + rng.normal(0, 1, 200)
    Ets(m=7, model="AAA").fit(y)
    return os.getpid()


def part_workers(args) -> None:
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    header("workers (loky start-up, n_jobs=3)")
    get_reusable_executor().shutdown(wait=True)
    t0 = time.perf_counter()
    Parallel(n_jobs=3)(delayed(_tiny_fit)(i) for i in range(3))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    Parallel(n_jobs=3)(delayed(_tiny_fit)(i) for i in range(3))
    t_second = time.perf_counter() - t0
    print(f"  first Parallel call (spawn + import + numba cache load): {t_first:.2f} s")
    print(f"  second call (workers reused): {t_second:.2f} s")


# ---------------------------------------------------------------------------
# bt
# ---------------------------------------------------------------------------
def bt_cases():
    from skforecast.stats import Arima, Ets, Sarimax
    return {
        "Arima(2,1,1)(1,0,1)[7]": lambda: Arima(
            order=(2, 1, 1), seasonal_order=(1, 0, 1), m=7
        ),
        "Ets auto (m=7)": lambda: Ets(m=7),
        "Ets AAA (m=7)": lambda: Ets(m=7, model="AAA"),
        "Sarimax(2,1,1)(1,0,1,7)": lambda: Sarimax(
            order=(2, 1, 1), seasonal_order=(1, 0, 1, 7)
        ),
    }


def run_bt(make_est, y, n_jobs, refit=True, blas_limit=False):
    from contextlib import nullcontext

    from threadpoolctl import threadpool_limits

    from skforecast.model_selection import TimeSeriesFold, backtesting_stats
    from skforecast.recursive import ForecasterStats

    cv = TimeSeriesFold(steps=7, initial_train_size=790, refit=refit)
    ctx = threadpool_limits(1, user_api="blas") if blas_limit else nullcontext()
    t0 = time.perf_counter()
    with ctx:
        metric, preds = backtesting_stats(
            forecaster=ForecasterStats(estimator=make_est()),
            y=y, cv=cv, metric="mean_absolute_error", n_jobs=n_jobs,
            show_progress=False, suppress_warnings=True,
        )
    return time.perf_counter() - t0, metric.iloc[0, -1], preds


def part_bt(args) -> None:
    from joblib import Parallel, delayed

    header(f"bt (1000 points, steps=7, 30 folds, refit=True, repeats={args.repeats})")
    y = make_series(1000)
    # Warm up numba (cache) in the parent and the loky workers.
    run_bt(lambda: __import__("skforecast.stats", fromlist=["Ets"]).Ets(
        m=7, model="AAA"), y.iloc[:820], 1)
    Parallel(n_jobs=4)(delayed(_tiny_fit)(i) for i in range(4))

    print(f"  {'case':26s} {'n_jobs':>6s} {'median s':>9s} {'runs':>22s} "
          f"{'MAE':>10s} {'same preds':>10s} {'speed-up':>8s}")
    configs = [(n, False) for n in args.n_jobs]
    configs.insert(1, (1, True))  # n_jobs=1 with BLAS limited to 1 thread
    for name, make_est in bt_cases().items():
        base_time, base_preds = None, None
        for n_jobs, blas_limit in configs:
            times, mae, preds = [], None, None
            for _ in range(args.repeats):
                t, mae, preds = run_bt(make_est, y, n_jobs, blas_limit=blas_limit)
                times.append(t)
            med = statistics.median(times)
            if base_preds is None:
                base_time, base_preds = med, preds
                same = "-"
            else:
                same = str(np.allclose(
                    base_preds["pred"].to_numpy(), preds["pred"].to_numpy(),
                    rtol=0, atol=1e-8))
            runs = ", ".join(f"{t:.1f}" for t in times)
            label = f"{n_jobs}+lim" if blas_limit else str(n_jobs)
            print(f"  {name:26s} {label:>6s} {med:9.2f} {runs:>22s} "
                  f"{mae:10.5f} {same:>10s} {base_time / med:7.2f}x", flush=True)

    header("bt reference: refit=False, n_jobs=1 (forced)")
    for name, make_est in bt_cases().items():
        times = [run_bt(make_est, y, 1, refit=False)[0] for _ in range(args.repeats)]
        print(f"  {name:26s} median {statistics.median(times):.2f} s", flush=True)


# ---------------------------------------------------------------------------
# blas
# ---------------------------------------------------------------------------
BLAS_WORKER = r"""
import sys, time, warnings, statistics
warnings.filterwarnings("ignore")
import numpy as np
from threadpoolctl import threadpool_limits
from skforecast.stats import Ets, Sarimax
kind, limit, reps = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
rng = np.random.default_rng(123)
t = np.arange(1000)
y = 50 + 0.02*t + 6*np.sin(2*np.pi*t/7) + np.cumsum(rng.normal(0, .3, 1000)) + rng.normal(0, 1.5, 1000)
make = {
    "ets": lambda: Ets(m=7),
    "sarimax": lambda: Sarimax(order=(2, 1, 1), seasonal_order=(1, 0, 1, 7)),
}[kind]
make().fit(y)  # warm-up (numba cache, imports)
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    if limit:
        with threadpool_limits(1, user_api="blas"):
            make().fit(y)
    else:
        make().fit(y)
    times.append(time.perf_counter() - t0)
print(statistics.median(times))
"""


BURNER = "while True: pass"


def run_blas(kind, limit, n_procs, reps, extra_env=None, n_burners=0):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    burners = [subprocess.Popen([sys.executable, "-c", BURNER])
               for _ in range(n_burners)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", BLAS_WORKER, kind, "1" if limit else "0",
             str(reps)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for _ in range(n_procs)
    ]
    try:
        return [float(p.communicate()[0]) for p in procs]
    finally:
        for b in burners:
            b.kill()
            b.wait()


def part_blas(args) -> None:
    header(f"blas (single fit, n=1000, median of {args.repeats} fits per process)")
    # (label, simultaneous fit processes, extra env, busy-loop processes)
    configs = [
        ("1 process", 1, None, 0),
        ("1 process + 1 busy-loop process", 1, None, 1),
        ("1 process + 2 busy-loop processes", 1, None, 2),
        ("2 processes at once", 2, None, 0),
        ("4 processes at once", 4, None, 0),
    ]
    # Note: OPENBLAS_NUM_THREADS above the number of cores is capped by
    # OpenBLAS (checked with threadpool_info: 16 -> 4), so it cannot emulate a
    # container whose CPU quota is lower than the cores OpenBLAS detects.
    for kind in ("ets", "sarimax"):
        for label, n_procs, extra_env, n_burners in configs:
            for limit in (False, True):
                res = run_blas(kind, limit, n_procs, args.repeats, extra_env,
                               n_burners)
                print(
                    f"  {kind:7s} {label:36s} limit={'yes' if limit else 'no ':3s} "
                    f"s/fit per process: {', '.join(f'{r:.2f}' for r in res)} "
                    f"(load {load()})", flush=True)


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------
def _fit_ets(seed):
    from skforecast.stats import Ets
    y = make_series(1000, seed).to_numpy()
    Ets(m=7).fit(y)
    return seed


def _fit_ets_limited(seed):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="blas"):
        return _fit_ets(seed)


def part_pool(args) -> None:
    from joblib import Parallel, delayed

    header("pool (8 series Ets auto, n=1000)")
    _fit_ets(0)
    seeds = list(range(8))
    res = {}
    t0 = time.perf_counter()
    for s in seeds:
        _fit_ets(s)
    res["sequential"] = time.perf_counter() - t0
    Parallel(n_jobs=4)(delayed(_fit_ets)(s) for s in range(4))  # warm workers
    t0 = time.perf_counter()
    Parallel(n_jobs=4)(delayed(_fit_ets)(s) for s in seeds)
    res["joblib n_jobs=4 (loky)"] = time.perf_counter() - t0
    ctx = mp.get_context("fork")
    for label, fn in (("multiprocessing.Pool(4)", _fit_ets),
                      ("multiprocessing.Pool(4) + threadpool_limits(1)",
                       _fit_ets_limited)):
        with ctx.Pool(4) as pool:
            t0 = time.perf_counter()
            pool.map(fn, seeds)
            res[label] = time.perf_counter() - t0
    for k, v in res.items():
        print(f"  {k:48s} {v:6.2f} s", flush=True)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
PROFILE_WORKER = r"""
import cProfile, pstats, io, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from skforecast.stats import Ets
rng = np.random.default_rng(123)
t = np.arange(1000)
y = 50 + 0.02*t + 6*np.sin(2*np.pi*t/7) + np.cumsum(rng.normal(0, .3, 1000)) + rng.normal(0, 1.5, 1000)
Ets(m=7).fit(y)
pr = cProfile.Profile(); pr.enable(); Ets(m=7).fit(y); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(6)
if sys.argv[1] == "0":
    print(s.getvalue())
"""


def part_profile(args) -> None:
    header("profile (Ets auto fit, 2 simultaneous processes, tottime top 6)")
    procs = [
        subprocess.Popen([sys.executable, "-c", PROFILE_WORKER, str(i)],
                         stdout=subprocess.PIPE, text=True)
        for i in range(2)
    ]
    out = [p.communicate()[0] for p in procs]
    lines = [ln for ln in out[0].splitlines() if ln.strip()]
    print("\n".join("  " + ln for ln in lines[:16]))


PARTS = {
    "env": part_env, "compile": part_compile, "workers": part_workers,
    "bt": part_bt, "blas": part_blas, "pool": part_pool, "profile": part_profile,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", default=list(PARTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, nargs="+", default=[1, 3, 4],
                        dest="n_jobs")
    args = parser.parse_args()
    for p in args.parts:
        PARTS[p](args)
