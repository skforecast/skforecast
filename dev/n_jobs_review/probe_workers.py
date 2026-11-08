"""
Probe what loky workers see: affinity, CPU count, thread limits set through
environment variables and the thread pools reported by threadpoolctl.

Gate G1: with a restricted affinity, `joblib.cpu_count()` must return the
number of allowed logical processors both in the main process and inside a
worker, otherwise `n_jobs='auto'` would oversubscribe and the affinity
results would not be valid.

Usage: python probe_workers.py --affinity 4
"""

from __future__ import annotations

import argparse
import sys

from common import RESULTS_DIR, CsvLog, set_affinity

THREAD_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
]


def worker_view(_):
    import os

    import joblib
    import lightgbm  # noqa: F401  (registers its OpenMP runtime)
    import numpy  # noqa: F401
    import psutil
    import sklearn.ensemble  # noqa: F401
    from threadpoolctl import threadpool_info

    pools = [
        f"{p['user_api']}:{p['internal_api']}:{p['num_threads']}"
        for p in threadpool_info()
    ]

    return {
        "pid": os.getpid(),
        "affinity": len(psutil.Process().cpu_affinity()),
        "cpu_count": joblib.cpu_count(),
        "cpu_count_physical": joblib.cpu_count(only_physical_cores=True),
        "os_cpu_count": os.cpu_count(),
        "process_cpu_count": getattr(os, "process_cpu_count", lambda: None)(),
        "env": ";".join(f"{v}={os.environ.get(v)}" for v in THREAD_VARS),
        "threadpools": ";".join(pools),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity", type=int, default=None)
    args = parser.parse_args()

    allowed = set_affinity(args.affinity)

    import joblib
    from joblib import Parallel, delayed

    n_allowed = len(allowed)
    main_view = worker_view(None)
    main_view["where"] = "main"

    fields = [
        "affinity_requested",
        "where",
        "n_jobs",
        "pid",
        "affinity",
        "cpu_count",
        "cpu_count_physical",
        "os_cpu_count",
        "process_cpu_count",
        "env",
        "threadpools",
    ]
    tag = "" if args.affinity is None else f"_aff{args.affinity}"
    log = CsvLog(RESULTS_DIR / f"probe{tag}.csv", fields)
    req = args.affinity or n_allowed
    log.write({"affinity_requested": req, "n_jobs": 0, **main_view})

    ok = main_view["cpu_count"] == n_allowed
    for n_jobs in sorted({1, 2, max(1, joblib.cpu_count() - 1)}):
        views = Parallel(n_jobs=n_jobs)(delayed(worker_view)(i) for i in range(n_jobs))
        for v in {v["pid"]: v for v in views}.values():
            v["where"] = "worker"
            log.write({"affinity_requested": req, "n_jobs": n_jobs, **v})
            ok &= v["affinity"] == n_allowed and v["cpu_count"] == n_allowed
            print(f"n_jobs={n_jobs} worker {v}")
    log.close()

    print(f"main {main_view}")
    print(f"allowed logical processors: {sorted(allowed)}")
    print(f"G1 {'PASS' if ok else 'FAIL'}: expected cpu_count {n_allowed}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
