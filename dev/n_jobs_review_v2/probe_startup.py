"""
E0: cost of starting the loky workers, cold (fresh process) and warm (reused),
with a task that imports what a backtesting fold needs. Prints one JSON line.

Usage: taskset -c 0-3 python probe_startup.py N_JOBS
"""

import json
import sys
import time

t0 = time.perf_counter()
import skforecast.model_selection  # noqa: E402,F401
from joblib import Parallel, delayed  # noqa: E402
t_import = time.perf_counter() - t0


def task(i):
    import skforecast.model_selection._validation  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    return i


def task_light(i):
    return i


n_jobs = int(sys.argv[1])
n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else n_jobs

t0 = time.perf_counter()
Parallel(n_jobs=n_jobs)(delayed(task)(i) for i in range(n_tasks))
cold = time.perf_counter() - t0

t0 = time.perf_counter()
Parallel(n_jobs=n_jobs)(delayed(task)(i) for i in range(n_tasks))
warm = time.perf_counter() - t0

t0 = time.perf_counter()
Parallel(n_jobs=n_jobs)(delayed(task_light)(i) for i in range(100))
warm100 = time.perf_counter() - t0

print(json.dumps({
    "n_jobs": n_jobs, "import_main_s": round(t_import, 3), "cold_s": round(cold, 3),
    "warm_s": round(warm, 4), "warm_100_tasks_s": round(warm100, 4),
}))
