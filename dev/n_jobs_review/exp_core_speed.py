"""
Single-thread speed of each core type of this hybrid CPU (performance,
efficiency and low power efficiency cores), and of a process without
affinity, to know where a single-threaded run is scheduled.

Each measurement runs in a fresh subprocess pinned to one logical processor.

Usage: python exp_core_speed.py
Output: results/exp_core_speed.csv
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

from common import RESULTS_DIR, CsvLog, cpu_topology

FIELDS = ["cpu", "core_type", "rep", "lgbm_fit_s", "python_loop_s"]


def child(cpu: int | None) -> None:
    import psutil

    if cpu is not None:
        psutil.Process().cpu_affinity([cpu])
    import numpy as np
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1713, 288))
    y = X[:, 0] + rng.normal(size=1713)
    model = LGBMRegressor(n_estimators=100, n_jobs=1, verbose=-1)
    model.fit(X, y)
    fits = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.fit(X, y)
        fits.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    s = 0.0
    for i in range(2_000_000):
        s += i * 0.5
    loop = time.perf_counter() - t0
    print(json.dumps({"lgbm_fit_s": min(fits), "python_loop_s": loop}))


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(None if sys.argv[2] == "none" else int(sys.argv[2]))
        return 0

    cores = cpu_topology()
    p_cores = [c for c in cores if c["efficiency_class"] == 1]
    e_cores = [c for c in cores if c["efficiency_class"] == 0]
    # On the Core Ultra 185H the two low power E cores are the last two
    # logical processors; Windows reports them with the same efficiency class
    # as the E cores, so they are told apart by position.
    targets = [
        ("none", "no affinity"),
        (str(p_cores[-1]["logical"][0]), "P"),
        (str(e_cores[2]["logical"][0]), "E"),
        (str(e_cores[-1]["logical"][0]), "LP-E"),
    ]
    log = CsvLog(RESULTS_DIR / "exp_core_speed.csv", FIELDS)
    for rep in range(3):
        for cpu, kind in targets:
            out = subprocess.run(
                [sys.executable, __file__, "--child", cpu],
                capture_output=True,
                text=True,
                check=True,
            )
            res = json.loads(out.stdout.strip().splitlines()[-1])
            log.write({"cpu": cpu, "core_type": kind, "rep": rep, **res})
            print(f"rep={rep} cpu={cpu:>4} {kind:<12} {res}", flush=True)
    log.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
