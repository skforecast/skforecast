"""
Block A: fixed cost of parallelizing with joblib (loky) on this machine.
On Windows the workers are started with `spawn`, so each new worker starts
a Python interpreter and imports the modules needed to unpickle its tasks.

Measured for k workers:

- cold: first `Parallel(n_jobs=k)` call of a fresh process (the libraries
  are already imported in the parent, as in a user session). Each cold
  measurement is a new subprocess.
- warm: later calls with the same k, which reuse the loky executor.

Payloads:

- noop: k trivial tasks (the worker only imports this module).
- forecaster: k predictions of a fitted ForecasterRecursive with LightGBM
  (the worker imports skforecast, pandas, scikit-learn and lightgbm).
- few: 4 trivial tasks with k workers (fewer tasks than workers).

Usage: python bench_spawn.py [--affinity 4] [--pilot]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import N_REPS, N_WARMUP, RESULTS_DIR, CsvLog, set_affinity, shuffled

FIELDS = [
    "timestamp",
    "affinity",
    "cpu_count",
    "mode",
    "payload",
    "k",
    "n_tasks",
    "phase",
    "rep",
    "seconds",
    "status",
    "message",
]


def _noop(i):
    return i


def _predict(forecaster, steps):
    return forecaster.predict(steps=steps).to_numpy()


def _payload(name: str, k: int):
    """
    Return the list of delayed tasks of a payload.
    """

    from joblib import delayed

    if name == "noop":
        return [delayed(_noop)(i) for i in range(k)]
    if name == "few":
        return [delayed(_noop)(i) for i in range(4)]
    if name == "forecaster":
        from lightgbm import LGBMRegressor

        from cases import single_data
        from skforecast.recursive import ForecasterRecursive

        y, _ = single_data(1_000)
        forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(n_estimators=100, n_jobs=1, verbose=-1), lags=24
        )
        forecaster.fit(y=y)
        return [delayed(_predict)(forecaster, 24) for _ in range(k)]
    raise ValueError(name)


def _imports() -> None:
    # Libraries already loaded in a typical session before parallelizing.
    import joblib  # noqa: F401
    import lightgbm  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    import skforecast  # noqa: F401


def k_values(cpu: int) -> list[int]:
    values = {2, 4, 8, 16, 21, 22, cpu - 1, cpu}

    return sorted(k for k in values if 2 <= k <= cpu)


def cold_child(k: int, payload: str) -> None:
    from joblib import Parallel

    _imports()
    tasks = _payload(payload, k)
    t0 = time.perf_counter()
    Parallel(n_jobs=k)(tasks)
    seconds = time.perf_counter() - t0
    print(
        "RESULT " + json.dumps({"seconds": seconds, "n_tasks": len(tasks)}), flush=True
    )


def warm_child(args, log: CsvLog, ctx: dict) -> None:
    from joblib import Parallel

    _imports()
    reps = 1 if args.pilot else N_REPS
    warmup = 1 if args.pilot else N_WARMUP
    cells = [
        (k, p)
        for k in k_values(ctx["cpu_count"])
        for p in ("noop", "forecaster", "few")
    ]
    for k, payload in shuffled(cells):
        tasks = _payload(payload, k)
        for phase, i in [("warmup", j) for j in range(warmup)] + [
            ("rep", j) for j in range(reps)
        ]:
            t0 = time.perf_counter()
            Parallel(n_jobs=k)(tasks)
            log.write(
                {
                    **ctx,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "mode": "warm",
                    "payload": payload,
                    "k": k,
                    "n_tasks": len(tasks),
                    "phase": phase,
                    "rep": i,
                    "seconds": time.perf_counter() - t0,
                    "status": "ok",
                }
            )
        print(f"warm k={k} {payload}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cold", type=int, default=None, help="internal")
    parser.add_argument("--payload", default=None, help="internal")
    parser.add_argument("--warm", action="store_true", help="internal")
    args = parser.parse_args()

    allowed = set_affinity(args.affinity)
    if args.cold is not None:
        cold_child(args.cold, args.payload)
        return 0

    import joblib

    cpu = joblib.cpu_count()
    out_dir = RESULTS_DIR / ("pilot" if args.pilot else "")
    tag = f"_aff{args.affinity}" if args.affinity else ""
    ctx = {"affinity": len(allowed), "cpu_count": cpu}

    if args.warm:
        log = CsvLog(out_dir / f"block_A{tag}_warm.csv", FIELDS)
        warm_child(args, log, ctx)
        log.close()
        return 0

    log = CsvLog(out_dir / f"block_A{tag}_cold.csv", FIELDS)
    reps = 1 if args.pilot else N_REPS
    script = str(Path(__file__).resolve())
    aff = ["--affinity", str(args.affinity)] if args.affinity else []
    for rep in range(reps):
        cells = [(k, p) for k in k_values(cpu) for p in ("noop", "forecaster", "few")]
        for k, payload in shuffled(cells, seed=rep):
            cmd = [sys.executable, script, "--cold", str(k), "--payload", payload, *aff]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
            row = {
                **ctx,
                "timestamp": time.strftime("%H:%M:%S"),
                "mode": "cold",
                "payload": payload,
                "k": k,
                "phase": "rep",
                "rep": rep,
            }
            if line:
                res = json.loads(line[0][7:])
                row.update(seconds=res["seconds"], n_tasks=res["n_tasks"], status="ok")
            else:
                row.update(status="error", message=out.stderr[-300:])
            log.write(row)
            print(f"cold rep={rep} k={k} {payload}: {row.get('seconds')}", flush=True)
    log.close()

    cmd = [sys.executable, script, "--warm", *aff, *(["--pilot"] if args.pilot else [])]
    subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
