"""
Block V1: one backtesting per fresh process (cold start), the situation of a
user script that runs a single backtesting: the loky workers do not exist
yet, so the call pays starting them.

In each child process the libraries are imported and the data are built
before timing; only the backtesting call is timed. After it, the child times
a sequential fit of the forecaster on the initial training set (the fit that
the backtesting already performs in the main process), which is the input of
the adaptive strategy E3.

With skforecast 0.27.0 and later (work threshold of `n_jobs='auto'`), the
child also records the number of jobs that 'auto' used (`auto_resolved`). To
calibrate the threshold, measure also `cpu-1` (`cpu_count() - 1` workers,
always in parallel, what 'auto' did before the threshold).

Usage: python bench_cold.py [--reps 5] [--values 1,auto,cpu-1] [--affinity N]
Output: results/block_V_cold.csv

With `--affinity N` (Windows only) the parent and every child run on N
logical processors (a laptop emulated on the big machine, as in bench.py), so
`cpu_count() - 1` and 'auto' see N processors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import RESULTS_DIR, CsvLog, set_affinity, shuffled

FIELDS = [
    "timestamp",
    "case_id",
    "op",
    "forecaster",
    "estimator",
    "size",
    "refit",
    "n_jobs_label",
    "n_jobs",
    "inner_n_jobs",
    "n_jobs_resolved",
    "inner_resolved",
    "n_folds",
    "rep",
    "seconds",
    "t_initial_fit",
    "auto_resolved",
    "status",
    "message",
]

CASES = [
    ("backtest", "recursive", "lgbm_1", "S", False),
    ("backtest", "recursive", "lgbm_1", "S", True),
    ("backtest", "recursive", "lgbm_1", "L", False),
    ("backtest", "recursive", "lgbm_1", "L", True),
    ("backtest", "recursive", "hgb", "L", True),
    ("backtest", "recursive", "rf_none", "L", True),
    ("backtest_ms", "multiseries", "lgbm_1", "S", True),
    ("backtest_ms", "multiseries", "lgbm_1", "L", True),
    ("backtest", "direct_fix", "lgbm_1", "S", True),
]
# For Direct the parallel level of 'auto' is the fit (inner); the outer
# n_jobs of the backtesting stays at 1 except for 'auto'. Only 1 and 'auto'
# are measured: the dependence of the start cost on the number of workers
# is measured directly in block A, and V1 validates the model
# "warm time + start cost" (8 and 4 workers and Direct without the fix were
# dropped before running, as they would not change any conclusion).
VALUES = {"1": 1, "auto": "auto", "cpu-1": "cpu-1"}


def metas(values: list[str]) -> list[dict]:
    from joblib import cpu_count

    out = []
    for op, forecaster, est, size, refit in CASES:
        for label in values:
            value = VALUES[label]
            if value == "cpu-1":
                value = max(1, cpu_count() - 1)
            meta = {
                "block": "V",
                "op": op,
                "forecaster": forecaster,
                "estimator": est,
                "size": size,
                "refit": refit,
                "n_jobs_label": label,
            }
            if forecaster.startswith("direct"):
                # 'auto' and cpu-1 parallelize the fit of the steps.
                meta["n_jobs"] = "auto" if value == "auto" else 1
                meta["inner_n_jobs"] = value
            else:
                meta["n_jobs"] = value
            meta["case_id"] = f"{op}|{forecaster}|{est}|{size}|refit={refit}"
            out.append(meta)
    return out


def child(meta: dict, affinity: int | None) -> None:
    allowed = set_affinity(affinity)

    import joblib  # noqa: F401
    import lightgbm  # noqa: F401
    import pandas  # noqa: F401
    import sklearn.ensemble  # noqa: F401

    if affinity is not None:
        # As in bench.py: on a real machine with this many processors, OpenMP
        # and BLAS would start this many threads in the main process.
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=len(allowed))

    import skforecast  # noqa: F401
    from cases import DIRECT_BT, MULTI, SINGLE, build_cell, multi_data, single_data

    # Record the jobs that 'auto' used, if this skforecast has the threshold.
    import skforecast.model_selection._validation as validation

    resolved = []
    if hasattr(validation, "limit_n_jobs_by_workload"):
        limit = validation.limit_n_jobs_by_workload

        def spy(**kwargs):
            out = limit(**kwargs)
            resolved.append(out)
            return out

        validation.limit_n_jobs_by_workload = spy

    setup, run, info = build_cell(meta)
    state = setup()
    t0 = time.perf_counter()
    run(state)
    seconds = time.perf_counter() - t0

    # Sequential fit on the initial training set (input of strategy E3).
    forecaster = setup()
    if meta["op"] == "backtest_ms":
        cfg = MULTI[meta["size"]]
        series, exog = multi_data(cfg["n_series"], cfg["n"])
        data = {
            "series": series.iloc[: cfg["initial"]],
            "exog": exog.iloc[: cfg["initial"]],
        }
    else:
        cfg = (DIRECT_BT if meta["forecaster"].startswith("direct") else SINGLE)[
            meta["size"]
        ]
        y, exog = single_data(cfg["n"])
        data = {"y": y.iloc[: cfg["initial"]], "exog": exog.iloc[: cfg["initial"]]}
    if hasattr(forecaster, "n_jobs") and meta["forecaster"].startswith("direct"):
        forecaster.n_jobs = 1
    t0 = time.perf_counter()
    forecaster.fit(**data)
    t_fit = time.perf_counter() - t0

    print(
        "RESULT "
        + json.dumps(
            {
                "seconds": seconds,
                "t_initial_fit": t_fit,
                "auto_resolved": resolved[-1] if resolved else "",
                **{
                    k: (v if not hasattr(v, "item") else v.item())
                    for k, v in info.items()
                },
            },
            default=str,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--values", default="1,auto", help="comma separated: 1, auto, cpu-1"
    )
    parser.add_argument(
        "--affinity", type=int, default=None, help="logical processors (Windows)"
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="comma separated substrings of the case ids to run (default: all)",
    )
    parser.add_argument("--child", default=None, help="internal")
    args = parser.parse_args()

    if args.child is not None:
        child(json.loads(args.child), args.affinity)
        return 0

    # The children inherit the affinity of the parent; `metas` needs it for
    # `cpu_count() - 1`.
    set_affinity(args.affinity)
    child_args = [] if args.affinity is None else ["--affinity", str(args.affinity)]

    log = CsvLog(RESULTS_DIR / "block_V_cold.csv", FIELDS)
    script = str(Path(__file__).resolve())
    t_start = time.perf_counter()
    all_metas = metas(args.values.split(","))
    if args.cases:
        wanted = args.cases.split(",")
        all_metas = [m for m in all_metas if any(w in m["case_id"] for w in wanted)]
    for rep in range(args.reps):
        for k, meta in enumerate(shuffled(all_metas, seed=100 + rep), 1):
            out = subprocess.run(
                [sys.executable, script, "--child", json.dumps(meta), *child_args],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
            row = {
                **{k2: meta.get(k2, "") for k2 in FIELDS if k2 in meta},
                "timestamp": time.strftime("%H:%M:%S"),
                "rep": rep,
            }
            if line:
                row.update(json.loads(line[0][7:]), status="ok")
            else:
                row.update(status="error", message=out.stderr[-300:])
            log.write(row)
            print(
                f"rep={rep} [{k}/{len(all_metas)}] {meta['case_id']} "
                f"nj={meta['n_jobs_label']}: "
                f"{row.get('seconds')} (total {time.perf_counter() - t_start:.0f} s)",
                flush=True,
            )
    log.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
