"""
Benchmark blocks B to G of the n_jobs review (block A, the fixed cost of
starting the workers, is in bench_spawn.py).

Usage
-----
python bench.py --block B [--affinity 4] [--pilot] [--resume]

Each cell is one operation (backtesting, fit of a Direct forecaster or grid
search) with one value of `n_jobs`. A cell runs `N_WARMUP` untimed-for-the-
median calls (stored as phase "warmup", which includes starting the workers
when `n_jobs` changes) and `N_REPS` timed calls. Cells are run in a seeded
shuffled order. Block C (oversubscription) runs every cell in its own
subprocess with a timeout, because some combinations can take minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

from common import (
    N_REPS,
    N_WARMUP,
    RESULTS_DIR,
    CsvLog,
    check_repo_skforecast,
    n_jobs_values,
    physical_cores_in_affinity,
    set_affinity,
    shuffled,
)

FIELDS = [
    "timestamp",
    "block",
    "cell_id",
    "case_id",
    "affinity",
    "cpu_count",
    "physical",
    "op",
    "forecaster",
    "estimator",
    "size",
    "refit",
    "n_jobs_label",
    "n_jobs",
    "n_jobs_resolved",
    "inner_n_jobs",
    "inner_resolved",
    "n_folds",
    "phase",
    "rep",
    "seconds",
    "status",
    "message",
    "preds_file",
]
CASE_KEYS = ["op", "forecaster", "estimator", "size", "refit"]
PILOT_LABELS = {"1", "cpu-1", "1|1", "cpu-1|1", "auto|auto"}
# --lean (applied after blocks B and C of the full machine were measured):
# -1, physical cores and the cap of 8 gave no new information there
# (21 ~ 22 >= 16 > 8), and 'auto' resolves deterministically to 1 or
# cpu_count - 1, so its time is that of the twin cell (the resolution is
# checked in summarize.py). With affinity, -1 and physical are kept because
# on 4 or 8 logical processors they differ meaningfully from n - 1.
LEAN_DROP_FULL = {"-1", "physical", "cap8", "-1|1", "physical|1", "auto"}
LEAN_DROP_AFFINITY = {"cap8", "auto"}


# Blocks
# ==============================================================================
def _cells(base: dict, values: list[tuple]) -> list[dict]:
    return [{**base, "n_jobs_label": label, "n_jobs": value} for label, value in values]


def block_b(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    ForecasterRecursive and ForecasterRecursiveClassifier with
    backtesting_forecaster: linear, non-linear, Pipeline and intermittent
    refit rules.
    """

    regs = [
        "ridge",
        "pipe_ridge",
        "hgb",
        "lgbm_1",
        "xgb_1",
        "rf_none",
        "et_none",
        "cat_1",
    ]
    clfs = ["logreg", "hgbc", "lgbmc_1"]
    sizes = ["S", "L"]
    if reduced:
        regs, clfs, sizes = ["ridge", "hgb", "lgbm_1"], [], ["L"]
    cells = []
    for est in regs + clfs:
        forecaster = "classifier" if est in clfs else "recursive"
        for size in sizes:
            if est == "hgbc" and size == "L":
                continue  # 45-64 s per sequential call in the pilot
            for refit in [False, True]:
                extra = (("cap8", 8),) if size == "L" and refit and cpu > 8 else ()
                base = {
                    "block": "B",
                    "op": "backtest",
                    "forecaster": forecaster,
                    "estimator": est,
                    "size": size,
                    "refit": refit,
                }
                cells += _cells(base, n_jobs_values(cpu, phys, extra))
    if not reduced:
        for est in ["hgb", "lgbm_1"]:
            base = {
                "block": "B",
                "op": "backtest",
                "forecaster": "recursive",
                "estimator": est,
                "size": "L",
                "refit": 3,
            }
            cells += _cells(base, [("1", 1), ("auto", "auto"), ("-1", -1)])

    return cells


def block_c(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    Oversubscription: estimators with their own threads inside the joblib
    workers of a backtesting with refit (where the estimator is fitted).
    """

    ests = [
        "lgbm_none",
        "lgbm_m1",
        "lgbm_4",
        "xgb_none",
        "xgb_m1",
        "rf_m1",
        "et_m1",
        "cat_m1",
        "hgb",
    ]
    if reduced:
        ests = ["lgbm_none", "xgb_none", "rf_m1", "hgb"]
    extra = tuple((str(k), k) for k in (4, 8) if k < cpu)
    cells = []
    for est in ests:
        base = {
            "block": "C",
            "op": "backtest",
            "forecaster": "recursive",
            "estimator": est,
            "size": "L",
            "refit": True,
        }
        cells += _cells(base, n_jobs_values(cpu, phys, extra))

    return cells


def block_d(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    ForecasterDirect and ForecasterDirectMultiVariate: parallel fit of the
    step estimators and its interaction with the backtesting workers.
    """

    cells = []
    fit_ests = ["ridge", "hgb", "lgbm_1", "lgbm_none", "rf_none"]
    sizes = ["S", "L"]
    if reduced:
        fit_ests, sizes = ["ridge", "lgbm_1"], ["L"]
    for est in fit_ests:
        for size in sizes:
            if est == "rf_none" and size == "L":
                continue  # 150 s per sequential call in the pilot
            extra = (("cap8", 8),) if size == "L" and cpu > 8 else ()
            base = {
                "block": "D",
                "op": "fit_direct",
                "forecaster": "direct",
                "estimator": est,
                "size": size,
                "refit": "",
            }
            cells += _cells(base, n_jobs_values(cpu, phys, extra))
    if reduced:
        return cells

    for est in ["ridge", "lgbm_1"]:
        for size in ["S", "L"]:
            base = {
                "block": "D",
                "op": "fit_dmv",
                "forecaster": "dmv",
                "estimator": est,
                "size": size,
                "refit": "",
            }
            cells += _cells(base, n_jobs_values(cpu, phys))

    # Backtesting: outer n_jobs (folds) and inner n_jobs (forecaster fit).
    # With the current rule, 'auto|auto' is outer 1 and inner cpu-1 for a
    # non-linear estimator. The "_fix" forecasters empty `estimators_` before
    # each fit (proposed fix, see patched.py); only refits are affected.
    combos = [
        ("auto|auto", "auto", "auto"),
        ("1|1", 1, 1),
        ("cpu-1|1", cpu - 1, 1),
        ("-1|1", -1, 1),
        ("physical|1", phys, 1),
        ("cpu-1|cpu-1", cpu - 1, cpu - 1),
    ]

    def add(op, forecaster, est, size, refit, selected):
        for label, outer, inner in selected:
            cells.append(
                {
                    "block": "D",
                    "op": op,
                    "forecaster": forecaster,
                    "estimator": est,
                    "size": size,
                    "refit": refit,
                    "n_jobs_label": label,
                    "n_jobs": outer,
                    "inner_n_jobs": inner,
                }
            )

    for est in ["ridge", "hgb", "lgbm_1"]:
        for size in ["S", "L"]:
            for refit in [False, True]:
                add("backtest", "direct", est, size, refit, combos)
            add("backtest", "direct_fix", est, size, True, combos[:1])
    for est in ["ridge", "lgbm_1"]:
        add("backtest_ms", "dmv", est, "S", True, combos[:3])
        add("backtest_ms", "dmv_fix", est, "S", True, combos[:1])

    return cells


def block_e(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    ForecasterRecursiveMultiSeries with backtesting_forecaster_multiseries.
    """

    ests = ["ridge", "hgb", "lgbm_1", "lgbm_none"]
    sizes = ["S", "L"]
    if reduced:
        ests, sizes = ["lgbm_1"], ["L"]
    cells = []
    for est in ests:
        for size in sizes:
            for refit in [False, True]:
                extra = (("cap8", 8),) if size == "L" and refit and cpu > 8 else ()
                base = {
                    "block": "E",
                    "op": "backtest_ms",
                    "forecaster": "multiseries",
                    "estimator": est,
                    "size": size,
                    "refit": refit,
                }
                cells += _cells(base, n_jobs_values(cpu, phys, extra))

    return cells


def block_f(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    ForecasterStats (backtesting_stats) and ForecasterEquivalentDate.
    """

    cells = []
    for est in ["arima", "sarimax"]:
        for size in ["S", "L"]:
            base = {
                "block": "F",
                "op": "backtest_stats",
                "forecaster": "stats",
                "estimator": est,
                "size": size,
                "refit": True,
            }
            cells += _cells(base, n_jobs_values(cpu, phys))
    for size in ["S", "L"]:
        for refit in [False, True]:
            base = {
                "block": "F",
                "op": "backtest",
                "forecaster": "equivalent_date",
                "estimator": "none",
                "size": size,
                "refit": refit,
            }
            cells += _cells(base, [("1", 1), ("auto", "auto"), ("cpu-1", cpu - 1)])

    return cells


def block_g(cpu: int, phys: int, reduced: bool) -> list[dict]:
    """
    Grid searches: one backtesting per candidate, the loky executor is
    reused between candidates.
    """

    cells = []
    for est in ["hgb", "lgbm_1"]:
        for refit in [False, True]:
            base = {
                "block": "G",
                "op": "grid",
                "forecaster": "recursive",
                "estimator": est,
                "size": "L",
                "refit": refit,
            }
            cells += _cells(base, n_jobs_values(cpu, phys))
    base = {
        "block": "G",
        "op": "grid_ms",
        "forecaster": "multiseries",
        "estimator": "lgbm_1",
        "size": "L",
        "refit": False,
    }
    cells += _cells(base, n_jobs_values(cpu, phys))

    return cells


BLOCKS = {
    "B": block_b,
    "C": block_c,
    "D": block_d,
    "E": block_e,
    "F": block_f,
    "G": block_g,
}


# Runner
# ==============================================================================
def case_id(meta: dict) -> str:
    return "|".join(f"{k}={meta.get(k, '')}" for k in CASE_KEYS)


def cell_id(meta: dict) -> str:
    return f"{meta['block']}|{case_id(meta)}|nj={meta['n_jobs_label']}"


def run_cell(meta: dict, log: CsvLog, ctx: dict, warmup: int, reps: int) -> None:
    """
    Run the warmup and timed calls of one cell and store its predictions.
    """

    from cases import build_cell, to_array

    import numpy as np

    cid = cell_id(meta)
    row = {
        **{k: meta.get(k, "") for k in FIELDS if k in meta},
        "cell_id": cid,
        "case_id": case_id(meta),
        **ctx,
    }
    try:
        setup, run, info = build_cell(meta)
    except Exception as exc:  # noqa: BLE001
        log.write(
            {**row, "phase": "setup", "status": "error", "message": repr(exc)[:300]}
        )
        return
    row.update(info)

    result = None
    phases = [("warmup", i) for i in range(warmup)] + [("rep", i) for i in range(reps)]
    for phase, i in phases:
        try:
            state = setup()
            t0 = time.perf_counter()
            result = run(state)
            seconds = time.perf_counter() - t0
            status, message = "ok", ""
        except Exception as exc:  # noqa: BLE001
            seconds, status, message = "", "error", repr(exc)[:300]
        log.write(
            {
                **row,
                "timestamp": time.strftime("%H:%M:%S"),
                "phase": phase,
                "rep": i,
                "seconds": seconds,
                "status": status,
                "message": message,
            }
        )
        if status != "ok":
            return

    preds_dir = ctx["preds_dir"]
    preds_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(cid.encode()).hexdigest()[:16] + ".npy"
    np.save(preds_dir / name, to_array(result))
    log.write({**row, "phase": "preds", "status": "ok", "preds_file": name})


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def _median_rep(csv_path: Path, cid: str) -> float | None:
    import csv

    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        times = [
            float(r["seconds"])
            for r in csv.DictReader(f)
            if r["cell_id"] == cid
            and r["phase"] in ("rep", "warmup")
            and r["status"] == "ok"
        ]

    return statistics.median(times) if times else None


def run_isolated(
    cells: list[dict],
    args,
    csv_path: Path,
    log_fields: list[str],
    ctx: dict,
    warmup: int,
    reps: int,
) -> None:
    """
    Run each cell in its own subprocess with a timeout of 10 times the
    sequential time of the same estimator (at least 30 s per call), so an
    oversubscribed cell cannot block the study.
    """

    by_est: dict[str, list[dict]] = {}
    for c in cells:
        by_est.setdefault(c["estimator"], []).append(c)

    done = CsvLog(csv_path, log_fields).done
    for est, group in by_est.items():
        seq = [c for c in group if c["n_jobs_label"] == "1"]
        rest = shuffled([c for c in group if c["n_jobs_label"] != "1"])
        t1 = None
        for c in seq + rest:
            cid = cell_id(c)
            if args.resume and cid in done:
                print(f"skip {cid}", flush=True)
            else:
                per_call = 600 if t1 is None else max(30.0, 10 * t1)
                timeout = per_call * (warmup + reps) + 90
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--block",
                    args.block,
                    "--cell",
                    cid,
                ]
                if args.affinity:
                    cmd += ["--affinity", str(args.affinity)]
                if args.pilot:
                    cmd += ["--pilot"]
                print(f"cell {cid} (timeout {timeout:.0f} s)", flush=True)
                proc = subprocess.Popen(cmd)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_tree(proc)
                    log = CsvLog(csv_path, log_fields)
                    log.write(
                        {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "block": c["block"],
                            "cell_id": cid,
                            "case_id": case_id(c),
                            **ctx,
                            **{k: c.get(k, "") for k in log_fields if k in c},
                            "phase": "timeout",
                            "status": "timeout",
                            "message": f"killed after {timeout:.0f} s "
                            f"(per call budget {per_call:.0f} s)",
                        }
                    )
                    log.close()
            if c["n_jobs_label"] == "1":
                t1 = _median_rep(csv_path, cid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True, choices=sorted(BLOCKS))
    parser.add_argument("--affinity", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--subset",
        action="store_true",
        help="reduced cells (those measured with affinity) on the whole machine",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--lean", action="store_true", help="drop redundant n_jobs values"
    )
    parser.add_argument("--cell", default=None, help="internal: run one cell")
    args = parser.parse_args()

    allowed = set_affinity(args.affinity)

    # Load the OpenMP and BLAS runtimes before limiting them.
    import catboost  # noqa: F401
    import joblib
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import xgboost  # noqa: F401
    from threadpoolctl import threadpool_limits

    check_repo_skforecast()
    limiter = None
    if args.affinity:
        # On a real machine with this many cores, OpenBLAS and OpenMP would
        # start this many threads in the main process. Environment variables
        # are not used because loky would pass them to the workers.
        limiter = threadpool_limits(limits=len(allowed))

    cpu = joblib.cpu_count()
    phys = physical_cores_in_affinity()
    cells = BLOCKS[args.block](cpu, phys, reduced=args.affinity is not None or args.subset)
    if args.lean and args.cell is None:
        drop = LEAN_DROP_AFFINITY if args.affinity else LEAN_DROP_FULL
        cells = [c for c in cells if c["n_jobs_label"] not in drop]
    if args.pilot:
        cells = [c for c in cells if c["n_jobs_label"] in PILOT_LABELS]
        warmup, reps = 0, 1
    else:
        warmup, reps = N_WARMUP, N_REPS

    out_dir = RESULTS_DIR / ("pilot" if args.pilot else "")
    tag = f"_aff{args.affinity}" if args.affinity else ""
    csv_path = out_dir / f"block_{args.block}{tag}.csv"
    ctx = {
        "affinity": len(allowed),
        "cpu_count": cpu,
        "physical": phys,
        "preds_dir": out_dir / "preds" / f"{args.block}{tag}",
    }

    if args.block == "C" and args.cell is None:
        run_isolated(cells, args, csv_path, FIELDS, ctx, warmup, reps)
        return 0

    if args.cell is not None:
        cells = [c for c in cells if cell_id(c) == args.cell]
        if not cells:
            raise SystemExit(f"unknown cell {args.cell}")
    else:
        cells = shuffled(cells)

    log = CsvLog(csv_path, FIELDS)
    t_start = time.perf_counter()
    for k, meta in enumerate(cells, 1):
        cid = cell_id(meta)
        if args.resume and cid in log.done:
            continue
        t0 = time.perf_counter()
        run_cell(meta, log, ctx, warmup, reps)
        print(
            f"[{k}/{len(cells)}] {cid}: {time.perf_counter() - t0:.1f} s "
            f"(total {time.perf_counter() - t_start:.0f} s)",
            flush=True,
        )
    log.close()
    del limiter

    return 0


if __name__ == "__main__":
    sys.exit(main())
