"""
Run the whole n_jobs review with one command.

Usage (from any directory, with the conda environment of the repository):

    set PYTHONPATH=<repository root>
    python dev/n_jobs_review/run_all.py                    # full machine, blocks A to G
    python dev/n_jobs_review/run_all.py --affinity 4,8     # laptops, blocks A to E
    python dev/n_jobs_review/run_all.py --pilot --affinity 22,4,8   # calibration run
    python dev/n_jobs_review/run_all.py --resume           # continue an interrupted run

For every affinity it first runs the worker probe (gate G1): if the CPU
count seen by joblib does not match the affinity, the blocks of that
affinity are skipped. Results go to results/ (results/pilot/ with --pilot).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import RESULTS_DIR, dump_json, machine_info  # noqa: E402

ALL_BLOCKS = ["A", "B", "C", "D", "E", "F", "G"]
AFFINITY_BLOCKS = ["A", "B", "C", "D", "E"]


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--lean", action="store_true", help="drop redundant n_jobs values"
    )
    parser.add_argument(
        "--affinity", default="22", help="comma separated, 22 = whole machine"
    )
    parser.add_argument("--blocks", default=None, help="comma separated subset of A-G")
    args = parser.parse_args()

    py = sys.executable
    out_dir = RESULTS_DIR / ("pilot" if args.pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(machine_info(), out_dir / "machine.json")

    t_start = time.perf_counter()
    for aff in [int(a) for a in args.affinity.split(",")]:
        full = aff >= 22
        aff_args = [] if full else ["--affinity", str(aff)]
        if run([py, "probe_workers.py", *aff_args]) != 0:
            print(f"G1 FAIL for affinity {aff}: blocks skipped", flush=True)
            continue
        blocks = ALL_BLOCKS if full else AFFINITY_BLOCKS
        if args.blocks:
            blocks = [b for b in blocks if b in args.blocks.split(",")]
        for block in blocks:
            t0 = time.perf_counter()
            print(
                f"BLOCK {block} affinity {aff} START {time.strftime('%H:%M:%S')}",
                flush=True,
            )
            extra = ["--pilot"] if args.pilot else []
            if block == "A":
                rc = run([py, "bench_spawn.py", *aff_args, *extra])
            else:
                extra += ["--resume"] if args.resume else []
                extra += ["--lean"] if args.lean else []
                rc = run([py, "bench.py", "--block", block, *aff_args, *extra])
            print(
                f"BLOCK {block} affinity {aff} END rc={rc} "
                f"{(time.perf_counter() - t0) / 60:.1f} min",
                flush=True,
            )

    print(f"ALL DONE {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
