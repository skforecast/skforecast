"""
Orchestrate a small benchmark of the n_jobs review (v2).

Example:
    python bench.py --name h1_rf --spec '{"kind": "grid", ...}' \
        --configs 1,auto --reps 5 --mode cold --cpus 4

Cold mode launches a fresh process per measure (configs interleaved in random
order within each repetition). Warm mode launches one process that warms up
every config and then times them interleaved. `--cpus k` pins the processes
to the first k CPUs with `taskset`, so `joblib.cpu_count()` returns k.

Results are appended to `results/<name>.csv` and a summary (median and min per
config) is printed and appended to `results/<name>.md`.
"""

import argparse
import csv
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def uptime():
    with open("/proc/loadavg") as f:
        return " ".join(f.read().split()[:3])


def launch(spec, configs, mode, cpus, reps, seed, mem, timeout):
    cmd = []
    if cpus:
        cmd += ["taskset", "-c", f"0-{cpus - 1}"]
    cmd += [
        sys.executable, os.path.join(HERE, "run_one.py"),
        "--spec", json.dumps(spec), "--configs", ",".join(map(str, configs)),
        "--mode", mode, "--reps", str(reps), "--seed", str(seed),
    ]
    if mem:
        cmd.append("--mem")
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    # NOTE: A new session, so that a timeout kills the loky workers too.
    proc = subprocess.Popen(
        cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    out = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    rows = []
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    if out.returncode != 0 or not rows:
        sys.stderr.write(out.stderr[-3000:])
        raise RuntimeError(f"run_one failed: {out.returncode}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--configs", required=True)
    p.add_argument("--mode", choices=["cold", "warm"], default="cold")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cpus", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem", action="store_true")
    p.add_argument("--timeout", type=float, default=170)
    p.add_argument("--note", default="")
    args = p.parse_args()

    spec = json.loads(args.spec)
    configs = args.configs.split(",")
    load_before = uptime()
    t_start = time.time()
    rows = []
    if args.mode == "cold":
        rng = random.Random(args.seed)
        for rep in range(args.reps):
            order = configs[:]
            rng.shuffle(order)
            for c in order:
                remaining = args.timeout - (time.time() - t_start)
                if remaining <= 0:
                    raise TimeoutError("budget exceeded")
                rows += launch(spec, [c], "cold", args.cpus, 1, rep, args.mem, remaining)
    else:
        rows = launch(
            spec, configs, "warm", args.cpus, args.reps, args.seed, args.mem, args.timeout
        )
    load_after = uptime()

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"{args.name}.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "mode", "cpus", "n_jobs", "rep", "seconds", "used",
                        "n_parallel", "fp", "peak_mb", "spec"])
        for r in rows:
            w.writerow([args.name, r["mode"], r["ncpu"], r["n_jobs"], r["rep"],
                        r["seconds"], r["used"], r["n_parallel"], r["fp"],
                        r["peak_mb"], json.dumps(spec)])

    lines = [
        f"### {args.name} ({args.mode}, {rows[0]['ncpu']} CPU, {args.reps} reps) {args.note}",
        "",
        f"spec: `{json.dumps(spec)}`  ",
        f"load before/after: {load_before} / {load_after}",
        "",
        "| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fps = set()
    for c in configs:
        rs = [r for r in rows if str(r["n_jobs"]) == c]
        secs = [r["seconds"] for r in rs]
        used = sorted({str(u) for r in rs for u in r["used"]})
        peaks = [r["peak_mb"] for r in rs if r["peak_mb"] is not None]
        fp = sorted({r["fp"] for r in rs})
        works = [r.get("est_work_s") for r in rs if r.get("est_work_s") is not None]
        work = f"{statistics.median(works):.2f}" if works else "-"
        fps.update(fp)
        lines.append(
            f"| {c} | {','.join(used)} | {statistics.median(secs):.3f} | "
            f"{min(secs):.3f} | {max(secs):.3f} | "
            f"{(max(peaks) if peaks else float('nan')):.0f} | {work} | {','.join(fp)} |"
        )
    lines.append("")
    lines.append(
        "Same result in every config: " + ("yes" if len(fps) == 1 else f"NO {fps}")
    )
    lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, f"{args.name}.md"), "a") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
