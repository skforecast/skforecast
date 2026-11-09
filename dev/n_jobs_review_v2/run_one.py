"""
Run one benchmark case in this process and print one JSON line per measure.

Cold mode (`--mode cold`): a single config (n_jobs) is timed once. The parent
launches a fresh process for each measure, so the workers of loky are started
and import skforecast inside the timed call.

Warm mode (`--mode warm`): every config is run once untimed (warm-up), then
`--reps` rounds in random order are timed in this process, reusing the workers.

The effective `n_jobs` of each `joblib.Parallel` created by skforecast is
recorded by patching the `Parallel` names of its modules.
"""

import argparse
import json
import os
import random
import threading
import time

import joblib

USED = []


class RecParallel(joblib.Parallel):
    def __init__(self, n_jobs=None, **kwargs):
        USED.append(n_jobs)
        super().__init__(n_jobs=n_jobs, **kwargs)


def _patch():
    import skforecast.direct._forecaster_direct as fd
    import skforecast.direct._forecaster_direct_multivariate as fdm
    import skforecast.model_selection._validation as val
    for mod in (val, fd, fdm):
        mod.Parallel = RecParallel


def _rss_tree_mb():
    """RSS (MB) of this process and all its descendants, read from /proc."""
    me = os.getpid()
    children = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(pid))
        except (OSError, IndexError, ValueError):
            pass
    total, stack = 0, [me]
    while stack:
        pid = stack.pop()
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except OSError:
            pass
        stack.extend(children.get(pid, []))
    return total / 1024


class MemSampler(threading.Thread):
    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period, self.peak, self._stop_evt = period, 0.0, threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, _rss_tree_mb())
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join()
        self.peak = max(self.peak, _rss_tree_mb())
        return self.peak


def _parse_n_jobs(value):
    return value if value == "auto" else int(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--configs", required=True, help="comma separated n_jobs values")
    p.add_argument("--mode", choices=["cold", "warm"], default="cold")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem", action="store_true")
    args = p.parse_args()

    t_import = time.perf_counter()
    import cases
    _patch()
    spec = json.loads(args.spec)
    run = cases.build(spec)
    t_import = time.perf_counter() - t_import

    configs = [_parse_n_jobs(c) for c in args.configs.split(",")]
    ncpu = len(os.sched_getaffinity(0))

    def measure(n_jobs, rep):
        USED.clear()
        sampler = MemSampler() if args.mem else None
        if sampler:
            sampler.start()
        t0 = time.perf_counter()
        fp = run(n_jobs)
        dt = time.perf_counter() - t0
        peak = sampler.stop() if sampler else None
        print(json.dumps({
            "n_jobs": n_jobs, "rep": rep, "seconds": round(dt, 4),
            "used": sorted(set(USED)), "n_parallel": len(USED), "fp": fp,
            "ncpu": ncpu, "mode": args.mode, "peak_mb": peak,
            "setup_s": round(t_import, 3),
        }), flush=True)

    if args.mode == "cold":
        assert len(configs) == 1
        measure(configs[0], args.seed)
        return

    rng = random.Random(args.seed)
    for c in configs:
        run(c)  # warm-up: starts the workers and fills caches
    for rep in range(args.reps):
        order = configs[:]
        rng.shuffle(order)
        for c in order:
            measure(c, rep)


if __name__ == "__main__":
    main()
