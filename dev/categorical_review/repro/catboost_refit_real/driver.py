"""
BUG-catboost-refit-categorical with real data: runs a list of (case, variant)
of run_variant.py one after another, each in its own process with a time
limit. On timeout the process tree is killed (the loky workers included, which
otherwise keep a core busy) and the status is recorded as `timeout`.

Usage: python driver.py <stream> <seconds> <case>:<variant>[,<variant>...] ...

Example (the session of 2026-11-25 ran six streams like this at the same time):
    python driver.py A 2400 bikeS_guide_fixed:ref,manual,bt1,bt2,prefit1,ref_s7

The status of each run is appended to `_out/stream_<stream>.jsonl` and its
output goes to `_out/logs/`. compare.py reads both. In that session the `bt2`
runs were stopped from outside after 600 s (1,600 s for the case with 1,000
iterations) instead of waiting for the limit of the driver, so they appear as
`error 1` in the stream files and are listed in `watchdog.log`.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CB_WORK", HERE / "_out"))
(WORK / "logs").mkdir(parents=True, exist_ok=True)
stream, timeout = sys.argv[1], int(sys.argv[2])
folder = f"preds_pandas{pd.__version__.split('.')[0]}{os.environ.get('CB_TAG', '')}"
log = WORK / f"stream_{stream}.jsonl"
tasks = []
for spec in sys.argv[3:]:
    case, variants = spec.split(":")
    tasks += [(case, v) for v in variants.split(",")]

for case, variant in tasks:
    t0 = time.perf_counter()
    with open(WORK / "logs" / f"{stream}__{case}__{variant}.log", "w") as out:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "run_variant.py"), case, variant],
            stdout=out, stderr=subprocess.STDOUT, cwd=str(HERE),
        )
        try:
            code = proc.wait(timeout=timeout)
            status = "ok" if code == 0 else f"error {code}"
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                proc.kill()
            status = "timeout"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "case": case, "variant": variant, "status": status, "folder": folder,
            "seconds": round(time.perf_counter() - t0, 1),
        }) + "\n")
with open(log, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"stream": stream, "status": "DONE"}) + "\n")
