"""
BUG-catboost-refit-categorical with real data: compares the predictions saved
by run_variant.py against their reference and prints one block per case.

Usage: python compare.py [<work folder>]   (default `_out/`)

Only the folders `preds_pandas<major>*` of the pandas that runs the script are
read: the pickles of pandas 3 cannot be read with pandas 2. Run it once with
each environment. RESULTADOS.txt is the output of the session of 2026-11-25.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_out"
MAJOR = pd.__version__.split(".")[0]
REF = {
    "manual": "ref", "bt1": "ref", "bt2": "ref", "bt_auto": "ref", "prefit1": "ref",
    "ref_s7": "ref", "prefit_norefit": "norefit", "oh_bt1": "oh_ref", "oh_ref": "ref",
}
ORDER = ["ref", "manual", "bt1", "bt2", "bt_auto", "prefit1", "ref_s7", "norefit",
         "prefit_norefit", "oh_ref", "oh_bt1"]
# Streams of the session of 2026-11-25, whose files do not have the folder.
STREAM_FOLDER = {"P3": "preds_pandas3", "V27": "preds_pandas2_v0270"}

status = {}
for f in sorted(WORK.glob("stream_*.jsonl")):
    stream = f.stem.replace("stream_", "")
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if "case" in r:
            folder = r.get("folder") or STREAM_FOLDER.get(stream, "preds_pandas2")
            status[(folder, r["case"], r["variant"])] = r

# Runs stopped from outside (see driver.py): "p2 <case> bt2 killed after N s; ..."
stopped = {}
if (WORK / "watchdog.log").exists():
    for line in open(WORK / "watchdog.log", encoding="utf-8-sig"):
        m = re.match(r"(p\d) (\S+) (\S+) killed after (\d+) s", line)
        if m:
            stopped[(f"preds_pandas{m[1][1:]}", m[2], m[3])] = int(m[4])


def mae(r):
    return float(np.mean(np.abs(r["wide"].to_numpy() - r["truth"].to_numpy())))


print("python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__)
for folder in sorted({k[0] for k in status if k[0].startswith(f"preds_pandas{MAJOR}")}):
    res = {}
    for f in sorted((WORK / folder).glob("*.pkl")):
        r = pickle.load(open(f, "rb"))
        res[(r["case"], r["variant"])] = r
    print(f"\n===== {folder} =====")
    for case in sorted({c for fo, c, _ in status if fo == folder}):
        base = res.get((case, "ref"))
        head = ""
        if base is not None:
            head = (f"folds {int(base['fold'].nunique())}, points {base['wide'].size}, "
                    f"mean |y| {np.mean(np.abs(base['truth'].to_numpy())):.1f}")
        print(f"\n{case}  ({head})")
        for v in [v for v in ORDER if (folder, case, v) in status]:
            st = status[(folder, case, v)]
            r = res.get((case, v))
            if st["status"] != "ok" or r is None:
                if (folder, case, v) in stopped:
                    why = f"BLOCKED (stopped from outside after {stopped[(folder, case, v)]} s)"
                elif st["status"] == "timeout":
                    why = f"BLOCKED (time limit of the driver, {st['seconds']:.0f} s)"
                else:
                    why = f"FAILED ({st['status']} after {st['seconds']:.0f} s, see its log)"
                print(f"  {v:15s} {why}")
                continue
            line = f"  {v:15s} MAE {mae(r):10.4f}  ({st['seconds']:6.0f} s)"
            ref = res.get((case, REF.get(v, "")))
            if ref is not None:
                a, b = r["wide"].to_numpy(float), ref["wide"].to_numpy(float)
                if np.array_equal(a, b):
                    line += f"  identical to {REF[v]}"
                else:
                    d = np.abs(a - b)
                    line += (f"  vs {REF[v]}: mean |diff| {d.mean():.4f}, max {d.max():.4f}, "
                             f"MAE ratio {mae(r) / mae(ref):.3f}")
                    if r["fold"] is not None and v in ("bt1", "bt2", "bt_auto", "prefit1"):
                        fold = r["fold"].loc[r["wide"].index].to_numpy()
                        same = [
                            bool(np.array_equal(a[fold == k], b[fold == k]))
                            for k in np.unique(fold)
                        ]
                        line += (f", identical folds {sum(same)}/{len(same)} "
                                 f"{''.join('=' if s else 'x' for s in same)}")
            print(line)
