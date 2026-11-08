"""
Estimate the duration of the full run from the pilot (results/pilot/).

For every cell of the full matrix, the time of one call is taken from the
pilot cell of the same case: the sequential one (n_jobs 1) for cells that
resolve to one job, and the parallel one (cpu-1) otherwise. A cell costs
(N_WARMUP + N_REPS) calls plus a fixed overhead.

Usage: python estimate.py
"""

from __future__ import annotations

import sys

import pandas as pd

from bench import BLOCKS, case_id, cell_id  # noqa: F401
from common import N_REPS, N_WARMUP, RESULTS_DIR

OVERHEAD_S = 2.0
SPAWN_S = 2.5
SEQ_LABELS = {"1", "1|1"}
PAR_LABELS = ["cpu-1", "cpu-1|1", "auto|auto"]


def pilot_times(tag: str) -> pd.DataFrame:
    frames = []
    for path in (RESULTS_DIR / "pilot").glob(f"block_[B-G]{tag}.csv"):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[df["phase"].isin(["rep", "timeout"])].copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    return df


def estimate(tag: str, cpu: int, phys: int, reduced: bool) -> pd.DataFrame:
    df = pilot_times(tag)
    rows = []
    for block, builder in BLOCKS.items():
        if reduced and block in ("F", "G"):
            continue
        for meta in builder(cpu, phys, reduced):
            cid = case_id(meta)
            # "_fix" cases were not in the pilot: use the current-code case.
            lookup = cid.replace("direct_fix", "direct").replace("dmv_fix", "dmv")
            sub = df[df["case_id"] == lookup]
            # The large Direct backtesting went from 10 to 5 folds after the pilot.
            scale = (
                0.5
                if (
                    meta["op"] == "backtest"
                    and meta["forecaster"].startswith("direct")
                    and meta["size"] == "L"
                )
                else 1.0
            )
            # Estimators halved after the pilot: hgb 100 -> 50 iterations,
            # forests 50 -> 25 trees; DirectMultiVariate L 20 -> 10 series.
            if meta["estimator"] in ("hgb", "rf_none", "rf_m1", "et_none", "et_m1"):
                scale *= 0.5
            if meta["forecaster"].startswith("dmv") and meta["size"] == "L":
                scale *= 0.5

            def med(labels, sub=sub):
                s = sub[sub["n_jobs_label"].isin(labels)]["seconds"]
                return s.median() * scale if len(s) else None

            t_seq = med(SEQ_LABELS)
            t_par = med(["cpu-1", "cpu-1|1"])
            t_auto = med(["auto|auto"])
            if (sub["status"] == "timeout").any() and t_par is None:
                t_par = 10 * (t_seq or 30)
            label = meta["n_jobs_label"]
            known = [x for x in (t_seq, t_par) if x is not None]
            if label in SEQ_LABELS:
                t = t_seq
            elif label == "auto|auto":
                t = t_auto if t_auto is not None else t_par
            elif label == "auto":
                t = max(known) if known else None
            else:
                t = t_par if t_par is not None else t_seq
            rows.append(
                {
                    "block": block,
                    "case_id": cid,
                    "label": label,
                    "t_call": t,
                    "missing": t is None,
                }
            )
    out = pd.DataFrame(rows)
    # Pilot calls of parallel cells sometimes include starting the workers
    # (up to about 5 s with 21 workers). In the full run only the warmup
    # call pays it, so the timed calls are corrected by SPAWN_S.
    par = ~out["label"].isin(SEQ_LABELS)
    t = out["t_call"].fillna(0)
    t_warm = (t - SPAWN_S).clip(lower=0.2 * t)
    out["cell_s"] = N_WARMUP * t + N_REPS * t.where(~par, t_warm) + OVERHEAD_S

    return out


def main() -> int:
    total = 0.0
    for tag, cpu, phys, reduced in [
        ("", 22, 16, False),
        ("_aff4", 4, 2, True),
        ("_aff8", 8, 4, True),
    ]:
        est = estimate(tag, cpu, phys, reduced)
        if est.empty:
            continue
        by_block = est.groupby("block").agg(
            cells=("cell_s", "size"),
            minutes=("cell_s", lambda s: s.sum() / 60),
            missing=("missing", "sum"),
        )
        spawn = 3.0 if not reduced else 2.0
        print(f"\naffinity {tag or 'full'} (block A about {spawn:.0f} min)")
        print(by_block.round(1).to_string())
        sub = by_block["minutes"].sum() + spawn
        print(f"subtotal {sub:.0f} min")
        total += sub
    print(f"\nTOTAL {total:.0f} min ({total / 60:.1f} h)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
