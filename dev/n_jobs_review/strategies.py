"""
Block V2: evaluation of strategies to choose n_jobs automatically, on the
measured cells (no new measurements).

Scenarios
---------
- warm: the loky workers already exist (searches, repeated calls). Time of a
  choice = median of the measured cell (full machine, blocks B, D, E, F, G).
- cold: a single call in a fresh session. Time = warm time + cost of starting
  k workers that import skforecast (block A, interpolated in k). Block V1
  measures some cases directly to validate this model.

Strategies
----------
- E0: current `select_n_jobs_backtesting` / `select_n_jobs_fit_forecaster`.
- E1: E0 with the Direct fix (estimators_ emptied before fit, "direct_fix"
  cells) and "cheap" instead of "linear": closed-form linear regressors
  (LinearModel, not wrapped in a Pipeline) use 1; LogisticRegression and
  pipelines are treated like any other estimator.
- E2: E1 with the number of workers capped by the number of tasks
  (folds in backtesting, steps in the fit of Direct).
- E3(T): E2, but 1 if the estimated sequential work is below T seconds. The
  estimate is the sequential time of the cell; a real implementation would
  extrapolate it from the first fold, run in the main process.

Usage: python strategies.py
Writes results/strategies_cells.csv and results/strategies_summary.md
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RESULTS_DIR, md_table
from summarize import cell_table, load_cells

CPU = 22
CHEAP_LINEAR = {"ridge"}  # closed-form LinearModel regressors measured
THRESHOLDS = [1.0, 2.0, 5.0, 10.0]


def spawn_model() -> callable:
    """
    Cold cost of starting k workers that import skforecast (block A,
    payload "forecaster", full machine), linear interpolation in k.
    """

    a = pd.read_csv(RESULTS_DIR / "block_A_cold.csv")
    a = a[
        (a["payload"] == "forecaster") & (a["phase"] == "rep") & (a["status"] == "ok")
    ]
    med = a.groupby("k")["seconds"].median()
    ks, ts = med.index.to_numpy(dtype=float), med.to_numpy()

    factor = cold_factor(float(np.interp(21, ks, ts)))

    def cost(k: float) -> float:
        if k <= 1:
            return 0.0
        return factor * float(np.interp(k, ks, ts))

    return cost


def cold_factor(spawn_21: float) -> float:
    """
    Share of the start cost of block A that a real backtesting pays, measured
    in block V1: median of (cold time - warm time) / start cost of 21
    workers, over the V1 cases with 'auto'. The first workers start working
    while the rest are still starting, so the share is below 1.
    """

    path = RESULTS_DIR / "block_V_cold.csv"
    if not path.exists():
        return 1.0
    v = pd.read_csv(path)
    v = v[(v["status"] == "ok") & (v["n_jobs_label"] == "auto")]
    cold = v.groupby("case_id")["seconds"].median()
    cells = cell_table(load_cells(RESULTS_DIR))
    cells = cells[cells["affinity"] >= CPU]
    shares = []
    for case, t_cold in cold.items():
        op, fc, est, size, refit = case.split("|")
        label = "auto"
        if fc == "direct_fix":
            label = "auto|auto"
        cid = f"op={op}|forecaster={fc}|estimator={est}|size={size}|{refit}"
        row = cells[
            (cells["case_id"] == cid) & (cells["n_jobs_label"].isin([label, "cpu-1"]))
        ]
        if len(row):
            shares.append((t_cold - row["median"].iloc[0]) / spawn_21)

    return float(np.median(shares)) if shares else 1.0


def label_k(label: str, cpu: int = CPU) -> int:
    """
    Number of workers of a label (outer level for Direct backtesting).
    """

    outer = label.split("|")[0]
    return {
        "1": 1,
        "cpu-1": cpu - 1,
        "-1": cpu,
        "physical": 16,
        "cap8": 8,
        "4": 4,
        "8": 8,
    }.get(outer, 1)


def load() -> pd.DataFrame:
    df = load_cells(RESULTS_DIR)
    cells = cell_table(df)
    cells = cells[(cells["affinity"] >= CPU) & cells["block"].isin(list("BDEFG"))]
    return cells


def case_rows(cells: pd.DataFrame) -> dict:
    """
    Per case: measured median by label, auto resolution and metadata.
    """

    out = {}
    for case, g in cells.groupby("case_id"):
        times = dict(zip(g["n_jobs_label"], g["median"]))
        first = g.iloc[0]
        auto_row = g[g["n_jobs_label"].isin(["auto"])]
        out[case] = {
            "block": first["block"],
            "op": first["op"],
            "forecaster": first["forecaster"],
            "estimator": first["estimator"],
            "size": first["size"],
            "refit": first["refit"],
            "n_folds": first["n_folds"],
            "times": times,
            "auto_resolved": (
                int(auto_row["n_jobs_resolved"].iloc[0])
                if len(auto_row) and pd.notna(auto_row["n_jobs_resolved"].iloc[0])
                else None
            ),
        }
    return out


def current_auto_label(c: dict) -> str | None:
    """
    Label whose time is that of the current 'auto' (E0).
    """

    if c["forecaster"] in ("direct", "dmv") and c["op"].startswith("backtest"):
        return "auto|auto"
    if "auto" in c["times"]:
        return "auto"
    res = c["auto_resolved"]
    if res is None:
        res = resolve_auto(c)
    return "1" if res == 1 else "cpu-1"


def resolve_auto(c: dict) -> int:
    """
    Resolution of 'auto' for cells measured without it (lean run), with the
    same functions skforecast uses.
    """

    from cases import build_cell

    meta = {
        "op": c["op"],
        "forecaster": c["forecaster"],
        "estimator": c["estimator"],
        "size": c["size"],
        "refit": _refit(c["refit"]),
        "n_jobs": "auto",
    }
    if c["op"] in ("fit_direct", "fit_dmv"):
        meta["refit"] = ""
    _, _, info = build_cell(meta)
    return int(info["n_jobs_resolved"])


def _refit(value):
    if value in ("True", True):
        return True
    if value in ("False", False):
        return False
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def choose(strategy: str, c: dict, threshold: float | None = None) -> tuple[str, int]:
    """
    Label of the measured cell whose time a strategy gets, and the number of
    workers it starts (for the cold cost).
    """

    times = c["times"]
    e0 = current_auto_label(c)
    # Number of workers that the current 'auto' starts, and the equivalent
    # measured label when 'auto' was measured on its own.
    if e0 == "auto|auto":
        k0 = _direct_k(c)
        twin = e0
    elif e0 == "auto":
        res = c["auto_resolved"] if c["auto_resolved"] is not None else resolve_auto(c)
        k0 = res
        twin = "1" if res == 1 else "cpu-1"
    else:
        k0 = label_k(e0)
        twin = e0
    if strategy == "E0":
        return e0, k0

    # E1: Direct fix and "cheap" instead of "linear".
    label = twin
    if (
        c["forecaster"] in ("direct", "dmv")
        and c["op"].startswith("backtest")
        and str(c["refit"]) == "True"
    ):
        label = "fix:auto|auto"
    is_cheap = c["estimator"] in CHEAP_LINEAR
    if c["estimator"] in ("logreg", "pipe_ridge") and label == "1":
        label = "cpu-1"
    if c["forecaster"] == "multiseries" and is_cheap:
        label = "1"
    k = (
        1
        if label in ("1", "1|1")
        else (_direct_k(c) if label.endswith("auto|auto") else label_k(label))
    )

    # E2: cap by the number of tasks (folds or steps).
    if strategy in ("E2", "E3") and k > 1:
        n_tasks = int(c["n_folds"])
        if label.endswith("auto|auto"):
            # Direct backtesting: the parallel level is the fit, one task per step.
            from cases import DIRECT_BT, DMV

            table = DMV if c["forecaster"].startswith("dmv") else DIRECT_BT
            n_tasks = table[c["size"]]["steps"]
        k = min(k, n_tasks)

    # E3: sequential when the estimated sequential work is small.
    if strategy == "E3":
        t_seq = times.get("1", times.get("1|1"))
        if t_seq is not None and t_seq < threshold:
            return ("1" if "1" in times else "1|1"), 1

    return label, k


def _direct_k(c: dict) -> int:
    # Inner parallel fit of Direct: 'auto' starts cpu - 1 workers unless the
    # estimator is linear or LightGBM with its own threads.
    return 1 if c["estimator"] in ("ridge", "lgbm_none") else CPU - 1


def time_of(c: dict, label: str, fix_times: dict) -> float | None:
    if label.startswith("fix:"):
        return fix_times.get(label[4:])
    t = c["times"].get(label)
    if t is None and label == "cpu-1":
        t = c["times"].get("auto")
    return t


def evaluate() -> pd.DataFrame:
    cells = load()
    cases = case_rows(cells)
    spawn = spawn_model()
    rows = []
    for case, c in cases.items():
        if c["forecaster"] in ("direct_fix", "dmv_fix"):
            continue
        fix_case = case.replace("forecaster=direct", "forecaster=direct_fix").replace(
            "forecaster=dmv", "forecaster=dmv_fix"
        )
        fix_times = cases.get(fix_case, {}).get("times", {})
        measured = {lab: t for lab, t in c["times"].items() if pd.notna(t)}
        if not measured:
            continue
        t_seq = measured.get("1", measured.get("1|1"))
        strategies = [("E0", None), ("E1", None), ("E2", None)] + [
            ("E3", t) for t in THRESHOLDS
        ]
        for scenario in ("warm", "cold"):

            def total(label, k):
                t = time_of(c, label, fix_times)
                if t is None:
                    return None
                return t + (spawn(k) if scenario == "cold" else 0.0)

            options = {
                lab: total(lab, _direct_k(c) if lab == "auto|auto" else label_k(lab))
                for lab in measured
            }
            options = {k: v for k, v in options.items() if v is not None}
            best = min(options.values())
            seq = (
                total("1" if "1" in measured else "1|1", 1)
                if t_seq is not None
                else None
            )
            for name, thr in strategies:
                label, k = choose(name, c, thr)
                t = total(label, k)
                rows.append(
                    {
                        "case_id": case,
                        "block": c["block"],
                        "op": c["op"],
                        "forecaster": c["forecaster"],
                        "estimator": c["estimator"],
                        "size": c["size"],
                        "refit": c["refit"],
                        "scenario": scenario,
                        "strategy": name if thr is None else f"E3({thr:g}s)",
                        "label": label,
                        "k": k,
                        "seconds": t,
                        "best": best,
                        "seq": seq,
                    }
                )
    out = pd.DataFrame(rows)
    out["vs_best"] = out["seconds"] / out["best"]
    out["vs_seq"] = out["seconds"] / out["seq"]
    return out


def summary(ev: pd.DataFrame) -> str:
    ev = ev.dropna(subset=["seconds"]).copy()
    ev["lost_s"] = ev["seconds"] - ev["best"]
    ev["lost_vs_seq_s"] = (ev["seconds"] - ev["seq"]).clip(lower=0)
    lines = ["# Evaluación de estrategias (V2)\n"]
    for scenario in ("warm", "cold"):
        s = ev[ev["scenario"] == scenario]
        g = (
            s.groupby(["strategy", "size"])
            .agg(
                geo_vs_best=("vs_best", lambda x: float(np.exp(np.log(x).mean()))),
                worst_vs_best=("vs_best", "max"),
                worst_vs_seq=("vs_seq", "max"),
                total_s=("seconds", "sum"),
                best_total_s=("best", "sum"),
                lost_s=("lost_s", "sum"),
                max_lost_s=("lost_s", "max"),
                max_lost_vs_seq_s=("lost_vs_seq_s", "max"),
                n=("seconds", "size"),
            )
            .round(2)
        )
        lines.append(f"\n## Escenario {scenario}\n")
        lines.append(md_table(g))
    return "\n".join(lines)


def main() -> int:
    ev = evaluate()
    ev.to_csv(RESULTS_DIR / "strategies_cells.csv", index=False)
    md = summary(ev)
    (RESULTS_DIR / "strategies_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
