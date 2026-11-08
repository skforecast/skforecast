"""
Summaries of the n_jobs review: median and range of the timed calls of each
cell, speedup against n_jobs=1, the value chosen by 'auto', the best value,
the first call (warmup, which includes starting the workers when n_jobs
changes) and the equivalence of the predictions against n_jobs=1.

Usage: python summarize.py [--pilot]
Writes results/summary_cells.csv, results/summary_spawn.csv and
results/summary.md (tables used in the report).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import RESULTS_DIR, md_table

ORDER = [
    "1",
    "auto",
    "cpu-1",
    "-1",
    "physical",
    "cap8",
    "4",
    "8",
    "auto|auto",
    "1|1",
    "1|cpu-1",
    "cpu-1|1",
    "physical|1",
    "-1|1",
    "cpu-1|cpu-1",
]


def base_case(case: str) -> str:
    """
    Case of the current code that a "_fix" case is compared with (same data,
    estimator and refit; its n_jobs=1 cell is the reference).
    """

    return case.replace("forecaster=direct_fix", "forecaster=direct").replace(
        "forecaster=dmv_fix", "forecaster=dmv"
    )


def load_cells(base) -> pd.DataFrame:
    frames = []
    for path in sorted(base.glob("block_[B-G]*.csv")):
        df = pd.read_csv(path, dtype={"refit": str, "n_jobs": str})
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def equivalence(df: pd.DataFrame, base) -> pd.DataFrame:
    """
    Maximum absolute difference of the stored predictions of each cell
    against the cell with n_jobs=1 (or 1|1) of the same case.
    """

    preds = df[df["phase"] == "preds"][
        ["block", "affinity", "case_id", "n_jobs_label", "preds_file"]
    ]
    preds = preds.assign(base=preds["case_id"].map(base_case))
    out = []
    for (block, aff, case), g in preds.groupby(["block", "affinity", "case_id"]):
        tag = "" if aff >= 22 else f"_aff{aff}"
        same_base = preds[
            (preds["block"] == block)
            & (preds["affinity"] == aff)
            & (preds["base"] == base_case(case))
        ]
        ref = same_base[same_base["n_jobs_label"].isin(["1", "1|1"])]
        if ref.empty:
            continue
        ref_arr = np.load(base / "preds" / f"{block}{tag}" / ref.iloc[0]["preds_file"])
        for _, r in g.iterrows():
            arr = np.load(base / "preds" / f"{block}{tag}" / r["preds_file"])
            diff = (
                float(np.nanmax(np.abs(arr - ref_arr)))
                if arr.shape == ref_arr.shape
                else np.inf
            )
            out.append(
                {
                    "block": block,
                    "affinity": aff,
                    "case_id": case,
                    "n_jobs_label": r["n_jobs_label"],
                    "max_abs_diff": diff,
                }
            )
    return pd.DataFrame(out)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    keys = ["block", "affinity", "case_id", "n_jobs_label"]
    reps = df[(df["phase"] == "rep") & (df["status"] == "ok")]
    agg = reps.groupby(keys).agg(
        median=("seconds", "median"),
        min=("seconds", "min"),
        max=("seconds", "max"),
        n=("seconds", "size"),
    )
    first = (
        df[df["phase"] == "warmup"]
        .groupby(keys)["seconds"]
        .first()
        .rename("first_call")
    )
    meta_cols = [
        "op",
        "forecaster",
        "estimator",
        "size",
        "refit",
        "n_jobs_resolved",
        "inner_resolved",
        "n_folds",
        "cpu_count",
    ]
    meta = (
        df[df["phase"].isin(["rep", "timeout", "setup"])]
        .groupby(keys)[meta_cols]
        .first()
    )
    status = df.groupby(keys)["status"].agg(
        lambda s: (
            "timeout"
            if (s == "timeout").any()
            else ("error" if (s == "error").any() else "ok")
        )
    )
    out = meta.join(agg).join(first).join(status).reset_index()

    out["base_case"] = out["case_id"].map(base_case)
    seq = (
        out[out["n_jobs_label"].isin(["1", "1|1"])]
        .set_index(["block", "affinity", "case_id"])["median"]
        .rename("t_seq")
    )
    out = out.join(seq, on=["block", "affinity", "base_case"])
    out["speedup"] = out["t_seq"] / out["median"]

    best = (
        out.dropna(subset=["median"])
        .sort_values("median")
        .groupby(["block", "affinity", "case_id"])
        .first()
    )
    out = out.join(
        best[["n_jobs_label", "median"]].rename(
            columns={"n_jobs_label": "best_label", "median": "best_median"}
        ),
        on=["block", "affinity", "case_id"],
    )
    auto = (
        out[out["n_jobs_label"].isin(["auto", "auto|auto"])]
        .set_index(["block", "affinity", "case_id"])["median"]
        .rename("auto_median")
    )
    out = out.join(auto, on=["block", "affinity", "case_id"])
    out["auto_vs_best"] = out["auto_median"] / out["best_median"]
    out["label_order"] = out["n_jobs_label"].map({k: i for i, k in enumerate(ORDER)})

    return out.sort_values(["block", "affinity", "case_id", "label_order"])


def fmt(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}"


def markdown(cells: pd.DataFrame, eq: pd.DataFrame) -> str:
    lines = []
    for (block, aff), g in cells.groupby(["block", "affinity"], sort=True):
        lines.append(
            f"\n### Bloque {block}, afinidad {aff} "
            f"(cpu_count {g['cpu_count'].iloc[0]})\n"
        )
        for case, c in g.groupby("case_id", sort=False):
            c = c.sort_values("label_order")
            short = (
                case.replace("op=", "")
                .replace("forecaster=", "")
                .replace("estimator=", "")
                .replace("size=", "")
                .replace("refit=", "refit ")
            )
            lines.append(f"\n`{short}`, folds/estimadores {c['n_folds'].iloc[0]}\n")
            lines.append(
                "| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |"
            )
            lines.append("|---|---|---|---|---|")
            for _, r in c.iterrows():
                resolved = r["n_jobs_resolved"]
                if isinstance(r["inner_resolved"], str) or not pd.isna(
                    r["inner_resolved"]
                ):
                    if str(r["inner_resolved"]) not in ("", "nan"):
                        resolved = f"{resolved} / {r['inner_resolved']}"
                if r["status"] == "timeout":
                    timing = "timeout"
                else:
                    timing = f"{fmt(r['median'])} [{fmt(r['min'])}-{fmt(r['max'])}]"
                label = (
                    f"**{r['n_jobs_label']}**"
                    if r["n_jobs_label"] == r["best_label"]
                    else r["n_jobs_label"]
                )
                lines.append(
                    f"| {label} | {resolved} | {timing} | "
                    f"{fmt(r['speedup'])} | {fmt(r['first_call'])} |"
                )
    if not eq.empty:
        bad = eq[eq["max_abs_diff"] > 1e-8]
        lines.append("\n### Equivalencia de predicciones frente a n_jobs=1\n")
        lines.append(
            f"{len(eq)} celdas comparadas; {len(bad)} con diferencia > 1e-8.\n"
        )
        if len(bad):
            lines.append(md_table(bad, index=False))

    return "\n".join(lines)


def spawn_summary(base) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(base.glob("block_A*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[(df["phase"] == "rep") & (df["status"] == "ok")]

    return (
        df.groupby(["affinity", "mode", "payload", "k"])
        .agg(
            median=("seconds", "median"),
            min=("seconds", "min"),
            max=("seconds", "max"),
            n=("seconds", "size"),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    base = RESULTS_DIR / ("pilot" if args.pilot else "")

    df = load_cells(base)
    cells = cell_table(df) if not df.empty else pd.DataFrame()
    # The stored predictions (results/preds/, about 110 MB) were removed after
    # the study; without them, the saved result of the check is reused.
    if not df.empty and (base / "preds").exists():
        eq = equivalence(df, base)
    elif (base / "summary_equivalence.csv").exists():
        eq = pd.read_csv(base / "summary_equivalence.csv")
    else:
        eq = pd.DataFrame()
    spawn = spawn_summary(base)

    if not cells.empty:
        cells.to_csv(base / "summary_cells.csv", index=False)
    if not eq.empty:
        eq.to_csv(base / "summary_equivalence.csv", index=False)
    if not spawn.empty:
        spawn.to_csv(base / "summary_spawn.csv", index=False)

    md = ["# Resumen de medidas\n"]
    if not spawn.empty:
        md.append("## Bloque A: coste fijo de arrancar workers\n")
        piv = spawn.assign(
            cell=lambda d: (
                d["median"].map(fmt)
                + " ["
                + d["min"].map(fmt)
                + "-"
                + d["max"].map(fmt)
                + "]"
            )
        ).pivot_table(
            index=["affinity", "mode", "payload"],
            columns="k",
            values="cell",
            aggfunc="first",
        )
        md.append(md_table(piv))
    if not cells.empty:
        md.append("\n## Bloques B a G\n")
        md.append(markdown(cells, eq))
    (base / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"written {base / 'summary.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
