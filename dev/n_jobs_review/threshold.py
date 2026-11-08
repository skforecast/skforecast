"""
Work threshold of `n_jobs='auto'` for one platform, from block A (cold start
of the workers) and V1 (single backtestings in a fresh process).

For each V1 case it shows the sequential time (`n_jobs=1`), the time with
`cpu_count() - 1` workers (always parallel; before the threshold, what 'auto'
did), the time of 'auto' and the jobs it used, and the work that skforecast
estimates from the initial fit: `t_fit * n_fits`, where `n_fits` is the
number of folds refitted (all but the first with `refit=True`, 0 without
refit).

Threshold (cases with refit, without Direct, whose backtesting is sequential
and parallelizes the fit of the steps instead):

1. Fit `parallel = a + b * sequential` on those cases: `a` is the cold
   overhead of starting the workers and `b` the inverse of the speedup.
2. Break-even in real sequential time: `T = a / (1 - b)`.
3. In units of the estimate: `T * median(estimate / sequential)`, rounded
   to 0.5 s. That is the value of `_MIN_PARALLEL_WORK_SECONDS` for the
   platform (`skforecast/model_selection/_utils.py`).

The script also checks the threshold of the installed skforecast: 'auto'
should never be much slower than the best of `n_jobs=1` and `cpu-1`.

Usage: python threshold.py [--results-dir results/macos]
Output: printed and saved as <results-dir>/threshold.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR, md_table


def cold_start_table(results_dir: Path) -> str:
    path = results_dir / "block_A_cold.csv"
    if not path.exists():
        return "No block_A_cold.csv.\n"
    d = pd.read_csv(path)
    d = d[(d["status"] == "ok") & (d["payload"].isin(["noop", "forecaster"]))]
    table = d.pivot_table(
        index="payload", columns="k", values="seconds", aggfunc="median"
    ).round(2)

    return "Cold start of k workers, median of the repetitions (s):\n\n" + (
        md_table(table) + "\n"
    )


def cases_table(results_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(results_dir / "block_V_cold.csv")
    d = d[d["status"] == "ok"]
    seconds = d.pivot_table(
        index="case_id", columns="n_jobs_label", values="seconds", aggfunc="median"
    )
    seq = d[d["n_jobs_label"] == "1"].groupby("case_id")
    out = pd.DataFrame(
        {
            "seq": seconds["1"],
            "t_fit": seq["t_initial_fit"].median(),
            "n_folds": seq["n_folds"].first(),
        }
    )
    # Before the threshold, 'auto' was always parallel in these cases (but
    # Direct, where both run the fit of the steps in parallel).
    if "cpu-1" in seconds:
        out["par"] = seconds["cpu-1"]
    elif "auto_resolved" not in d or d["auto_resolved"].isna().all():
        out["par"] = seconds["auto"]
    if "auto" in seconds and "auto_resolved" in d and d["auto_resolved"].notna().any():
        out["auto"] = seconds["auto"]
        out["auto_jobs"] = (
            d[d["n_jobs_label"] == "auto"]
            .groupby("case_id")["auto_resolved"]
            .agg(lambda s: "/".join(sorted({str(int(v)) for v in s.dropna()})))
        )
    out["refit"] = out.index.str.contains("refit=True")
    out["direct"] = out.index.str.contains("direct")
    out["estimate"] = out["t_fit"] * (out["n_folds"] - 1) * out["refit"]

    return out.sort_values("estimate")


def threshold(cases: pd.DataFrame) -> tuple[str, float | None]:
    if "par" not in cases:
        return "No parallel measurement (`cpu-1`): run V1 with it.\n", None
    fit = cases[cases["refit"] & ~cases["direct"]].dropna(subset=["par"])
    if len(fit) < 3:
        return "Fewer than 3 cases with refit: cannot fit the line.\n", None
    b, a = np.polyfit(fit["seq"], fit["par"], 1)
    if not 0 < b < 1:
        return f"Unexpected slope b = {b:.2f}: review the data.\n", None
    break_even = a / (1 - b)
    ratio = float((fit["estimate"] / fit["seq"]).median())
    value = np.round(break_even * ratio * 2) / 2
    winners = fit[fit["par"] < fit["seq"]]
    losers = fit[fit["par"] >= fit["seq"]]
    text = (
        f"- Line fitted on {len(fit)} cases with refit: parallel = {a:.2f} + "
        f"{b:.3f} x sequential (cold overhead {a:.2f} s, speedup {1 / b:.1f}).\n"
        f"- Break-even: {break_even:.2f} s of real sequential work.\n"
        f"- Estimate / real sequential time: median {ratio:.2f} "
        f"[{(fit['estimate'] / fit['seq']).min():.2f}-"
        f"{(fit['estimate'] / fit['seq']).max():.2f}].\n"
        f"- Threshold in units of the estimate: {break_even * ratio:.2f} s, "
        f"rounded to **{value:.1f} s**.\n"
        f"- Bracket from the cases: parallel loses up to an estimate of "
        f"{losers['estimate'].max() if len(losers) else float('nan'):.2f} s and "
        f"wins from {winners['estimate'].min() if len(winners) else float('nan'):.2f} s.\n"
    )

    return text, float(value)


def check_auto(cases: pd.DataFrame) -> str:
    if "auto" not in cases:
        return (
            "No 'auto' measured with a skforecast that has the threshold "
            "(no `auto_resolved`).\n"
        )
    best = cases[["seq", "par"]].min(axis=1) if "par" in cases else cases["seq"]
    loss = cases["auto"] / best
    worst = loss.idxmax()

    return (
        f"- 'auto' / best of 1 and cpu-1: median {loss.median():.2f}, worst "
        f"{loss.max():.2f} ({worst}).\n"
        f"- 'auto' / n_jobs=1: worst {(cases['auto'] / cases['seq']).max():.2f}.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    cases = cases_table(args.results_dir)
    columns = [
        c
        for c in ["seq", "par", "auto", "auto_jobs", "t_fit", "n_folds", "estimate"]
        if c in cases
    ]
    shown = cases[columns].copy()
    if "par" in cases:
        shown["par/seq"] = cases["par"] / cases["seq"]
    if "auto" in cases:
        shown["auto/seq"] = cases["auto"] / cases["seq"]
    threshold_text, _ = threshold(cases)
    report = (
        f"# Work threshold: {args.results_dir}\n\n"
        + cold_start_table(args.results_dir)
        + "\nSingle backtestings in a fresh process, median (s). `par`: "
        "`cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.\n\n"
        + md_table(shown.round(2))
        + "\n\n## Threshold\n\n"
        + threshold_text
        + "\n## 'auto' with the threshold of the installed skforecast\n\n"
        + check_auto(cases)
    )
    print(report)
    (args.results_dir / "threshold.md").write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
