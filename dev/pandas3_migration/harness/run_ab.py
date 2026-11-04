"""
A/B comparison of two skforecast trees in one or more environments.

- base: a git ref (default `origin/0.27.x`), exported with `git archive` to
  `<out-dir>/base_tree` (only `skforecast/`, no git state).
- head: a directory (default: the working tree of this repository).

Steps, in each environment given with `--python NAME=PATH`:

1. Identity (`--identity`): `identity_cases.py` on base and head, compared
   with `identity_compare.py --mode strict`. Between environments, the head
   results are compared with `--mode normalized` (informative).
2. Time (`--bench`): `bench_hotspots.py` in alternating processes, base and
   head (A B B A ...), `--rounds` times. Minimum and median of all samples.
3. Memory (`--memory`): one `bench_hotspots.py --mode memory` process per
   tree (tracemalloc peak, minimum of 2 calls).

Everything is written to `--out-dir` (keep it outside the package; the
default `results/` next to this script is git-ignored), with one Markdown
summary per step.

Example (Windows, from the repository root):
    python dev/pandas3_migration/harness/run_ab.py --identity --bench control
        --python p2=C:/Users/me/miniconda3/envs/skforecast_27_py13/python.exe
        --python p3=C:/Users/me/miniconda3/envs/skforecast_27_py13_p3/python.exe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[2]


def export_base(ref: str, out_dir: Path) -> Path:
    target = out_dir / "base_tree"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    archive = out_dir / "base.zip"
    subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=zip", "-o", str(archive),
         ref, "skforecast"],
        check=True,
    )
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink()
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", ref],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (target / "COMMIT").write_text(f"{ref} {commit}\n", encoding="utf-8")
    return target


def run_script(python: str, tree: Path, script: str, args: list[str],
               out_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tree),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
    })
    cmd = [python, str(HARNESS / script), *args, "--expect-tree", str(tree)]
    return subprocess.run(cmd, env=env, cwd=out_dir, text=True,
                          capture_output=True, encoding="utf-8")


def check(proc: subprocess.CompletedProcess, what: str) -> None:
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{what} failed with exit code {proc.returncode}")


def step_identity(pythons, trees, out_dir: Path, index_unit: str) -> list[str]:
    lines = ["## Identity", ""]
    head_files = {}
    for env_name, python in pythons.items():
        files = {}
        for variant, tree in trees.items():
            out = out_dir / f"identity_{env_name}_{variant}.json"
            print(f"identity {env_name} {variant} ...", flush=True)
            proc = run_script(python, tree, "identity_cases.py",
                              ["--out", str(out), "--label", f"{env_name}-{variant}",
                               "--index-unit", index_unit], out_dir)
            check(proc, f"identity_cases.py ({env_name}, {variant})")
            files[variant] = out
        head_files[env_name] = files["head"]
        proc = subprocess.run(
            [sys.executable, str(HARNESS / "identity_compare.py"),
             str(files["base"]), str(files["head"]), "--mode", "strict"],
            text=True, capture_output=True, encoding="utf-8",
        )
        lines += [f"### {env_name}: base vs head (strict)", "", "```",
                  proc.stdout.strip(), "```", ""]
        print(f"identity {env_name}: {'OK' if proc.returncode == 0 else 'DIFFERENT'}")

    env_names = list(pythons)
    for other in env_names[1:]:
        proc = subprocess.run(
            [sys.executable, str(HARNESS / "identity_compare.py"),
             str(head_files[env_names[0]]), str(head_files[other]),
             "--mode", "normalized"],
            text=True, capture_output=True, encoding="utf-8",
        )
        lines += [f"### head: {env_names[0]} vs {other} (normalized, informative)",
                  "", "```", proc.stdout.strip(), "```", ""]
    return lines


def _ratio(b, a):
    return b / a if a else float("nan")


def step_time(pythons, trees, out_dir: Path, bench: str, rounds: int,
              warmup: int, repeat: int) -> list[str]:
    lines = ["## Time", "",
             f"{rounds} rounds of alternating processes (A B B A ...), "
             f"{warmup} warm-up and {repeat} measures per process: "
             f"{rounds * repeat} samples per tree. Ratio = head / base.", ""]
    for env_name, python in pythons.items():
        samples = {"base": {}, "head": {}}
        outputs = {"base": {}, "head": {}}
        errors = {}
        cpu = []
        for r in range(rounds):
            order = ["base", "head"] if r % 2 == 0 else ["head", "base"]
            for variant in order:
                out = out_dir / f"time_{env_name}_{variant}_r{r}.json"
                print(f"time {env_name} round {r + 1}/{rounds} {variant} ...", flush=True)
                proc = run_script(python, trees[variant], "bench_hotspots.py",
                                  ["--out", str(out), "--bench", bench, "--mode", "time",
                                   "--warmup", str(warmup), "--repeat", str(repeat),
                                   "--label", f"{env_name}-{variant}-r{r}"], out_dir)
                check(proc, f"bench_hotspots.py ({env_name}, {variant}, round {r})")
                data = json.loads(out.read_text(encoding="utf-8"))
                for name, entry in data["benchmarks"].items():
                    cpu.append(entry["cpu_percent_before"])
                    if "error" in entry:
                        errors[(name, variant)] = entry["error"]
                        continue
                    samples[variant].setdefault(name, []).extend(entry["times_ms"])
                    outputs[variant].setdefault(name, set()).add(entry["output"])
                env_info = data["environment"]

        pk = env_info["packages"]
        lines += [
            f"### {env_name}",
            "",
            f"Python {env_info['python']}, pandas {pk['pandas']}, numpy {pk['numpy']}, "
            f"scikit-learn {pk['scikit-learn']}, numba {pk['numba']}, "
            f"{env_info['platform']}. CPU use before each benchmark (whole machine): "
            f"median {statistics.median(cpu):.0f} %, max {max(cpu):.0f} %.",
            "",
            "| benchmark | base min ms | head min ms | ratio min | base median ms "
            "| head median ms | ratio median | same output |",
            "|---|---|---|---|---|---|---|---|",
        ]
        names = list(dict.fromkeys(list(samples["base"]) + list(samples["head"])
                                   + [n for n, _ in errors]))
        for name in names:
            if (name, "base") in errors or (name, "head") in errors:
                lines.append(f"| {name} | error: base={errors.get((name, 'base'))} "
                             f"head={errors.get((name, 'head'))} | | | | | | |")
                continue
            a, b = samples["base"][name], samples["head"][name]
            out_a, out_b = outputs["base"][name], outputs["head"][name]
            same = "n/a" if out_a == {None} else ("yes" if out_a == out_b and len(out_a) == 1
                                                   else "NO")
            lines.append(
                f"| {name} | {min(a):.1f} | {min(b):.1f} | {_ratio(min(b), min(a)):.3f} "
                f"| {statistics.median(a):.1f} | {statistics.median(b):.1f} "
                f"| {_ratio(statistics.median(b), statistics.median(a)):.3f} | {same} |"
            )
        lines.append("")
    return lines


def step_memory(pythons, trees, out_dir: Path, bench: str) -> list[str]:
    lines = ["## Memory", "",
             "tracemalloc peak above the memory in use before the call, minimum of 2 "
             "calls, one process per tree.", ""]
    for env_name, python in pythons.items():
        peaks = {}
        for variant, tree in trees.items():
            out = out_dir / f"memory_{env_name}_{variant}.json"
            print(f"memory {env_name} {variant} ...", flush=True)
            proc = run_script(python, tree, "bench_hotspots.py",
                              ["--out", str(out), "--bench", bench, "--mode", "memory",
                               "--label", f"{env_name}-{variant}"], out_dir)
            check(proc, f"bench_hotspots.py memory ({env_name}, {variant})")
            peaks[variant] = json.loads(out.read_text(encoding="utf-8"))["benchmarks"]
        lines += [f"### {env_name}", "",
                  "| benchmark | base MiB | head MiB | ratio |", "|---|---|---|---|"]
        for name in peaks["base"]:
            a, b = peaks["base"][name], peaks["head"].get(name, {})
            if "peak_mib" not in a or "peak_mib" not in b:
                lines.append(f"| {name} | {a.get('error', '')} | {b.get('error', '')} | |")
                continue
            lines.append(f"| {name} | {a['peak_mib']:.2f} | {b['peak_mib']:.2f} "
                         f"| {_ratio(b['peak_mib'], a['peak_mib']):.3f} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base", default="origin/0.27.x", help="git ref of the base tree.")
    parser.add_argument("--head", default=str(REPO), help="Root of the head tree.")
    parser.add_argument("--python", action="append", required=True, metavar="NAME=PATH",
                        help="Environment to use; repeat for several.")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--index-unit", choices=["native", "ns"], default="native")
    parser.add_argument("--bench", default=None,
                        help="all, control or a comma separated list (time).")
    parser.add_argument("--memory", action="store_true",
                        help="Also measure memory of the --bench selection.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", default=None,
                        help="Default: results/<timestamp> next to this script.")
    args = parser.parse_args()

    pythons = dict(item.split("=", 1) for item in args.python)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        HARNESS / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {"base": export_base(args.base, out_dir), "head": Path(args.head).resolve()}
    base_commit = (trees["base"] / "COMMIT").read_text(encoding="utf-8").strip()
    head_status = subprocess.run(
        ["git", "-C", str(trees["head"]), "log", "-1", "--format=%h %s"],
        capture_output=True, text=True,
    ).stdout.strip()
    lines = [
        "# A/B comparison",
        "",
        f"- base: `{base_commit}` exported to `{trees['base']}`",
        f"- head: `{trees['head']}` (last commit: {head_status}; uncommitted "
        "changes included)",
        f"- environments: {', '.join(f'{k}={v}' for k, v in pythons.items())}",
        "",
    ]
    if args.identity:
        lines += step_identity(pythons, trees, out_dir, args.index_unit)
    if args.bench:
        lines += step_time(pythons, trees, out_dir, args.bench, args.rounds,
                           args.warmup, args.repeat)
        if args.memory:
            lines += step_memory(pythons, trees, out_dir, args.bench)

    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
