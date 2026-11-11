"""
Compare two JSON files written by `identity_cases.py`.

Typical comparisons:

- base vs head in the same environment (`--mode strict`): every output must
  have the same strict fingerprint, unless the PR changes it on purpose.
- pandas 2 vs pandas 3 with the same code (`--mode normalized`): the values
  must be equal; dtype differences (`object` vs `str`, `[ns]` vs `[us]`) are
  reported separately.

The checks on inputs, repeated calls and memory sharing are compared too.
Exit code 1 when something differs at the chosen level.

Usage:
    python identity_compare.py A.json B.json [--mode strict|normalized]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _describe(data: dict, path: str) -> str:
    env = data["environment"]
    return (
        f"{path}\n    label={data.get('label')} index_unit={data['index_unit']} "
        f"pandas={env['packages']['pandas']} numpy={env['packages']['numpy']} "
        f"python={env['python']}\n    skforecast={env['skforecast_file']}"
    )


def _flatten(desc, prefix: str = "") -> dict[str, str]:
    if isinstance(desc, dict):
        flat = {}
        for key, value in desc.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(desc, list) and desc and isinstance(desc[0], (dict, list)):
        flat = {}
        for i, value in enumerate(desc):
            flat.update(_flatten(value, f"{prefix}[{i}]"))
        return flat
    return {prefix: json.dumps(desc)}


def _describe_diff(oa: dict, ob: dict) -> str:
    """
    The parts of the description (types, dtypes, index) that differ.
    """
    da, db = _flatten(oa.get("describe")), _flatten(ob.get("describe"))
    parts = [
        f"{key or 'object'}: {da.get(key)} -> {db.get(key)}"
        for key in dict.fromkeys(list(da) + list(db))
        if da.get(key) != db.get(key)
    ]
    if not parts:
        return ""
    shown = parts[:4] + ([f"... {len(parts) - 4} more"] if len(parts) > 4 else [])
    return "\n    " + "\n    ".join(shown)


def compare(a: dict, b: dict, mode: str) -> tuple[list[str], bool]:
    lines: list[str] = []
    failed = False
    counts = {"outputs": 0, "strict_equal": 0, "values_equal_dtype_differs": 0,
              "different": 0, "only_in_a": 0, "only_in_b": 0}
    differences: list[str] = []
    check_lines: list[str] = []

    cases = list(dict.fromkeys(list(a["cases"]) + list(b["cases"])))
    for case in cases:
        ca, cb = a["cases"].get(case), b["cases"].get(case)
        if ca is None or cb is None:
            differences.append(f"{case}: only in {'B' if ca is None else 'A'}")
            failed = True
            continue
        err_a = (ca["error"] or "").splitlines()[0] if ca["error"] else None
        err_b = (cb["error"] or "").splitlines()[0] if cb["error"] else None
        if err_a != err_b:
            differences.append(f"{case}: error A={err_a!r} B={err_b!r}")
            failed = True
        elif err_a:
            check_lines.append(f"{case}: fails in both: {err_a}")

        outputs = list(dict.fromkeys(list(ca["outputs"]) + list(cb["outputs"])))
        for out in outputs:
            oa, ob = ca["outputs"].get(out), cb["outputs"].get(out)
            if oa is None:
                counts["only_in_b"] += 1
                differences.append(f"{case}.{out}: only in B")
                failed = True
                continue
            if ob is None:
                counts["only_in_a"] += 1
                differences.append(f"{case}.{out}: only in A")
                failed = True
                continue
            counts["outputs"] += 1
            if oa["strict"] == ob["strict"]:
                counts["strict_equal"] += 1
            elif oa["normalized"] == ob["normalized"]:
                counts["values_equal_dtype_differs"] += 1
                differences.append(
                    f"{case}.{out}: same values, dtype/index differs"
                    f"{_describe_diff(oa, ob)}"
                )
                if mode == "strict":
                    failed = True
            else:
                counts["different"] += 1
                differences.append(f"{case}.{out}: VALUES DIFFER{_describe_diff(oa, ob)}")
                failed = True

        for key in ["inputs_modified", "repeat_mismatch"]:
            if cb[key]:
                check_lines.append(f"{case}: {key} in B: {cb[key]}")
                if set(cb[key]) - set(ca[key]):
                    failed = True
            if ca[key] and not cb[key]:
                check_lines.append(f"{case}: {key} only in A: {ca[key]}")
        if ca["shared_memory"] != cb["shared_memory"]:
            new = sorted(set(cb["shared_memory"]) - set(ca["shared_memory"]))
            gone = sorted(set(ca["shared_memory"]) - set(cb["shared_memory"]))
            check_lines.append(f"{case}: shared_memory new in B: {new} | gone: {gone}")
            # Between pandas versions this is informative (pandas 3 copies
            # lazily and Copy-on-Write protects the data); between base and
            # head in the same environment, new sharing must be explained.
            if new and mode == "strict":
                failed = True
        if ca["warnings"] != cb["warnings"]:
            new = sorted(set(cb["warnings"]) - set(ca["warnings"]))
            gone = sorted(set(ca["warnings"]) - set(cb["warnings"]))
            for w in new:
                check_lines.append(f"{case}: warning new in B: {w}")
            for w in gone:
                check_lines.append(f"{case}: warning gone in B: {w}")

    lines.append("| Outputs | Strict equal | Same values, dtype differs | "
                 "Different | Only in A | Only in B |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| {counts['outputs']} | {counts['strict_equal']} | "
        f"{counts['values_equal_dtype_differs']} | {counts['different']} | "
        f"{counts['only_in_a']} | {counts['only_in_b']} |"
    )
    if differences:
        lines.append("")
        lines.append("Differences:")
        lines.extend(f"- {x}" for x in differences)
    if check_lines:
        lines.append("")
        lines.append("Checks (inputs, repeated calls, memory sharing, warnings):")
        lines.extend(f"- {x}" for x in check_lines)
    return lines, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--mode", choices=["strict", "normalized"], default="strict")
    args = parser.parse_args()

    a, b = _load(args.a), _load(args.b)
    print("A:", _describe(a, args.a))
    print("B:", _describe(b, args.b))
    print(f"Mode: {args.mode}\n")
    lines, failed = compare(a, b, args.mode)
    print("\n".join(lines))
    print("\nResult:", "DIFFERENT" if failed else "OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
