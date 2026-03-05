"""
Generate AI context files from llms-base.txt + skills/ + ai_context_header.md.

This script is the single entry point for producing all derived AI context
files used by IDEs, the web site, and LLMs.  The source files that are
maintained by hand are:

  1. tools/ai/llms-base.txt        – core API reference (~730 lines)
  2. llms.txt                      – public index per llmstxt.org spec (~120 lines)
  3. skills/*/SKILL.md             – modular Agent Skills (one per directory)
  4. tools/ai/ai_context_header.md – dev-only header (testing, code style)

Everything else is generated.

Usage
-----
    python tools/ai/generate_ai_context_files.py          # generate all files
    python tools/ai/generate_ai_context_files.py --check  # CI mode: fail if stale
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
AI_DIR = Path(__file__).resolve().parent

SKILL_ORDER: list[str] = [
    "forecasting-single-series",
    "forecasting-multiple-series",
    "statistical-models",
    "hyperparameter-optimization",
    "prediction-intervals",
    "feature-engineering",
    "feature-selection",
    "drift-detection",
    "deep-learning-forecasting",
    "choosing-a-forecaster",
    "troubleshooting-common-errors",
    "complete-api-reference",
]

# IDE targets — each tuple is (relative_path, needs_cursor_frontmatter)
IDE_TARGETS: list[tuple[str, bool]] = [
    (".github/copilot-instructions.md", False),
    ("AGENTS.md", False),
    (".claude/CLAUDE.md", False),
    (".windsurfrules", False),
    (".cursor/rules/skforecast.mdc", True),
]

AUTOGEN_NOTICE_IDE = textwrap.dedent("""\
    <!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->
    <!-- Source: tools/ai/llms-base.txt + tools/ai/ai_context_header.md -->
    <!-- Regenerate with: python tools/ai/generate_ai_context_files.py -->

""")

AUTOGEN_NOTICE_FULL = textwrap.dedent("""\
    <!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->
    <!-- Source: tools/ai/llms-base.txt + skills/ -->
    <!-- Regenerate with: python tools/ai/generate_ai_context_files.py -->

""")

CURSOR_FRONTMATTER = textwrap.dedent("""\
    ---
    description: Skforecast Python library for time series forecasting with ML models
    globs: "**/*.py"
    ---
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML front-matter delimited by ``---`` from *text*."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text


def read_source(path: Path, label: str) -> str:
    if not path.exists():
        sys.exit(f"ERROR: source file not found: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        sys.exit(f"ERROR: source file is empty: {path}")
    return content


def validate_skill(skill_dir: Path) -> list[str]:
    """Return a list of validation errors (empty == OK)."""
    errors: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    name = skill_dir.name

    if not skill_md.exists():
        errors.append(f"  {name}: SKILL.md not found")
        return errors

    raw = skill_md.read_text(encoding="utf-8")

    # --- frontmatter presence & required fields ---
    if not raw.startswith("---"):
        errors.append(f"  {name}: missing YAML frontmatter (must start with ---)")
    else:
        end = raw.find("---", 3)
        if end == -1:
            errors.append(f"  {name}: malformed frontmatter (no closing ---)")
        else:
            fm = raw[3:end]
            if not re.search(r"^name:", fm, re.MULTILINE):
                errors.append(f"  {name}: frontmatter missing required field 'name'")
            if not re.search(r"^description:", fm, re.MULTILINE):
                errors.append(f"  {name}: frontmatter missing required field 'description'")
            # Validate name matches directory
            m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
            if m:
                fm_name = m.group(1).strip().strip('"').strip("'")
                if fm_name != name:
                    errors.append(
                        f"  {name}: frontmatter name '{fm_name}' != directory name '{name}'"
                    )

    # --- line count ---
    body = strip_yaml_frontmatter(raw)
    line_count = body.count("\n") + 1
    if line_count > 500:
        errors.append(f"  {name}: SKILL.md body is {line_count} lines (max 500)")

    # --- name must be in SKILL_ORDER ---
    if name not in SKILL_ORDER:
        errors.append(f"  {name}: not listed in SKILL_ORDER")

    return errors


def validate_version_consistency() -> list[str]:
    """Check that llms-base.txt version matches skforecast/__init__.py."""
    errors: list[str] = []
    init_path = ROOT / "skforecast" / "__init__.py"
    llms_path = AI_DIR / "llms-base.txt"

    if not init_path.exists():
        errors.append("  skforecast/__init__.py not found")
        return errors

    init_text = init_path.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
    if not m:
        errors.append("  cannot parse __version__ from skforecast/__init__.py")
        return errors

    pkg_version = m.group(1)

    llms_text = llms_path.read_text(encoding="utf-8")
    if f"Version: {pkg_version}" not in llms_text:
        errors.append(
            f"  llms-base.txt does not contain 'Version: {pkg_version}'"
            f" (from skforecast/__init__.py)"
        )
    return errors


def validate_imports_consistency() -> list[str]:
    """Check that llms-base.txt imports match public __init__.py exports.

    For each subpackage, extract the names exported in its ``__init__.py``
    and verify that every public name appears somewhere in ``llms-base.txt``
    as ``from skforecast.<pkg> import <name>``.
    """
    errors: list[str] = []
    llms_path = AI_DIR / "llms-base.txt"
    if not llms_path.exists():
        return errors

    llms_text = llms_path.read_text(encoding="utf-8")

    # Collect all "from skforecast.<mod> import <name(s)>" in llms.txt.
    # Handles single imports and comma-separated imports on one line:
    #   from skforecast.stats import Arima, Ets, Sarimax, Arar
    llms_imports: dict[str, set[str]] = {}
    for match in re.finditer(
        r"from\s+skforecast\.(\S+)\s+import\s+(.+)$", llms_text, re.MULTILINE
    ):
        mod = match.group(1)
        names = [n.strip() for n in match.group(2).split(",")]
        for name in names:
            # Take only the identifier (ignore "# comment" after)
            ident = name.split()[0] if name.split() else ""
            if ident and ident.isidentifier():
                llms_imports.setdefault(mod, set()).add(ident)

    # Subpackages to check
    subpackages = [
        "recursive", "direct", "preprocessing", "model_selection",
        "feature_selection", "metrics", "datasets", "stats",
        "drift_detection", "deep_learning",
    ]

    for pkg in subpackages:
        init_path = ROOT / "skforecast" / pkg / "__init__.py"
        if not init_path.exists():
            continue
        init_text = init_path.read_text(encoding="utf-8")

        # Extract imported names from __init__.py
        exported: set[str] = set()

        # Match "from .X import (A, B, ...)"
        for match in re.finditer(
            r"from\s+\.[\w.]*\s+import\s+\(([^)]+)\)", init_text, re.DOTALL
        ):
            for token in re.findall(r"\b(\w+)\b", match.group(1)):
                exported.add(token)

        # Match "from .X import A, B, C" (no parentheses)
        for match in re.finditer(
            r"from\s+\.[\w.]*\s+import\s+(?!\()(.+)$", init_text, re.MULTILINE
        ):
            for token in re.findall(r"\b(\w+)\b", match.group(1)):
                exported.add(token)

        # Remove submodule imports ("from . import submod")
        submod_imports = set(
            re.findall(r"^from\s+\.\s+import\s+(\w+)", init_text, re.MULTILINE)
        )
        exported -= submod_imports

        # Filter: only public, non-private identifiers.
        # Exclude names that shadow the parent package (e.g. `datasets.datasets`
        # is a raw dict, not a user-facing class/function).
        exported = {
            n for n in exported
            if not n.startswith("_") and n.isidentifier() and n != pkg
        }

        in_llms = llms_imports.get(pkg, set())
        missing = exported - in_llms
        if missing:
            errors.append(
                f"  llms-base.txt missing imports from skforecast.{pkg}: "
                f"{', '.join(sorted(missing))}"
            )

    return errors


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_llms_full(llms_base_txt: str) -> str:
    """Assemble llms-full.txt = llms-base.txt + all skills (no frontmatter)."""
    parts: list[str] = [AUTOGEN_NOTICE_FULL.rstrip("\n"), "", llms_base_txt.rstrip("\n")]

    skills_dir = ROOT / "skills"
    for skill_name in SKILL_ORDER:
        skill_dir = skills_dir / skill_name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        raw = skill_md.read_text(encoding="utf-8")
        body = strip_yaml_frontmatter(raw).strip()

        parts.append("")
        parts.append("=" * 80)
        parts.append(f"# SKILL: {skill_name}")
        parts.append("=" * 80)
        parts.append("")
        parts.append(body)

        # Include references/ if present
        refs_dir = skill_dir / "references"
        if refs_dir.exists():
            for ref_file in sorted(refs_dir.glob("*.md")):
                ref_body = ref_file.read_text(encoding="utf-8").strip()
                parts.append("")
                parts.append("---")
                parts.append(f"### Reference: {ref_file.stem}")
                parts.append("")
                parts.append(ref_body)

    parts.append("")  # trailing newline
    return "\n".join(parts)


def build_ide_content(header: str, llms_base_txt: str) -> str:
    """Build IDE context file = notice + header + llms-base.txt."""
    return AUTOGEN_NOTICE_IDE + header.rstrip("\n") + "\n\n" + llms_base_txt.rstrip("\n") + "\n"


def build_cursor_content(header: str, llms_base_txt: str) -> str:
    """Build Cursor IDE file with its required frontmatter."""
    return CURSOR_FRONTMATTER + AUTOGEN_NOTICE_IDE + header.rstrip("\n") + "\n\n" + llms_base_txt.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(*, check_only: bool = False) -> bool:
    """Generate (or check) all derived files.  Returns True if all OK."""
    llms_base_txt = read_source(AI_DIR / "llms-base.txt", "llms-base.txt")
    llms_index_txt = read_source(ROOT / "llms.txt", "llms.txt")
    header = read_source(AI_DIR / "ai_context_header.md", "ai_context_header.md")

    # ── validate skills ──────────────────────────────────────────────
    all_errors: list[str] = []
    skills_dir = ROOT / "skills"
    if skills_dir.exists():
        # Check every skill dir has valid SKILL.md
        for skill_dir in sorted(skills_dir.iterdir()):
            if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                all_errors.extend(validate_skill(skill_dir))
        # Check each entry in SKILL_ORDER has a directory
        for skill_name in SKILL_ORDER:
            sd = skills_dir / skill_name
            if not sd.exists():
                all_errors.append(f"  {skill_name}: directory not found in skills/")

    # ── validate version consistency ─────────────────────────────────
    all_errors.extend(validate_version_consistency())

    # ── validate imports consistency ─────────────────────────────────
    all_errors.extend(validate_imports_consistency())

    if all_errors:
        print("Validation errors:")
        for e in all_errors:
            print(e)
        if check_only:
            return False
        else:
            print("\nWARNING: proceeding with generation despite validation errors.\n")

    # ── build outputs ────────────────────────────────────────────────
    outputs: dict[Path, str] = {}

    # llms-full.txt
    outputs[ROOT / "llms-full.txt"] = build_llms_full(llms_base_txt)

    # IDE files
    ide_content = build_ide_content(header, llms_base_txt)
    cursor_content = build_cursor_content(header, llms_base_txt)

    for relpath, is_cursor in IDE_TARGETS:
        target = ROOT / relpath
        if is_cursor:
            outputs[target] = cursor_content
        else:
            outputs[target] = ide_content

    # docs/ copies — index + full
    outputs[ROOT / "docs" / "llms.txt"] = llms_index_txt
    outputs[ROOT / "docs" / "llms-full.txt"] = outputs[ROOT / "llms-full.txt"]

    # ── check or write ───────────────────────────────────────────────
    if check_only:
        stale: list[str] = []
        for path, expected in outputs.items():
            if not path.exists():
                stale.append(f"  MISSING: {path.relative_to(ROOT)}")
            elif path.read_text(encoding="utf-8") != expected:
                stale.append(f"  STALE:   {path.relative_to(ROOT)}")
        if stale:
            print("The following generated files are out of date:")
            for s in stale:
                print(s)
            print("\nRun: python tools/ai/generate_ai_context_files.py")
            return False
        print("All generated files are up to date.")
        return True

    # Write files
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}")

    print(f"\nGenerated {len(outputs)} files successfully.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI context files from llms-base.txt + skills/."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if generated files are stale (for CI).",
    )
    args = parser.parse_args()

    ok = generate(check_only=args.check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
