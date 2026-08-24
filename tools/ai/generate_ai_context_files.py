"""
Generate AI context files from llms-base.txt + skills/ + ai_context_header.md.

This script is the single entry point for producing all derived AI context
files used by IDEs, the web site, and LLMs.  The source files that are
maintained by hand are:

  1. tools/ai/llms-base.txt        - core API reference
  2. llms.txt                      - public index per llmstxt.org spec
  3. skills/*/SKILL.md             - modular Agent Skills (one per directory)
  4. tools/ai/ai_context_header.md - dev-only header (testing, code style)

Everything else is generated.

Usage
-----
    python tools/ai/generate_ai_context_files.py          # generate all files
    python tools/ai/generate_ai_context_files.py --check  # CI mode: fail if stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
AI_DIR = Path(__file__).resolve().parent
PYPROJECT_LLM_CONTEXT_RE = re.compile(
    r'^"LLM Context"\s*=\s*"([^"]+)"', re.MULTILINE
)
ALLOWED_REDIRECT_PREFIXES = ("https://doi.org/",)

# Reading order for llms-full.txt, following the prerequisite edges the skills
# declare in their "### Related skills" sections.
SKILL_ORDER: list[str] = [
    "choosing-a-forecaster",
    "autocorrelation-and-lag-selection",
    "feature-engineering",
    "forecasting-single-series",
    "forecasting-multiple-series",
    "foundation-forecasting",
    "baseline-forecasting",
    "metric-selection",
    "backtesting-configuration",
    "hyperparameter-optimization",
    "feature-selection",
    "prediction-intervals",
    "statistical-models",
    "deep-learning-forecasting",
    "drift-detection",
    "troubleshooting-common-errors",
    "complete-api-reference",
]

# IDE targets
IDE_TARGETS: list[str] = [
    ".github/copilot-instructions.md",
    "AGENTS.md",
]

# Marker delimiting the generated skill roster inside CLAUDE.md
SKILLS_LIST_START = "<!-- SKILLS-LIST:START"
SKILLS_LIST_END = "<!-- SKILLS-LIST:END -->"

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
            # Descriptions are injected into the system prompt, so they must be
            # third person: a first/second person voice degrades skill discovery.
            desc_match = re.search(
                r"^description:(.*?)(?=^[A-Za-z][\w-]*:|\Z)",
                fm,
                re.MULTILINE | re.DOTALL,
            )
            description = desc_match.group(1) if desc_match else ""
            for phrase in ("I can ", "I will ", "You can ", "You should ", "We "):
                if phrase in description:
                    errors.append(
                        f"  {name}: description must be written in third person "
                        f"(found {phrase.strip()!r})"
                    )

    # --- required section headings ---
    body = strip_yaml_frontmatter(raw)
    renamed = False
    for banned, expected in (
        ("## When to Use This Skill", "## When to Use"),
        ("## Common Pitfalls", "## Common Mistakes"),
    ):
        if re.search(rf"^{re.escape(banned)}[ \t]*$", body, re.MULTILINE):
            errors.append(f"  {name}: use '{expected}' instead of '{banned}'")
            renamed = renamed or expected == "## When to Use"
    if not renamed and not re.search(r"^## When to Use[ \t]*$", body, re.MULTILINE):
        errors.append(f"  {name}: SKILL.md is missing a '## When to Use' section")

    # --- line count ---
    line_count = body.count("\n") + 1
    if line_count > 500:
        errors.append(f"  {name}: SKILL.md body is {line_count} lines (max 500)")

    # --- references ---
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        if not re.search(r"^## References[ \t]*$", body, re.MULTILINE):
            errors.append(
                f"  {name}: has references/ but no '## References' section in SKILL.md"
            )

        for ref_file in sorted(refs_dir.glob("*.md")):
            ref_body = ref_file.read_text(encoding="utf-8")
            if not ref_body.strip():
                errors.append(f"  {name}: references/{ref_file.name} is empty")
            ref_line_count = ref_body.count("\n") + 1
            if ref_line_count > 1000:
                errors.append(
                    f"  {name}: references/{ref_file.name} is "
                    f"{ref_line_count} lines (max 1000)"
                )
            # Claude previews long files with partial reads, so the full scope
            # has to be visible from the top.
            if ref_line_count > 100 and not re.search(
                r"^## Contents[ \t]*$", ref_body, re.MULTILINE
            ):
                errors.append(
                    f"  {name}: references/{ref_file.name} is {ref_line_count} "
                    f"lines and needs a '## Contents' section"
                )
            # References must stay one level deep from SKILL.md.
            for linked in re.findall(r"\]\(([^)]+\.md)\)", ref_body):
                if not linked.startswith(("http://", "https://")):
                    errors.append(
                        f"  {name}: references/{ref_file.name} links to "
                        f"'{linked}'; reference files must not link to other files"
                    )
            if f"references/{ref_file.name}" not in raw:
                errors.append(
                    f"  {name}: references/{ref_file.name} is not linked from SKILL.md"
                )

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
        "drift_detection", "deep_learning", "foundation", "plot",
        "exceptions", "experimental",
    ]

    for pkg in subpackages:
        init_path = ROOT / "skforecast" / pkg / "__init__.py"
        if not init_path.exists():
            continue
        init_text = init_path.read_text(encoding="utf-8")

        # Extract imported names from __init__.py
        exported: set[str] = set()

        # Match "from .X import (A, B, ...)" — exclude parent imports (from ..X)
        for match in re.finditer(
            r"from\s+\.(?!\.)[\w.]*\s+import\s+\(([^)]+)\)", init_text, re.DOTALL
        ):
            for token in re.findall(r"\b(\w+)\b", match.group(1)):
                exported.add(token)

        # Match "from .X import A, B, C" (no parentheses) — exclude parent imports
        for match in re.finditer(
            r"from\s+\.(?!\.)[\w.]*\s+import\s+(?!\()(.+)$", init_text, re.MULTILINE
        ):
            for token in re.findall(r"\b(\w+)\b", match.group(1)):
                exported.add(token)

        # Remove submodule imports ("from . import submod1, submod2, ...")
        submod_imports: set[str] = set()
        for submod_match in re.finditer(
            r"from\s+\.\s+import\s+(.+)$", init_text, re.MULTILINE
        ):
            for token in re.findall(r"\b(\w+)\b", submod_match.group(1)):
                submod_imports.add(token)
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


def extract_urls(text: str) -> list[str]:
    """Extract markdown and bare HTTP(S) URLs from text."""
    urls: list[str] = re.findall(r"\(\s*(https?://[^)\s]+)\s*\)", text)
    urls.extend(re.findall(r"\bhttps?://[^\s)>\"']+", text))

    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        clean_url = url.rstrip(".,;:)'\"")
        if clean_url not in seen:
            seen.add(clean_url)
            unique_urls.append(clean_url)

    return unique_urls


def check_url(url: str, label: str) -> list[str]:
    """Check a URL with HEAD and fallback to GET."""
    errors: list[str] = []
    allow_redirect = any(
        url.startswith(prefix) for prefix in ALLOWED_REDIRECT_PREFIXES
    )
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "skforecast-link-checker/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.getcode()
            if code and code >= 400:
                errors.append(f"  {label}: HTTP {code} - {url}")
            elif resp.url != url and not allow_redirect:
                errors.append(f"  {label}: redirected {url} -> {resp.url}")
    except urllib.error.HTTPError as exc:
        # Some servers reject HEAD; retry with GET
        try:
            req_get = urllib.request.Request(url, method="GET")
            req_get.add_header("User-Agent", "skforecast-link-checker/1.0")
            with urllib.request.urlopen(req_get, timeout=15) as resp:
                code = resp.getcode()
                if code and code >= 400:
                    errors.append(f"  {label}: HTTP {code} - {url}")
                elif resp.url != url and not allow_redirect:
                    errors.append(f"  {label}: redirected {url} -> {resp.url}")
        except Exception:
            errors.append(f"  {label}: HTTP {exc.code} - {url}")
    except urllib.error.URLError as exc:
        errors.append(f"  {label}: Connection error ({exc.reason}) - {url}")
    except Exception as exc:
        errors.append(f"  {label}: {exc} - {url}")

    return errors


def validate_urls_in_file(
    path: Path,
    label: str,
    ignore_patterns: list[str] | None = None,
) -> list[str]:
    """Check all URLs in a text file are reachable and canonical."""
    errors: list[str] = []
    if not path.exists():
        return [f"  {label}: file not found at {path.relative_to(ROOT)}"]

    urls = extract_urls(path.read_text(encoding="utf-8"))
    if ignore_patterns:
        urls = [
            url for url in urls
            if not any(pattern in url for pattern in ignore_patterns)
        ]

    if not urls:
        return [f"  {label}: no URLs found"]

    print(f"  Checking {len(urls)} URLs in {label} ...")
    for url in urls:
        errors.extend(check_url(url, label))

    if not errors:
        print(f"  All {len(urls)} URLs in {label} are reachable.")

    return errors


def validate_pyproject_llm_context_url() -> list[str]:
    """Check the PyPI metadata LLM Context URL."""
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return ["  pyproject.toml: file not found"]

    text = pyproject_path.read_text(encoding="utf-8")
    match = PYPROJECT_LLM_CONTEXT_RE.search(text)
    if not match:
        return ["  pyproject.toml: missing project URL 'LLM Context'"]

    url = match.group(1)
    print("  Checking LLM Context URL in pyproject.toml ...")
    errors = check_url(url, "pyproject.toml LLM Context")
    if not errors:
        print("  LLM Context URL in pyproject.toml is reachable.")

    return errors


def validate_python_snippets() -> list[str]:
    """Syntax-check executable Python snippets in AI context source files."""
    errors: list[str] = []
    source_files: list[Path] = [AI_DIR / "llms-base.txt"]
    source_files.extend((ROOT / "skills").glob("*/SKILL.md"))
    source_files.extend((ROOT / "skills").glob("*/references/*.md"))

    # Signature reference blocks intentionally use non-executable notation.
    skip_dirs = {ROOT / "skills" / "complete-api-reference" / "references"}

    for path in sorted(source_files):
        if path.parent in skip_dirs or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"^[ \t]*```python\n(.*?)\n[ \t]*```",
            text,
            re.DOTALL | re.MULTILINE,
        ):
            snippet = textwrap.dedent(match.group(1))
            try:
                ast.parse(snippet)
            except SyntaxError as exc:
                relpath = path.relative_to(ROOT)
                errors.append(
                    f"  {relpath}: invalid Python snippet near line "
                    f"{text[:match.start()].count(chr(10)) + exc.lineno + 1}: "
                    f"{exc.msg}"
                )

    return errors


def validate_ai_context_urls(
    ignore_patterns: list[str] | None = None,
) -> list[str]:
    """Check AI context URLs exposed to users and package metadata."""
    errors: list[str] = []
    errors.extend(
        validate_urls_in_file(ROOT / "llms.txt", "llms.txt", ignore_patterns)
    )
    errors.extend(
        validate_urls_in_file(
            AI_DIR / "llms-base.txt", "tools/ai/llms-base.txt", ignore_patterns
        )
    )
    errors.extend(validate_pyproject_llm_context_url())
    return errors


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def discover_skills(skills_dir: Path) -> list[str]:
    """Return every skill directory containing a SKILL.md, alphabetically."""
    if not skills_dir.exists():
        return []
    return sorted(
        p.name
        for p in skills_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").exists()
    )


def parse_related_skills(skill_md: Path) -> dict[str, set[str]]:
    """Return {'Prerequisite': {...}, 'Next': {...}} from '### Related skills'."""
    body = skill_md.read_text(encoding="utf-8")
    # The block may be the last section, so the trailing '## ' is optional.
    block = re.search(
        r"^### Related skills(.*?)(?=^## |\Z)", body, re.DOTALL | re.MULTILINE
    )
    edges: dict[str, set[str]] = {"Prerequisite": set(), "Next": set()}
    if not block:
        return edges
    for line in block.group(1).splitlines():
        m = re.match(r"-\s+\*\*(Prerequisite|Next)\*\*:\s*(.*)", line.strip())
        if m:
            edges[m.group(1)].update(re.findall(r"`([a-z][a-z-]+)`", m.group(2)))
    return edges


def validate_related_skills(skills_dir: Path) -> list[str]:
    """Check the declared skill graph against SKILL_ORDER and itself."""
    errors: list[str] = []
    skills = discover_skills(skills_dir)
    known = set(skills)
    position = {name: i for i, name in enumerate(resolve_skill_order(skills_dir))}
    graph = {
        name: parse_related_skills(skills_dir / name / "SKILL.md") for name in skills
    }

    banned = ("Before", "After", "With")
    for name in skills:
        raw = (skills_dir / name / "SKILL.md").read_text(encoding="utf-8")
        for label in banned:
            if re.search(rf"^-\s+\*\*{label}\*\*:", raw, re.MULTILINE):
                errors.append(
                    f"  {name}: '**{label}**:' is ambiguous; use "
                    f"'**Prerequisite**', '**Next**', '**Alongside**' or '**Related**'"
                )

        for target in graph[name]["Prerequisite"]:
            if target not in known:
                continue
            # A prerequisite has to be readable before the skill that needs it.
            if position[target] > position[name]:
                errors.append(
                    f"  {name}: declares prerequisite '{target}', which comes "
                    f"later in SKILL_ORDER (#{position[target] + 1} > #{position[name] + 1})"
                )
            if name in graph[target]["Prerequisite"]:
                errors.append(
                    f"  {name}: mutual prerequisite with '{target}'; one of them "
                    f"should be '**Alongside**' or '**Next**'"
                )

    return errors


def resolve_skill_order(skills_dir: Path) -> list[str]:
    """Curated reading order first, then any skill missing from SKILL_ORDER.

    Omission from SKILL_ORDER is a validation error; appending is only a safety
    net so that generation forced past that error still emits every skill.
    """
    found = discover_skills(skills_dir)
    ordered = [s for s in SKILL_ORDER if s in found]
    return ordered + [s for s in found if s not in ordered]


def build_claude_md(claude_md: str, skills: list[str]) -> str:
    """Refresh the generated skill roster delimited by the SKILLS-LIST markers."""
    start = claude_md.find(SKILLS_LIST_START)
    end = claude_md.find(SKILLS_LIST_END)
    if start == -1 or end == -1:
        raise SystemExit(
            "CLAUDE.md: SKILLS-LIST:START / SKILLS-LIST:END markers not found"
        )
    block = (
        f"{SKILLS_LIST_START} - generated, do not edit by hand -->\n"
        f"  {', '.join(skills)}\n"
        f"  {SKILLS_LIST_END}"
    )
    return claude_md[:start] + block + claude_md[end + len(SKILLS_LIST_END):]


def build_llms_full(llms_base_txt: str) -> str:
    """Assemble llms-full.txt = llms-base.txt + all skills (no frontmatter)."""
    parts: list[str] = [AUTOGEN_NOTICE_FULL.rstrip("\n"), "", llms_base_txt.rstrip("\n")]

    skills_dir = ROOT / "skills"
    for skill_name in resolve_skill_order(skills_dir):
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

        for skill_name in discover_skills(skills_dir):
            if skill_name not in SKILL_ORDER:
                all_errors.append(
                    f"  {skill_name}: not listed in SKILL_ORDER "
                    f"(add it in the position it should be read)"
                )

        all_errors.extend(validate_related_skills(skills_dir))

    # ── validate version consistency ─────────────────────────────────
    all_errors.extend(validate_version_consistency())

    # ── validate imports consistency ─────────────────────────────────
    all_errors.extend(validate_imports_consistency())

    # ── validate Python snippets ─────────────────────────────────────
    all_errors.extend(validate_python_snippets())

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

    for relpath in IDE_TARGETS:
        outputs[ROOT / relpath] = ide_content

    # docs/ copies — index + full
    outputs[ROOT / "docs" / "llms.txt"] = llms_index_txt
    outputs[ROOT / "docs" / "llms-full.txt"] = outputs[ROOT / "llms-full.txt"]

    # CLAUDE.md — only the marked skill roster is generated
    outputs[ROOT / "CLAUDE.md"] = build_claude_md(
        read_source(ROOT / "CLAUDE.md", "CLAUDE.md"), discover_skills(skills_dir)
    )

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
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Validate all URLs in llms.txt are reachable (requires network).",
    )
    parser.add_argument(
        "--ignore-urls",
        nargs="*",
        default=None,
        metavar="SUBSTR",
        help="Skip URLs containing these substrings (e.g. llms-full.txt).",
    )
    args = parser.parse_args()

    ok = generate(check_only=args.check)

    if args.check_urls:
        url_errors = validate_ai_context_urls(ignore_patterns=args.ignore_urls)
        if url_errors:
            print('\nURL validation errors in AI context files:')
            for e in url_errors:
                print(e)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
