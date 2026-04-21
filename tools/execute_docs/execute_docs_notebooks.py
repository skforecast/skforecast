# Execute and save all documentation notebooks, collecting warnings
# ======================================================================================
# Usage:
#   python tools/execute_docs/execute_docs_notebooks.py                  # All docs/
#   python tools/execute_docs/execute_docs_notebooks.py user_guides      # Only docs/user_guides/
#   python tools/execute_docs/execute_docs_notebooks.py quick-start faq  # Multiple subdirectories
# ======================================================================================
import argparse
import papermill as pm
import nbformat
import os
import re
import sys
import logging
import importlib.metadata
from pathlib import Path
from datetime import datetime
import time

CONDA_ENV = "skforecast_py14"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
LOG_DIR = REPO_ROOT / "tools/execute_docs/logs"

# Notebooks to exclude (by name or relative path within docs/)
EXCLUDE_NOTEBOOKS = []


# ANSI color codes
class C:
    HEADER = "\033[96m"
    OK     = "\033[92m"
    FAIL   = "\033[91m"
    INFO   = "\033[93m"
    END    = "\033[0m"


_ANSI_ESCAPE = re.compile(r'\033\[[0-9;]*m')


def _strip_ansi(text):
    return _ANSI_ESCAPE.sub('', text)


def _setup_logger(log_path):
    logger = logging.getLogger("docs_nb_runner")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — preserves ANSI colours
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler — strips ANSI so the file is plain-text readable
    class _PlainFormatter(logging.Formatter):
        def format(self, record):
            return _strip_ansi(super().format(record))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter("%(message)s"))
    logger.addHandler(fh)

    return logger


def _get_pkg_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "n/a"


def _extract_warnings(notebook_path):
    """Parse a notebook file and return deduplicated warning entries from cell outputs.

    Each entry combines the warning header line and the first non-empty message
    line that follows it, e.g. 'file:line: DeprecationWarning: — message text'.
    """
    warning_entries = []
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                text = ""
                if output.get("output_type") == "stream" and output.get("name") == "stderr":
                    text = "".join(output.get("text", []))
                elif output.get("output_type") in ("display_data", "execute_result"):
                    text = "".join(output.get("data", {}).get("text/plain", []))
                if not text:
                    continue
                lines = text.splitlines()
                for idx, line in enumerate(lines):
                    if re.search(r'Warning', line):
                        entry = line.strip()
                        for next_line in lines[idx + 1:idx + 4]:
                            if next_line.strip():
                                entry += " — " + next_line.strip()
                                break
                        if entry not in warning_entries:
                            warning_entries.append(entry)
    except Exception:
        pass
    return warning_entries


def _collect_notebooks(subdirs):
    """Collect notebooks from the specified subdirectories of docs/."""

    exclude_set = {str(Path(x)) for x in EXCLUDE_NOTEBOOKS}

    if subdirs:
        search_dirs = []
        for sub in subdirs:
            d = DOCS_DIR / sub
            if not d.is_dir():
                print(f"Warning: directory docs/{sub}/ does not exist, skipping.")
            else:
                search_dirs.append(d)
    else:
        search_dirs = [DOCS_DIR]

    notebooks = []
    for search_dir in search_dirs:
        notebooks.extend([
            nb for nb in search_dir.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in str(nb)
            and (nb.name not in exclude_set)
            and (str(nb.relative_to(DOCS_DIR)) not in exclude_set)
        ])

    return sorted(set(notebooks))


def run_notebooks(subdirs=None):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{'_'.join(subdirs)}" if subdirs else ""
    log_path  = LOG_DIR / f"docs_execution_log{suffix}_{timestamp}.txt"
    log = _setup_logger(log_path)

    notebooks = _collect_notebooks(subdirs)

    if not notebooks:
        log.info(f"{C.FAIL}No notebooks found.{C.END}")
        return

    results = {"success": [], "failed": [], "warnings": []}

    # ── Header ───────────────────────────────────────────────────────────────
    scope = ", ".join(f"docs/{s}/" for s in subdirs) if subdirs else "docs/"
    log.info(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Log file   : {log_path.resolve()}")
    log.info(f"Directory  : {DOCS_DIR.resolve()}")
    log.info(f"Scope      : {scope}")
    log.info(f"Env        : {CONDA_ENV}")
    log.info("Kernel     : python3")
    log.info(f"Python     : {sys.version.split()[0]}")
    log.info(f"papermill  : {_get_pkg_version('papermill')}")
    log.info(f"skforecast : {_get_pkg_version('skforecast')}")
    log.info(f"Notebooks  : {len(notebooks)}")

    if EXCLUDE_NOTEBOOKS:
        log.info(f"\n{C.INFO}Excluded notebooks:{C.END}")
        for exc in EXCLUDE_NOTEBOOKS:
            log.info(f"  - {exc}")

    total_start = time.time()

    # ── Execution loop ────────────────────────────────────────────────────────
    for i, notebook in enumerate(notebooks, 1):

        nb_start_time = datetime.now().strftime("%H:%M:%S")
        rel_path = notebook.relative_to(DOCS_DIR)
        log.info(f"\n{C.HEADER}{'=' * 60}")
        log.info(f"[{i}/{len(notebooks)}] {nb_start_time} — {C.INFO}{rel_path}{C.END}")
        log.info(f"{'=' * 60}{C.END}")

        temp_output_path = notebook.with_name(f"{notebook.stem}_temp_exec{notebook.suffix}")
        start = time.time()

        try:
            pm.execute_notebook(
                input_path=str(notebook),
                output_path=temp_output_path,
                kernel_name='python3',
                progress_bar=False,
                log_output=True
            )
            elapsed = time.time() - start

            # Atomically replace the original with the successfully executed file
            os.replace(temp_output_path, str(notebook))

            # Extract warnings from the saved notebook outputs
            nb_warnings = _extract_warnings(notebook)
            results["success"].append((notebook, elapsed))

            if nb_warnings:
                results["warnings"].append((notebook, nb_warnings))
                log.info(
                    f"{C.OK}✓ Success:{C.END} {rel_path} ({elapsed:.2f}s) | "
                    f"{C.INFO}{len(nb_warnings)} warning(s){C.END}"
                )
                for w in nb_warnings:
                    log.info(f"  {C.INFO}WARNING:{C.END} {w}")
            else:
                log.info(f"{C.OK}✓ Success:{C.END} {rel_path} ({elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - start

            # Extract warnings from partial output before removing the temp file
            # (papermill writes the output file even on failure, up to the failing cell)
            partial_warnings = (
                _extract_warnings(temp_output_path)
                if os.path.exists(temp_output_path) else []
            )
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

            full_error = str(e)
            results["failed"].append((notebook, elapsed, full_error, partial_warnings))
            log.info(f"{C.FAIL}✗ FAILED:{C.END} {rel_path} ({elapsed:.2f}s)")
            log.info(f"  {C.FAIL}Error:{C.END}\n{full_error}")
            if partial_warnings:
                log.info(f"  {C.INFO}Warnings before failure ({len(partial_warnings)}):{C.END}")
                for w in partial_warnings:
                    log.info(f"    {C.INFO}WARNING:{C.END} {w}")

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(f"\n{C.HEADER}{'=' * 60}")
    log.info("EXECUTION SUMMARY")
    log.info(f"{'=' * 60}{C.END}")
    log.info(f"Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Total time   : {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    log.info(f"Scope        : {scope}")
    log.info(f"Total        : {C.INFO}{len(notebooks)}{C.END}")
    log.info(f"Excluded     : {C.INFO}{len(EXCLUDE_NOTEBOOKS)}{C.END}")
    log.info(f"Successful   : {C.OK}{len(results['success'])}{C.END}")
    log.info(f"Failed       : {C.FAIL}{len(results['failed'])}{C.END}")
    log.info(f"With warnings: {C.INFO}{len(results['warnings'])}{C.END}")

    if results["failed"]:
        log.info(f"\n{C.FAIL}Failed notebooks:{C.END}")
        for nb, elapsed, _, partial_warns in results["failed"]:
            suffix = f" | {len(partial_warns)} warning(s) before failure" if partial_warns else ""
            log.info(f"  {C.FAIL}✗ {nb.relative_to(DOCS_DIR)}{C.END} ({elapsed:.2f}s){suffix}")
        log.info(f"\n{C.FAIL}Errors (full):{C.END}")
        for nb, elapsed, err, _ in results["failed"]:
            log.info(f"\n  {C.FAIL}── {nb.relative_to(DOCS_DIR)} ──{C.END}")
            log.info(err)

    if results["warnings"]:
        log.info(f"\n{C.INFO}Notebooks with warnings:{C.END}")
        for nb, warns in results["warnings"]:
            log.info(f"  {C.INFO}⚠ {nb.relative_to(DOCS_DIR)}{C.END} ({len(warns)} warning(s))")
            for w in warns:
                log.info(f"    {w}")

    log.info(f"\nFull log saved to: {log_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute documentation notebooks and collect warnings.",
        epilog=(
            "Examples:\n"
            "  python tools/execute_docs/execute_docs_notebooks.py                  # All docs/\n"
            "  python tools/execute_docs/execute_docs_notebooks.py user_guides      # Only docs/user_guides/\n"
            "  python tools/execute_docs/execute_docs_notebooks.py quick-start faq  # Multiple subdirs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subdirs",
        nargs="*",
        default=None,
        help="Subdirectories of docs/ to execute (e.g. user_guides quick-start). "
             "If omitted, all notebooks under docs/ are executed.",
    )
    args = parser.parse_args()
    run_notebooks(subdirs=args.subdirs or None)
