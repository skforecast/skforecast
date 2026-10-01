"""
PostToolUse hook (Edit|Write): report ruff errors introduced in an edited .py file.

The repository has pre-existing ruff findings, so the hook compares the edited
file against its version in HEAD and only reports findings that are new. It
never blocks and never formats: the code base aligns assignments on purpose,
which `ruff format` would undo. If ruff is not installed, the hook does nothing.
Standard library only (runs on any Python 3).
"""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter


def find_ruff():
    ruff = shutil.which("ruff")
    if ruff:
        return [ruff]
    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, check=True
        )
        return [sys.executable, "-m", "ruff"]
    except (OSError, subprocess.CalledProcessError):
        return None


def ruff_findings(ruff, rel_path, source, project_dir):
    """Run ruff on `source` as if it were `rel_path`, return a list of findings."""
    result = subprocess.run(
        ruff + ["check", "--output-format", "json", "--stdin-filename", rel_path, "-"],
        input=source, capture_output=True, text=True, cwd=project_dir
    )
    try:
        return json.loads(result.stdout or "[]")
    except ValueError:
        return []


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not file_path.endswith(".py") or not os.path.isfile(file_path):
        return 0

    ruff = find_ruff()
    if ruff is None:
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    try:
        rel_path = os.path.relpath(os.path.abspath(file_path), os.path.abspath(project_dir))
    except ValueError:
        return 0
    rel_path = rel_path.replace(os.sep, "/")
    if rel_path.startswith("../"):
        return 0

    with open(file_path, encoding="utf-8") as f:
        current = f.read()
    head = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        capture_output=True, text=True, cwd=project_dir
    )
    baseline = head.stdout if head.returncode == 0 else ""

    def key(finding):
        return (finding.get("code"), finding.get("message"))

    before = Counter(key(f) for f in ruff_findings(ruff, rel_path, baseline, project_dir))
    new_findings = []
    for finding in ruff_findings(ruff, rel_path, current, project_dir):
        k = key(finding)
        if before[k] > 0:
            before[k] -= 1
        else:
            new_findings.append(finding)

    if not new_findings:
        return 0

    lines = [
        f"{rel_path}:{(f.get('location') or {}).get('row', '?')}: "
        f"{f.get('code')} {f.get('message')}"
        for f in new_findings
    ]
    message = (
        "ruff check found new issues in the edited file (fix them unless they are "
        "intentional):\n" + "\n".join(lines)
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": message,
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
