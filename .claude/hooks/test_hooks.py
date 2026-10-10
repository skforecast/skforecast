"""
Tests of the Claude Code hooks in this directory. They are outside the package
test suite (`testpaths = ["skforecast"]`); run them after changing a hook:

    python -m pytest .claude/hooks -q -p no:cacheprovider
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent
PROJECT_DIR = HOOKS_DIR.parent.parent


def run_hook(script, payload, **env):
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_DIR), **env},
    )
    return result.returncode, result.stdout, result.stderr


def bash(command):
    return {"tool_name": "Bash", "tool_input": {"command": command}}


# attribution_guard.py
# ==============================================================================


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "Fix lags\n\nCo-Authored-By: Claude <noreply@anthropic.com>"',
        "git commit -m \"$(cat <<'EOF'\nFix lags\n\nCo-authored-by: Claude Opus <x@y.z>\nEOF\n)\"",
        'git -C repo commit -m "Fix\n\nClaude-Session: https://claude.ai/code/abc"',
        'gh pr create --base 0.26.x --title "Fix" --body "🤖 Generated with [Claude Code](https://claude.com/claude-code)"',
        'git add . && git commit -m "x\n\nCo-Authored-By: Anthropic bot <a@b.c>"',
    ],
    ids=["co-author", "heredoc", "session-trailer", "gh-pr-body", "chained"],
)
def test_attribution_guard_blocks_ai_attribution_in_command(command):
    code, _, stderr = run_hook("attribution_guard.py", bash(command))
    assert code == 2
    assert "authored by the user alone" in stderr


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "Fix lags in ForecasterRecursive"',
        'git commit -m "Fix\n\nCo-authored-by: Joaquin Amat <joaquin@example.com>"',
        "git log --grep 'Co-Authored-By: Claude'",
        'grep -rn "Generated with Claude Code" .',
        "pytest skforecast/metrics/tests -q",
    ],
    ids=["plain-commit", "human-co-author", "git-log", "grep", "not-git"],
)
def test_attribution_guard_allows_commands_without_ai_attribution(command):
    code, _, _ = run_hook("attribution_guard.py", bash(command))
    assert code == 0


@pytest.mark.parametrize("option", ["-F", "--body-file"])
def test_attribution_guard_reads_message_files(tmp_path, option):
    message = tmp_path / "msg.txt"
    message.write_text("Fix lags\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n")
    command = (
        f"git commit -F {message}" if option == "-F"
        else f"gh pr create --base 0.26.x --title Fix --body-file {message}"
    )
    code, _, _ = run_hook("attribution_guard.py", bash(command))
    assert code == 2


def test_attribution_guard_blocks_github_mcp_pull_request_body():
    payload = {
        "tool_name": "mcp__plugin_github_github__create_pull_request",
        "tool_input": {
            "title": "Fix lags",
            "body": "Summary\n\nClaude-Session: https://claude.ai/code/abc",
        },
    }
    code, _, _ = run_hook("attribution_guard.py", payload)
    assert code == 2


def test_attribution_guard_ignores_github_mcp_read_tools():
    payload = {
        "tool_name": "mcp__plugin_github_github__search_issues",
        "tool_input": {"query": "Generated with Claude Code"},
    }
    code, _, _ = run_hook("attribution_guard.py", payload)
    assert code == 0


# protect_generated.py
# ==============================================================================


@pytest.mark.parametrize(
    "rel_path",
    ["AGENTS.md", ".github/copilot-instructions.md", "llms-full.txt",
     "docs/llms-full.txt", "docs/llms.txt"],
)
def test_protect_generated_blocks_generated_files(rel_path):
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(PROJECT_DIR / rel_path)}}
    code, _, stderr = run_hook("protect_generated.py", payload)
    assert code == 2
    assert "generate_ai_context_files.py" in stderr


@pytest.mark.parametrize(
    "rel_path", ["CLAUDE.md", "llms.txt", "tools/ai/llms-base.txt", "skforecast/__init__.py"]
)
def test_protect_generated_allows_source_files(rel_path):
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(PROJECT_DIR / rel_path)}}
    code, _, _ = run_hook("protect_generated.py", payload)
    assert code == 0


# session_start_remote.py
# ==============================================================================


def test_session_start_remote_does_nothing_locally(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("")
    code, stdout, _ = run_hook(
        "session_start_remote.py", {}, CLAUDE_CODE_REMOTE="", CLAUDE_ENV_FILE=str(env_file)
    )
    assert code == 0
    assert stdout == ""
    assert env_file.read_text() == ""


def test_session_start_remote_skips_deep_learning_packages_by_default():
    sys.path.insert(0, str(HOOKS_DIR))
    try:
        import session_start_remote
    finally:
        sys.path.remove(str(HOOKS_DIR))

    requirements = session_start_remote.requirements_without_deep_learning(
        PROJECT_DIR / "pyproject.toml"
    )
    names = [req.split(">")[0].split("[")[0].strip().lower() for req in requirements]
    assert "torch" not in names
    assert "keras" not in names
    assert "pytest" in names
    assert "statsmodels" in names
