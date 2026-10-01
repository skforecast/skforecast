"""
PreToolUse hook (Bash and GitHub MCP write tools): commits and pull requests
are authored by the user alone.

Blocks a commit message, PR title or PR body that carries a `Co-Authored-By`
trailer naming Claude or Anthropic, a `Claude-Session` trailer or a
"Generated with Claude Code" line. It checks the command text (`-m`, heredoc),
the message files it reads (`git commit -F`, `gh pr create --body-file`) and
the arguments of GitHub MCP tools that create commits or pull requests. The
`attribution` setting in `.claude/settings.json` already turns the automatic
lines off; this catches messages written by hand.

Adapted from `.claude/hooks/pre_bash_guard.py` in skforecast/skforecast-ai.
Exit code 2 blocks the call and sends the reason back to Claude. Standard
library only (runs on any Python 3).
"""

import json
import os
import re
import shlex
import sys
from pathlib import Path

SEGMENT_SEPARATOR = re.compile(r"\n|&&|\|\||;|\|")
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# git options that take a separate value before the subcommand.
GIT_OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
AI_ATTRIBUTION = re.compile(
    r"co-authored-by:[^\n]*(claude|anthropic)|generated with \[?claude code"
    r"|^claude-session:",
    re.IGNORECASE | re.MULTILINE,
)
# A `git commit` at the start of a command, found on the raw text because
# `shlex` cannot parse a first line such as `git commit -m "$(cat <<'EOF'`.
COMMIT_COMMAND = re.compile(
    r"(?:^|[;&|(])\s*(?:[A-Za-z_]\w*=\S*\s+)*git(?:\s+-\S+(?:\s+[^-\s]\S*)?)*"
    r"\s+commit(?![\w-])",
    re.MULTILINE,
)
# Options that read a commit message or a PR body from a file.
MESSAGE_FILE_OPTIONS = ("-F", "--file", "--body-file")
# GitHub MCP tools that write commits, pull requests or comments on them.
MCP_WRITE_TOOL = re.compile(
    r"^mcp__.*github.*__(create_pull_request|update_pull_request|push_files"
    r"|create_or_update_file|delete_file|merge_pull_request)$"
)
BLOCK_MESSAGE = (
    "Blocked: commits and pull requests are authored by the user alone "
    "(AGENTS.md). Remove the Co-Authored-By trailer naming Claude or "
    "Anthropic, the Claude-Session trailer and any 'Generated with Claude "
    "Code' line, and do not mention Claude in the message.\n"
)


def segments(command):
    """
    Split `command` into simple commands and return the words of each, with
    leading env assignments removed. A segment that is not valid shell
    (for example a line inside a heredoc) is skipped.
    """
    result = []
    for segment in SEGMENT_SEPARATOR.split(command):
        try:
            words = shlex.split(segment.strip().lstrip("({ "))
        except ValueError:
            continue
        while words and ENV_ASSIGNMENT.match(words[0]):
            words.pop(0)
        if words:
            result.append(words)
    return result


def git_commit_args(command):
    """Return the arguments of every `git ... commit` simple command."""
    commits = []
    for words in segments(command):
        if os.path.basename(words[0]) != "git":
            continue
        i = 1
        while i < len(words) and words[i].startswith("-"):
            i += 2 if words[i] in GIT_OPTIONS_WITH_VALUE else 1
        if i < len(words) and words[i] == "commit":
            commits.append(words[i + 1:])
    return commits


def message_files(args):
    """
    Paths given to `-F`, `--file` or `--body-file`, in the forms `-F path`,
    `-Fpath` and `--file=path`. `-` (stdin) is skipped.
    """
    paths = []
    for i, arg in enumerate(args):
        for option in MESSAGE_FILE_OPTIONS:
            if arg == option and i + 1 < len(args):
                paths.append(args[i + 1])
            elif arg.startswith(option + "="):
                paths.append(arg.split("=", 1)[1])
            elif option == "-F" and arg.startswith("-F") and len(arg) > 2:
                paths.append(arg[2:])
    return [p for p in paths if p != "-"]


def writes_pr(words):
    """Whether a simple command creates or edits a pull request."""
    return (
        os.path.basename(words[0]) == "gh"
        and words[1:2] == ["pr"]
        and words[2:3] in (["create"], ["edit"])
    )


def bash_has_ai_attribution(command, cwd):
    """
    Whether a command that writes a commit message or a PR body carries an
    AI attribution line, in its own text (which includes any heredoc) or in
    the message files it reads.
    """
    writers = git_commit_args(command) + [
        words[3:] for words in segments(command) if writes_pr(words)
    ]
    if not writers and not COMMIT_COMMAND.search(command):
        return False
    if AI_ATTRIBUTION.search(command):
        return True
    for args in writers:
        for path in message_files(args):
            try:
                text = (Path(cwd) / path).read_text(errors="ignore")
            except OSError:
                continue
            if AI_ATTRIBUTION.search(text):
                return True
    return False


def strings(value):
    """Yield every string inside a JSON-like value (title, body, message...)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    if tool_name == "Bash":
        cwd = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or "."
        blocked = bash_has_ai_attribution(tool_input.get("command", ""), cwd)
    elif MCP_WRITE_TOOL.match(tool_name):
        blocked = any(AI_ATTRIBUTION.search(text) for text in strings(tool_input))
    else:
        blocked = False

    if blocked:
        sys.stderr.write(BLOCK_MESSAGE)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
