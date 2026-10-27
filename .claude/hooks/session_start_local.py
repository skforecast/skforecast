"""
SessionStart hook: put the developer's Python environment on PATH in local sessions.

The shell that runs the agent's Bash commands does not inherit the conda
environment of the developer's terminal (no `python`, `pytest` or `ruff`), and
sourcing the shell profile does not activate it. This hook does it explicitly.

Does nothing in cloud sessions (CLAUDE_CODE_REMOTE == "true", handled by
session_start_remote.py) or when SKFORECAST_PYTHON_ENV is not set. Set it in the
`env` of `.claude/settings.local.json` (git-ignored) to the prefix of the
environment, e.g. `/opt/homebrew/Caskroom/miniconda/base/envs/skforecast_py14`.
When the environment changes (new Python version), only that value changes.

Through $CLAUDE_ENV_FILE it exports PATH, CONDA_PREFIX, CONDA_DEFAULT_ENV and
PYTHONPATH=. (so scripts like `python tools/...` import the skforecast of the
checkout or worktree they run from, not an installed copy), and
DYLD_LIBRARY_PATH when SKFORECAST_DYLD_LIBRARY_PATH is set. The value travels
under another name because macOS SIP removes DYLD_* variables when a protected
shell (/bin/sh, /bin/zsh) starts; it only survives when exported inside the
shell, which is what the env file does. It prints the versions of Python and the
main dependencies. Failures are reported but never block the session. Standard
library only.
"""

import os
import subprocess
import sys

PYTHON_ENV_VAR = "SKFORECAST_PYTHON_ENV"
DYLD_ENV_VAR = "SKFORECAST_DYLD_LIBRARY_PATH"
REPORTED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "numba", "statsmodels", "optuna"
)
VERSIONS_SCRIPT = (
    "import importlib.metadata as m, platform, sys\n"
    "out = ['Python ' + platform.python_version()]\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        out.append(name + ' ' + m.version(name))\n"
    "    except m.PackageNotFoundError:\n"
    "        pass\n"
    "print(', '.join(out))\n"
)


def env_lines(prefix, dyld_library_path):
    """Return the export lines written to $CLAUDE_ENV_FILE."""
    lines = [
        f'export PATH="{os.path.join(prefix, "bin")}:$PATH"',
        f'export CONDA_PREFIX="{prefix}"',
        f'export CONDA_DEFAULT_ENV="{os.path.basename(os.path.normpath(prefix))}"',
        # Relative on purpose: it resolves against the directory of each command.
        'export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"',
    ]
    if dyld_library_path:
        lines.append(f'export DYLD_LIBRARY_PATH="{dyld_library_path}"')
    return lines


def main():
    if os.environ.get("CLAUDE_CODE_REMOTE") == "true":
        return 0
    prefix = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if not prefix:
        return 0

    python = os.path.join(prefix, "bin", "python")
    if not os.path.exists(python):
        print(
            f"WARNING: {PYTHON_ENV_VAR}={prefix} has no bin/python. Fix it in "
            "the `env` of .claude/settings.local.json; meanwhile run Python "
            "through `conda run -n <env>`."
        )
        return 0

    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if env_file:
        lines = env_lines(prefix, os.environ.get(DYLD_ENV_VAR, "").strip())
        with open(env_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    try:
        versions = subprocess.run(
            [python, "-c", VERSIONS_SCRIPT, *REPORTED_PACKAGES],
            capture_output=True, text=True, timeout=60, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        versions = "versions not available"
    print(f"skforecast local environment: {prefix}.\n{versions}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
