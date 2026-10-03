"""
SessionStart hook: prepare the Python environment in cloud sessions (claude.ai/code).

Does nothing unless CLAUDE_CODE_REMOTE == "true", so local sessions keep using
the developer's conda environment. In the cloud (Ubuntu, no conda) it:

1. Creates a virtual environment outside the repository (~/.venvs/skforecast),
   because the `.venv` directory at the repository root must not be used.
2. Installs skforecast in editable mode with the `test` extras except the deep
   learning packages (torch, keras). They are the slowest part of the install
   and only the deep learning tests need them, so they are skipped to make the
   install faster. Set SKFORECAST_CLOUD_DL=1 in the cloud environment variables
   to install them too (CPU-only torch wheels from download.pytorch.org, as the
   unit-tests.yml workflow does), e.g. to work on ForecasterRnn or foundation
   models. A session can also install them later by hand.
3. Exports VIRTUAL_ENV and PATH through $CLAUDE_ENV_FILE so later Bash commands
   use that environment.

The install is skipped when pyproject.toml and SKFORECAST_CLOUD_DL have not
changed since the last run (resumed sessions). Failures are reported but never
block the session. Standard library only.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

VENV_DIR = os.path.join(os.path.expanduser("~"), ".venvs", "skforecast")
MARKER = os.path.join(VENV_DIR, ".skforecast-install-hash")
PYTHON_VERSION = "3.12"
DL_PACKAGES = ("torch", "keras")
DL_ENV_VAR = "SKFORECAST_CLOUD_DL"


def deep_learning_enabled():
    return os.environ.get(DL_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def requirements_without_deep_learning(pyproject_path):
    """
    Return the `test` extras of pyproject.toml without the deep learning packages.
    """
    import tomllib

    with open(pyproject_path, "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]["test"]

    requirements = []
    for requirement in extras:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group(0)
        if name.lower().replace("_", "-") not in DL_PACKAGES:
            requirements.append(requirement)

    return requirements


def run(cmd, cwd, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed:\n{(result.stderr or result.stdout)[-2000:]}"
        )


def write_env_file():
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if not env_file:
        return
    bin_dir = os.path.join(VENV_DIR, "bin")
    with open(env_file, "a", encoding="utf-8") as f:
        f.write(f'export VIRTUAL_ENV="{VENV_DIR}"\n')
        f.write(f'export PATH="{bin_dir}:$PATH"\n')
        f.write('export KERAS_BACKEND="torch"\n')


def main():
    if os.environ.get("CLAUDE_CODE_REMOTE") != "true":
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    with_dl = deep_learning_enabled()
    with open(pyproject_path, "rb") as f:
        pyproject_hash = hashlib.sha256(f.read()).hexdigest()
    pyproject_hash += "-dl" if with_dl else "-nodl"
    if with_dl:
        mode = "`test` extras with CPU torch and keras"
    else:
        mode = (
            "`test` extras without torch and keras, for a faster install. Deep "
            f"learning tests are not needed unless you work on them; set {DL_ENV_VAR}=1 "
            "in the cloud environment variables, or install them by hand, if you "
            "need them (e.g. for foundation models)"
        )

    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.exists(MARKER) and os.path.exists(venv_python):
        with open(MARKER, encoding="utf-8") as f:
            if f.read().strip() == pyproject_hash:
                write_env_file()
                print(f"skforecast cloud environment ready (cached): {VENV_DIR} ({mode})")
                return 0

    start = time.time()
    try:
        uv = shutil.which("uv")
        if uv is None:
            run([sys.executable, "-m", "pip", "install", "--user", "uv"], project_dir)
            uv = shutil.which("uv") or os.path.join(os.path.expanduser("~"), ".local", "bin", "uv")

        if not os.path.exists(venv_python):
            try:
                run([uv, "venv", "--python", PYTHON_VERSION, VENV_DIR], project_dir)
            except RuntimeError:
                # Fall back to the default interpreter if uv cannot provide 3.12.
                run([uv, "venv", VENV_DIR], project_dir)

        if with_dl:
            env = dict(os.environ, UV_TORCH_BACKEND="cpu")
            requirements = ["-e", ".[test]"]
        else:
            env = None
            requirements = ["-e", "."] + requirements_without_deep_learning(
                pyproject_path
            )
        run(
            [uv, "pip", "install", "--python", venv_python] + requirements,
            project_dir, env=env
        )
        with open(MARKER, "w", encoding="utf-8") as f:
            f.write(pyproject_hash)
    except Exception as exc:  # Never block the session on an install failure.
        print(
            "WARNING: the skforecast cloud environment could not be prepared. "
            f"Install it manually before running tests.\n{exc}"
        )
        return 0

    write_env_file()
    print(
        f"skforecast cloud environment installed in {time.time() - start:.0f}s: "
        f"{VENV_DIR} (editable install, {mode})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
