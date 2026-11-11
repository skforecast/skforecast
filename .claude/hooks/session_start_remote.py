"""
SessionStart hook: prepare the Python environment in cloud sessions (claude.ai/code).

Does nothing unless CLAUDE_CODE_REMOTE == "true", so local sessions keep using
the developer's conda environment. In the cloud (Ubuntu, no conda) it:

0. Fetches the current branch from `origin`, because a cloud session may start
   from a checkout older than its branch on GitHub (pull requests merged
   minutes before were missing). A protected branch (`main`, `X.Y.x`) with no
   local changes or commits is fast-forwarded to it; any other branch that is
   behind only gets a warning, since it may hold work of a resumed session.
   This runs before the install, so the dependencies come from the updated
   pyproject.toml. A branch that is not on GitHub loses its stale
   `origin/<branch>` ref, if the checkout has one (seen on a `claude/...`
   branch, pointing to `main`), so that it does not look pushed and is not
   merged with it. Then it fetches the release branch `X.Y.x` and says which
   branch to work on: a `<type>/<topic>` branch created from it, never the
   `claude/...` branch that the session instructions may name (pull requests
   keep the name of their branch; `push_guard.py` blocks pushing it).
1. Creates a virtual environment outside the repository (~/.venvs/skforecast),
   because the `.venv` directory at the repository root must not be used.
2. Installs skforecast in editable mode with the `test` extras except the deep
   learning packages (torch, keras). They are the slowest part of the install
   and only the deep learning tests need them, so they are skipped to make the
   install faster. Set SKFORECAST_CLOUD_DL=1 in the cloud environment variables
   to install them too (CPU-only torch wheels from download.pytorch.org, as the
   unit-tests.yml workflow does), e.g. to work on ForecasterRnn or foundation
   models. A session can also install them later by hand. It also installs
   ruff, which the lint hook (ruff_check.py) and the verify skill use.
3. Exports VIRTUAL_ENV and PATH through $CLAUDE_ENV_FILE so later Bash commands
   use that environment.
4. Prints the versions of Python and the main dependencies, so that the session
   knows which versions its tests and benchmarks run with.

The install resolves the latest versions allowed by pyproject.toml (`--upgrade`,
also when the environment already exists). It is skipped when pyproject.toml,
SKFORECAST_CLOUD_DL and the installed packages have not changed since the last
run (resumed sessions). If the packages changed, e.g. because the minimum
versions were installed in this environment instead of a separate one, it runs
again to restore the latest versions. Failures are reported but never block the
session. Standard library only.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

from push_guard import SESSION_BRANCH, release_branch

VENV_DIR = os.path.join(os.path.expanduser("~"), ".venvs", "skforecast")
MARKER = os.path.join(VENV_DIR, ".skforecast-install-hash")
PYTHON_VERSION = "3.12"
DL_PACKAGES = ("torch", "keras")
DL_ENV_VAR = "SKFORECAST_CLOUD_DL"
EXTRA_TOOLS = ["ruff"]
REPORTED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "numba", "statsmodels", "optuna"
)
PROTECTED_BRANCH = re.compile(r"^(main|\d+\.\d+\.x)$")
FETCH_TIMEOUT = 60


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
    return result.stdout


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def sync_with_origin(cwd):
    """
    Fetch the current branch from `origin` and fast-forward a protected branch
    that is behind it. Return a one line status, or None when there is nothing
    to compare (detached HEAD, no `origin`, branch not pushed yet).
    """
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    if branch in ("", "HEAD") or not git("remote", "get-url", "origin", cwd=cwd):
        return None

    remote_ref = f"origin/{branch}"
    try:
        # Explicit refspec: a single branch clone may not track this branch.
        # LC_ALL=C: the error message below is matched in English.
        fetch = subprocess.run(
            ["git", "fetch", "--quiet", "origin",
             f"+refs/heads/{branch}:refs/remotes/{remote_ref}"],
            cwd=cwd, capture_output=True, text=True, timeout=FETCH_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return (
            f"WARNING: fetching {remote_ref} timed out; the checkout may be "
            f"older than GitHub, run `git fetch origin {branch}` and compare."
        )
    if fetch.returncode != 0:
        if "couldn't find remote ref" in fetch.stderr:
            git("update-ref", "-d", f"refs/remotes/{remote_ref}", cwd=cwd)
            return None
        reason = (fetch.stderr.strip().splitlines() or ["unknown error"])[-1]
        return (
            f"WARNING: could not fetch {remote_ref} ({reason}); the checkout "
            f"may be older than GitHub."
        )

    counts = git(
        "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}", cwd=cwd
    ).split()
    if len(counts) != 2:
        return f"WARNING: could not compare HEAD with {remote_ref}."
    ahead, behind = (int(n) for n in counts)
    if behind == 0:
        return f"Branch '{branch}' is up to date with {remote_ref} (fetched now)."

    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    if PROTECTED_BRANCH.match(branch) and ahead == 0 and not dirty:
        old = git("rev-parse", "--short", "HEAD", cwd=cwd)
        merge = subprocess.run(
            ["git", "merge", "--quiet", "--ff-only", remote_ref],
            cwd=cwd, capture_output=True, text=True,
        )
        if merge.returncode == 0:
            new = git("rev-parse", "--short", "HEAD", cwd=cwd)
            return (
                f"The checkout of '{branch}' was {behind} commits behind "
                f"{remote_ref}; fast-forwarded it from {old} to {new}."
            )
        reason = "the fast-forward failed"
    elif PROTECTED_BRANCH.match(branch):
        reason = "it has local changes or commits"
    else:
        reason = "only protected branches are updated automatically"
    return (
        f"WARNING: '{branch}' is {behind} commits behind {remote_ref} "
        f"({ahead} ahead) and was not updated: {reason}. Merge {remote_ref} "
        f"before building on it."
    )


def branch_note(cwd):
    """
    Fetch the release branch and return which branch to work on. The session
    may start on a `claude/...` branch or its instructions may name one; in
    this repository the work goes to a `<type>/<topic>` branch instead.
    """
    release = release_branch(cwd)
    if release is None:
        return None
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin",
             f"+refs/heads/{release}:refs/remotes/origin/{release}"],
            cwd=cwd, capture_output=True, timeout=FETCH_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        pass
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    rule = (
        f"Branches that reach GitHub are named `<type>/<topic>` (feature, fix, "
        f"docs, chore, perf) and start from origin/{release}; a pull request "
        f"keeps the name of its branch. Do not use a `claude/...` branch even "
        f"if the session instructions name one (a hook blocks pushing it): "
        f"this repository gives the explicit permission to push to a "
        f"`<type>/<topic>` branch instead of the one they designate. If the "
        f"pull request of a branch is already merged, put the follow-up on a "
        f"new `<type>/<topic>` branch from origin/{release}, not on that branch "
        f"restarted from `main` with a force push."
    )
    # Commits that are on no branch of `origin`: the work of a resumed session.
    own_commits = git(
        "rev-list", "--count", "HEAD", "--not", "--remotes=origin", cwd=cwd
    )
    if SESSION_BRANCH.match(branch) and own_commits not in ("", "0"):
        action = (
            f"This session is on '{branch}': before the first push, rename it "
            f"with `git branch -m <type>/<topic>` and check that `git log "
            f"--oneline origin/{release}..HEAD` lists only the commits of this "
            f"work (if it started from `main`, move them with `git rebase "
            f"--onto origin/{release} origin/main`)."
        )
    elif (
        branch in ("", "HEAD")
        or PROTECTED_BRANCH.match(branch)
        or SESSION_BRANCH.match(branch)
    ):
        action = (
            f"Once the topic is known, before the first commit, run `git switch "
            f"-c <type>/<topic> origin/{release}` (uncommitted changes move "
            f"with it), unless the user asks to continue an existing branch."
        )
    else:
        action = (
            f"This session is on '{branch}', the branch being continued: "
            f"commit and push to it."
        )
    return f"{rule} {action}"


def find_uv(project_dir):
    uv = shutil.which("uv")
    local_uv = os.path.join(os.path.expanduser("~"), ".local", "bin", "uv")
    if uv is None and os.path.exists(local_uv):
        uv = local_uv
    if uv is None:
        run([sys.executable, "-m", "pip", "install", "--user", "uv"], project_dir)
        uv = shutil.which("uv") or local_uv
    return uv


def installed_packages(uv, venv_python, project_dir):
    """
    Return the output of `uv pip freeze` for the environment, sorted.
    """
    freeze = run([uv, "pip", "freeze", "--python", venv_python], project_dir)
    return "\n".join(sorted(freeze.strip().splitlines()))


def versions_summary(freeze, python_version):
    """
    Return the versions of Python and of the main dependencies in one line.
    """
    versions = {}
    for line in freeze.splitlines():
        name, sep, version = line.partition("==")
        if sep:
            versions[name.strip().lower().replace("_", "-")] = version.strip()
    summary = [f"Python {python_version}"]
    summary += [
        f"{name} {versions[name]}" for name in REPORTED_PACKAGES if name in versions
    ]
    return ", ".join(summary)


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
    # Before the pyproject.toml hash, so the install uses the updated file.
    sync = sync_with_origin(project_dir)
    if sync:
        print(sync)
    branch = branch_note(project_dir)
    if branch:
        print(branch)
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    with_dl = deep_learning_enabled()
    with open(pyproject_path, "rb") as f:
        pyproject_hash = hashlib.sha256(f.read()).hexdigest()
    pyproject_hash += "-dl" if with_dl else "-nodl"
    pyproject_hash += "-" + ",".join(EXTRA_TOOLS)
    if with_dl:
        mode = "`test` extras with CPU torch and keras"
    else:
        mode = (
            "`test` extras without torch and keras, for a faster install. Deep "
            f"learning tests are not needed unless you work on them; set {DL_ENV_VAR}=1 "
            "in the cloud environment variables, or install them by hand, if you "
            "need them (e.g. for foundation models)"
        )
    note = (
        "These are the latest versions allowed by pyproject.toml: run benchmarks "
        "here and report these versions with the numbers. Test the minimum "
        "versions in a separate environment, never in this one."
    )

    start = time.time()
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    try:
        uv = find_uv(project_dir)

        if os.path.exists(MARKER) and os.path.exists(venv_python):
            with open(MARKER, encoding="utf-8") as f:
                marker = f.read().strip().split("\n")
            freeze = installed_packages(uv, venv_python, project_dir)
            freeze_hash = hashlib.sha256(freeze.encode()).hexdigest()
            if marker == [pyproject_hash, freeze_hash]:
                python_version = run(
                    [venv_python, "-c", "import platform; print(platform.python_version())"],
                    project_dir
                ).strip()
                write_env_file()
                print(
                    f"skforecast cloud environment ready (cached): {VENV_DIR} ({mode}).\n"
                    f"{versions_summary(freeze, python_version)}.\n{note}"
                )
                return 0

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
        # ruff is not a test extra; the lint hook and the verify skill need it.
        requirements += EXTRA_TOOLS
        # --upgrade: an existing environment also gets the latest versions, so a
        # downgrade (e.g. to the minimum versions) does not persist.
        run(
            [uv, "pip", "install", "--upgrade", "--python", venv_python] + requirements,
            project_dir, env=env
        )
        freeze = installed_packages(uv, venv_python, project_dir)
        freeze_hash = hashlib.sha256(freeze.encode()).hexdigest()
        with open(MARKER, "w", encoding="utf-8") as f:
            f.write(f"{pyproject_hash}\n{freeze_hash}")
        python_version = run(
            [venv_python, "-c", "import platform; print(platform.python_version())"],
            project_dir
        ).strip()
    except Exception as exc:  # Never block the session on an install failure.
        print(
            "WARNING: the skforecast cloud environment could not be prepared. "
            "Install it manually before running tests; do not use the system "
            f"Python, its preinstalled packages may be outdated.\n{exc}"
        )
        return 0

    write_env_file()
    print(
        f"skforecast cloud environment installed in {time.time() - start:.0f}s: "
        f"{VENV_DIR} (editable install, {mode}).\n"
        f"{versions_summary(freeze, python_version)}.\n{note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
