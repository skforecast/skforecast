"""
Read and check the version to publish. Run by `.github/workflows/publish-pypi.yml`
from the root of the repository, before the build. Needs `packaging`.

Usage:

    python tools/release/prepare_version.py pypi <tag> <prerelease>
    python tools/release/prepare_version.py testpypi <run number>

Both targets check that `version` in `pyproject.toml` and `__version__` in
`skforecast/__init__.py` are the same canonical PEP 440 version, without a dev
or local part.

- `pypi` (a published GitHub release): the tag must be `v<version>`, and the
  release must be marked as a prerelease on GitHub if, and only if, the version
  is a prerelease (`0.27.0rc1`).
- `testpypi` (a rehearsal): the version becomes `<version>.dev<run number>` in
  both files, so that every rehearsal uploads a new version to TestPyPI and the
  name of the release is never taken there.

Prints the version to publish and, on GitHub Actions, writes it to the step
outputs as `version`, with `branch`, the release branch `X.Y.x`.
"""

from __future__ import annotations

import ast
import os
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

PYPROJECT = Path("pyproject.toml")
INIT = Path("skforecast/__init__.py")


class VersionError(Exception):
    pass


def init_version() -> str | None:
    """
    `__version__` in `skforecast/__init__.py`.
    """

    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                getattr(target, "id", None) == "__version__" for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    return None


def replace_once(path: Path, old: str, new: str) -> None:
    """
    Replace the only occurrence of `old` in a file.
    """

    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise VersionError(f"{path} does not contain {old!r} exactly once")
    path.write_text(text.replace(old, new), encoding="utf-8")


def read_version() -> Version:
    """
    The version of the repository, checked.
    """

    version = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    in_init = init_version()
    if in_init != version:
        raise VersionError(
            f"pyproject.toml says {version} and skforecast/__init__.py says {in_init}"
        )
    try:
        parsed = Version(version)
    except InvalidVersion:
        raise VersionError(f"{version} is not a valid PEP 440 version") from None
    if str(parsed) != version:
        raise VersionError(f"{version} is not in canonical form, write {parsed}")
    if parsed.is_devrelease or parsed.local:
        raise VersionError(f"{version} has a dev or local part")

    return parsed


def main(argv: list[str]) -> str:
    version = read_version()

    if argv[:1] == ["pypi"] and len(argv) == 3:
        tag, prerelease = argv[1], argv[2]
        if tag != f"v{version}":
            raise VersionError(f"the tag is {tag} and the version is {version}")
        if prerelease not in {"true", "false"}:
            raise VersionError(f"prerelease must be true or false, not {prerelease}")
        if (prerelease == "true") != version.is_prerelease:
            kind = "a prerelease" if version.is_prerelease else "a final version"
            marked = "marked" if prerelease == "true" else "not marked"
            raise VersionError(
                f"{version} is {kind}, but the GitHub release is {marked} as a "
                "prerelease"
            )
        publish = str(version)
    elif argv[:1] == ["testpypi"] and len(argv) == 2 and argv[1].isdigit():
        publish = f"{version}.dev{argv[1]}"
        replace_once(PYPROJECT, f'version = "{version}"', f'version = "{publish}"')
        replace_once(INIT, f'__version__ = "{version}"', f'__version__ = "{publish}"')
        if read_version_unchecked() != publish:
            raise VersionError(f"the version was not rewritten to {publish}")
    else:
        raise VersionError(__doc__)

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as file:
            file.write(f"version={publish}\n")
            file.write(f"branch={version.major}.{version.minor}.x\n")
    return publish


def read_version_unchecked() -> str | None:
    """
    The version of `pyproject.toml` if `__version__` agrees, else None.
    """

    version = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    return version if init_version() == version else None


if __name__ == "__main__":
    try:
        print(main(sys.argv[1:]))
    except VersionError as error:
        print(f"::error::{error}")
        sys.exit(1)
