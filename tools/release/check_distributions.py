"""
Check the distributions of skforecast built for a release before they are
uploaded to an index. Run by `.github/workflows/publish-pypi.yml` from the root
of the repository, after `python -m build`.

Usage:

    python tools/release/check_distributions.py <dist dir> <version>

Checks:

- `<dist dir>` holds exactly the wheel and the sdist of `<version>`, and the
  metadata of both says that version.
- Neither carries tests, nor anything from `dev/`, `docs/`, `benchmarks/`,
  `tools/` or other folders of the repository: the wheel only holds the package
  and its `.dist-info`, the sdist only the files that build it.
- The sdist has what the conda-forge feedstock needs (it builds from the sdist
  on PyPI and ships `LICENSE`).
- The wheel and the sdist hold the same package files, which are every file
  tracked by git under `skforecast/` outside the test folders. This catches a
  subpackage left out by `[tool.setuptools.packages.find]` (for example, a
  folder without `__init__.py`).
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath

PACKAGE = "skforecast"

# Top level entries allowed in the sdist (inside `skforecast-<version>/`).
SDIST_ALLOWED = {
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
    PACKAGE,
    f"{PACKAGE}.egg-info",
}
SDIST_REQUIRED = {"LICENSE", "PKG-INFO", "README.md", "pyproject.toml"}


def is_test_path(path: PurePosixPath) -> bool:
    """
    Whether a path belongs to the tests (a `tests` or `tests_*` folder, or a
    `test_*.py` or `conftest.py` file).
    """

    folders = path.parts[:-1]
    if any(part == "tests" or part.startswith("tests_") for part in folders):
        return True
    return path.name == "conftest.py" or (
        path.name.startswith("test_") and path.suffix == ".py"
    )


def package_files(paths: list[PurePosixPath]) -> set[str]:
    """
    Files under `skforecast/`, relative to the root of the distribution.
    """

    return {
        path.as_posix()
        for path in paths
        if path.parts and path.parts[0] == PACKAGE and path.suffix != ".pyc"
    }


def read_version(metadata: str) -> tuple[str, str]:
    """
    Name and version from a METADATA or PKG-INFO file.
    """

    headers = HeaderParser().parsestr(metadata)
    return headers["Name"], headers["Version"]


def check_wheel(wheel: Path, version: str, errors: list[str]) -> set[str]:
    """
    Check the wheel and return its package files.
    """

    dist_info = f"{PACKAGE}-{version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")

    paths = [PurePosixPath(name) for name in names]
    for path in paths:
        if path.parts[0] not in {PACKAGE, dist_info}:
            errors.append(f"wheel: unexpected file {path}")
        elif is_test_path(path):
            errors.append(f"wheel: test file {path}")

    if f"{dist_info}/licenses/LICENSE" not in names:
        errors.append(f"wheel: {dist_info}/licenses/LICENSE is missing")
    if read_version(metadata) != (PACKAGE, version):
        errors.append(f"wheel: METADATA says {read_version(metadata)}")

    return package_files(paths)


def check_sdist(sdist: Path, version: str, errors: list[str]) -> set[str]:
    """
    Check the sdist and return its package files.
    """

    root = f"{PACKAGE}-{version}"
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        pkg_info = archive.extractfile(f"{root}/PKG-INFO")
        metadata = pkg_info.read().decode("utf-8") if pkg_info else ""

    paths = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.parts[0] != root:
            errors.append(f"sdist: {path} is outside {root}/")
            continue
        if not (member.isfile() or member.isdir()):
            errors.append(f"sdist: {path} is not a regular file or folder")
        relative = PurePosixPath(*path.parts[1:])
        if not relative.parts:
            continue
        if relative.parts[0] not in SDIST_ALLOWED:
            errors.append(f"sdist: unexpected entry {relative}")
        elif is_test_path(relative):
            errors.append(f"sdist: test file {relative}")
        if member.isfile():
            paths.append(relative)

    top_level = {path.parts[0] for path in paths}
    for required in sorted(SDIST_REQUIRED - top_level):
        errors.append(f"sdist: {required} is missing")
    if read_version(metadata) != (PACKAGE, version):
        errors.append(f"sdist: PKG-INFO says {read_version(metadata)}")

    return package_files(paths)


def source_files() -> set[str]:
    """
    Files tracked by git under `skforecast/`, outside the test folders.
    """

    output = subprocess.run(
        ["git", "ls-files", "-z", "--", PACKAGE],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [PurePosixPath(name) for name in output.split("\0") if name]
    return {path.as_posix() for path in paths if not is_test_path(path)}


def compare(name: str, found: set[str], expected: set[str], errors: list[str]) -> None:
    """
    Report the package files missing from a distribution, or that it has in
    excess.
    """

    for path in sorted(expected - found):
        errors.append(f"{name}: {path} is missing")
    for path in sorted(found - expected):
        errors.append(f"{name}: {path} is not in the source tree")


def main(dist_dir: Path, version: str) -> int:
    errors: list[str] = []

    wheel = dist_dir / f"{PACKAGE}-{version}-py3-none-any.whl"
    sdist = dist_dir / f"{PACKAGE}-{version}.tar.gz"
    found = {path.name for path in dist_dir.iterdir()}
    expected = {wheel.name, sdist.name}
    if found != expected:
        print(f"::error::{dist_dir} holds {sorted(found)}, expected {sorted(expected)}")
        return 1

    wheel_files = check_wheel(wheel, version, errors)
    sdist_files = check_sdist(sdist, version, errors)
    sources = source_files()
    compare("wheel", wheel_files, sources, errors)
    compare("sdist", sdist_files, sources, errors)

    for error in errors:
        print(f"::error::{error}")
    if errors:
        return 1

    print(f"{wheel.name} and {sdist.name}: {len(sources)} package files each")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), sys.argv[2]))
