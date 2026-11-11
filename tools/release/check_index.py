"""
Check that an index serves exactly the distributions of skforecast that were
built. Run by `.github/workflows/publish-pypi.yml` after the upload.

The upload uses `skip-existing`, so a file already on the index is kept. This
check makes sure that what the index serves for the version is byte for byte
what the build job produced (both the wheel and the sdist, which the conda-forge
feedstock takes from PyPI). An index takes a while to list a new version, hence
the retries.

Usage:

    python tools/release/check_index.py <json api> <dist dir> <version>

where `<json api>` is `https://pypi.org/pypi` or `https://test.pypi.org/pypi`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ATTEMPTS = 20
WAIT = 30  # seconds


def fetch(url: str) -> dict[str, str] | None:
    """
    sha256 of each file the index lists for the version, or None if the index
    does not list the version yet.
    """

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            release = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    return {file["filename"]: file["digests"]["sha256"] for file in release["urls"]}


def main(api: str, dist_dir: Path, version: str) -> int:
    built = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dist_dir.iterdir())
    }
    url = f"{api.rstrip('/')}/skforecast/{version}/json"

    served: dict[str, str] | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            served = fetch(url)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"attempt {attempt}: {error}")
            served = None
        if served is not None and set(built) <= set(served):
            break
        print(f"attempt {attempt}: {url} does not list every file yet")
        time.sleep(WAIT)
    else:
        print(f"::error::{url} does not list {sorted(built)} after {ATTEMPTS} attempts")
        return 1

    errors = [
        f"{name} on the index is not the one built (sha256 {served[name]}, "
        f"built {digest})"
        for name, digest in built.items()
        if served[name] != digest
    ]
    errors += [
        f"the index has {name}, which was not built"
        for name in sorted(set(served) - set(built))
    ]
    for error in errors:
        print(f"::error::{error}")
    if errors:
        return 1

    for name, digest in built.items():
        print(f"{name}: sha256 {digest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), sys.argv[3]))
