"""
Download Mermaid.js into docs/vendor/mermaid, so the documentation draws
diagrams without loading anything from a CDN.

The documentation draws the ```mermaid fences of the Markdown pages with
javascripts/mermaid.js (see pymdownx.superfences in mkdocs.yml). Material's own
Mermaid integration is not used: the privacy plugin rewrites the URL it loads
Mermaid.js from to the root of site_url, where the versioned site published by
mike (latest/, 0.28.0/...) has no copy of it.

Usage
-----
    python tools/docs/vendor_mermaid.py                     # version below
    python tools/docs/vendor_mermaid.py --version 11.17.2   # another version
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs" / "vendor" / "mermaid"
VERSION = "11.17.2"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--version", default=VERSION, help="Mermaid version.")
    args = parser.parse_args()

    cdn = f"https://cdn.jsdelivr.net/npm/mermaid@{args.version}"

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    (TARGET / "mermaid.min.js").write_bytes(_get(f"{cdn}/dist/mermaid.min.js"))
    (TARGET / "LICENSE").write_bytes(_get(f"{cdn}/LICENSE"))
    (TARGET / "VERSION").write_text(f"Mermaid {args.version}\n", encoding="utf-8")
    print(f"Mermaid {args.version} in {TARGET}")


if __name__ == "__main__":
    main()
