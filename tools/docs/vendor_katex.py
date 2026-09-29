"""
Download KaTeX into docs/vendor/katex, so the documentation renders math
without loading anything from a CDN.

The documentation renders LaTeX with KaTeX (see javascripts/katex.js and the
KaTeX entries in mkdocs.yml). KaTeX is served from the site itself for privacy
(no third-party request) and because its stylesheet loads the fonts with paths
relative to itself, which the Material privacy plugin does not download. Only
the woff2 fonts are kept (supported by every current browser), and the woff and
ttf fallbacks are removed from the stylesheet.

Usage
-----
    python tools/docs/vendor_katex.py                    # version below
    python tools/docs/vendor_katex.py --version 0.18.9   # another version
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs" / "vendor" / "katex"
VERSION = "0.18.9"
FILES = ["dist/katex.min.js", "dist/contrib/auto-render.min.js", "dist/katex.min.css"]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--version", default=VERSION, help="KaTeX version.")
    args = parser.parse_args()

    cdn = f"https://cdn.jsdelivr.net/npm/katex@{args.version}"
    listing = json.loads(
        _get(
            f"https://data.jsdelivr.com/v1/packages/npm/katex@{args.version}?structure=flat"
        )
    )
    fonts = [
        f["name"].lstrip("/")
        for f in listing["files"]
        if f["name"].startswith("/dist/fonts/") and f["name"].endswith(".woff2")
    ]

    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "fonts").mkdir(parents=True)

    for path in FILES + fonts:
        content = _get(f"{cdn}/{path}")
        if path.endswith(".css"):
            # Keep only the woff2 source of each @font-face
            content = re.sub(
                rb',url\(fonts/[^)]+\.(?:woff|ttf)\) format\("(?:woff|truetype)"\)',
                b"",
                content,
            )
        (TARGET / path.removeprefix("dist/contrib/").removeprefix("dist/")).write_bytes(
            content
        )

    (TARGET / "LICENSE").write_bytes(_get(f"{cdn}/LICENSE"))
    (TARGET / "VERSION").write_text(f"KaTeX {args.version}\n", encoding="utf-8")
    print(
        f"KaTeX {args.version}: {len(FILES)} files and {len(fonts)} fonts in {TARGET}"
    )


if __name__ == "__main__":
    main()
