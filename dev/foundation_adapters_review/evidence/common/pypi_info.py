"""
Print the release history of a package on PyPI: latest version, release dates,
`requires_python` and the requirements of the latest release.

Usage:
    python pypi_info.py <package> [--since YYYY-MM-DD]
"""

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--since", default="2000-01-01")
    args = parser.parse_args()

    url = f"https://pypi.org/pypi/{args.package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)

    info = data["info"]
    print(f"package         : {info['name']}")
    print(f"latest          : {info['version']}")
    print(f"requires_python : {info['requires_python']}")
    license_text = info.get("license_expression") or info.get("license") or ""
    print(f"license         : {license_text.strip().splitlines()[0][:80] if license_text.strip() else None}")
    print("requires_dist   :")
    for requirement in info.get("requires_dist") or []:
        print(f"    {requirement}")
    print("releases (upload date of the first file):")
    rows = []
    for version, files in data["releases"].items():
        if not files:
            continue
        date = min(f["upload_time"] for f in files)[:10]
        yanked = any(f.get("yanked") for f in files)
        rows.append((date, version, yanked))
    for date, version, yanked in sorted(rows):
        if date >= args.since:
            print(f"    {date}  {version}{'  (yanked)' if yanked else ''}")


if __name__ == "__main__":
    main()
