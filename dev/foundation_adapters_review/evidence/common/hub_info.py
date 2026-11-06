"""
List the checkpoints of an author on the Hugging Face Hub whose id starts with
a prefix, with their revision, dates, license, gating and size.

The token, when needed, is injected by the network proxy: do not set HF_TOKEN.

Usage:
    python hub_info.py <author> [--prefix PREFIX] [--search TEXT]
    python hub_info.py --repo <owner/name>
"""

import argparse
import json
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"


def get(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def describe(repo_id: str) -> dict:
    info = get(f"{API}/{repo_id}?blobs=true")
    card = info.get("cardData") or {}
    size = sum((s.get("size") or 0) for s in info.get("siblings", []))
    return {
        "id": info["id"],
        "sha": info.get("sha"),
        "created": (info.get("createdAt") or "")[:10],
        "last_modified": (info.get("lastModified") or "")[:10],
        "gated": info.get("gated"),
        "private": info.get("private"),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "license_link": card.get("license_link"),
        "pipeline_tag": info.get("pipeline_tag"),
        "library_name": info.get("library_name"),
        "downloads": info.get("downloads"),
        "size_mb": round(size / 1e6, 1),
        "files": sorted(s["rfilename"] for s in info.get("siblings", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("author", nargs="?")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--search", default=None)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    if args.repo:
        repos = [args.repo]
    else:
        query = {"author": args.author, "limit": 500}
        if args.search:
            query["search"] = args.search
        listing = get(f"{API}?{urllib.parse.urlencode(query)}")
        repos = sorted(
            m["id"] for m in listing
            if m["id"].split("/", 1)[1].lower().startswith(args.prefix.lower())
        )

    for repo_id in repos:
        try:
            print(json.dumps(describe(repo_id), ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"id": repo_id, "error": str(exc)[:200]}))


if __name__ == "__main__":
    main()
