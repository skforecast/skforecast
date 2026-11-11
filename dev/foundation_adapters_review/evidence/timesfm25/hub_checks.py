"""
Hub checks for the TimesFM 2.5 checkpoints: commit history of each repository,
config.json, model card metadata, and the tensor names of model.safetensors
(read from the safetensors header, without downloading the weights).

Run with the review environment:
    ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/hub_checks.py
"""

import json

import huggingface_hub
from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

REPOS = [
    "google/timesfm-2.5-200m-pytorch",
    "google/timesfm-2.5-200m-transformers",
    "google/timesfm-2.5-200m-flax",
]

print(f"huggingface_hub {huggingface_hub.__version__}")
api = HfApi()
names = {}
for repo in REPOS:
    print(f"\n== {repo}")
    try:
        commits = api.list_repo_commits(repo)
        for c in commits:
            print(f"  commit {c.commit_id[:12]} {str(c.created_at)[:10]} {c.title}")
    except Exception as exc:
        print(f"  commits: {type(exc).__name__}: {exc}")
    try:
        path = hf_hub_download(repo, "config.json")
        with open(path) as f:
            print("  config.json:", json.dumps(json.load(f))[:600])
    except Exception as exc:
        print(f"  config.json: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        meta = get_safetensors_metadata(repo)
        tensors = {}
        for file_meta in meta.files_metadata.values():
            tensors.update(file_meta.tensors)
        names[repo] = tensors
        print(f"  safetensors: {len(tensors)} tensors, params {meta.parameter_count}")
        for k in sorted(tensors)[:6]:
            print(f"    {k} {tensors[k].shape} {tensors[k].dtype}")
    except Exception as exc:
        print(f"  safetensors: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        readme = hf_hub_download(repo, "README.md")
        with open(readme) as f:
            text = f.read()
        print("  README front matter:", text.split("---")[1].strip().replace("\n", " | ")[:400])
    except Exception as exc:
        print(f"  README: {type(exc).__name__}: {str(exc)[:200]}")

a = names.get("google/timesfm-2.5-200m-pytorch", {})
b = names.get("google/timesfm-2.5-200m-transformers", {})
if a and b:
    print("\n== tensor names pytorch vs transformers")
    print(f"  common: {len(set(a) & set(b))}, only pytorch: {len(set(a) - set(b))}, "
          f"only transformers: {len(set(b) - set(a))}")
    print("  example only transformers:", sorted(set(b) - set(a))[:3])
