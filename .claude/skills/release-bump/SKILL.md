---
name: release-bump
description: Bumps the skforecast version in every place that carries it and prepares the release notes section for the new version.
disable-model-invocation: true
argument-hint: "<new version, e.g. 0.27.0>"
---

# Bump the version

New version: $ARGUMENTS (ask for it if empty). The current one is `__version__` in `skforecast/__init__.py`.

## 1. Branch

A new minor version starts a new release branch `X.Y.x` from the previous one (ask before creating or pushing it). A patch version stays on its release branch.

## 2. Update every version reference

Replace the old version with the new one in:

- `pyproject.toml` (`version = ...`)
- `skforecast/__init__.py` (`__version__`)
- `tests/test_skforecast_version.py`
- `tools/ai/llms-base.txt` (`This document is for skforecast vX.Y.Z+` and `- Version:`)
- `.claude-plugin/marketplace.json` (plugin `version`)
- `llms.txt` (root)
- `docs/quick-start/how-to-install.md` and `docs/quick-start/ai-assisted-forecasting.md`

Then grep the old version across the repository (excluding `docs/releases/`, `dev/`, `site/`, `build/` and generated files) and review each remaining hit: some are historical and must stay, others (tests that mention the version, TODOs such as "Review in skforecast X.Y.Z") need a decision. List them for the user instead of changing them blindly.

`CITATION.cff` has no version on purpose (Zenodo takes it from the GitHub release).

## 3. Release notes

In `docs/releases/releases.md`, add a section above the previous one following the existing format:

```markdown
## X.Y.Z <small>In development</small> { id="X.Y.Z" }

The main changes in this release are:


**Added**


**Changed**


**Fixed**
```

## 4. Regenerate and check

```bash
python tools/ai/generate_ai_context_files.py
python tools/ai/generate_ai_context_files.py --check
pytest tests/test_skforecast_version.py
```

Report the files changed and the remaining hits of the old version that need a decision.
