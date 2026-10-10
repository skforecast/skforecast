---
name: verify
description: Definition of done for skforecast changes. Lints the changed files, runs only the tests affected by the change, and checks the AI context files, release notes and hooks when those areas changed. Use before reporting any code, test or docs change as finished, and before opening a pull request. Never runs the full test suite.
---

# Verify the current change

Run every step in order and report each result. Stop at the first failure, fix it, and start again from that step. Never report the task as done with a failing step; if a failure is unrelated to the change, say so explicitly with the output.

## 0. How to run Python

- Local session: the active conda environment (see "Python environment" in the contributor rules).
- Cloud session (`CLAUDE_CODE_REMOTE=true`): `python` and `pytest` directly; the SessionStart hook already installed the package with the `test` extras.

## 1. Scope

```bash
git status --short
git diff --name-only HEAD
git ls-files --others --exclude-standard
```

On a feature branch, also include the commits not yet in the release branch (`X.Y.x`, from `__version__` in `skforecast/__init__.py`): `git diff --name-only <release-branch>...HEAD`.

Classify the changed files: package code (`skforecast/<pkg>/`), tests, public API (signatures, defaults, return values, exports in `__init__.py`), docs (`docs/`, `mkdocs.yml`), AI context sources (`tools/ai/`, `skills/`, `llms.txt`), harness (`.claude/`).

## 2. Lint

```bash
ruff check <changed .py files>
```

Fix the issues introduced by the change. Pre-existing findings in touched files are out of scope; do not fix them unless asked.

## 3. Affected tests

Map each changed module to its tests and run them sequentially with `-q` (never `-n`, never the full suite):

- `skforecast/<pkg>/_<module>.py` → `skforecast/<pkg>/tests/tests_<module>/` (for example `recursive/_forecaster_recursive.py` → `recursive/tests/tests_forecaster_recursive/`). Run first the `test_<method>.py` files of the methods touched, with `-x`, then the whole `tests_<module>/` folder once.
- A shared helper (`skforecast/utils/`, `skforecast/base/`, `skforecast/model_selection/_utils.py`, `skforecast/preprocessing/`) is used by many forecasters: run its own tests plus the test folders of the callers whose behavior you changed, and list in the report the packages you did not run. If the change is broad enough to need the full suite, say so and let the user decide; do not launch it.
- Changed test files → those files.

```bash
pytest <paths> -q
```

## 4. Conditional checks

- Public API or AI context sources changed: follow the `ai-context-sync` skill and run `python tools/ai/generate_ai_context_files.py --check`.
- Harness changed (`.claude/hooks/`): `python -m pytest .claude/hooks -q -p no:cacheprovider` (outside `testpaths`).
- A documentation notebook's code changed: re-execute only that notebook with `python tools/docs/execute_notebooks/execute_notebooks.py <notebook>` and check its log in `tools/docs/execute_notebooks/logs/` (exit code 1 means a failure). If it cannot be executed (slow, GPU, gated models), at least run the same command with `--check` to catch error outputs and unexecuted cells in the saved outputs.
- User-facing change with no entry in `docs/releases/releases.md` (section of the version in development): follow the `release-note` skill.

## 5. Report

A short list: each step, the command, pass or fail, and the number of tests run. Mention anything skipped and why, including the packages not covered by the tests you ran.
