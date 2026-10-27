---
name: verify
description: Definition of done for skforecast changes. Lints the changed files, runs only the tests affected by the change, and checks the AI context files, release notes and hooks when those areas changed. Use before reporting any code, test or docs change as finished, and before opening a pull request. Never runs the full test suite.
---

# Verify the current change

Run every step in order and report each result. Stop at the first failure, fix it, and start again from that step. Never report the task as done with a failing step; if a failure is unrelated to the change, say so explicitly with the output.

Reuse: if step 1 prints `unchanged since the last green verify` and that verify ran in this session, report its result again instead of running steps 2 to 4, for example right before a commit or when opening the pull request after committing. The stamp compares the content of every non-ignored file, so a commit of the verified files keeps it valid and any edit invalidates it.

## 0. How to run Python

- Local session: the SessionStart hook `session_start_local.py` puts the environment named by `SKFORECAST_PYTHON_ENV` (in `.claude/settings.local.json`) on `PATH`, so call `python` and `pytest` directly. If `python` is not found, say so, use `conda run -n <env>` and suggest setting that variable; do not search for environments.
- Cloud session (`CLAUDE_CODE_REMOTE=true`): `python` and `pytest` directly; the SessionStart hook already installed the package with the `test` extras and ruff, at the latest versions allowed by `pyproject.toml`. Minimum-version checks go in a separate environment (see "Python environment" in the contributor rules).

## 1. Scope

```bash
python .claude/skills/verify/affected.py
```

It lists the files changed on the branch (since the merge base with the release branch `X.Y.x`, plus uncommitted and untracked files), classifies them and prints the `lint`, `pytest` and `checks` commands that cover them, plus `notes` that need a decision. Out-of-scope paths (`dev/`, `benchmarks/`, `images/`, governance files) get nothing. If it prints `nothing to verify`, report that and stop.

## 2. Lint

Run the `lint` command (`ruff check <changed .py files>`). Fix the issues introduced by the change. Pre-existing findings in touched files are out of scope; do not fix them unless asked.

## 3. Affected tests

Run the `pytest` command once, sequentially, with `-q` (never `-n`, never the full suite). Do not rerun the method test files with `-x` first: that belongs to the iteration loop (see `.claude/rules/tests.md`), not to verify.

- Shared code (a `shared code` note: `skforecast/utils/`, `skforecast/base/`, `skforecast/model_selection/_utils.py`, `skforecast/preprocessing/`): also run the test folders of the callers whose behavior you changed, and list in the report the packages you did not run. If the change is broad enough to need the full suite, say so and let the user decide; do not launch it.
- A module with no tests by name runs the whole test folder of its package (`no tests by name` note). If it should map to specific tests, add it to `EXCEPTIONS` in `affected.py`.

## 4. Checks and notes

Run every command under `checks`. They cover:

- AI context sources, generated files or a package `__init__.py` (exports): follow the `ai-context-sync` skill when a public API changed, and run `python tools/ai/generate_ai_context_files.py --check`.
- Harness (`.claude/`): `python -m pytest .claude/hooks .claude/skills -q -p no:cacheprovider` (outside `testpaths`; never `pytest .claude`, which would collect the copies of the repository in `.claude/worktrees/`).
- Docs (`docs/`, `mkdocs.yml`, `README.md`, `tools/docs/hooks/`): `python tools/docs/hooks/page_meta.py`. A new page needs its search engine title and description in `tools/docs/hooks/page_meta.yml` (CI runs the same check on pull requests).
- Notebooks: `execute_notebooks.py --check` on the saved outputs. If a notebook's code changed, re-execute only that notebook with `python tools/docs/execute_notebooks/execute_notebooks.py <path relative to docs/>` and check its log in `tools/docs/execute_notebooks/logs/` (exit code 1 means a failure). If it cannot be executed (slow, GPU, gated models), the `--check` result is the minimum.
- `tools/foundation_compat/`: its own tests.

Then act on each note: `pyproject.toml` (propose the minimum versions run; the user decides on the full suite), `.github/` (review the workflows by reading), files with no automatic check, and a user-facing change with no entry in `docs/releases/releases.md` (follow the `release-note` skill).

## 5. Report

A short list: each step, the command, pass or fail, and the number of tests run. Mention anything skipped and why, including the packages not covered by the tests you ran.

When every step is green, record it for reuse with `python .claude/skills/verify/affected.py --stamp`.
