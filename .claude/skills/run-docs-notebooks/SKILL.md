---
name: run-docs-notebooks
description: Executes the documentation notebooks with the repository runner on a release branch (X.Y.x) and reviews the result before a release: failures, warnings, outputs that changed and sentences of the guides that no longer match their outputs. Commits the notebooks that run cleanly and reports the rest. Use for the pre-release pass over the docs, or over a subset of them.
disable-model-invocation: true
argument-hint: "[docs/ subfolders or notebooks relative to docs/, e.g. user_guides faq/forecasting-time-series-with-missing-values.ipynb; empty = all]"
---

# Execute and review the documentation notebooks

Scope: $ARGUMENTS (empty: every notebook under `docs/`, slow ones included).

The runner is `tools/docs/execute_notebooks/execute_notebooks.py`; its usage is in `.claude/rules/docs.md` ("Executing notebooks"). This skill runs it and reviews what it produced. It never changes the package code, the runner or the text of a guide without the user's approval.

## 1. Branch and environment

1. Release branch: `X.Y.x` from `__version__` in `skforecast/__init__.py` (e.g. `0.27.0` → `0.27.x`). Create `docs/execute-notebooks-X.Y` from the local `X.Y.x` (it may hold commits not yet on origin). If other sessions work in the main checkout, work in a worktree of your own.
2. Commits are fine; do not push or open a pull request without asking.
3. The kernel runs from each notebook's folder, where the repository is not on the path. Check it runs the code of this checkout:
   ```bash
   cd docs/user_guides && python -c "import skforecast; print(skforecast.__version__, skforecast.__file__)"
   ```
   The path must be this checkout's `skforecast/` folder. If the import fails or points to `site-packages` or another checkout (a worktree shares the environment with the main checkout), run every command below with `PYTHONPATH=<root of this checkout>` and tell the user; do not install or uninstall anything in the environment without asking. The runner header line `Kernel skf` must show this checkout; if it does not, stop.
4. On macOS, if LightGBM and torch are both installed, export `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib` (duplicate libomp: crashes or hangs).
5. Cloud sessions (`CLAUDE_CODE_REMOTE=true`): the SessionStart virtual environment skips torch and keras unless `SKFORECAST_CLOUD_DL=1`; the notebooks that need them fail. Say so instead of installing them, unless the user asked for it.

## 2. Execute

- Run in the background, it takes a long time:
  ```bash
  python tools/docs/execute_notebooks/execute_notebooks.py --include-slow $ARGUMENTS
  ```
  With an explicit notebook as target, `--include-slow` is not needed.
- `EXCLUDE_NOTEBOOKS` (read the list in the script, it changes) are skipped when scanning folders: they need a specific environment (foundation models, GPU, packages that are not dependencies) or are benchmarks. Do not run them as part of this pass. If one of them needs refreshing and the environment has its packages, offer to run it explicitly at the end; never install packages for it without asking.
- Do not change `EXECUTION_TIMEOUT` or the lists of the script. A cell over the limit aborts its notebook: record which cell and how long it took.
- If a notebook fails, let the run finish; do not fix and rerun in the middle.

## 3. Review

Use the log in `tools/docs/execute_notebooks/logs/` (its summary at the end) and `git diff` of each notebook. To explain changes, read the section of the version in development in `docs/releases/releases.md`.

1. **Failures**: notebook, cell, full error and cause. A failure in the package code (not in the notebook) is not fixed here: describe it with a minimal example.
2. **Warnings**, grouped by category and origin:
   - `FutureWarning` and `DeprecationWarning` of pandas, scikit-learn, numpy, Matplotlib... raised from skforecast code or from the code of a guide: pending work, with the file and line that triggers it;
   - skforecast warnings, which the runner suppresses from the per-notebook logs and counts in the final summary: say if one is new or appears where it makes no sense;
   - third-party warnings that end up in the published page and do not depend on the guide: candidates for `REMOVED_WARNINGS` (propose, do not change).
3. **Outputs that change meaningfully** compared with the committed ones: metrics, tables, number of rows, figures that disappear. Give the reason: a change of this release (from the release notes), new versions of the dependencies, or randomness (unseeded code). Differences in the last decimal do not count.
4. **Sentences of a guide that quote a result that no longer matches** its new output ("the MAE drops to 12.3", "the best model is..."). List the sentence, the old and the new value and the proposed text. Do not edit the text without approval; once approved, keep it short (`.claude/rules/docs.md`, "Keep user guides simple").
5. At the end, `python tools/docs/execute_notebooks/execute_notebooks.py --check`: error outputs and unexecuted cells of every notebook in scope, excluded ones included, and leftover `*_temp_exec.ipynb` files.

## 4. Commit

- The notebooks that ran without errors, in one or a few commits per folder (`Re-execute the user guides with ...`). Restore the committed version of the notebooks that failed (`git restore <notebook>`): they are not committed half-executed.
- Run the `verify` skill before each commit. No AI attribution.
- No release note: re-executing a notebook is not a user-facing change. Approved text fixes in a guide do not need one either, unless they change what the guide teaches.

## 5. Report

In the language of the conversation:

- Duration of the run and the versions in the log header (Python, skforecast, pandas, numpy, scikit-learn, lightgbm, keras, torch).
- A table of notebooks with their status: ok, with warnings, failed, over the time limit, not executed (and why).
- Failures with their cause.
- Warnings grouped, each with its proposal.
- Outputs that changed and why.
- Sentences that no longer match, with the proposed text.
- What is left: notebooks not executed, decisions for the user, commits not pushed.
